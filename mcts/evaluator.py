"""Batched evaluator contract and PyTorch policy-value implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import torch

from game import GameState
from mcts.errors import InvalidEvaluationError
from mcts.gpu_ops import GpuMctsOps
from network import (
    NetworkSpecification,
    PolicyValueNetwork,
    encode_batch,
    encode_batch_gpu,
    encode_packed_batch,
    masked_policy_probabilities,
)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One masked policy and current-player value in ``[-1, 1]``."""

    policy: tuple[float, ...]
    value: float


@dataclass(frozen=True, slots=True)
class ScorePrediction:
    """One neural estimate of the final Black and White territory scores."""

    black_score: float
    white_score: float

    @property
    def margin(self) -> float:
        """Return the estimated Black-minus-White score margin."""

        return self.black_score - self.white_score


@dataclass(frozen=True, slots=True)
class PreparedEvaluationBatch:
    """CPU-resident features ready for the serialized device stage."""

    features: torch.Tensor
    legal_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class CompletedEvaluationBatch:
    """Model outputs ready for Python result materialization.

    The tensors may remain on CUDA until ``materialize`` is called.  Keeping
    them device-resident lets the shared scheduler release its lease before
    the unavoidable CPU conversion needed by native MCTS.
    """

    policy: torch.Tensor
    values: torch.Tensor


@dataclass(slots=True)
class EvaluatorTiming:
    """Optional accumulated timings for one centralized evaluator lifetime.

    Values are collected only by explicitly profiled benchmarks. CUDA timings
    use CUDA events, while CPU-side stages use ``perf_counter``.
    """

    feature_encoding_seconds: float = 0.0
    gpu_model_seconds: float = 0.0
    gpu_postprocess_seconds: float = 0.0
    host_transfer_seconds: float = 0.0
    result_materialization_seconds: float = 0.0
    batches: int = 0
    positions: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-compatible copy of the accumulated timings."""

        return {
            "feature_encoding_seconds": self.feature_encoding_seconds,
            "gpu_model_seconds": self.gpu_model_seconds,
            "gpu_postprocess_seconds": self.gpu_postprocess_seconds,
            "host_transfer_seconds": self.host_transfer_seconds,
            "result_materialization_seconds": self.result_materialization_seconds,
            "batches": self.batches,
            "positions": self.positions,
        }


class BatchEvaluator(Protocol):
    """Interface used by MCTS to evaluate one non-empty state batch."""

    def evaluate(self, states: Sequence[GameState]) -> Sequence[Evaluation]:
        """Return evaluations in the exact input order."""


class TorchBatchEvaluator:
    """Run version-compatible PyTorch inference with optional CUDA AMP."""

    def __init__(
        self,
        model: PolicyValueNetwork,
        specification: NetworkSpecification,
        *,
        device: torch.device | str,
        use_amp: bool,
        timing: EvaluatorTiming | None = None,
    ) -> None:
        specification.validate()
        model.specification.ensure_compatible(specification)
        self._device = torch.device(device)
        if use_amp and self._device.type != "cuda":
            raise InvalidEvaluationError("AMP inference currently requires CUDA")
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise InvalidEvaluationError("CUDA inference requested but unavailable")
        self._model = model.to(self._device).eval()
        self._specification = specification
        self._use_amp = use_amp
        self._batch_count = 0
        self._position_count = 0
        self._timing = timing
        # The CUDA MCTS helper is owned by the evaluator so every search
        # sharing this model uses the same device and counters.  It performs
        # dense leaf-batch arithmetic and root PUCT post-processing; the
        # mutable tree remains in the Native C++ extension.
        self._gpu_mcts = GpuMctsOps(self._device)

    @property
    def device(self) -> torch.device:
        """Return the evaluator's execution device."""

        return self._device

    @property
    def gpu_mcts_ops(self) -> GpuMctsOps | None:
        """Return CUDA MCTS tensor operations when CUDA is active."""

        return self._gpu_mcts if self._gpu_mcts.enabled else None

    @property
    def batch_count(self) -> int:
        """Return the lifetime number of completed evaluator calls."""

        return self._batch_count

    @property
    def position_count(self) -> int:
        """Return the lifetime number of evaluated positions."""

        return self._position_count

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        """Encode, infer, mask, and transfer one state batch to CPU values."""

        prepared = self.prepare(states)
        completed = self.evaluate_prepared(prepared)
        return self.materialize(completed)

    def prepare_on_device(self, states: Sequence[GameState]) -> PreparedEvaluationBatch:
        """Build all leaf feature planes directly on the CUDA device."""

        if not self._gpu_mcts.enabled:
            return self.prepare(states)
        if not states:
            raise InvalidEvaluationError("Cannot evaluate an empty state batch")
        features = encode_batch_gpu(states, self._specification, device=self._device)
        return PreparedEvaluationBatch(
            features=features,
            legal_mask=features[:, 2].flatten(start_dim=1).to(dtype=torch.bool),
        )

    def evaluate_on_device(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        """Convenience API for a CUDA-native feature/inference round."""

        return self.materialize(self.evaluate_prepared(self.prepare_on_device(states)))

    def prepare(self, states: Sequence[GameState]) -> PreparedEvaluationBatch:
        """Encode states on CPU without occupying a shared CUDA scheduler."""

        if not states:
            raise InvalidEvaluationError("Cannot evaluate an empty state batch")
        encoding_started = perf_counter() if self._timing is not None else 0.0
        features = encode_batch(states, self._specification, device="cpu")
        legal_mask = torch.tensor(
            [state.legal_mask for state in states],
            dtype=torch.bool,
            device="cpu",
        )
        if self._timing is not None:
            self._record_feature_encoding(perf_counter() - encoding_started)
        return PreparedEvaluationBatch(features=features, legal_mask=legal_mask)

    def evaluate_packed(self, payload: bytes) -> tuple[Evaluation, ...]:
        """Evaluate compact self-play states without rebuilding Python objects."""

        prepared = self.prepare_packed(payload)
        completed = self.evaluate_prepared(prepared)
        return self.materialize(completed)

    def prepare_packed(self, payload: bytes) -> PreparedEvaluationBatch:
        """Decode compact states into CPU features outside the device stage."""

        encoding_started = perf_counter() if self._timing is not None else 0.0
        try:
            features = encode_packed_batch(payload, self._specification, device="cpu")
        except Exception as error:
            from network.errors import FeatureEncodingError

            if not isinstance(error, FeatureEncodingError):
                raise
            from training.inference_transport import unpack_state_batch

            return self.prepare(
                unpack_state_batch(
                    payload,
                    ruleset_id=self._specification.ruleset_id,
                    board_size=self._specification.board_size,
                )
            )
        legal_mask = features[:, 2].flatten(start_dim=1).to(dtype=torch.bool)
        if self._timing is not None:
            self._record_feature_encoding(perf_counter() - encoding_started)
        return PreparedEvaluationBatch(features=features, legal_mask=legal_mask)

    def evaluate_prepared(
        self, batch: PreparedEvaluationBatch
    ) -> CompletedEvaluationBatch:
        """Run transfer, CUDA inference, and masking without host sync."""

        features = batch.features.to(device=self._device)
        legal_mask = batch.legal_mask.to(device=self._device)
        return self._evaluate_features(features, legal_mask)

    def materialize(self, batch: CompletedEvaluationBatch) -> tuple[Evaluation, ...]:
        """Convert CPU tensors to evaluator objects outside the CUDA lease."""

        materialization_started = perf_counter() if self._timing is not None else 0.0
        transfer_started = perf_counter() if self._timing is not None else 0.0
        policy_cpu = batch.policy.detach().cpu()
        values_cpu = batch.values.detach().cpu()
        if self._timing is not None:
            self._record_host_transfer(perf_counter() - transfer_started)
        batch_size = int(policy_cpu.shape[0])
        evaluations = tuple(
            Evaluation(
                policy=tuple(float(value) for value in policy_cpu[index].tolist()),
                value=float(values_cpu[index].item()),
            )
            for index in range(batch_size)
        )
        if self._timing is not None:
            self._record_materialization(
                perf_counter() - materialization_started, batch_size
            )
        return evaluations

    def predict_scores(
        self, states: Sequence[GameState]
    ) -> tuple[ScorePrediction, ...]:
        """Predict final territory scores without changing MCTS value semantics."""

        if not states:
            raise InvalidEvaluationError("Cannot predict scores for an empty batch")
        features = encode_batch(states, self._specification, device=self._device)
        with torch.inference_mode():
            if self._use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self._model(features)
            else:
                output = self._model(features)

        normalizer = self._specification.score_normalizer
        black_scores = output.black_score.float().cpu() * normalizer
        white_scores = output.white_score.float().cpu() * normalizer
        self._batch_count += 1
        self._position_count += len(states)
        return tuple(
            ScorePrediction(
                black_score=float(black_score.item()),
                white_score=float(white_score.item()),
            )
            for black_score, white_score in zip(black_scores, white_scores, strict=True)
        )

    def _evaluate_features(
        self, features: torch.Tensor, legal_mask: torch.Tensor
    ) -> CompletedEvaluationBatch:
        """Run one already-encoded, legal-masked inference batch."""

        if self._timing is None or self._device.type != "cuda":
            with torch.inference_mode():
                if self._use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        policy, values = self._forward(features, legal_mask)
                else:
                    policy, values = self._forward(features, legal_mask)
        else:
            policy, values = self._profiled_cuda_forward(features, legal_mask)

        self._batch_count += 1
        batch_size = int(features.shape[0])
        self._position_count += batch_size
        return CompletedEvaluationBatch(policy=policy, values=values)

    def _profiled_cuda_forward(
        self, features: torch.Tensor, legal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Measure CUDA model and post-processing stages without changing outputs."""

        model_started = torch.cuda.Event(enable_timing=True)
        model_finished = torch.cuda.Event(enable_timing=True)
        postprocess_finished = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            model_started.record()
            if self._use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self._model(features)
                    model_finished.record()
                    policy = masked_policy_probabilities(
                        output.policy_logits.float(), legal_mask
                    )
                    values = output.win_probability.float() * 2.0 - 1.0
                    postprocess_finished.record()
            else:
                output = self._model(features)
                model_finished.record()
                policy = masked_policy_probabilities(output.policy_logits, legal_mask)
                values = output.win_probability * 2.0 - 1.0
                postprocess_finished.record()
        torch.cuda.synchronize(self._device)
        self._timing.gpu_model_seconds += (
            model_started.elapsed_time(model_finished) / 1_000.0
        )
        self._timing.gpu_postprocess_seconds += (
            model_finished.elapsed_time(postprocess_finished) / 1_000.0
        )
        return policy, values

    def _record_feature_encoding(self, elapsed_seconds: float) -> None:
        if self._timing is not None:
            self._timing.feature_encoding_seconds += elapsed_seconds

    def _record_host_transfer(self, elapsed_seconds: float) -> None:
        if self._timing is not None:
            self._timing.host_transfer_seconds += elapsed_seconds

    def _record_materialization(self, elapsed_seconds: float, positions: int) -> None:
        if self._timing is not None:
            self._timing.result_materialization_seconds += elapsed_seconds
            self._timing.batches += 1
            self._timing.positions += positions

    def _forward(
        self, features: torch.Tensor, legal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self._model(features)
        policy = masked_policy_probabilities(output.policy_logits.float(), legal_mask)
        values = output.win_probability.float() * 2.0 - 1.0
        return policy, values

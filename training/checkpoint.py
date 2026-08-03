"""Immutable, checksummed checkpoint bundles with atomic aliases."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

import torch
from torch.optim import Optimizer

from config.schema import AppConfig
from network import NetworkSpecification, PolicyValueNetwork
from training.errors import CheckpointCompatibilityError, CheckpointError

CHECKPOINT_SCHEMA_VERSION = 2
_STATE_FILE_NAME = "state.pt"
_MANIFEST_FILE_NAME = "manifest.json"
_ALIAS_SCHEMA_VERSION = 1
_SUPPORTED_ALIASES = frozenset({"latest", "best"})
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MANIFEST_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "checkpoint_id",
        "created_at",
        "iteration",
        "network_specification",
        "configuration",
        "metrics",
        "parent_checkpoint_id",
        "state_file",
        "state_sha256",
        "has_optimizer",
        "has_scheduler",
        "python_random_state",
    }
)
_MANIFEST_FIELDS_V2 = _MANIFEST_FIELDS_V1 | {"has_scaler"}
_STATE_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "torch_rng_state",
        "cuda_rng_states",
    }
)
_STATE_FIELDS_V2 = _STATE_FIELDS_V1 | {"scaler_state"}
_SUPPORTED_CHECKPOINT_SCHEMAS = frozenset({1, CHECKPOINT_SCHEMA_VERSION})

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
MetricValue = str | int | float | bool | None


class Stateful(Protocol):
    """Minimal optimizer-like state contract used by learning-rate schedulers."""

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> object: ...


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Validated human-readable metadata for one immutable checkpoint."""

    schema_version: int
    checkpoint_id: str
    created_at: str
    iteration: int
    network_specification: NetworkSpecification
    configuration: dict[str, JsonValue]
    metrics: dict[str, MetricValue]
    parent_checkpoint_id: str | None
    state_sha256: str
    has_optimizer: bool
    has_scheduler: bool
    has_scaler: bool


@dataclass(frozen=True, slots=True)
class _ValidatedRngState:
    python: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]


class CheckpointManager:
    """Publish and restore immutable checkpoint bundles under one run root."""

    __slots__ = ("_aliases_directory", "_bundles_directory", "_directory", "_spec")

    def __init__(self, directory: Path, specification: NetworkSpecification) -> None:
        specification.validate()
        self._directory = directory.resolve()
        self._bundles_directory = self._directory / "bundles"
        self._aliases_directory = self._directory / "aliases"
        self._spec = specification

    def save(
        self,
        checkpoint_id: str,
        *,
        model: PolicyValueNetwork,
        optimizer: Optimizer | None,
        scheduler: Stateful | None,
        scaler: Stateful | None = None,
        iteration: int,
        config: AppConfig,
        metrics: Mapping[str, MetricValue],
        parent_checkpoint_id: str | None = None,
        aliases: Sequence[str] = ("latest",),
    ) -> CheckpointMetadata:
        """Write an immutable bundle, then atomically publish requested aliases."""

        _validate_identifier(checkpoint_id, "checkpoint_id")
        if checkpoint_id in _SUPPORTED_ALIASES:
            raise CheckpointError("Checkpoint ID must not use a reserved alias")
        if type(iteration) is not int or iteration < 0:
            raise CheckpointError("Checkpoint iteration must be nonnegative")
        if parent_checkpoint_id is not None:
            _validate_identifier(parent_checkpoint_id, "parent_checkpoint_id")
            if not (
                self._bundle_path(parent_checkpoint_id) / _MANIFEST_FILE_NAME
            ).is_file():
                raise CheckpointError(
                    f"Parent checkpoint does not exist: {parent_checkpoint_id}"
                )
        validated_aliases = tuple(_validate_alias(alias) for alias in aliases)
        if len(set(validated_aliases)) != len(validated_aliases):
            raise CheckpointError("Checkpoint aliases must be unique")
        self._validate_model_and_config(model, config)
        clean_metrics = _validate_metrics(metrics)

        self._bundles_directory.mkdir(parents=True, exist_ok=True)
        self._aliases_directory.mkdir(parents=True, exist_ok=True)
        destination = self._bundle_path(checkpoint_id)
        if destination.exists():
            raise CheckpointError(f"Checkpoint already exists: {checkpoint_id}")
        temporary = self._bundles_directory / (
            f".{checkpoint_id}.{uuid4().hex}.temporary"
        )
        temporary.mkdir()
        try:
            state_path = temporary / _STATE_FILE_NAME
            state_payload: dict[str, object] = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_state": model.state_dict(),
                "optimizer_state": (
                    optimizer.state_dict() if optimizer is not None else None
                ),
                "scheduler_state": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            }
            torch.save(state_payload, state_path)
            _fsync_file(state_path)
            state_sha256 = _sha256_file(state_path)
            manifest: dict[str, JsonValue] = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_id": checkpoint_id,
                "created_at": datetime.now(UTC).isoformat(),
                "iteration": iteration,
                "network_specification": _to_json_value(self._spec.to_dict()),
                "configuration": _to_json_value(config),
                "metrics": _to_json_value(clean_metrics),
                "parent_checkpoint_id": parent_checkpoint_id,
                "state_file": _STATE_FILE_NAME,
                "state_sha256": state_sha256,
                "has_optimizer": optimizer is not None,
                "has_scheduler": scheduler is not None,
                "has_scaler": scaler is not None,
                "python_random_state": _to_json_value(random.getstate()),
            }
            manifest_path = temporary / _MANIFEST_FILE_NAME
            _write_json_file(manifest_path, manifest)
            os.replace(temporary, destination)
        except CheckpointError:
            _remove_temporary_directory(temporary, self._bundles_directory)
            raise
        except Exception as error:
            _remove_temporary_directory(temporary, self._bundles_directory)
            raise CheckpointError(
                f"Failed to save checkpoint: {checkpoint_id}"
            ) from error

        metadata = self.read_metadata(checkpoint_id)
        for alias in validated_aliases:
            self.publish_alias(alias, checkpoint_id)
        return metadata

    def load(
        self,
        identifier: str,
        *,
        model: PolicyValueNetwork,
        optimizer: Optimizer | None = None,
        scheduler: Stateful | None = None,
        scaler: Stateful | None = None,
        restore_rng: bool = True,
        map_location: torch.device | str = "cpu",
    ) -> CheckpointMetadata:
        """Verify one bundle before restoring requested state and RNG streams."""

        checkpoint_id = self.resolve_identifier(identifier)
        metadata, manifest = self._read_metadata_and_manifest(checkpoint_id)
        try:
            model.specification.ensure_compatible(self._spec)
        except Exception as error:
            raise CheckpointCompatibilityError(
                "Target model specification is incompatible"
            ) from error
        bundle = self._bundle_path(checkpoint_id)
        state_path = bundle / _STATE_FILE_NAME
        if _sha256_file(state_path) != metadata.state_sha256:
            raise CheckpointError("Checkpoint state checksum does not match manifest")

        try:
            raw_payload = torch.load(
                state_path, map_location=map_location, weights_only=True
            )
            payload = _validate_state_payload(raw_payload)
            if payload["schema_version"] != metadata.schema_version:
                raise CheckpointError(
                    "Checkpoint manifest and state schema versions disagree"
                )
            optimizer_state = payload["optimizer_state"]
            scheduler_state = payload["scheduler_state"]
            scaler_state = payload["scaler_state"]
            if metadata.has_optimizer is not (optimizer_state is not None):
                raise CheckpointError(
                    "Checkpoint optimizer metadata disagrees with tensor state"
                )
            if metadata.has_scheduler is not (scheduler_state is not None):
                raise CheckpointError(
                    "Checkpoint scheduler metadata disagrees with tensor state"
                )
            if metadata.has_scaler is not (scaler_state is not None):
                raise CheckpointError(
                    "Checkpoint scaler metadata disagrees with tensor state"
                )
            if optimizer is not None and optimizer_state is None:
                raise CheckpointCompatibilityError(
                    "Checkpoint does not contain optimizer state"
                )
            if scheduler is not None and scheduler_state is None:
                raise CheckpointCompatibilityError(
                    "Checkpoint does not contain scheduler state"
                )
            if scaler is not None and scaler_state is None:
                raise CheckpointCompatibilityError(
                    "Checkpoint does not contain AMP scaler state"
                )
            model_state = _required_mapping(payload, "model_state")
            validated_rng = _validate_rng(payload, manifest) if restore_rng else None
            with torch.random.fork_rng(devices=[]):
                validation_model = PolicyValueNetwork(self._spec)
                validation_model.load_state_dict(model_state, strict=True)
            model.load_state_dict(model_state, strict=True)
            if optimizer is not None:
                optimizer.load_state_dict(
                    cast(dict[str, Any], _required_mapping(payload, "optimizer_state"))
                )
            if scheduler is not None:
                scheduler.load_state_dict(
                    cast(dict[str, Any], _required_mapping(payload, "scheduler_state"))
                )
            if scaler is not None:
                scaler.load_state_dict(
                    cast(dict[str, Any], _required_mapping(payload, "scaler_state"))
                )
            if validated_rng is not None:
                _apply_rng(validated_rng)
        except CheckpointError:
            raise
        except Exception as error:
            raise CheckpointError(
                f"Failed to load checkpoint state: {checkpoint_id}"
            ) from error
        return metadata

    def read_metadata(self, identifier: str) -> CheckpointMetadata:
        """Read and validate metadata without loading tensor state."""

        checkpoint_id = self.resolve_identifier(identifier)
        metadata, _ = self._read_metadata_and_manifest(checkpoint_id)
        return metadata

    def publish_alias(self, alias: str, checkpoint_id: str) -> None:
        """Atomically point ``latest`` or ``best`` at an existing bundle."""

        alias = _validate_alias(alias)
        _validate_identifier(checkpoint_id, "checkpoint_id")
        bundle = self._bundle_path(checkpoint_id)
        manifest_path = bundle / _MANIFEST_FILE_NAME
        if not manifest_path.is_file():
            raise CheckpointError(f"Checkpoint does not exist: {checkpoint_id}")
        pointer: dict[str, JsonValue] = {
            "schema_version": _ALIAS_SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "manifest_sha256": _sha256_file(manifest_path),
        }
        self._aliases_directory.mkdir(parents=True, exist_ok=True)
        destination = self._aliases_directory / f"{alias}.json"
        temporary = destination.with_name(
            f".{destination.name}.{uuid4().hex}.temporary"
        )
        try:
            _write_json_file(temporary, pointer)
            os.replace(temporary, destination)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise CheckpointError(
                f"Failed to publish checkpoint alias: {alias}"
            ) from error

    def has_alias(self, alias: str) -> bool:
        """Return whether an alias pointer exists without masking corruption."""

        validated = _validate_alias(alias)
        return (self._aliases_directory / f"{validated}.json").is_file()

    def resolve_identifier(self, identifier: str) -> str:
        """Resolve a supported alias or validate an immutable checkpoint ID."""

        if identifier not in _SUPPORTED_ALIASES:
            _validate_identifier(identifier, "checkpoint identifier")
            return identifier
        pointer_path = self._aliases_directory / f"{identifier}.json"
        pointer = _read_json_object(pointer_path, "checkpoint alias")
        if set(pointer) != {"schema_version", "checkpoint_id", "manifest_sha256"}:
            raise CheckpointError("Checkpoint alias fields are incomplete or unknown")
        if _exact_int(pointer, "schema_version") != _ALIAS_SCHEMA_VERSION:
            raise CheckpointError("Checkpoint alias schema version is unsupported")
        checkpoint_id = _exact_string(pointer, "checkpoint_id")
        _validate_identifier(checkpoint_id, "checkpoint_id")
        manifest_path = self._bundle_path(checkpoint_id) / _MANIFEST_FILE_NAME
        if not manifest_path.is_file():
            raise CheckpointError("Checkpoint alias points to a missing bundle")
        if _sha256_file(manifest_path) != _exact_string(pointer, "manifest_sha256"):
            raise CheckpointError("Checkpoint alias manifest checksum is invalid")
        return checkpoint_id

    def list_checkpoint_ids(self) -> tuple[str, ...]:
        """Return valid immutable bundle IDs in stable lexical order."""

        if not self._bundles_directory.exists():
            return ()
        return tuple(
            sorted(
                path.name
                for path in self._bundles_directory.iterdir()
                if path.is_dir()
                and _IDENTIFIER_PATTERN.fullmatch(path.name) is not None
                and (path / _MANIFEST_FILE_NAME).is_file()
            )
        )

    def _validate_model_and_config(
        self, model: PolicyValueNetwork, config: AppConfig
    ) -> None:
        try:
            model.specification.ensure_compatible(self._spec)
            configured = NetworkSpecification.from_config(config.rules, config.network)
            configured.ensure_compatible(self._spec)
        except Exception as error:
            raise CheckpointCompatibilityError(
                "Checkpoint model or configuration is incompatible"
            ) from error

    def _read_metadata_and_manifest(
        self, checkpoint_id: str
    ) -> tuple[CheckpointMetadata, dict[str, JsonValue]]:
        _validate_identifier(checkpoint_id, "checkpoint_id")
        manifest_path = self._bundle_path(checkpoint_id) / _MANIFEST_FILE_NAME
        manifest = _read_json_object(manifest_path, "checkpoint manifest")
        schema_version = _exact_int(manifest, "schema_version")
        expected_fields = {
            1: _MANIFEST_FIELDS_V1,
            CHECKPOINT_SCHEMA_VERSION: _MANIFEST_FIELDS_V2,
        }.get(schema_version)
        if expected_fields is None:
            raise CheckpointError("Checkpoint schema version is unsupported")
        if set(manifest) != expected_fields:
            raise CheckpointError(
                "Checkpoint manifest fields are incomplete or unknown"
            )
        if _exact_string(manifest, "checkpoint_id") != checkpoint_id:
            raise CheckpointError("Checkpoint manifest ID does not match its directory")
        if _exact_string(manifest, "state_file") != _STATE_FILE_NAME:
            raise CheckpointError("Checkpoint state file name is unsupported")
        try:
            specification = NetworkSpecification.from_dict(
                cast(
                    dict[str, object],
                    _exact_object(manifest, "network_specification"),
                )
            )
            self._spec.ensure_compatible(specification)
        except Exception as error:
            raise CheckpointCompatibilityError(
                "Checkpoint network specification is incompatible"
            ) from error

        iteration = _exact_int(manifest, "iteration")
        if iteration < 0:
            raise CheckpointError("Checkpoint iteration must be nonnegative")
        parent = manifest.get("parent_checkpoint_id")
        if parent is not None:
            if not isinstance(parent, str):
                raise CheckpointError("Checkpoint parent ID must be a string or null")
            _validate_identifier(parent, "parent_checkpoint_id")
        metrics_raw = _exact_object(manifest, "metrics")
        metrics = _validate_metrics(metrics_raw)
        configuration = _exact_object(manifest, "configuration")
        return (
            CheckpointMetadata(
                schema_version=schema_version,
                checkpoint_id=checkpoint_id,
                created_at=_exact_string(manifest, "created_at"),
                iteration=iteration,
                network_specification=specification,
                configuration=configuration,
                metrics=metrics,
                parent_checkpoint_id=parent,
                state_sha256=_validate_sha256(_exact_string(manifest, "state_sha256")),
                has_optimizer=_exact_bool(manifest, "has_optimizer"),
                has_scheduler=_exact_bool(manifest, "has_scheduler"),
                has_scaler=(
                    _exact_bool(manifest, "has_scaler")
                    if schema_version >= 2
                    else False
                ),
            ),
            manifest,
        )

    def _bundle_path(self, checkpoint_id: str) -> Path:
        return self._bundles_directory / checkpoint_id


def _validate_state_payload(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise CheckpointError("Checkpoint state payload must be a string-keyed mapping")
    payload = cast(dict[str, object], raw)
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version not in (
        _SUPPORTED_CHECKPOINT_SCHEMAS
    ):
        raise CheckpointError("Checkpoint state schema version is unsupported")
    expected_fields = (
        _STATE_FIELDS_V2
        if schema_version == CHECKPOINT_SCHEMA_VERSION
        else _STATE_FIELDS_V1
    )
    if set(payload) != expected_fields:
        raise CheckpointError("Checkpoint state fields are incomplete or unknown")
    if not isinstance(payload.get("torch_rng_state"), torch.Tensor):
        raise CheckpointError("Checkpoint torch RNG state must be a tensor")
    cuda_states = payload.get("cuda_rng_states")
    if not isinstance(cuda_states, list) or any(
        not isinstance(state, torch.Tensor) for state in cuda_states
    ):
        raise CheckpointError("Checkpoint CUDA RNG states must be tensors")
    if schema_version == 1:
        payload = {**payload, "scaler_state": None}
    return payload


def _validate_rng(
    payload: dict[str, object], manifest: dict[str, JsonValue]
) -> _ValidatedRngState:
    python_state = _decode_python_random_state(manifest.get("python_random_state"))
    random_probe = random.Random()
    random_probe.setstate(python_state)
    torch_rng_state = cast(torch.Tensor, payload["torch_rng_state"]).cpu()
    torch_probe = torch.Generator(device="cpu")
    torch_probe.set_state(torch_rng_state)
    cuda_states = cast(list[torch.Tensor], payload["cuda_rng_states"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise CheckpointCompatibilityError(
                "Checkpoint has CUDA RNG state but CUDA is unavailable"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise CheckpointCompatibilityError(
                "Checkpoint CUDA RNG device count does not match this host"
            )
    return _ValidatedRngState(
        python=python_state,
        torch_cpu=torch_rng_state,
        torch_cuda=tuple(state.cpu() for state in cuda_states),
    )


def _apply_rng(state: _ValidatedRngState) -> None:
    random.setstate(state.python)
    torch.set_rng_state(state.torch_cpu)
    if state.torch_cuda:
        torch.cuda.set_rng_state_all(list(state.torch_cuda))


def _decode_python_random_state(value: JsonValue | None) -> tuple[Any, ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise CheckpointError("Checkpoint Python RNG state is invalid")
    version, internal, gaussian = value
    if type(version) is not int or not isinstance(internal, list):
        raise CheckpointError("Checkpoint Python RNG state is invalid")
    if any(type(item) is not int for item in internal):
        raise CheckpointError("Checkpoint Python RNG state is invalid")
    if gaussian is not None and not isinstance(gaussian, (int, float)):
        raise CheckpointError("Checkpoint Python RNG state is invalid")
    return (version, tuple(internal), gaussian)


def _required_mapping(payload: dict[str, object], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping) or any(
        not isinstance(item, str) for item in value
    ):
        raise CheckpointCompatibilityError(
            f"Checkpoint {key} must be a string-keyed mapping"
        )
    return cast(Mapping[str, Any], value)


def _validate_metrics(metrics: Mapping[str, object]) -> dict[str, MetricValue]:
    validated: dict[str, MetricValue] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise CheckpointError("Checkpoint metric names must be non-empty strings")
        if value is not None and type(value) not in {str, int, float, bool}:
            raise CheckpointError(f"Checkpoint metric {key} has an unsupported value")
        if isinstance(value, float) and not isfinite(value):
            raise CheckpointError(f"Checkpoint metric {key} must be finite")
        validated[key] = cast(MetricValue, value)
    return validated


def _to_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise CheckpointError("Checkpoint JSON values must be finite")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _to_json_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        converted: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointError("Checkpoint JSON object keys must be strings")
            converted[key] = _to_json_value(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    raise CheckpointError(
        f"Checkpoint value is not JSON serializable: {type(value).__name__}"
    )


def _write_json_file(path: Path, content: Mapping[str, JsonValue]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(content, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _read_json_object(path: Path, description: str) -> dict[str, JsonValue]:
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"Cannot read {description}: {path}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CheckpointError(f"{description.capitalize()} must be a JSON object")
    return cast(dict[str, JsonValue], value)


def _exact_object(mapping: Mapping[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise CheckpointError(f"Checkpoint {key} must be an object")
    return value


def _exact_string(mapping: Mapping[str, JsonValue], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise CheckpointError(f"Checkpoint {key} must be a non-empty string")
    return value


def _exact_int(mapping: Mapping[str, JsonValue], key: str) -> int:
    value = mapping.get(key)
    if type(value) is not int:
        raise CheckpointError(f"Checkpoint {key} must be an integer")
    return value


def _exact_bool(mapping: Mapping[str, JsonValue], key: str) -> bool:
    value = mapping.get(key)
    if type(value) is not bool:
        raise CheckpointError(f"Checkpoint {key} must be a boolean")
    return value


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise CheckpointError(
            f"{name} must contain only letters, digits, dot, underscore, or hyphen"
        )


def _validate_alias(alias: str) -> str:
    if alias not in _SUPPORTED_ALIASES:
        allowed = ", ".join(sorted(_SUPPORTED_ALIASES))
        raise CheckpointError(f"Checkpoint alias must be one of: {allowed}")
    return alias


def _validate_sha256(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CheckpointError("Checkpoint SHA-256 value is invalid")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CheckpointError(f"Cannot read checkpoint file: {path}") from error
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb+") as source:
        os.fsync(source.fileno())


def _remove_temporary_directory(path: Path, expected_parent: Path) -> None:
    if not path.exists():
        return
    if path.parent != expected_parent or not path.name.endswith(".temporary"):
        raise CheckpointError("Refusing to remove an unexpected checkpoint path")
    shutil.rmtree(path)

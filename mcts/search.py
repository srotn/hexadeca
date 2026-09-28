"""Exact-simulation neural MCTS with batched virtual-loss reservations."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from time import perf_counter

from config.schema import MctsConfig, RulesConfig
from game import Board, GameState, Move, Player, score_terminal
from mcts.endgame import ExactEndgameResult, ExactEndgameSolver
from mcts.errors import InvalidEvaluationError, SearchStateError
from mcts.evaluator import BatchEvaluator, Evaluation
from mcts.node import MctsEdge, MctsNode, backup_path, cancel_path
from mcts.policy import (
    SearchMode,
    add_root_dirichlet_noise,
    configured_temperature,
    sample_action,
    simulation_count,
    visit_probabilities,
)
from native import NativeSearchTree, is_available


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Root statistics and immutable policy arrays from one completed search."""

    root: MctsNode
    root_zobrist_hash: int
    mode: SearchMode
    simulations: int
    visit_counts: tuple[int, ...]
    root_priors: tuple[float, ...]
    policy_target: tuple[float, ...]
    action_probabilities: tuple[float, ...]
    root_value: float
    inference_batches: int
    inference_positions: int
    maximum_batch_size: int
    solved_outcome: int | None = None
    exact_action_outcomes: tuple[tuple[int, int], ...] = ()
    exact_states_evaluated: int = 0
    exact_cache_hits: int = 0

    def select_action(self, random_source: random.Random) -> int:
        """Sample the configured visit distribution using an explicit RNG."""

        return sample_action(self.action_probabilities, random_source)


@dataclass(frozen=True, slots=True)
class _PendingLeaf:
    node: MctsNode
    state: GameState
    nodes: tuple[MctsNode, ...]
    edges: tuple[MctsEdge, ...]
    terminal_value: float | None


@dataclass(slots=True)
class MctsTiming:
    """Optional aggregate timing for one or more MCTS searches.

    All values are worker CPU wall time. In a multi-process self-play run they
    must therefore be interpreted as aggregate worker time, not portions of
    coordinator wall time.
    """

    root_validation_seconds: float = 0.0
    root_setup_seconds: float = 0.0
    tree_selection_seconds: float = 0.0
    leaf_snapshot_seconds: float = 0.0
    evaluator_roundtrip_seconds: float = 0.0
    tree_commit_seconds: float = 0.0
    tree_export_seconds: float = 0.0
    root_statistics_seconds: float = 0.0
    gpu_mcts_seconds: float = 0.0
    gpu_mcts_calls: int = 0
    search_calls: int = 0
    evaluator_batches: int = 0
    evaluator_positions: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Return JSON-compatible aggregate search timing values."""

        return {
            "root_validation_seconds": self.root_validation_seconds,
            "root_setup_seconds": self.root_setup_seconds,
            "tree_selection_seconds": self.tree_selection_seconds,
            "leaf_snapshot_seconds": self.leaf_snapshot_seconds,
            "evaluator_roundtrip_seconds": self.evaluator_roundtrip_seconds,
            "tree_commit_seconds": self.tree_commit_seconds,
            "tree_export_seconds": self.tree_export_seconds,
            "root_statistics_seconds": self.root_statistics_seconds,
            "gpu_mcts_seconds": self.gpu_mcts_seconds,
            "gpu_mcts_calls": self.gpu_mcts_calls,
            "search_calls": self.search_calls,
            "evaluator_batches": self.evaluator_batches,
            "evaluator_positions": self.evaluator_positions,
        }

    def add(self, other: MctsTiming) -> None:
        """Accumulate another worker's independent measurements."""

        self.root_validation_seconds += other.root_validation_seconds
        self.root_setup_seconds += other.root_setup_seconds
        self.tree_selection_seconds += other.tree_selection_seconds
        self.leaf_snapshot_seconds += other.leaf_snapshot_seconds
        self.evaluator_roundtrip_seconds += other.evaluator_roundtrip_seconds
        self.tree_commit_seconds += other.tree_commit_seconds
        self.tree_export_seconds += other.tree_export_seconds
        self.root_statistics_seconds += other.root_statistics_seconds
        self.gpu_mcts_seconds += other.gpu_mcts_seconds
        self.gpu_mcts_calls += other.gpu_mcts_calls
        self.search_calls += other.search_calls
        self.evaluator_batches += other.evaluator_batches
        self.evaluator_positions += other.evaluator_positions


class MctsSearch:
    """Hybrid MCTS facade with native C++ tree state and CUDA leaf work.

    ``engine="native"`` keeps mutable selection, expansion, virtual-loss
    accounting, and backup in :class:`native.NativeSearchTree`.  The selected
    leaf batch is encoded directly on CUDA and evaluated in one GPU lease;
    visit-target normalisation and dense root-score diagnostics also use CUDA.
    This split preserves exact native tree semantics while moving the dense
    part of every MCTS round off the CPU.
    """

    def __init__(
        self,
        rules: RulesConfig,
        config: MctsConfig,
        evaluator: BatchEvaluator,
        *,
        timing: MctsTiming | None = None,
        materialize_native_tree: bool = True,
    ) -> None:
        """Bind validated rules, all MCTS behavior, and one batch evaluator."""

        if type(materialize_native_tree) is not bool:
            raise TypeError("materialize_native_tree must be a bool")
        self._rules = rules
        self._config = config
        self._evaluator = evaluator
        self._timing = timing
        self._materialize_native_tree = materialize_native_tree
        self._endgame_solver = ExactEndgameSolver(rules)
        # ``TorchBatchEvaluator`` and the experiment's scheduled wrapper both
        # expose this optional helper.  Recording/reference evaluators simply
        # leave it absent, preserving the exact CPU path used by tests.
        self._gpu_mcts = getattr(evaluator, "gpu_mcts_ops", None)
        self._exact_states_evaluated = 0
        self._exact_cache_hits = 0

    def run(
        self,
        root_state: GameState,
        mode: SearchMode,
        *,
        random_source: random.Random | None = None,
    ) -> SearchResult:
        """Run the configured number of simulations from one non-terminal state."""

        if not isinstance(mode, SearchMode):
            raise SearchStateError("Search mode must be a SearchMode value")
        self._exact_states_evaluated = 0
        self._exact_cache_hits = 0
        validation_started = perf_counter() if self._timing is not None else 0.0
        root_actions = self._validate_root_state(root_state)
        if self._timing is not None:
            self._timing.root_validation_seconds += perf_counter() - validation_started
            self._timing.search_calls += 1
        if root_state.terminal:
            raise SearchStateError("Cannot search a terminal root state")
        if (
            mode is SearchMode.TRAINING
            and self._config.root_noise_enabled
            and random_source is None
        ):
            raise SearchStateError("Training search requires an explicit random source")
        # Validate native availability before any optional exact-endgame
        # shortcut.  A production caller that selected ``engine="native"``
        # must fail clearly when the extension is missing; it must never
        # silently execute a Python-only search path at the end of a game.
        if self._config.engine == "native" and not is_available():
            raise SearchStateError("The configured native MCTS engine is unavailable")
        root_board = self._replay(root_actions)
        if self._should_solve_exact(root_board):
            return self._run_exact_root(root_state, root_board, mode)
        if self._config.engine == "native":
            return self._run_native(root_state, root_actions, mode, random_source)

        root = MctsNode(
            to_play=root_state.to_play,
            zobrist_hash=root_state.zobrist_hash,
            terminal=False,
        )
        root_evaluation = self._evaluate_timed((root_state,))[0]
        self._expand(root, root_state, root_evaluation)
        inference_batches = 1
        inference_positions = 1
        maximum_batch_size = 1
        if mode is SearchMode.TRAINING and self._config.root_noise_enabled:
            if random_source is None:
                raise SearchStateError("Training search requires a random source")
            add_root_dirichlet_noise(root, self._config, random_source)

        target_simulations = simulation_count(self._config, mode)
        completed_simulations = 0
        while completed_simulations < target_simulations:
            completed_before_batch = completed_simulations
            pending: list[_PendingLeaf] = []
            while (
                len(pending) < self._config.max_inference_batch_size
                and completed_simulations + len(pending) < target_simulations
            ):
                selected = self._select_leaf(root, root_actions)
                if selected is None:
                    break
                if selected.terminal_value is None:
                    pending.append(selected)
                else:
                    backup_path(
                        selected.nodes,
                        selected.edges,
                        selected.terminal_value,
                        virtual_loss=self._config.virtual_loss,
                    )
                    self._propagate_solved(selected.nodes, selected.edges)
                    completed_simulations += 1

            if pending:
                states = tuple(item.state for item in pending)
                try:
                    evaluations = self._evaluate_timed(states)
                except Exception:
                    self._cancel_pending(pending)
                    raise

                for item, evaluation in zip(pending, evaluations, strict=True):
                    item.node.evaluation_in_flight = False
                    self._expand(item.node, item.state, evaluation)
                    backup_path(
                        item.nodes,
                        item.edges,
                        evaluation.value,
                        virtual_loss=self._config.virtual_loss,
                    )
                    completed_simulations += 1
                inference_batches += 1
                inference_positions += len(pending)
                maximum_batch_size = max(maximum_batch_size, len(pending))
            elif completed_simulations == completed_before_batch:
                raise SearchStateError("Search tree made no progress and has no batch")

        solved_outcome = self._finalize_solved_root(root, completed_simulations)
        visit_counts = tuple(
            root.children[action].visit_count if action in root.children else 0
            for action in range(root_state.action_size)
        )
        root_priors = tuple(
            root.children[action].prior if action in root.children else 0.0
            for action in range(root_state.action_size)
        )
        temperature = configured_temperature(self._config, mode, root_state.ply)
        probabilities = self._visit_probabilities(visit_counts, temperature)
        return SearchResult(
            root=root,
            root_zobrist_hash=root_state.zobrist_hash,
            mode=mode,
            simulations=completed_simulations,
            visit_counts=visit_counts,
            root_priors=root_priors,
            policy_target=probabilities,
            action_probabilities=probabilities,
            root_value=root.mean_value,
            inference_batches=inference_batches,
            inference_positions=inference_positions,
            maximum_batch_size=maximum_batch_size,
            solved_outcome=solved_outcome,
            exact_action_outcomes=self._root_solved_actions(root),
            exact_states_evaluated=self._exact_states_evaluated,
            exact_cache_hits=self._exact_cache_hits,
        )

    def _visit_probabilities(
        self, visit_counts: tuple[int, ...], temperature: float
    ) -> tuple[float, ...]:
        """Use CUDA for dense MCTS target normalisation when available."""

        gpu = self._gpu_mcts
        if gpu is None:
            return visit_probabilities(visit_counts, temperature)
        started = perf_counter()
        try:
            result = gpu.visit_probabilities(visit_counts, temperature)
        except (RuntimeError, ValueError):
            # CUDA can disappear during an orderly service shutdown.  The
            # native tree result remains valid, so retain the CPU fallback.
            return visit_probabilities(visit_counts, temperature)
        if self._timing is not None:
            self._timing.gpu_mcts_calls += 1
            self._timing.gpu_mcts_seconds += perf_counter() - started
        return result

    def _score_root_on_gpu(self, root: MctsNode) -> None:
        """Run the complete root PUCT score vector on CUDA for diagnostics.

        Native C++ remains authoritative for reservation/tie-breaking.  This
        call deliberately has no side effects on the tree; it makes the dense
        PUCT arithmetic visible in timing counters and provides a safe seam
        for the future fully CUDA-resident selector.
        """

        gpu = self._gpu_mcts
        if gpu is None or not root.children:
            return
        actions = tuple(sorted(root.children))
        edges = tuple(root.children[action] for action in actions)
        started = perf_counter()
        try:
            gpu.puct_scores(
                priors=tuple(edge.prior for edge in edges),
                visit_counts=tuple(edge.visit_count for edge in edges),
                value_sums=tuple(edge.value_sum for edge in edges),
                virtual_visit_counts=tuple(edge.virtual_visit_count for edge in edges),
                virtual_value_sums=tuple(edge.virtual_value_sum for edge in edges),
                solved_values=tuple(
                    float("nan") if edge.solved_value is None else edge.solved_value
                    for edge in edges
                ),
                parent_visit_count=root.visit_count,
                parent_value_sum=root.value_sum,
                c_puct=self._config.c_puct,
                fpu_reduction=self._config.fpu_reduction,
            )
        except (RuntimeError, ValueError):
            return
        if self._timing is not None:
            self._timing.gpu_mcts_calls += 1
            self._timing.gpu_mcts_seconds += perf_counter() - started

    def _run_exact_root(
        self, root_state: GameState, board: Board, mode: SearchMode
    ) -> SearchResult:
        """Return a proven root result without neural evaluation or PUCT sampling."""

        exact = self._solve_exact(board)
        action_outcomes = dict(exact.action_outcomes)
        optimal_actions = tuple(
            action
            for action, outcome in exact.action_outcomes
            if outcome == exact.outcome
        )
        if not optimal_actions:
            raise SearchStateError("Exact non-terminal root has no optimal action")

        root = MctsNode(
            to_play=root_state.to_play,
            zobrist_hash=root_state.zobrist_hash,
            terminal=False,
            expanded=True,
            solved_value=float(exact.outcome),
        )
        prior = 1.0 / len(optimal_actions)
        root.children = {
            action: MctsEdge(
                prior=prior if action in optimal_actions else 0.0,
                solved_value=float(outcome),
            )
            for action, outcome in exact.action_outcomes
        }
        nominal_simulations = simulation_count(self._config, mode)
        for index in range(nominal_simulations):
            action = optimal_actions[index % len(optimal_actions)]
            edge = root.children[action]
            edge.visit_count += 1
            edge.value_sum += float(action_outcomes[action])
        root.visit_count = nominal_simulations
        root.value_sum = float(nominal_simulations * exact.outcome)

        visit_counts = tuple(
            root.children[action].visit_count if action in root.children else 0
            for action in range(root_state.action_size)
        )
        root_priors = tuple(
            root.children[action].prior if action in root.children else 0.0
            for action in range(root_state.action_size)
        )
        temperature = configured_temperature(self._config, mode, root_state.ply)
        probabilities = self._visit_probabilities(visit_counts, temperature)
        return SearchResult(
            root=root,
            root_zobrist_hash=root_state.zobrist_hash,
            mode=mode,
            simulations=nominal_simulations,
            visit_counts=visit_counts,
            root_priors=root_priors,
            policy_target=probabilities,
            action_probabilities=probabilities,
            root_value=float(exact.outcome),
            inference_batches=0,
            inference_positions=0,
            maximum_batch_size=0,
            solved_outcome=exact.outcome,
            exact_action_outcomes=exact.action_outcomes,
            exact_states_evaluated=self._exact_states_evaluated,
            exact_cache_hits=self._exact_cache_hits,
        )

    def _run_native(
        self,
        root_state: GameState,
        root_actions: tuple[int, ...],
        mode: SearchMode,
        random_source: random.Random | None,
    ) -> SearchResult:
        """Run native tree operations with CUDA leaf evaluation."""

        if NativeSearchTree is None:
            raise SearchStateError(
                "Native MCTS was reported available but did not load"
            )
        root = MctsNode(
            to_play=root_state.to_play,
            zobrist_hash=root_state.zobrist_hash,
            terminal=False,
        )
        root_evaluation = self._evaluate_timed((root_state,))[0]
        setup_started = perf_counter() if self._timing is not None else 0.0
        self._expand(root, root_state, root_evaluation)
        if mode is SearchMode.TRAINING and self._config.root_noise_enabled:
            if random_source is None:
                raise SearchStateError(
                    "Training search requires an explicit random source"
                )
            add_root_dirichlet_noise(root, self._config, random_source)

        root_priors = tuple(
            root.children[action].prior if action in root.children else 0.0
            for action in range(root_state.action_size)
        )
        tree = NativeSearchTree(
            self._rules.board_size,
            self._rules.neighborhood_radius,
            self._rules.zobrist_seed,
            root_actions,
            self._config.c_puct,
            self._config.fpu_reduction,
            self._config.virtual_loss,
        )
        tree.initialize_root(root_priors)
        if self._timing is not None:
            self._timing.root_setup_seconds += perf_counter() - setup_started

        target_simulations = simulation_count(self._config, mode)
        completed_simulations = 0
        inference_batches = 1
        inference_positions = 1
        maximum_batch_size = 1
        while completed_simulations < target_simulations:
            selection_started = perf_counter() if self._timing is not None else 0.0
            selection = tree.select_batch(
                self._config.max_inference_batch_size,
                target_simulations - completed_simulations,
            )
            if self._timing is not None:
                self._timing.tree_selection_seconds += (
                    perf_counter() - selection_started
                )
            leaf_ids = tuple(int(leaf_id) for leaf_id in selection["leaf_ids"])
            terminal_simulations = int(selection["terminal_simulations"])
            completed_simulations += terminal_simulations
            if leaf_ids:
                snapshot_started = perf_counter() if self._timing is not None else 0.0
                exact_by_leaf: dict[int, ExactEndgameResult] = {}
                neural_leaf_ids: list[int] = []
                neural_states: list[GameState] = []
                leaf_states = getattr(tree, "leaf_states", None)
                packed_leaf_states = getattr(tree, "leaf_packed_states", None)
                packed_evaluate = getattr(
                    self._evaluator, "evaluate_native_packed", None
                )
                raw_legal_counts = selection.get("leaf_legal_counts")
                packed_path = (
                    callable(packed_leaf_states)
                    and callable(packed_evaluate)
                    and bool(getattr(self._evaluator, "supports_native_packed", False))
                    and isinstance(raw_legal_counts, list)
                    and len(raw_legal_counts) == len(leaf_ids)
                )
                if packed_path:
                    exact_leaf_ids = tuple(
                        leaf_id
                        for leaf_id, legal_count in zip(
                            leaf_ids, raw_legal_counts, strict=True
                        )
                        if self._config.exact_endgame_enabled
                        and int(legal_count)
                        <= self._config.exact_endgame_max_legal_moves
                    )
                    exact_states: tuple[GameState, ...] = ()
                    if exact_leaf_ids:
                        raw_states = (
                            leaf_states(exact_leaf_ids)
                            if callable(leaf_states)
                            else tuple(
                                tree.leaf_state(leaf_id) for leaf_id in exact_leaf_ids
                            )
                        )
                        exact_states = tuple(
                            self._native_snapshot(raw_state) for raw_state in raw_states
                        )
                    for leaf_id, state in zip(
                        exact_leaf_ids, exact_states, strict=True
                    ):
                        board = self._replay(
                            tuple(move.action for move in state.history)
                        )
                        exact_by_leaf[leaf_id] = self._solve_exact(board)
                    exact_id_set = set(exact_leaf_ids)
                    neural_leaf_ids.extend(
                        leaf_id for leaf_id in leaf_ids if leaf_id not in exact_id_set
                    )
                else:
                    # Compatibility path for an older native extension.  It
                    # still avoids replaying a board unless the leaf's legal
                    # count can actually fall inside the exact-solver horizon.
                    raw_states = (
                        leaf_states(leaf_ids)
                        if callable(leaf_states)
                        else tuple(tree.leaf_state(leaf_id) for leaf_id in leaf_ids)
                    )
                    states = tuple(
                        self._native_snapshot(raw_state) for raw_state in raw_states
                    )
                    for leaf_id, state in zip(leaf_ids, states, strict=True):
                        if (
                            self._config.exact_endgame_enabled
                            and sum(state.legal_mask)
                            <= self._config.exact_endgame_max_legal_moves
                        ):
                            board = self._replay(
                                tuple(move.action for move in state.history)
                            )
                            exact_by_leaf[leaf_id] = self._solve_exact(board)
                        else:
                            neural_leaf_ids.append(leaf_id)
                            neural_states.append(state)
                if self._timing is not None:
                    self._timing.leaf_snapshot_seconds += (
                        perf_counter() - snapshot_started
                    )
                try:
                    if neural_leaf_ids and packed_path:
                        packed_payload = packed_leaf_states(tuple(neural_leaf_ids))
                        evaluations = self._evaluate_native_packed_timed(
                            packed_evaluate,
                            packed_payload,
                            len(neural_leaf_ids),
                        )
                    else:
                        evaluations = (
                            self._evaluate_timed(tuple(neural_states))
                            if neural_states
                            else ()
                        )
                except Exception:
                    tree.cancel(leaf_ids)
                    raise
                committed_count = 0
                try:
                    commit_started = perf_counter() if self._timing is not None else 0.0
                    evaluation_by_leaf = dict(
                        zip(neural_leaf_ids, evaluations, strict=True)
                    )
                    batch_commit = getattr(tree, "commit_batch", None)
                    if not exact_by_leaf and callable(batch_commit):
                        # Probe searches disable the exact solver, so all
                        # selected leaves are neural leaves.  Commit the whole
                        # reservation batch in one native call; C++ validates
                        # every policy before mutating any leaf and then loops
                        # in the same order as the scalar implementation.
                        batch_commit(
                            leaf_ids,
                            tuple(
                                evaluation_by_leaf[leaf_id].policy
                                for leaf_id in leaf_ids
                            ),
                            tuple(
                                evaluation_by_leaf[leaf_id].value
                                for leaf_id in leaf_ids
                            ),
                        )
                        committed_count = len(leaf_ids)
                    else:
                        # Keep scalar ordering when exact leaves are mixed in;
                        # this preserves solved-value propagation semantics.
                        for leaf_id in leaf_ids:
                            exact = exact_by_leaf.get(leaf_id)
                            if exact is not None:
                                tree.solve(leaf_id, float(exact.outcome))
                            else:
                                tree.commit(
                                    leaf_id,
                                    evaluation_by_leaf[leaf_id].policy,
                                    evaluation_by_leaf[leaf_id].value,
                                )
                            committed_count += 1
                    if self._timing is not None:
                        self._timing.tree_commit_seconds += (
                            perf_counter() - commit_started
                        )
                except Exception:
                    # ``commit_batch`` has a full pre-validation pass and is
                    # therefore all-or-nothing; the scalar path increments
                    # after each successful leaf.  Cancel only reservations
                    # that are still active, avoiding the old undefined
                    # ``leaf_id`` variable when a batch call failed before a
                    # loop variable was bound.
                    remaining = leaf_ids[committed_count:]
                    if remaining:
                        tree.cancel(remaining)
                    raise
                completed_simulations += len(leaf_ids)
                if neural_leaf_ids:
                    inference_batches += 1
                    inference_positions += len(neural_leaf_ids)
                    maximum_batch_size = max(maximum_batch_size, len(neural_leaf_ids))
            elif terminal_simulations == 0:
                raise SearchStateError("Search tree made no progress and has no batch")

        if self._materialize_native_tree:
            export_started = perf_counter() if self._timing is not None else 0.0
            native_root = self._native_root(tree.export_nodes())
            if self._timing is not None:
                self._timing.tree_export_seconds += perf_counter() - export_started
        else:
            statistics_started = perf_counter() if self._timing is not None else 0.0
            native_root = self._native_root_statistics(tree.root_statistics())
            if self._timing is not None:
                self._timing.root_statistics_seconds += (
                    perf_counter() - statistics_started
                )
        solved_outcome = self._finalize_solved_root(native_root, completed_simulations)
        visit_counts = tuple(
            native_root.children[action].visit_count
            if action in native_root.children
            else 0
            for action in range(root_state.action_size)
        )
        output_priors = tuple(
            native_root.children[action].prior
            if action in native_root.children
            else 0.0
            for action in range(root_state.action_size)
        )
        temperature = configured_temperature(self._config, mode, root_state.ply)
        probabilities = self._visit_probabilities(visit_counts, temperature)
        return SearchResult(
            root=native_root,
            root_zobrist_hash=root_state.zobrist_hash,
            mode=mode,
            simulations=completed_simulations,
            visit_counts=visit_counts,
            root_priors=output_priors,
            policy_target=probabilities,
            action_probabilities=probabilities,
            root_value=native_root.mean_value,
            inference_batches=inference_batches,
            inference_positions=inference_positions,
            maximum_batch_size=maximum_batch_size,
            solved_outcome=solved_outcome,
            exact_action_outcomes=self._root_solved_actions(native_root),
            exact_states_evaluated=self._exact_states_evaluated,
            exact_cache_hits=self._exact_cache_hits,
        )

    def _native_snapshot(self, raw_state: object) -> GameState:
        """Translate one native leaf into the immutable evaluator contract."""

        if not isinstance(raw_state, dict):
            raise SearchStateError("Native MCTS returned an invalid leaf snapshot")
        actions = tuple(int(action) for action in raw_state["history"])
        history = tuple(
            Move(
                action=action,
                row=action // self._rules.board_size,
                column=action % self._rules.board_size,
                player=(Player.BLACK if index % 2 == 0 else Player.WHITE),
            )
            for index, action in enumerate(actions)
        )
        return GameState(
            ruleset_id=self._rules.ruleset_id,
            board_size=self._rules.board_size,
            cells=tuple(int(cell) for cell in raw_state["cells"]),
            to_play=Player(int(raw_state["to_play"])),
            ply=len(history),
            history=history,
            legal_mask=tuple(bool(value) for value in raw_state["legal_mask"]),
            zobrist_hash=int(raw_state["zobrist_hash"]),
            terminal=bool(raw_state["terminal"]),
        )

    @staticmethod
    def _native_root(raw_nodes: object) -> MctsNode:
        """Materialize immutable-at-search-end Python diagnostics from native nodes."""

        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise SearchStateError("Native MCTS exported no tree nodes")
        if any(not isinstance(raw_node, dict) for raw_node in raw_nodes):
            raise SearchStateError("Native MCTS exported an invalid tree node")
        nodes = [
            MctsNode(
                to_play=Player(int(raw_node["to_play"])),
                zobrist_hash=int(raw_node["zobrist_hash"]),
                terminal=bool(raw_node["terminal"]),
                visit_count=int(raw_node["visit_count"]),
                value_sum=float(raw_node["value_sum"]),
                expanded=bool(raw_node["expanded"]),
                solved_value=(
                    None
                    if not isfinite(float(raw_node["solved_value"]))
                    else float(raw_node["solved_value"])
                ),
            )
            for raw_node in raw_nodes
        ]
        for node, raw_node in zip(nodes, raw_nodes, strict=True):
            children = raw_node["children"]
            if not isinstance(children, dict):
                raise SearchStateError("Native MCTS exported invalid child edges")
            node.children = {
                int(action): MctsEdge(
                    prior=float(edge[0]),
                    child=(nodes[int(edge[1])] if int(edge[1]) >= 0 else None),
                    visit_count=int(edge[2]),
                    value_sum=float(edge[3]),
                    virtual_visit_count=int(edge[4]),
                    virtual_value_sum=float(edge[5]),
                    solved_value=(
                        None if not isfinite(float(edge[6])) else float(edge[6])
                    ),
                )
                for action, edge in children.items()
            }
        return nodes[0]

    @staticmethod
    def _native_root_statistics(raw_statistics: object) -> MctsNode:
        """Materialize root-only native statistics for production self-play."""

        if not isinstance(raw_statistics, dict):
            raise SearchStateError("Native MCTS returned invalid root statistics")
        children = raw_statistics.get("children")
        if not isinstance(children, dict):
            raise SearchStateError("Native MCTS returned invalid root child statistics")
        try:
            root = MctsNode(
                to_play=Player(int(raw_statistics["to_play"])),
                zobrist_hash=int(raw_statistics["zobrist_hash"]),
                terminal=bool(raw_statistics["terminal"]),
                visit_count=int(raw_statistics["visit_count"]),
                value_sum=float(raw_statistics["value_sum"]),
                expanded=bool(raw_statistics["expanded"]),
                solved_value=(
                    None
                    if not isfinite(float(raw_statistics["solved_value"]))
                    else float(raw_statistics["solved_value"])
                ),
            )
            root.children = {
                int(action): MctsEdge(
                    prior=float(edge[0]),
                    visit_count=int(edge[1]),
                    value_sum=float(edge[2]),
                    virtual_visit_count=int(edge[3]),
                    virtual_value_sum=float(edge[4]),
                    solved_value=(
                        None if not isfinite(float(edge[5])) else float(edge[5])
                    ),
                )
                for action, edge in children.items()
            }
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise SearchStateError(
                "Native MCTS returned malformed root statistics"
            ) from error
        return root

    def _select_leaf(
        self, root: MctsNode, root_actions: tuple[int, ...]
    ) -> _PendingLeaf | None:
        board = self._replay(root_actions)
        node = root
        nodes = [root]
        edges: list[MctsEdge] = []
        try:
            while True:
                selected = node.select_edge(
                    self._config.c_puct, self._config.fpu_reduction
                )
                if selected is None:
                    cancel_path(tuple(edges), virtual_loss=self._config.virtual_loss)
                    return None
                action, edge = selected
                edge.reserve(self._config.virtual_loss)
                edges.append(edge)
                board.apply(action)
                state = self._snapshot(board)

                if edge.child is None:
                    edge.child = MctsNode(
                        to_play=state.to_play,
                        zobrist_hash=state.zobrist_hash,
                        terminal=state.terminal,
                    )
                child = edge.child
                self._validate_child(child, state)
                node = child
                nodes.append(node)

                if state.terminal:
                    node.solved_value = self._terminal_value(board)
                    return _PendingLeaf(
                        node=node,
                        state=state,
                        nodes=tuple(nodes),
                        edges=tuple(edges),
                        terminal_value=node.solved_value,
                    )
                if node.solved_value is not None:
                    return _PendingLeaf(
                        node=node,
                        state=state,
                        nodes=tuple(nodes),
                        edges=tuple(edges),
                        terminal_value=node.solved_value,
                    )
                if self._should_solve_exact(board):
                    exact = self._solve_exact(board)
                    node.solved_value = float(exact.outcome)
                    return _PendingLeaf(
                        node=node,
                        state=state,
                        nodes=tuple(nodes),
                        edges=tuple(edges),
                        terminal_value=float(exact.outcome),
                    )
                if not node.expanded:
                    if node.evaluation_in_flight:
                        cancel_path(
                            tuple(edges), virtual_loss=self._config.virtual_loss
                        )
                        return None
                    node.evaluation_in_flight = True
                    return _PendingLeaf(
                        node=node,
                        state=state,
                        nodes=tuple(nodes),
                        edges=tuple(edges),
                        terminal_value=None,
                    )
        except Exception:
            cancel_path(tuple(edges), virtual_loss=self._config.virtual_loss)
            raise

    def _evaluate_exact(self, states: tuple[GameState, ...]) -> tuple[Evaluation, ...]:
        evaluations = tuple(self._evaluator.evaluate(states))
        if len(evaluations) != len(states):
            raise InvalidEvaluationError(
                "Evaluator output count must match its input batch"
            )
        for evaluation, state in zip(evaluations, states, strict=True):
            self._validate_evaluation(evaluation, state)
        return evaluations

    def _evaluate_timed(self, states: tuple[GameState, ...]) -> tuple[Evaluation, ...]:
        """Evaluate states and optionally account for worker round-trip time."""

        if self._timing is None:
            return self._evaluate_exact(states)
        started_at = perf_counter()
        evaluations = self._evaluate_exact(states)
        self._timing.evaluator_roundtrip_seconds += perf_counter() - started_at
        self._timing.evaluator_batches += 1
        self._timing.evaluator_positions += len(states)
        return evaluations

    def _evaluate_native_packed_timed(
        self,
        evaluator: Callable[[bytes, int, int], Sequence[Evaluation]],
        payload: bytes,
        state_count: int,
    ) -> tuple[Evaluation, ...]:
        """Evaluate C++-serialized leaves without materializing Python states."""

        started_at = perf_counter() if self._timing is not None else 0.0
        evaluations = tuple(evaluator(payload, state_count, self._rules.board_size**2))
        if len(evaluations) != state_count:
            raise InvalidEvaluationError(
                "Evaluator output count must match its packed input batch"
            )
        action_size = self._rules.board_size**2
        for evaluation in evaluations:
            if len(evaluation.policy) != action_size:
                raise InvalidEvaluationError(
                    "Evaluation policy length must equal the action size"
                )
            if not isfinite(evaluation.value) or not -1.0 <= evaluation.value <= 1.0:
                raise InvalidEvaluationError(
                    "Evaluation value must be finite and within [-1, 1]"
                )
            if any(
                not isfinite(probability) or probability < 0.0
                for probability in evaluation.policy
            ) or not isclose(sum(evaluation.policy), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise InvalidEvaluationError("Packed evaluation policy is invalid")
        if self._timing is not None:
            self._timing.evaluator_roundtrip_seconds += perf_counter() - started_at
            self._timing.evaluator_batches += 1
            self._timing.evaluator_positions += state_count
        return evaluations

    def _expand(self, node: MctsNode, state: GameState, evaluation: Evaluation) -> None:
        priors = {
            action: evaluation.policy[action]
            for action, legal in enumerate(state.legal_mask)
            if legal
        }
        node.expand(priors)

    def _validate_root_state(self, state: GameState) -> tuple[int, ...]:
        if state.ruleset_id != self._rules.ruleset_id:
            raise SearchStateError("Root ruleset does not match MCTS rules")
        if state.board_size != self._rules.board_size:
            raise SearchStateError("Root board size does not match MCTS rules")
        actions = tuple(move.action for move in state.history)
        try:
            reconstructed = self._snapshot(self._replay(actions))
        except Exception as error:
            raise SearchStateError("Root history cannot be replayed legally") from error
        if reconstructed != state:
            raise SearchStateError("Root snapshot does not match its replayed history")
        return actions

    def _replay(self, actions: tuple[int, ...]) -> Board:
        board = Board(self._rules)
        for action in actions:
            board.apply(action)
        return board

    def _snapshot(self, board: Board) -> GameState:
        return GameState(
            ruleset_id=self._rules.ruleset_id,
            board_size=board.size,
            cells=board.cells,
            to_play=board.to_play,
            ply=board.ply,
            history=board.history,
            legal_mask=board.legal_mask(),
            zobrist_hash=board.zobrist_hash,
            terminal=board.is_terminal,
        )

    @staticmethod
    def _validate_child(node: MctsNode, state: GameState) -> None:
        if (
            node.zobrist_hash != state.zobrist_hash
            or node.to_play is not state.to_play
            or node.terminal is not state.terminal
        ):
            raise SearchStateError("Tree child does not match its replayed state")

    @staticmethod
    def _validate_evaluation(evaluation: Evaluation, state: GameState) -> None:
        if len(evaluation.policy) != state.action_size:
            raise InvalidEvaluationError(
                "Evaluation policy length must equal the action size"
            )
        if not isfinite(evaluation.value) or not -1.0 <= evaluation.value <= 1.0:
            raise InvalidEvaluationError(
                "Evaluation value must be finite and within [-1, 1]"
            )
        if any(
            not isfinite(probability) or probability < 0.0
            for probability in evaluation.policy
        ):
            raise InvalidEvaluationError(
                "Evaluation policy must be finite and nonnegative"
            )
        if any(
            probability != 0.0
            for probability, legal in zip(
                evaluation.policy, state.legal_mask, strict=True
            )
            if not legal
        ):
            raise InvalidEvaluationError(
                "Evaluation policy must assign zero mass to illegal actions"
            )
        if not isclose(sum(evaluation.policy), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise InvalidEvaluationError("Evaluation policy must sum to one")

    def _terminal_value(self, board: Board) -> float:
        result = score_terminal(board)
        if result.winner is None:
            return 0.0
        return 1.0 if result.winner is board.to_play else -1.0

    def _should_solve_exact(self, board: Board) -> bool:
        """Return whether this position falls inside the configured exact horizon."""

        return (
            self._config.exact_endgame_enabled
            and not board.is_terminal
            and board.legal_count <= self._config.exact_endgame_max_legal_moves
        )

    def _solve_exact(self, board: Board) -> ExactEndgameResult:
        exact = self._endgame_solver.solve(board)
        self._exact_states_evaluated += exact.states_evaluated
        self._exact_cache_hits += exact.cache_hits
        return exact

    @staticmethod
    def _propagate_solved(
        nodes: tuple[MctsNode, ...], edges: tuple[MctsEdge, ...]
    ) -> None:
        """Propagate proven child outcomes toward the root by minimax rules."""

        for index in range(len(edges) - 1, -1, -1):
            child = nodes[index + 1]
            if child.solved_value is None:
                break
            edge = edges[index]
            edge.solved_value = -child.solved_value
            parent = nodes[index]
            solved_edges = tuple(
                candidate.solved_value for candidate in parent.children.values()
            )
            if any(value == 1.0 for value in solved_edges):
                parent.solved_value = 1.0
            elif all(value is not None for value in solved_edges):
                parent.solved_value = max(
                    float(value) for value in solved_edges if value is not None
                )
            else:
                break

    @staticmethod
    def _finalize_solved_root(root: MctsNode, simulations: int) -> int | None:
        """Replace sampling statistics with a policy over proven-optimal actions."""

        if root.solved_value is None:
            return None
        optimal = tuple(
            edge
            for edge in root.children.values()
            if edge.solved_value == root.solved_value
        )
        if not optimal:
            raise SearchStateError("Solved root has no proven-optimal action")
        for edge in root.children.values():
            edge.visit_count = 0
            edge.value_sum = 0.0
        for index in range(simulations):
            edge = optimal[index % len(optimal)]
            edge.visit_count += 1
            edge.value_sum += root.solved_value
        root.visit_count = simulations
        root.value_sum = simulations * root.solved_value
        return int(root.solved_value)

    @staticmethod
    def _root_solved_actions(root: MctsNode) -> tuple[tuple[int, int], ...]:
        return tuple(
            (action, int(edge.solved_value))
            for action, edge in root.children.items()
            if edge.solved_value is not None
        )

    def _cancel_pending(self, pending: list[_PendingLeaf]) -> None:
        for item in pending:
            item.node.evaluation_in_flight = False
            cancel_path(item.edges, virtual_loss=self._config.virtual_loss)

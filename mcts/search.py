"""Exact-simulation neural MCTS with batched virtual-loss reservations."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import isclose, isfinite

from config.schema import MctsConfig, RulesConfig
from game import Board, GameState, score_terminal
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


class MctsSearch:
    """Single-owner Python reference search with batched neural evaluation."""

    def __init__(
        self,
        rules: RulesConfig,
        config: MctsConfig,
        evaluator: BatchEvaluator,
    ) -> None:
        """Bind validated rules, all MCTS behavior, and one batch evaluator."""

        self._rules = rules
        self._config = config
        self._evaluator = evaluator

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
        root_actions = self._validate_root_state(root_state)
        if root_state.terminal:
            raise SearchStateError("Cannot search a terminal root state")
        if mode is SearchMode.TRAINING and random_source is None:
            raise SearchStateError("Training search requires an explicit random source")

        root = MctsNode(
            to_play=root_state.to_play,
            zobrist_hash=root_state.zobrist_hash,
            terminal=False,
        )
        root_evaluation = self._evaluate_exact((root_state,))[0]
        self._expand(root, root_state, root_evaluation)
        inference_batches = 1
        inference_positions = 1
        maximum_batch_size = 1
        if mode is SearchMode.TRAINING:
            if random_source is None:
                raise SearchStateError("Training search requires a random source")
            add_root_dirichlet_noise(root, self._config, random_source)

        target_simulations = simulation_count(self._config, mode)
        completed_simulations = 0
        while completed_simulations < target_simulations:
            completed_before_batch = completed_simulations
            pending: list[_PendingLeaf] = []
            while (
                len(pending) < self._config.inference_batch_size
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
                    completed_simulations += 1

            if pending:
                states = tuple(item.state for item in pending)
                try:
                    evaluations = self._evaluate_exact(states)
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

        visit_counts = tuple(
            root.children[action].visit_count if action in root.children else 0
            for action in range(root_state.action_size)
        )
        root_priors = tuple(
            root.children[action].prior if action in root.children else 0.0
            for action in range(root_state.action_size)
        )
        temperature = configured_temperature(self._config, mode, root_state.ply)
        policy_target = visit_probabilities(visit_counts, 1.0)
        probabilities = visit_probabilities(visit_counts, temperature)
        return SearchResult(
            root=root,
            root_zobrist_hash=root_state.zobrist_hash,
            mode=mode,
            simulations=completed_simulations,
            visit_counts=visit_counts,
            root_priors=root_priors,
            policy_target=policy_target,
            action_probabilities=probabilities,
            root_value=root.mean_value,
            inference_batches=inference_batches,
            inference_positions=inference_positions,
            maximum_batch_size=maximum_batch_size,
        )

    def _select_leaf(
        self, root: MctsNode, root_actions: tuple[int, ...]
    ) -> _PendingLeaf | None:
        board = self._replay(root_actions)
        node = root
        nodes = [root]
        edges: list[MctsEdge] = []
        try:
            while True:
                selected = node.select_edge(self._config.c_puct)
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
                    return _PendingLeaf(
                        node=node,
                        state=state,
                        nodes=tuple(nodes),
                        edges=tuple(edges),
                        terminal_value=self._terminal_value(board),
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

    def _cancel_pending(self, pending: list[_PendingLeaf]) -> None:
        for item in pending:
            item.node.evaluation_in_flight = False
            cancel_path(item.edges, virtual_loss=self._config.virtual_loss)

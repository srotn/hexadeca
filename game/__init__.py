"""Canonical Hexadeca game engine and self-play environment."""

from game.board import Board, Move, Player, decode_action, encode_action
from game.environment import GameEnvironment, GameState, StepResult
from game.errors import (
    BoardError,
    GameEnvironmentError,
    IllegalMoveError,
    InvalidActionError,
    NonTerminalBoardError,
    ScoringError,
    TerminalStateError,
    UndoError,
    UnsupportedScoringRuleError,
)
from game.scoring import CellScore, ScoreResult, score_cell, score_terminal

__all__ = [
    "Board",
    "BoardError",
    "CellScore",
    "GameEnvironment",
    "GameEnvironmentError",
    "GameState",
    "IllegalMoveError",
    "InvalidActionError",
    "Move",
    "NonTerminalBoardError",
    "Player",
    "ScoreResult",
    "ScoringError",
    "StepResult",
    "TerminalStateError",
    "UndoError",
    "UnsupportedScoringRuleError",
    "decode_action",
    "encode_action",
    "score_cell",
    "score_terminal",
]

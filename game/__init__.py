"""Canonical Hexadeca board engine."""

from game.board import Board, Move, Player, decode_action, encode_action
from game.errors import (
    BoardError,
    IllegalMoveError,
    InvalidActionError,
    NonTerminalBoardError,
    ScoringError,
    UndoError,
    UnsupportedScoringRuleError,
)
from game.scoring import CellScore, ScoreResult, score_cell, score_terminal

__all__ = [
    "Board",
    "BoardError",
    "CellScore",
    "IllegalMoveError",
    "InvalidActionError",
    "Move",
    "NonTerminalBoardError",
    "Player",
    "ScoreResult",
    "ScoringError",
    "UndoError",
    "UnsupportedScoringRuleError",
    "decode_action",
    "encode_action",
    "score_cell",
    "score_terminal",
]

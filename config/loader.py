"""TOML configuration loading, layering, and validation."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from config.schema import AppConfig, SchemaError, build_config

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.toml")


class ConfigError(ValueError):
    """Raised when a configuration source is malformed or incompatible."""


def load_config(
    default_path: Path = DEFAULT_CONFIG_PATH,
    profile_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Load the default TOML config with optional profile and value overrides.

    A profile may replace only the values it needs. Overrides use dotted keys,
    such as ``{"training.batch_size": 128}``, and are applied last.
    """

    merged = _read_toml(default_path)
    if profile_path is not None:
        merged = _deep_merge(merged, _read_toml(profile_path))
    if overrides is not None:
        merged = _apply_overrides(merged, overrides)
    try:
        return build_config(merged)
    except SchemaError as error:
        raise ConfigError(str(error)) from error


def _read_toml(path: Path) -> dict[str, Any]:
    """Read one TOML mapping or raise a domain-specific error."""

    try:
        with path.open("rb") as config_file:
            content = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise ConfigError(f"Configuration file does not exist: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"Invalid TOML in configuration file: {path}") from error

    if not isinstance(content, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    return content


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _apply_overrides(
    config: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply typed dotted-key overrides to a deep copy of a configuration."""

    merged = deepcopy(dict(config))
    for dotted_key, value in overrides.items():
        key_parts = dotted_key.split(".")
        if not dotted_key or any(not part for part in key_parts):
            raise ConfigError(f"Invalid override key: {dotted_key!r}")

        target = merged
        for key_part in key_parts[:-1]:
            current = target.get(key_part)
            if not isinstance(current, dict):
                raise ConfigError(f"Unknown override section: {dotted_key!r}")
            target = current
        target[key_parts[-1]] = deepcopy(value)
    return merged

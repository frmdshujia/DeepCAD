"""YAML-backed command-line configuration utilities."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Load flat YAML values as parser defaults, then apply CLI overrides."""
    if not any(action.dest == "config" for action in parser._actions):
        parser.add_argument(
            "--config", help="Flat YAML file containing training hyperparameters")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    preliminary, _ = pre_parser.parse_known_args()
    if preliminary.config:
        path = Path(preliminary.config).expanduser().resolve()
        with path.open() as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ValueError(f"Configuration must be a YAML mapping: {path}")
        valid = {action.dest for action in parser._actions}
        unknown = sorted(set(values) - valid)
        if unknown:
            raise ValueError(
                f"Unknown configuration keys in {path}: {', '.join(unknown)}")
        parser.set_defaults(**values)
    return parser.parse_args()

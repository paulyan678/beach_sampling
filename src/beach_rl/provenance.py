"""Immutable run receipts and integrity-checked checkpoint selection."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from beach_rl.config import ExperimentConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_receipt(path: Path, payload: Any) -> None:
    """Never replace an existing scientific receipt."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(
            sorted(
                (dist.metadata["Name"], dist.version)
                for dist in importlib.metadata.distributions()
                if dist.metadata.get("Name")
            )
        ),
    }


def source_snapshot() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    files = sorted(
        {
            *root.glob("src/**/*.py"),
            *root.glob("configs/*.yaml"),
            *root.glob("requirements*.txt"),
            root / "pyproject.toml",
        }
    )
    hashes = {str(path.relative_to(root)): sha256(path) for path in files if path.is_file()}
    result: dict[str, Any] = {
        "files": hashes,
        "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "git_commit": None,
        "git_dirty": None,
    }
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        result["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def split_manifest(config: ExperimentConfig) -> dict[str, Any]:
    """Record half-open procedural namespaces; reject overlapping experiment splits."""
    t, a, b = config.training, config.agent, config.beach
    groups: dict[str, tuple[int, int]] = {
        "validation": (t.validation_seed, t.validation_seed + t.validation_profiles),
        "test": (t.eval_seed, t.eval_seed + t.eval_profiles),
    }
    for seed in t.seeds:
        groups[f"train_{seed}"] = (
            seed * 1_000_000,
            seed * 1_000_000 + (t.total_steps + b.horizon - 1) // b.horizon,
        )
        if a.demonstration_episodes:
            groups[f"demonstrations_{seed}"] = (
                seed * 1_000_000 + 800_000,
                seed * 1_000_000 + 800_000 + a.demonstration_episodes,
            )
    items = list(groups.items())
    for i, (name, (start, stop)) in enumerate(items):
        for other, (left, right) in items[i + 1 :]:
            if max(start, left) < min(stop, right):
                raise ValueError(f"procedural seed namespaces overlap: {name} and {other}")
    return {"range_convention": "start inclusive, stop exclusive", "ranges": groups}


def begin_run(output: Path, config: ExperimentConfig, *, kind: str, seed: int | None = None) -> str:
    config.validate()
    split_config = (
        replace(config, training=replace(config.training, seeds=(seed,)))
        if seed is not None
        else config
    )
    splits = split_manifest(split_config)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"output directory is not empty; use a new run directory: {output}")
    run_id = str(uuid.uuid4())
    payload = config.to_dict()
    write_receipt(
        output / "run.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "kind": kind,
            "training_seed": seed,
            "config": payload,
            "config_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest(),
            "source": source_snapshot(),
            "environment": environment(),
            "splits": splits,
        },
    )
    return run_id


def checkpoint_candidates(output: Path, total_steps: int) -> list[Path]:
    receipt = json.loads((output / "completed.json").read_text())
    run = json.loads((output / "run.json").read_text())
    if receipt["run_id"] != run["run_id"] or receipt["total_steps"] != total_steps:
        raise ValueError("checkpoint receipt does not identify the current training run")
    by_step: dict[int, Path] = {}
    for item in receipt["checkpoints"]:
        name, step = item["file"], item["step"]
        if Path(name).name != name or not 0 < step <= total_steps:
            raise ValueError("invalid checkpoint path or training step")
        path = output / name
        if sha256(path) != item["sha256"]:
            raise ValueError(f"checkpoint integrity mismatch: {path}")
        by_step.setdefault(step, path)
    if total_steps not in by_step:
        raise ValueError("current final checkpoint is missing from the receipt")
    return [by_step[step] for step in sorted(by_step)]

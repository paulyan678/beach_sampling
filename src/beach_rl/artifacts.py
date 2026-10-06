"""Regenerate reported statistics and export inspectable scientific releases."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from beach_rl.evaluation import summarise_evaluation
from beach_rl.provenance import environment, sha256, source_snapshot, write_receipt


def regenerate(episodes_path: Path, output: Path) -> pd.DataFrame:
    episodes = pd.read_csv(episodes_path)
    required = [
        "information_gain",
        "return",
        "samples",
        "unique_samples",
        "path_length",
        "invalid_actions",
        "posterior_rmse",
        "profile_seed",
    ]
    if not np.isfinite(episodes[required].to_numpy(dtype=float)).all():
        raise ValueError("episode metrics must be finite")
    keys = ["policy", "profile_seed"] + (["agent_seed"] if "agent_seed" in episodes else [])
    if episodes.empty or episodes.duplicated(keys).any():
        raise ValueError("episode rows must be nonempty and unique per policy/agent/profile")
    profiles = set(episodes.profile_seed)
    for _, rows in episodes.groupby("policy"):
        if set(rows.profile_seed) != profiles:
            raise ValueError("policies must evaluate the same paired profiles")
        if "agent_seed" in rows and rows.agent_seed.notna().any():
            if rows.agent_seed.isna().any():
                raise ValueError("agent seeds must be present for every learned-policy row")
            for _, agent_rows in rows.groupby("agent_seed"):
                if set(agent_rows.profile_seed) != profiles:
                    raise ValueError("every agent must evaluate every profile")
    if output.exists() and any(output.iterdir()):
        raise ValueError("report output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    summary = summarise_evaluation(episodes)
    summary.to_csv(output / "summary.csv", index=False)
    write_receipt(
        output / "receipt.json",
        {
            "schema_version": 1,
            "kind": "raw-result-regeneration",
            "bootstrap_seed": 2026,
            "input_sha256": sha256(episodes_path),
            "rows": len(episodes),
            "profiles": len(profiles),
            "output_sha256": sha256(output / "summary.csv"),
            "source": source_snapshot(),
            "environment": environment(),
            "boundary": "Recomputes archived rows; does not retrain or re-evaluate missing agents.",
        },
    )
    return summary


def export_release(run: Path, output: Path, *, include_checkpoints: bool = False) -> None:
    run, output = run.resolve(), output.resolve()
    if output == run or run in output.parents:
        raise ValueError("release output must be outside the original run")
    if output.exists():
        raise ValueError("release output already exists")
    receipt = json.loads((run / "completed.json").read_text())
    identity = json.loads((run / "run.json").read_text())
    if receipt["run_id"] != identity["run_id"]:
        raise ValueError("run and completion identities disagree")
    expected = dict(receipt.get("artifacts", {}))
    if "checkpoints" in receipt:
        expected.update({item["file"]: item["sha256"] for item in receipt["checkpoints"]})
        expected["training.csv"] = receipt["training_sha256"]
    expected.update(
        {item["checkpoint"]: item["sha256"] for item in receipt.get("selected_checkpoints", [])}
    )
    # Study receipts reference each training receipt. Preserve every emitted
    # checkpoint, including candidates that validation did not select.
    for name in list(expected):
        if name.endswith("/completed.json"):
            child_path = run / name
            child = json.loads(child_path.read_text())
            prefix = str(Path(name).parent)
            for item in child.get("checkpoints", []):
                expected[f"{prefix}/{item['file']}"] = item["sha256"]
    expected["run.json"] = sha256(run / "run.json")
    expected["completed.json"] = sha256(run / "completed.json")
    for name, digest in expected.items():
        path = run / name
        if path.is_symlink() or run not in path.resolve().parents or sha256(path) != digest:
            raise ValueError(f"unsafe path or artifact integrity mismatch: {name}")
    output.mkdir(parents=True)
    included, omitted = {}, {}
    for name, digest in sorted(expected.items()):
        if name.endswith(".pt") and not include_checkpoints:
            omitted[name] = digest
            continue
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / name, destination)
        included[name] = digest
    write_receipt(
        output / "release-manifest.json",
        {
            "schema_version": 1,
            "run_id": identity["run_id"],
            "files": included,
            "omitted_checkpoints": omitted,
            "boundary": (
                "Checkpoint inclusion is explicit; publisher must hold redistribution rights."
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report")
    report.add_argument("--episodes", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    release = sub.add_parser("export-release")
    release.add_argument("--run", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    release.add_argument("--include-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.command == "report":
        print(regenerate(args.episodes, args.output).to_string(index=False))
    else:
        export_release(args.run, args.output, include_checkpoints=args.include_checkpoints)


if __name__ == "__main__":
    main()

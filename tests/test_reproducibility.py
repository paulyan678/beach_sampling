import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from beach_rl.artifacts import export_release, regenerate
from beach_rl.config import ExperimentConfig
from beach_rl.provenance import (
    begin_run,
    checkpoint_candidates,
    sha256,
    split_manifest,
    write_receipt,
)
from beach_rl.training import train_agent


def small_config(steps):
    cfg = ExperimentConfig.from_yaml("configs/smoke.yaml")
    return replace(
        cfg,
        agent=replace(cfg.agent, demonstration_episodes=0, pretrain_updates=0),
        training=replace(cfg.training, total_steps=steps, checkpoint_every=4),
    )


@pytest.mark.parametrize("steps,expected", [(3, [3]), (4, [4]), (5, [4, 5])])
def test_current_run_final_checkpoint_is_always_a_candidate(tmp_path, steps, expected):
    cfg = small_config(steps)
    train_agent(cfg, 7, tmp_path)
    candidates = checkpoint_candidates(tmp_path, steps)
    import torch

    assert [torch.load(p, weights_only=False)["metadata"]["step"] for p in candidates] == expected
    # A leftover file cannot enter selection: only the current receipt is authoritative.
    (tmp_path / "checkpoint_step_999.pt").write_bytes(b"stale")
    assert checkpoint_candidates(tmp_path, steps) == candidates
    with pytest.raises(ValueError, match="not empty"):
        train_agent(small_config(2), 7, tmp_path)


def test_study_rejects_output_before_overwriting_anything(tmp_path):
    cfg = small_config(5)
    begin_run(tmp_path, cfg, kind="study")
    original = (tmp_path / "run.json").read_bytes()
    with pytest.raises(ValueError, match="not empty"):
        begin_run(tmp_path, small_config(3), kind="study")
    assert (tmp_path / "run.json").read_bytes() == original


def test_checkpoint_tampering_and_export_policy(tmp_path):
    run = tmp_path / "run"
    train_agent(small_config(3), 7, run)
    export_release(run, tmp_path / "public")
    receipt = json.loads((tmp_path / "public/release-manifest.json").read_text())
    assert receipt["omitted_checkpoints"] and not list((tmp_path / "public").glob("*.pt"))
    export_release(run, tmp_path / "with-weights", include_checkpoints=True)
    assert list((tmp_path / "with-weights").glob("*.pt"))
    (run / "checkpoint_final.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        checkpoint_candidates(run, 3)


def test_overlapping_split_namespaces_are_rejected():
    cfg = small_config(3)
    with pytest.raises(ValueError, match="overlap"):
        split_manifest(replace(cfg, training=replace(cfg.training, eval_seed=8000)))


def test_study_export_accounts_for_unselected_checkpoints(tmp_path):
    run = tmp_path / "study"
    identity = begin_run(run, small_config(5), kind="study")
    child = run / "training/seed_7"
    train_agent(small_config(5), 7, child)
    write_receipt(
        run / "completed.json",
        {
            "run_id": identity,
            "artifacts": {
                str(p.relative_to(run)): sha256(p)
                for p in run.rglob("*")
                if p.is_file() and p.suffix != ".pt"
            },
        },
    )
    export_release(run, tmp_path / "without")
    omitted = json.loads((tmp_path / "without/release-manifest.json").read_text())
    assert len(omitted["omitted_checkpoints"]) == 2
    export_release(run, tmp_path / "with", include_checkpoints=True)
    assert len(list((tmp_path / "with").rglob("*.pt"))) == 2


def test_archived_dense_statistics_regenerate_from_raw_rows(tmp_path):
    actual = regenerate(Path("results/research/episodes.csv"), tmp_path).set_index("policy")
    expected = pd.read_csv("results/research/summary.csv").set_index("policy")
    columns = ["information_mean", "ratio_to_random", "ratio_ci95_low", "ratio_ci95_high"]
    np.testing.assert_allclose(actual[columns], expected[columns], rtol=1e-12, atol=1e-12)
    assert np.isclose(
        actual.loc["rainbow", "information_mean"]
        / actual.loc["greedy_information", "information_mean"],
        0.9729945654361151,
    )


def test_report_rejects_nonfinite_and_unpaired_rows(tmp_path):
    source = pd.read_csv("results/research/episodes.csv")
    path = tmp_path / "episodes.csv"
    source.loc[0, "information_gain"] = np.nan
    source.to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite"):
        regenerate(path, tmp_path / "report")
    source.drop(index=0).to_csv(path, index=False)
    with pytest.raises(ValueError, match="paired"):
        regenerate(path, tmp_path / "report")

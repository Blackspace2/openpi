import importlib.util
import json
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "resplit_episode_manifest", pathlib.Path(__file__).parent / "resplit_episode_manifest.py"
)
_resplit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_resplit)


def _write_manifest(root, records, *, seed=1, train_fraction=0.5):
    (root / "meta").mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "seed": seed, "train_fraction": train_fraction, "episodes": records}
    (root / "meta" / "openpi_episode_split.json").write_text(json.dumps(manifest))


def _records(task_to_count: dict[str, int]) -> list[dict[str, object]]:
    records = []
    next_id = 0
    for task, count in task_to_count.items():
        for _ in range(count):
            records.append(
                {
                    "converted_episode_id": next_id,
                    "source_task_directory": task,
                    "source_episode_file": f"e{next_id}",
                    "prompt": task,
                    "split": "train",
                }
            )
            next_id += 1
    return records


def test_main_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resplit.main(tmp_path, train_fraction=0.7, split_seed=42)


def test_main_invalid_train_fraction_raises(tmp_path):
    with pytest.raises(ValueError, match="train_fraction"):
        _resplit.main(tmp_path, train_fraction=1.5, split_seed=42)


def test_main_stratified_split_keeps_each_task_close_to_ratio(tmp_path):
    # Small, uneven per-task counts -- exactly the scenario where the converter's global split
    # drifts far from train_fraction for individual tasks.
    records = _records({"cabinet": 20, "shrimp": 6})
    _write_manifest(tmp_path, records)

    _resplit.main(tmp_path, train_fraction=0.7, split_seed=42, stratify_by_task=True)

    manifest = json.loads((tmp_path / "meta" / "openpi_episode_split.json").read_text())
    counts: dict[str, dict[str, int]] = {}
    for record in manifest["episodes"]:
        c = counts.setdefault(record["source_task_directory"], {"train": 0, "val": 0})
        c[record["split"]] += 1

    # int(20 * 0.7) == 14, int(6 * 0.7) == 4 -- each task independently hits its own 70/30 split.
    assert counts["cabinet"] == {"train": 14, "val": 6}
    assert counts["shrimp"] == {"train": 4, "val": 2}
    assert manifest["train_fraction"] == 0.7
    assert manifest["seed"] == 42


def test_main_global_split_matches_converters_algorithm(tmp_path):
    records = _records({"cabinet": 20, "shrimp": 6})
    _write_manifest(tmp_path, records)

    _resplit.main(tmp_path, train_fraction=0.7, split_seed=42, stratify_by_task=False)

    manifest = json.loads((tmp_path / "meta" / "openpi_episode_split.json").read_text())
    train_count = sum(1 for record in manifest["episodes"] if record["split"] == "train")
    val_count = sum(1 for record in manifest["episodes"] if record["split"] == "val")
    assert train_count + val_count == 26
    assert train_count == int(26 * 0.7)


def test_main_updates_audit_report_if_present(tmp_path):
    records = _records({"cabinet": 4, "shrimp": 4})
    _write_manifest(tmp_path, records)
    audit_path = tmp_path / "meta" / "openpi_mobile_aloha_audit.json"
    audit_path.write_text(json.dumps({"episode_count": 8, "episodes_per_prompt_and_split": {}}))

    _resplit.main(tmp_path, train_fraction=0.5, split_seed=7, stratify_by_task=True)

    audit = json.loads(audit_path.read_text())
    assert audit["episode_count"] == 8  # untouched fields survive
    assert "cabinet" in audit["episodes_per_prompt_and_split"]
    assert audit["split_summary"]["train_episode_count"] + audit["split_summary"]["val_episode_count"] == 8
    assert audit["split_summary"]["intersection"] == []

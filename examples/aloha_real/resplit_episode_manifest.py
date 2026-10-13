"""Regenerate the train/val split in `meta/openpi_episode_split.json` without re-running conversion.

`convert_aloha_data_to_lerobot.py` assigns train/val globally across all episodes (see its
`_assign_splits`), which can leave individual tasks far from the requested train_fraction when task
sizes are small and uneven. Since `source_task_directory` is already recorded per episode in the
manifest, a different split (e.g. stratified per task, a different seed, or a different ratio) can be
computed purely from the existing manifest -- no need to touch `data/`, `images/`, or any other
converted dataset file.

Usage:
    uv run examples/aloha_real/resplit_episode_manifest.py \
        --dataset-root /path/to/mobile-aloha-lerobot/mobile_aloha \
        --train-fraction 0.7 \
        --split-seed 42 \
        --stratify-by-task
"""

import json
from pathlib import Path
import random

import tyro

SPLIT_MANIFEST_NAME = "openpi_episode_split.json"
AUDIT_REPORT_NAME = "openpi_mobile_aloha_audit.json"


def _assign_splits_global(episode_ids: list[int], train_fraction: float, split_seed: int) -> dict[int, str]:
    ids = sorted(episode_ids)
    random.Random(split_seed).shuffle(ids)
    n = len(ids)
    train_count = n if n <= 1 else min(max(int(n * train_fraction), 1), n - 1)
    train_set = set(ids[:train_count])
    return {episode_id: "train" if episode_id in train_set else "val" for episode_id in ids}


def _assign_splits_stratified(
    records: list[dict[str, object]], train_fraction: float, split_seed: int
) -> dict[int, str]:
    grouped: dict[str, list[int]] = {}
    for record in records:
        grouped.setdefault(record["source_task_directory"], []).append(record["converted_episode_id"])

    rng = random.Random(split_seed)
    result: dict[int, str] = {}
    for task_directory in sorted(grouped):
        ids = sorted(grouped[task_directory])
        rng.shuffle(ids)
        n = len(ids)
        train_count = n if n <= 1 else min(max(int(n * train_fraction), 1), n - 1)
        train_set = set(ids[:train_count])
        for episode_id in ids:
            result[episode_id] = "train" if episode_id in train_set else "val"
    return result


def main(
    dataset_root: Path,
    train_fraction: float = 0.7,
    split_seed: int = 42,
    *,
    stratify_by_task: bool = True,
) -> None:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")

    manifest_path = dataset_root / "meta" / SPLIT_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No split manifest found at {manifest_path}. This script only rewrites an existing "
            "manifest written by convert_aloha_data_to_lerobot.py; it cannot create one from scratch."
        )
    manifest = json.loads(manifest_path.read_text())
    records = manifest["episodes"]

    splits = (
        _assign_splits_stratified(records, train_fraction, split_seed)
        if stratify_by_task
        else _assign_splits_global([r["converted_episode_id"] for r in records], train_fraction, split_seed)
    )
    for record in records:
        record["split"] = splits[record["converted_episode_id"]]

    manifest["seed"] = split_seed
    manifest["train_fraction"] = train_fraction
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    prompt_split_counts: dict[str, dict[str, int]] = {}
    for record in records:
        counts = prompt_split_counts.setdefault(record["prompt"], {"train": 0, "val": 0})
        counts[record["split"]] += 1

    strategy = "stratified per task" if stratify_by_task else "global"
    print(f"Rewrote {manifest_path} ({strategy} split, train_fraction={train_fraction}, seed={split_seed}):")
    for prompt in sorted(prompt_split_counts):
        counts = prompt_split_counts[prompt]
        total = counts["train"] + counts["val"]
        ratio = counts["train"] / total if total else 0.0
        print(f"  {prompt!r}: train={counts['train']} val={counts['val']} (train_ratio={ratio:.1%})")

    audit_path = dataset_root / "meta" / AUDIT_REPORT_NAME
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        train_ids = {r["converted_episode_id"] for r in records if r["split"] == "train"}
        val_ids = {r["converted_episode_id"] for r in records if r["split"] == "val"}
        audit["episodes_per_prompt_and_split"] = prompt_split_counts
        audit["split_summary"] = {
            "train_episode_count": len(train_ids),
            "val_episode_count": len(val_ids),
            "intersection": sorted(train_ids & val_ids),
        }
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
        print(f"Updated {audit_path}")


if __name__ == "__main__":
    tyro.cli(main)

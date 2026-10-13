"""Rename one or more tasks' prompt text across a converted LeRobot dataset.

Edit `NEW_TASK_PROMPT` below -- position `i` in the list is the new prompt for `task_index=i` -- then
run the script and pick which of those entries to actually apply via `--task-index` (comma-separated
task indices, or -1 to apply every entry in `NEW_TASK_PROMPT`). This lets you rename several tasks in
one edit instead of re-invoking the script once per prompt.

Only `meta/tasks.jsonl` is actually read at training time (via `openpi.transforms.PromptFromLeRobotTask`,
which looks up `task_index -> prompt` from `dataset_meta.tasks`). `task_index` itself never changes --
it's already baked into every episode's parquet file. This script edits the natural-language string
associated with each `task_index`, and always mirrors the rename into every other meta file that
references the old prompt text so nothing goes stale: `meta/episodes.jsonl`,
`meta/openpi_episode_split.json`, and `meta/openpi_mobile_aloha_audit.json`.

Usage:
    # 1. Edit NEW_TASK_PROMPT below.
    # 2. Apply only task 0 and 1:
    uv run examples/aloha_real/update_task_prompt.py \
        --dataset-root /path/to/mobile-aloha-lerobot/mobile_aloha \
        --task-index 0,1

    # Apply every entry in NEW_TASK_PROMPT:
    uv run examples/aloha_real/update_task_prompt.py \
        --dataset-root /path/to/mobile-aloha-lerobot/mobile_aloha \
        --task-index -1
"""

import json
from pathlib import Path

import tyro

# Edit this list with the new prompt for each task_index (position i -> task_index=i), then pick
# which entries to actually apply with --task-index (comma-separated indices, or -1 for all of them).
NEW_TASK_PROMPT = [
    # task_index: 0
    "open the cabinet and pick up the pot",  

    # task_index: 1
    "push the chairs in to the table",  

    # task_index: 2
    "call the elevator and take it to another floor", 

     # task_index: 3
    "cook the shrimp",  

    # task_index: 4
    "wash the pan",  

    # task_index: 5
    "wipe the spilled wine off the table",  
]

"""
details = [
    "drive to the cabinet, open the cabinet door, reach inside with the arm, grasp the pot by its handle, and lift the pot out of the cabinet" ,
    "approach the dining table, grasp each chair that has been pulled out, and push it back in so it is tucked neatly under the table",
    "navigate to the elevator, press the call button to summon it, wait for the doors to open, move inside, and press the button for the destination floor",
    "pick up the spatula, stir and flip the shrimp cooking in the pan on the stove, and continue cooking until they are done",
    "pick up the dirty pan from the stove, hold it under the running water in the sink, scrub it clean with the sponge, and set it down to dry",
    "pick up the cloth, wipe up the wine that has spilled on the table, and discard the soiled cloth once the surface is clean"
]
"""


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n")


def _parse_task_indices(task_index: str) -> list[int]:
    if task_index.strip() == "-1":
        return list(range(len(NEW_TASK_PROMPT)))
    selected = [int(part) for part in task_index.split(",")]
    invalid = [i for i in selected if i < 0 or i >= len(NEW_TASK_PROMPT)]
    if invalid:
        raise ValueError(f"task_index(es) {invalid} out of range for NEW_TASK_PROMPT (len={len(NEW_TASK_PROMPT)}).")
    return selected


def main(dataset_root: Path, task_index: str = "-1") -> None:
    task_indices = _parse_task_indices(task_index)

    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    tasks = _read_jsonl(tasks_path)
    tasks_by_index = {t["task_index"]: t for t in tasks}

    renames: dict[str, str] = {}  # old_prompt -> new_prompt
    for idx in task_indices:
        if idx not in tasks_by_index:
            raise ValueError(f"task_index={idx} not found in {tasks_path}.")
        new_prompt = NEW_TASK_PROMPT[idx]
        old_prompt = tasks_by_index[idx]["task"]
        if old_prompt == new_prompt:
            print(f"task_index={idx} already has this prompt, nothing to do.")
            continue
        tasks_by_index[idx]["task"] = new_prompt
        renames[old_prompt] = new_prompt
        print(f"task_index={idx}: {old_prompt!r} -> {new_prompt!r}")

    if not renames:
        return
    _write_jsonl(tasks_path, tasks)
    print(f"{tasks_path}: updated {len(renames)} task(s)")

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        episodes = _read_jsonl(episodes_path)
        changed = 0
        for episode in episodes:
            new_tasks = [renames.get(t, t) for t in episode["tasks"]]
            if new_tasks != episode["tasks"]:
                episode["tasks"] = new_tasks
                changed += 1
        _write_jsonl(episodes_path, episodes)
        print(f"{episodes_path}: updated {changed} episode record(s)")

    split_path = dataset_root / "meta" / "openpi_episode_split.json"
    if split_path.exists():
        manifest = json.loads(split_path.read_text())
        changed = 0
        for record in manifest["episodes"]:
            if record["prompt"] in renames:
                record["prompt"] = renames[record["prompt"]]
                changed += 1
        split_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"{split_path}: updated {changed} episode record(s)")

    audit_path = dataset_root / "meta" / "openpi_mobile_aloha_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        for key in ("episodes_per_prompt", "episodes_per_prompt_and_split"):
            if key not in audit:
                continue
            for old_prompt, new_prompt in renames.items():
                if old_prompt in audit[key]:
                    audit[key][new_prompt] = audit[key].pop(old_prompt)
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
        print(f"{audit_path}: updated prompt-keyed section(s)")


if __name__ == "__main__":
    tyro.cli(main)

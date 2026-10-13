import importlib.util
import json
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "update_task_prompt", pathlib.Path(__file__).parent / "update_task_prompt.py"
)
_update_task_prompt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_update_task_prompt)


def _write_dataset(root: pathlib.Path) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "tasks.jsonl").write_text(
        "\n".join(
            json.dumps(t)
            for t in [{"task_index": 0, "task": "wash the pan"}, {"task_index": 1, "task": "cook the shrimp"}]
        )
        + "\n"
    )
    (meta / "episodes.jsonl").write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {"episode_index": 0, "tasks": ["wash the pan"], "length": 1100},
                {"episode_index": 1, "tasks": ["cook the shrimp"], "length": 3750},
            ]
        )
        + "\n"
    )
    (meta / "openpi_episode_split.json").write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 42,
                "train_fraction": 0.7,
                "episodes": [
                    {
                        "converted_episode_id": 0,
                        "source_task_directory": "aloha_mobile_wash_pan",
                        "source_episode_file": "e0",
                        "prompt": "wash the pan",
                        "split": "train",
                    },
                    {
                        "converted_episode_id": 1,
                        "source_task_directory": "aloha_mobile_shrimp",
                        "source_episode_file": "e1",
                        "prompt": "cook the shrimp",
                        "split": "train",
                    },
                ],
            }
        )
    )
    (meta / "openpi_mobile_aloha_audit.json").write_text(
        json.dumps(
            {
                "episode_count": 2,
                "episodes_per_prompt": {"wash the pan": 1, "cook the shrimp": 1},
                "episodes_per_prompt_and_split": {
                    "wash the pan": {"train": 1, "val": 0},
                    "cook the shrimp": {"train": 1, "val": 0},
                },
            }
        )
    )


def test_updates_tasks_jsonl_and_mirrors_into_reports(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.setattr(
        _update_task_prompt, "NEW_TASK_PROMPT", ["wash the metal frying pan in the sink", "cook the shrimp"]
    )

    _update_task_prompt.main(tmp_path, task_index="0")

    tasks = _update_task_prompt._read_jsonl(tmp_path / "meta" / "tasks.jsonl")  # noqa: SLF001
    assert {t["task_index"]: t["task"] for t in tasks} == {
        0: "wash the metal frying pan in the sink",
        1: "cook the shrimp",
    }

    episodes = _update_task_prompt._read_jsonl(tmp_path / "meta" / "episodes.jsonl")  # noqa: SLF001
    assert episodes[0]["tasks"] == ["wash the metal frying pan in the sink"]
    assert episodes[1]["tasks"] == ["cook the shrimp"]  # untouched

    manifest = json.loads((tmp_path / "meta" / "openpi_episode_split.json").read_text())
    assert manifest["episodes"][0]["prompt"] == "wash the metal frying pan in the sink"
    assert manifest["episodes"][1]["prompt"] == "cook the shrimp"

    audit = json.loads((tmp_path / "meta" / "openpi_mobile_aloha_audit.json").read_text())
    assert audit["episodes_per_prompt"] == {
        "wash the metal frying pan in the sink": 1,
        "cook the shrimp": 1,
    }
    assert audit["episodes_per_prompt_and_split"] == {
        "wash the metal frying pan in the sink": {"train": 1, "val": 0},
        "cook the shrimp": {"train": 1, "val": 0},
    }


def test_task_index_minus_one_applies_all_entries(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.setattr(
        _update_task_prompt, "NEW_TASK_PROMPT", ["wash the frying pan", "cook the shrimp thoroughly"]
    )

    _update_task_prompt.main(tmp_path, task_index="-1")

    tasks = _update_task_prompt._read_jsonl(tmp_path / "meta" / "tasks.jsonl")  # noqa: SLF001
    assert {t["task_index"]: t["task"] for t in tasks} == {
        0: "wash the frying pan",
        1: "cook the shrimp thoroughly",
    }


def test_task_index_comma_separated_selects_subset(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.setattr(
        _update_task_prompt, "NEW_TASK_PROMPT", ["wash the frying pan", "cook the shrimp"]
    )

    _update_task_prompt.main(tmp_path, task_index="0")

    tasks = _update_task_prompt._read_jsonl(tmp_path / "meta" / "tasks.jsonl")  # noqa: SLF001
    assert {t["task_index"]: t["task"] for t in tasks} == {
        0: "wash the frying pan",
        1: "cook the shrimp",  # not selected, untouched
    }


def test_out_of_range_task_index_raises(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.setattr(_update_task_prompt, "NEW_TASK_PROMPT", ["anything"])

    with pytest.raises(ValueError, match="out of range"):
        _update_task_prompt.main(tmp_path, task_index="99")


def test_task_index_not_in_tasks_jsonl_raises(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.setattr(
        _update_task_prompt, "NEW_TASK_PROMPT", ["a", "b", "c", "d", "e", "anything"]
    )

    with pytest.raises(ValueError, match="not found"):
        _update_task_prompt.main(tmp_path, task_index="5")

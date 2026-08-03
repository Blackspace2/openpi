import dataclasses
import json
import types

import jax
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def _write_split_manifest(root, records):
    (root / "meta").mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "seed": 42, "train_fraction": 0.7, "episodes": records}
    (root / "meta" / "openpi_episode_split.json").write_text(json.dumps(manifest))


def test_load_episode_split_ids_filters_by_split(tmp_path):
    _write_split_manifest(
        tmp_path,
        [
            {"converted_episode_id": 2, "source_task_directory": "t", "source_episode_file": "e2", "prompt": "p", "split": "train"},
            {"converted_episode_id": 0, "source_task_directory": "t", "source_episode_file": "e0", "prompt": "p", "split": "train"},
            {"converted_episode_id": 1, "source_task_directory": "t", "source_episode_file": "e1", "prompt": "p", "split": "val"},
        ],
    )
    dataset_meta = types.SimpleNamespace(root=tmp_path)

    train_ids = _data_loader._load_episode_split_ids(dataset_meta, "train")  # noqa: SLF001
    val_ids = _data_loader._load_episode_split_ids(dataset_meta, "val")  # noqa: SLF001

    assert train_ids == [0, 2]
    assert val_ids == [1]
    # No leakage: train and val must never share an episode id.
    assert set(train_ids).isdisjoint(val_ids)


def test_load_episode_split_ids_missing_manifest_raises(tmp_path):
    dataset_meta = types.SimpleNamespace(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        _data_loader._load_episode_split_ids(dataset_meta, "train")  # noqa: SLF001


def test_load_episode_split_ids_empty_split_raises(tmp_path):
    _write_split_manifest(
        tmp_path,
        [{"converted_episode_id": 0, "source_task_directory": "t", "source_episode_file": "e0", "prompt": "p", "split": "train"}],
    )
    dataset_meta = types.SimpleNamespace(root=tmp_path)
    with pytest.raises(ValueError, match="No episodes found"):
        _data_loader._load_episode_split_ids(dataset_meta, "val")  # noqa: SLF001

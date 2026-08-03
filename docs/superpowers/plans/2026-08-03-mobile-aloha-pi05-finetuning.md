# Mobile ALOHA π0.5 Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an episode-disjoint Mobile ALOHA data pipeline and fine-tune the open-source π0.5 flow-matching action policy with 14D `qpos` state, 16D physical actions, language prompts, and decoded offline validation videos.

**Architecture:** Convert raw task-directory HDF5 files into one local LeRobot dataset and store a deterministic episode split manifest beside its metadata. A Mobile-specific policy adapter converts only the first 14 arm/gripper dimensions into π space, keeps the 2D base velocity semantic, applies delta encoding only to 12 arm joints, quantile-normalizes the 14D/16D semantic data, and pads to the unchanged 32D π0.5 checkpoint interface. A separate evaluator loads only validation episodes and compares post-transform 16D predictions with recorded actions.

**Tech Stack:** Python 3.11, NumPy, h5py, OpenCV, LeRobot v2 API pinned by this repository, JAX/OpenPI, Tyro, pytest, imageio/OpenCV video writing.

## Global Constraints

- Phase 1 uses Mobile ALOHA only; static ALOHA co-training is excluded.
- `state = qpos[14]`; `action = concat(action[14], base_action[2])` in the exact source ordering.
- Keep `Pi0Config(pi05=True, action_dim=32, action_horizon=50)` and load `gs://openpi-assets/checkpoints/pi05_base/params`.
- Normalize transformed 14D state and 16D action before right-padding; compute statistics from train episodes only.
- Delta-encode only the 12 arm joints. Both grippers and both base dimensions stay absolute.
- Preserve raw gripper/base targets: no clipping, base smoothing, or action timestamp shift in the baseline.
- Use one prompted VLA for all six tasks. The repack transform must preserve the language `prompt` key.
- Split complete episodes globally with seed `42` into 70% train and 30% validation; do not stratify by task.
- Evaluation errors use decoded physical 16D actions; never score 32D padding.
- Preserve all unrelated and user-authored working-tree changes. Do not commit, push, delete, or overwrite existing datasets without explicit user authorization.
- Do not modify `scripts/train.py`, `src/openpi/models/pi0_config.py`, or the π0.5 network architecture for this feature.

## File Map

- Create `src/openpi/training/episode_splits.py`: generic versioned episode split manifest types and validation.
- Create `src/openpi/training/episode_splits_test.py`: deterministic 70/30 and serialization tests.
- Create `src/openpi/training/mobile_aloha_dataset.py`: raw HDF5 discovery, audit, streaming frame conversion, and LeRobot writer.
- Create `src/openpi/training/mobile_aloha_dataset_test.py`: synthetic compressed-HDF5 tests.
- Create `examples/aloha_real/convert_mobile_aloha_data_to_lerobot.py`: thin Tyro CLI for audit/conversion.
- Modify `src/openpi/policies/aloha_policy.py`: expose copy-safe ALOHA coordinate conversion helpers.
- Create `src/openpi/policies/mobile_aloha_policy.py`: 14D state/16D action input-output transforms.
- Create `src/openpi/policies/mobile_aloha_policy_test.py`: transform, base preservation, and round-trip tests.
- Modify `src/openpi/training/config.py`: episode split option, Mobile data config factory, and `pi05_mobile_aloha` training config.
- Modify `src/openpi/training/data_loader.py`: load only manifest-selected LeRobot episodes.
- Modify `src/openpi/training/data_loader_test.py`: split selection and missing-manifest tests.
- Create `src/openpi/training/mobile_aloha_config_test.py`: configuration and prompt-preservation tests.
- Create `src/openpi/evaluation/__init__.py`: evaluation package.
- Create `src/openpi/evaluation/mobile_aloha.py`: decoded action metrics, teacher-forced inference, plot/video helpers.
- Create `src/openpi/evaluation/mobile_aloha_test.py`: alignment, masking, grouping, and rendering tests.
- Create `scripts/evaluate_mobile_aloha.py`: checkpoint evaluation CLI.
- Create `docs/mobile-aloha-pi05-finetuning.md`: commands and output interpretation.

---

### Task 1: Add a deterministic episode split manifest

**Files:**
- Create: `src/openpi/training/episode_splits.py`
- Test: `src/openpi/training/episode_splits_test.py`

**Interfaces:**
- Produces: `EpisodeSource`, `EpisodeSplitRecord`, `EpisodeSplitManifest`.
- Produces: `assign_episode_splits(sources, *, train_fraction, seed) -> EpisodeSplitManifest`.
- Produces: `save_episode_split_manifest(manifest, path)`, `load_episode_split_manifest(path)`, and `episode_indices_for_split(manifest, split)`.
- Manifest location used by later tasks: `meta/openpi_episode_split.json` inside the LeRobot dataset root.

- [ ] **Step 1: Write failing deterministic-split tests**

```python
def test_assign_episode_splits_is_deterministic_and_disjoint():
    sources = tuple(
        EpisodeSource(i, f"task_{i % 3}", f"episode_{i}.hdf5", f"prompt {i % 3}") for i in range(10)
    )
    first = assign_episode_splits(sources, train_fraction=0.7, seed=42)
    second = assign_episode_splits(sources, train_fraction=0.7, seed=42)

    assert first == second
    train = set(episode_indices_for_split(first, "train"))
    val = set(episode_indices_for_split(first, "val"))
    assert len(train) == 7
    assert len(val) == 3
    assert train.isdisjoint(val)
    assert train | val == set(range(10))


def test_manifest_round_trip(tmp_path):
    sources = tuple(EpisodeSource(i, "cabinet", f"episode_{i}.hdf5", "use the cabinet") for i in range(4))
    expected = assign_episode_splits(sources, train_fraction=0.7, seed=42)
    path = tmp_path / "meta" / "openpi_episode_split.json"
    save_episode_split_manifest(expected, path)
    assert load_episode_split_manifest(path) == expected
```

- [ ] **Step 2: Run tests and confirm the module is missing**

Run: `pytest -q src/openpi/training/episode_splits_test.py`

Expected: collection fails because `openpi.training.episode_splits` does not exist.

- [ ] **Step 3: Implement immutable records, seeded global shuffle, and strict validation**

Use `random.Random(seed).shuffle(indices)` so the split does not depend on NumPy RNG changes. Compute `train_count = int(len(sources) * train_fraction)`, require at least two episodes, and clamp the count to `[1, len(sources)-1]`. Serialize with an explicit schema version:

```python
MANIFEST_VERSION = 1
MANIFEST_RELATIVE_PATH = pathlib.Path("meta/openpi_episode_split.json")
SplitName = Literal["train", "val"]

@dataclasses.dataclass(frozen=True)
class EpisodeSource:
    episode_index: int
    source_task_directory: str
    source_episode_file: str
    prompt: str

@dataclasses.dataclass(frozen=True)
class EpisodeSplitRecord(EpisodeSource):
    split: SplitName

@dataclasses.dataclass(frozen=True)
class EpisodeSplitManifest:
    version: int
    seed: int
    train_fraction: float
    episodes: tuple[EpisodeSplitRecord, ...]
```

Validation must reject duplicate episode indices, unknown split strings, empty prompts, unsupported versions, and train/val overlap.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q src/openpi/training/episode_splits_test.py`

Expected: all tests pass.

- [ ] **Step 5: Review only the new manifest files**

Run: `git diff -- src/openpi/training/episode_splits.py src/openpi/training/episode_splits_test.py`

Expected: no production files outside the manifest unit changed; do not commit.

### Task 2: Convert raw Mobile ALOHA HDF5 without losing base actions or prompts

**Files:**
- Create: `src/openpi/training/mobile_aloha_dataset.py`
- Create: `src/openpi/training/mobile_aloha_dataset_test.py`
- Create: `examples/aloha_real/convert_mobile_aloha_data_to_lerobot.py`

**Interfaces:**
- Consumes: Task 1 manifest functions.
- Produces: `RawEpisode`, `EpisodeAudit`, `discover_raw_episodes`, `audit_raw_episode`, `iter_episode_frames`, and `convert_mobile_aloha_dataset`.
- CLI consumes `--raw-root`, `--repo-id`, `--task-prompts`, `--seed`, `--train-fraction`, `--audit-only`, `--overwrite`, and `--push-to-hub`.

- [ ] **Step 1: Write a synthetic compressed-HDF5 fixture and failing tests**

Create two 8-frame episodes using `cv2.imencode(".jpg", rgb[..., ::-1])`. Store padded encoded bytes at the same three camera keys as the real sample. Assert:

```python
def test_iter_episode_frames_concatenates_base_action(synthetic_episode):
    frames = list(iter_episode_frames(synthetic_episode, prompt="use the cabinet"))
    assert len(frames) == 8
    assert frames[0]["observation.state"].shape == (14,)
    assert frames[0]["action"].shape == (16,)
    np.testing.assert_allclose(frames[0]["action"][:14], synthetic_episode.action[0])
    np.testing.assert_allclose(frames[0]["action"][14:], synthetic_episode.base_action[0])
    assert frames[0]["task"] == "use the cabinet"
    for camera in CAMERA_NAMES:
        assert frames[0][f"observation.images.{camera}"].shape == (480, 640, 3)


def test_audit_rejects_missing_base_action(synthetic_episode_without_base):
    with pytest.raises(ValueError, match="/base_action"):
        audit_raw_episode(synthetic_episode_without_base)
```

Also test non-finite low-dimensional values, mismatched `T`, unmapped task directories, JPEG decode failure, and numeric episode sorting (`episode_2` before `episode_10`).

- [ ] **Step 2: Run tests and confirm the converter module is missing**

Run: `pytest -q src/openpi/training/mobile_aloha_dataset_test.py`

Expected: collection fails on the missing module.

- [ ] **Step 3: Implement discovery, validation, and streaming decoding**

Use these exact constants and ordering:

```python
FPS = 50
CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
STATE_NAMES = (
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle",
    "left_wrist_rotate", "left_gripper", "right_waist", "right_shoulder", "right_elbow",
    "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper",
)
ACTION_NAMES = (*STATE_NAMES, "base_linear_velocity", "base_angular_velocity")
```

Open one HDF5 file at a time and decode one frame at a time. For compressed rows, call `cv2.imdecode(np.asarray(row, dtype=np.uint8), cv2.IMREAD_COLOR)` and convert BGR to RGB. For uncompressed `[T,H,W,3]`, return a copied RGB array. Never load all three 1500-frame videos into RAM together.

- [ ] **Step 4: Implement the pinned LeRobot writer correctly**

Create features with state shape `(14,)`, action shape `(16,)`, and camera feature shape `(3, 480, 640)` using dtype `video` by default. For each frame, include the natural-language task:

```python
frame = {
    "observation.state": qpos.astype(np.float32, copy=False),
    "action": np.concatenate([arm_action, base_action]).astype(np.float32, copy=False),
    "task": prompt,
    **camera_frames,
}
dataset.add_frame(frame)
```

After the final frame of an episode call `dataset.save_episode()` with no `task=` argument. This repository's pinned LeRobot API reads the task from each frame. Do not copy the stale generic converter's `save_episode(task=...)` or `dataset.consolidate()` calls.

Write the split manifest after all converted episode indices are known. Generate `meta/openpi_mobile_aloha_audit.json` with frame counts, per-channel low-dimensional summary statistics, and per-prompt train/val counts. If global shuffling leaves a prompt with zero validation episodes, emit a warning in both the terminal and audit JSON without changing the split.

- [ ] **Step 5: Make overwrite explicit and recoverable at the interface boundary**

If the target `$LEROBOT_HOME/<repo-id>` exists and `overwrite=False`, raise `FileExistsError`. Only the explicit CLI flag may remove that exact resolved dataset directory. Reject paths resolving to `$LEROBOT_HOME` itself or its parent. Default `push_to_hub=False`.

- [ ] **Step 6: Add the thin Tyro CLI**

The example script imports `convert_mobile_aloha_dataset` and passes CLI values without duplicating conversion logic. `--audit-only` audits raw files and writes/prints a JSON report without creating or deleting a LeRobot dataset; this mode supports the single downloaded sample.

- [ ] **Step 7: Run conversion unit tests and inspect the real sample read-only**

Run:

```bash
pytest -q src/openpi/training/mobile_aloha_dataset_test.py
uv run examples/aloha_real/convert_mobile_aloha_data_to_lerobot.py \
  --raw-root dataset/mobile_aloha \
  --repo-id mobile_aloha \
  --task-prompts /private/tmp/mobile_aloha_task_prompts.json \
  --audit-only
```

Before the second command, create `/private/tmp/mobile_aloha_task_prompts.json` containing the exact observed directory mapping:

```json
{"aloha_mobile_cabinet": "use the cabinet"}
```

Expected: the sample reports 14D qpos, 14D arm action, 2D base action, three decodable 640×480 cameras, and no writes under the raw dataset directory.

### Task 3: Add a Mobile ALOHA 16D policy adapter

**Files:**
- Modify: `src/openpi/policies/aloha_policy.py:159-202`
- Create: `src/openpi/policies/mobile_aloha_policy.py`
- Create: `src/openpi/policies/mobile_aloha_policy_test.py`

**Interfaces:**
- Produces from `aloha_policy.py`: `aloha_state_to_pi`, `aloha_actions_to_pi`, and `pi_actions_to_aloha`; each copies its input before any in-place gripper conversion.
- Produces: `MobileAlohaInputs`, `MobileAlohaOutputs`, and `MOBILE_ALOHA_DELTA_MASK`.

- [ ] **Step 1: Write failing transform tests**

Test the following invariants:

```python
def test_mobile_input_keeps_base_absolute():
    raw = make_mobile_aloha_example(action_horizon=50)
    transformed = MobileAlohaInputs(adapt_to_pi=True)(raw)
    assert transformed["state"].shape == (14,)
    assert transformed["actions"].shape == (50, 16)
    np.testing.assert_array_equal(transformed["actions"][:, 14:16], raw["actions"][:, 14:16])
    assert transformed["prompt"] == raw["prompt"]


def test_mobile_output_returns_16_dims_and_round_trips():
    raw = make_mobile_aloha_example(action_horizon=50)
    encoded = MobileAlohaInputs(adapt_to_pi=True)(raw.copy())["actions"]
    decoded = MobileAlohaOutputs(adapt_to_pi=True)({"actions": encoded})["actions"]
    np.testing.assert_allclose(decoded, raw["actions"], atol=1e-5)


def test_mobile_delta_mask():
    assert MOBILE_ALOHA_DELTA_MASK == (True,) * 6 + (False,) + (True,) * 6 + (False,) * 3
```

Also assert that unknown cameras and wrong state/action dimensions fail with descriptive errors.

- [ ] **Step 2: Run tests and confirm the adapter is missing**

Run: `pytest -q src/openpi/policies/mobile_aloha_policy_test.py`

Expected: collection fails on the missing module.

- [ ] **Step 3: Expose copy-safe shared ALOHA primitives**

Add public wrappers around the existing private transformations instead of duplicating joint signs and gripper constants:

```python
def aloha_state_to_pi(state: np.ndarray, *, adapt_to_pi: bool = True) -> np.ndarray:
    return _decode_state(np.array(state, copy=True), adapt_to_pi=adapt_to_pi)

def aloha_actions_to_pi(actions: np.ndarray, *, adapt_to_pi: bool = True) -> np.ndarray:
    return _encode_actions_inv(np.array(actions, copy=True), adapt_to_pi=adapt_to_pi)

def pi_actions_to_aloha(actions: np.ndarray, *, adapt_to_pi: bool = True) -> np.ndarray:
    return _encode_actions(np.array(actions, copy=True), adapt_to_pi=adapt_to_pi)
```

Leave `AlohaInputs` and `AlohaOutputs` behavior unchanged.

- [ ] **Step 4: Implement Mobile inputs and outputs**

`MobileAlohaInputs` must:

1. Validate state last dimension 14 and optional actions last dimension 16.
2. Convert CHW images to HWC uint8 and map `cam_high`, `cam_left_wrist`, `cam_right_wrist` to `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`.
3. Apply `aloha_state_to_pi` to state.
4. Apply `aloha_actions_to_pi` only to `actions[..., :14]`, concatenate untouched `actions[..., 14:16]`, and preserve prompt.

`MobileAlohaOutputs` must first slice the semantic prefix `actions[..., :16]`, apply `pi_actions_to_aloha` to the first 14 dimensions, retain the two base dimensions, and return `[H,16]`.

- [ ] **Step 5: Run policy tests and existing ALOHA tests**

Run:

```bash
pytest -q src/openpi/policies/mobile_aloha_policy_test.py src/openpi/policies/policy_test.py
```

Expected: Mobile tests pass and existing 14D ALOHA behavior remains green.

### Task 4: Make LeRobot train/val episode selection part of DataConfig

**Files:**
- Modify: `src/openpi/training/config.py:64-98`
- Modify: `src/openpi/training/data_loader.py:130-151`
- Modify: `src/openpi/training/data_loader_test.py`

**Interfaces:**
- Adds: `DataConfig.episode_split: Literal["all", "train", "val"] = "all"`.
- Consumes: Task 1 manifest stored under `dataset_meta.root`.
- Existing configs remain behaviorally unchanged because their default is `all`.

- [ ] **Step 1: Write failing split-selection tests**

Monkeypatch `LeRobotDatasetMetadata` with a temporary root containing a manifest and monkeypatch `LeRobotDataset` to capture its constructor arguments:

```python
def test_create_torch_dataset_selects_manifest_train_episodes(monkeypatch, tmp_path):
    write_test_manifest(tmp_path, train=[0, 2, 3], val=[1])
    captured = {}
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", fake_metadata(tmp_path))
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", capture_dataset(captured))

    config = _config.DataConfig(repo_id="mobile_aloha", episode_split="train")
    _data_loader.create_torch_dataset(config, 50, pi0_config.Pi0Config(pi05=True))
    assert captured["episodes"] == [0, 2, 3]
```

Add tests for `all -> None`, validation selection, missing manifest, and an empty selected split.

- [ ] **Step 2: Run the focused test and observe the missing field failure**

Run: `pytest -q src/openpi/training/data_loader_test.py -k episode_split`

Expected: construction fails because `DataConfig` has no `episode_split` field.

- [ ] **Step 3: Add the config field and loader selection**

After metadata construction:

```python
episodes = None
if data_config.episode_split != "all":
    manifest_path = dataset_meta.root / episode_splits.MANIFEST_RELATIVE_PATH
    manifest = episode_splits.load_episode_split_manifest(manifest_path)
    episodes = list(episode_splits.episode_indices_for_split(manifest, data_config.episode_split))
    if not episodes:
        raise ValueError(f"No episodes found for split {data_config.episode_split!r}")

dataset = lerobot_dataset.LeRobotDataset(
    data_config.repo_id,
    episodes=episodes,
    delta_timestamps={...},
)
```

Keep `PromptFromLeRobotTask(dataset_meta.tasks)` after subset construction.

- [ ] **Step 4: Run all data loader tests**

Run: `pytest -q src/openpi/training/data_loader_test.py`

Expected: new split tests and existing fake/real loader tests pass in the supported server environment.

### Task 5: Register the Mobile ALOHA π0.5 data and training configuration

**Files:**
- Modify: `src/openpi/training/config.py:228-278`
- Modify: `src/openpi/training/config.py` near the ALOHA fine-tuning configs
- Create: `src/openpi/training/mobile_aloha_config_test.py`

**Interfaces:**
- Produces: `LeRobotMobileAlohaDataConfig`.
- Produces config name: `pi05_mobile_aloha`.
- Consumes the local LeRobot repo id `mobile_aloha` and train-only manifest selection.

- [ ] **Step 1: Write the failing configuration test**

```python
def test_pi05_mobile_aloha_config_contract():
    config = _config.get_config("pi05_mobile_aloha")
    assert config.model.pi05 is True
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 50
    data = config.data.create(config.assets_dirs, config.model)
    assert data.repo_id == "mobile_aloha"
    assert data.episode_split == "train"
    assert data.prompt_from_task is True
    assert data.action_sequence_keys == ("action",)
    assert data.use_quantile_norm is True
```

Add a repack test with flat LeRobot keys and assert that `prompt`, all three images, 14D state, and 16D actions survive.

- [ ] **Step 2: Run the test and confirm the config is absent**

Run: `pytest -q src/openpi/training/mobile_aloha_config_test.py`

Expected: `get_config` rejects `pi05_mobile_aloha`.

- [ ] **Step 3: Implement the data config factory**

Use a repack structure that explicitly retains prompt:

```python
{
    "images": {
        "cam_high": "observation.images.cam_high",
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    },
    "state": "observation.state",
    "actions": "action",
    "prompt": "prompt",
}
```

Build `data_transforms` with `MobileAlohaInputs` and `MobileAlohaOutputs`. Push `DeltaActions`/`AbsoluteActions` using `MOBILE_ALOHA_DELTA_MASK`. Use `ModelTransformFactory()(model_config)` so resize, prompt tokenization, and 32D padding remain standard OpenPI behavior.

- [ ] **Step 4: Add the full fine-tuning config**

```python
TrainConfig(
    name="pi05_mobile_aloha",
    model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
    data=LeRobotMobileAlohaDataConfig(
        repo_id="mobile_aloha",
        base_config=DataConfig(prompt_from_task=True, episode_split="train"),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    batch_size=32,
    num_train_steps=30_000,
    save_interval=1_000,
    keep_period=5_000,
)
```

Do not point `AssetsConfig` at standard ALOHA `trossen` statistics. The normal-stats command will create `assets/pi05_mobile_aloha/mobile_aloha/norm_stats.json` from the selected train episodes.

- [ ] **Step 5: Run transform/config tests**

Run:

```bash
pytest -q src/openpi/training/mobile_aloha_config_test.py src/openpi/policies/mobile_aloha_policy_test.py
```

Expected: repack preserves prompt, semantic actions are 16D before model padding, and the config is 32D/50-step π0.5.

- [ ] **Step 6: Verify train-only normalization without modifying the existing script**

Run after full dataset conversion:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_mobile_aloha
```

Expected output path: `assets/pi05_mobile_aloha/mobile_aloha/norm_stats.json`. Inspect it and assert state statistics have 14 entries and action statistics have 16 entries. Because the config carries `episode_split="train"`, `scripts/compute_norm_stats.py` and training select the same train episodes automatically.

### Task 6: Add decoded 16D offline metrics and horizon masking

**Files:**
- Create: `src/openpi/evaluation/__init__.py`
- Create: `src/openpi/evaluation/mobile_aloha.py`
- Create: `src/openpi/evaluation/mobile_aloha_test.py`

**Interfaces:**
- Produces: `compute_action_metrics(prediction, target, *, base_motion_threshold=0.01)`.
- Produces: `compute_horizon_metrics(predicted_chunks, target_chunks, valid_mask)`.
- Produces: `evaluate_episode(policy, dataset, *, episode_index, seed) -> EpisodeEvaluation`.

- [ ] **Step 1: Write failing metric/alignment tests**

Cover exact-zero predictions, known constant offsets, group dimensions, base sign, false starts, and padded horizon tails:

```python
def test_horizon_metrics_ignore_padded_tail():
    pred = np.zeros((2, 50, 16), dtype=np.float32)
    target = np.zeros_like(pred)
    target[1, 10:, :] = 1000.0
    valid = np.ones((2, 50), dtype=bool)
    valid[1, 10:] = False
    metrics = compute_horizon_metrics(pred, target, valid)
    assert metrics["overall_mae"] == 0.0


def test_one_step_trace_uses_chunk_index_zero():
    chunks = np.arange(3 * 50 * 16).reshape(3, 50, 16)
    np.testing.assert_array_equal(first_step_trace(chunks), chunks[:, 0, :])
```

Group slices are arms `[0:6, 7:13]`, grippers `[6,13]`, and base `[14:16]`. Horizon buckets are `h0`, `h1_9`, `h10_24`, and `h25_49`.

- [ ] **Step 2: Run tests and confirm the evaluation module is missing**

Run: `pytest -q src/openpi/evaluation/mobile_aloha_test.py -k 'metrics or horizon or first_step'`

Expected: collection fails on the missing module.

- [ ] **Step 3: Implement metrics only on finite decoded 16D arrays**

Validate shapes and finite values before computing. Base direction accuracy uses elements whose ground-truth absolute velocity is at least `0.01`. False-start rate uses frames where `norm(target_base) < 0.01` and counts `norm(predicted_base) >= 0.01`.

- [ ] **Step 4: Implement teacher-forced episode inference**

Instantiate a one-episode LeRobot dataset with:

```python
delta_timestamps={"action": [h / fps for h in range(50)]}
```

For every frame, construct the canonical policy observation without including target actions:

```python
observation = {
    "state": to_numpy(item["observation.state"]),
    "images": {
        "cam_high": to_numpy(item["observation.images.cam_high"]),
        "cam_left_wrist": to_numpy(item["observation.images.cam_left_wrist"]),
        "cam_right_wrist": to_numpy(item["observation.images.cam_right_wrist"]),
    },
    "prompt": item["task"],
}
```

Generate deterministic flow noise with `np.random.SeedSequence([seed, episode_index, frame_index])`, shape `(50,32)`, and pass it to `policy.infer(observation, noise=noise)`. Store returned post-transform `[50,16]`, raw target `item["action"]`, and `~item["action_is_pad"]`.

- [ ] **Step 5: Run numerical evaluation tests**

Run: `pytest -q src/openpi/evaluation/mobile_aloha_test.py -k 'not render and not video'`

Expected: alignment, tail masking, groups, and deterministic noise tests pass without loading a checkpoint.

### Task 7: Render prediction-versus-target artifacts and add the evaluator CLI

**Files:**
- Modify: `src/openpi/evaluation/mobile_aloha.py`
- Modify: `src/openpi/evaluation/mobile_aloha_test.py`
- Create: `scripts/evaluate_mobile_aloha.py`

**Interfaces:**
- Produces: `render_action_summary`, `render_episode_video`, and JSON-safe metrics serialization.
- CLI consumes `--config-name`, `--checkpoint-dir`, `--output-dir`, `--episode-index`, `--max-episodes`, `--seed`, and `--max-frames`.

- [ ] **Step 1: Write failing renderer tests**

With five synthetic frames, assert the static PNG is non-empty, the composed RGB video frame has the requested dimensions, all 16 channel labels are present in renderer metadata, and the output JSON contains checkpoint/config/manifest identifiers.

- [ ] **Step 2: Implement the 16-channel static chart**

Use a 4×4 subplot layout. Draw target in black and prediction in red. Use these labels in physical order:

```text
L waist, L shoulder, L elbow, L forearm, L wrist angle, L wrist rotate, L gripper,
R waist, R shoulder, R elbow, R forearm, R wrist angle, R wrist rotate, R gripper,
base linear, base angular
```

Compute each y-range from the combined finite target/prediction extent with a 5% margin; do not force gripper or base axes into arm joint ranges.

- [ ] **Step 3: Implement efficient rolling-curve MP4 composition**

Precompute screen-space coordinates for all 16 prediction/target sequences. For each video frame, select a clamped 10-second window centered on the current time, draw only those polyline segments into a 4×4 OpenCV chart grid, and draw the current-time cursor. Compose `cam_high` on the left, add two wrist thumbnails, and overlay prompt/episode/frame/checkpoint text. Write at the dataset fps with OpenCV `mp4v`; fail with a clear error if `VideoWriter.isOpened()` is false.

- [ ] **Step 4: Implement the CLI around the standard checkpoint policy loader**

The CLI must:

1. Require `config_name == "pi05_mobile_aloha"` for this first implementation.
2. Load the split manifest and select validation episode IDs unless one explicit validation ID is supplied.
3. Call `policy_config.create_trained_policy(config, checkpoint_dir)` so checkpoint normalization assets and Mobile output transforms are reused.
4. Write `episode_<id>_prediction.mp4`, `episode_<id>_actions.png`, and `episode_<id>_metrics.json`.
5. Write `summary_metrics.json` containing macro averages across evaluated episodes.

`--max-frames` is for smoke tests and must be recorded in metrics metadata so truncated results cannot be mistaken for full-episode evaluation.

- [ ] **Step 5: Run evaluator tests**

Run: `pytest -q src/openpi/evaluation/mobile_aloha_test.py`

Expected: all metric and rendering tests pass; checkpoint loading remains an explicit GPU-server smoke test.

### Task 8: Document and verify the end-to-end baseline

**Files:**
- Create: `docs/mobile-aloha-pi05-finetuning.md`
- Verify all files from Tasks 1-7.

**Interfaces:**
- Produces a single runbook from raw HDF5 to validation artifacts.

- [ ] **Step 1: Document the task-prompt file contract**

The runbook must show a JSON object whose keys exactly equal the six raw task directory basenames and whose values are natural English commands. It must explain that the converter reports missing/extra mappings and that all six tasks feed one model.

- [ ] **Step 2: Document full conversion and train-only statistics commands**

Use the shared repo id and cache location consistently:

```bash
export LEROBOT_HOME=/data/lerobot

uv run examples/aloha_real/convert_mobile_aloha_data_to_lerobot.py \
  --raw-root /data/mobile_aloha \
  --repo-id mobile_aloha \
  --task-prompts /data/mobile_aloha/task_prompts.json \
  --seed 42 \
  --train-fraction 0.7

uv run scripts/compute_norm_stats.py --config-name pi05_mobile_aloha
```

The runbook must show how to inspect the manifest and confirm the norm-stat dimensions before training.

- [ ] **Step 3: Document JAX full fine-tuning and resume commands**

```bash
uv run scripts/train.py pi05_mobile_aloha \
  --exp-name mobile_aloha_baseline \
  --overwrite

uv run scripts/train.py pi05_mobile_aloha \
  --exp-name mobile_aloha_baseline \
  --resume
```

Explain that full fine-tuning is intended for an A100/H100-class server; a LoRA configuration is not part of this baseline.

- [ ] **Step 4: Document one-episode and full-validation evaluation**

```bash
uv run scripts/evaluate_mobile_aloha.py \
  --config-name pi05_mobile_aloha \
  --checkpoint-dir checkpoints/pi05_mobile_aloha/mobile_aloha_baseline/30000 \
  --output-dir output/mobile_aloha_eval/step_30000 \
  --max-episodes 1

uv run scripts/evaluate_mobile_aloha.py \
  --config-name pi05_mobile_aloha \
  --checkpoint-dir checkpoints/pi05_mobile_aloha/mobile_aloha_baseline/30000 \
  --output-dir output/mobile_aloha_eval/step_30000_all
```

State plainly that these are teacher-forced imitation-gap results, not closed-loop task success.

- [ ] **Step 5: Run targeted formatting and unit verification**

Run:

```bash
ruff check \
  src/openpi/training/episode_splits.py \
  src/openpi/training/mobile_aloha_dataset.py \
  src/openpi/policies/mobile_aloha_policy.py \
  src/openpi/evaluation/mobile_aloha.py \
  scripts/evaluate_mobile_aloha.py \
  examples/aloha_real/convert_mobile_aloha_data_to_lerobot.py

pytest -q \
  src/openpi/training/episode_splits_test.py \
  src/openpi/training/mobile_aloha_dataset_test.py \
  src/openpi/policies/mobile_aloha_policy_test.py \
  src/openpi/training/mobile_aloha_config_test.py \
  src/openpi/training/data_loader_test.py \
  src/openpi/evaluation/mobile_aloha_test.py
```

On Apple Silicon, run the HDF5/manifest tests in an isolated CPU environment if the repository's CUDA-pinned JAX dependency cannot resolve. Run the complete import, checkpoint, one-batch, and inference tests on the supported Ubuntu NVIDIA server.

- [ ] **Step 6: Run server smoke checks before a long job**

1. Inspect one transformed sample: state `[32]`, action chunk `[50,32]`, non-padding action prefix `[50,16]`, and tokenized non-empty prompt.
2. Load `pi05_base` into `pi05_mobile_aloha` and verify parameter shapes without beginning a long run.
3. Overfit one small batch for enough steps to see loss decrease and inspect decoded base dimensions.
4. Run a short training job through one checkpoint save.
5. Evaluate one validation episode with `--max-frames 20` and inspect the three expected output files.

- [ ] **Step 7: Review the final diff without committing**

Run:

```bash
git status --short
git diff --check
git diff -- src/openpi examples/aloha_real scripts docs/mobile-aloha-pi05-finetuning.md
```

Expected: no unrelated user changes are reverted or reformatted. Report test evidence and any GPU-only checks that remain; do not commit or push without explicit authorization.

"""Convert static or Mobile ALOHA HDF5 episodes to LeRobot format.

For Mobile ALOHA, the physical schema is 14D qpos state and 16D action:
left arm/gripper, right arm/gripper, base linear velocity, and base angular velocity.


改动
- observation.state = qpos[14]
- action = concat(action[14], base_action[2])，得到 16 维原始物理动作
- motors 顺序统一为左臂 7 维、右臂 7 维、底盘线速度、角速度
- Mobile 模式只使用三路相机
- 每帧写入自然语言 task prompt
- 支持六个任务目录和独立 prompt JSON 映射
- 全局按 episode、seed 42、7:3 划分 train/val，不做任务分层
- 生成 split manifest 和数据审计报告
- 检查维度、时间长度、NaN/Inf、缺失相机、JPEG 解码
- 不进行底盘平滑、动作裁剪或时间偏移
- 默认不上传、不覆盖已有 LeRobot 数据集
- 修正了当前锁定 LeRobot 版本的 HF_LEROBOT_HOME 和 save_episode() API

"""

import dataclasses
import io
import json
import random
import re
import shutil
from pathlib import Path
from typing import Literal

import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import tqdm
import tyro


FPS = 50
ALOHA_MOTOR_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)
MOBILE_ACTION_NAMES = (*ALOHA_MOTOR_NAMES, "base_linear_velocity", "base_angular_velocity")
ALOHA_CAMERAS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
MOBILE_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
SPLIT_MANIFEST_NAME = "openpi_episode_split.json"
AUDIT_REPORT_NAME = "openpi_mobile_aloha_audit.json"


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    path: Path
    task_directory: str
    prompt: str


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    is_mobile: bool = False,
    cameras: tuple[str, ...] | list[str] | None = None,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    # Raw ALOHA records are ordered left arm/gripper first, then right arm/gripper.
    motors = list(ALOHA_MOTOR_NAMES)
    action_names = list(MOBILE_ACTION_NAMES if is_mobile else ALOHA_MOTOR_NAMES)
    cameras = tuple(cameras or (MOBILE_CAMERAS if is_mobile else ALOHA_CAMERAS))

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": [
                action_names,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    if not hdf5_files:
        raise ValueError("No HDF5 episodes were found")
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    if not hdf5_files:
        raise ValueError("No HDF5 episodes were found")
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    if not hdf5_files:
        raise ValueError("No HDF5 episodes were found")
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def _episode_number(path: Path) -> int:
    match = re.fullmatch(r"episode_(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"Unexpected episode filename: {path.name}")
    return int(match.group(1))


def _sorted_episodes(directory: Path) -> list[Path]:
    return sorted(directory.glob("episode_*.hdf5"), key=_episode_number)


def _load_task_prompts(path: Path) -> dict[str, str]:
    raw_mapping = json.loads(path.read_text())
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"Task prompt mapping must be a JSON object: {path}")

    mapping: dict[str, str] = {}
    for task_directory, prompt in raw_mapping.items():
        if not isinstance(task_directory, str) or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Task prompt entries must map non-empty strings to non-empty strings: {path}")
        mapping[task_directory] = prompt.strip()
    return mapping


def discover_episode_specs(raw_dir: Path, task: str, task_prompts_path: Path | None = None) -> list[EpisodeSpec]:
    """Discover either one flat task or multiple task subdirectories."""
    flat_episodes = _sorted_episodes(raw_dir)
    if flat_episodes:
        prompt = task.strip()
        task_directory = raw_dir.name
        if task_prompts_path is not None:
            mapping = _load_task_prompts(task_prompts_path)
            if task_directory not in mapping:
                raise ValueError(f"Missing prompt for task directory {task_directory!r}")
            prompt = mapping[task_directory]
        if not prompt:
            raise ValueError("Task prompt must not be empty")
        return [EpisodeSpec(path, task_directory, prompt) for path in flat_episodes]

    task_directories = sorted(path for path in raw_dir.iterdir() if path.is_dir() and _sorted_episodes(path))
    if not task_directories:
        raise ValueError(f"No episode_*.hdf5 files found under {raw_dir}")
    if task_prompts_path is None:
        raise ValueError("--task-prompts-path is required when raw data contains task subdirectories")

    mapping = _load_task_prompts(task_prompts_path)
    directory_names = {path.name for path in task_directories}
    missing = sorted(directory_names - mapping.keys())
    extra = sorted(mapping.keys() - directory_names)
    if missing or extra:
        raise ValueError(f"Task prompt mapping mismatch: missing={missing}, extra={extra}")

    return [
        EpisodeSpec(path, task_directory.name, mapping[task_directory.name])
        for task_directory in task_directories
        for path in _sorted_episodes(task_directory)
    ]


def _validate_episode(ep: h5py.File, ep_path: Path, cameras: tuple[str, ...], *, is_mobile: bool) -> int:
    required = ["/observations/qpos", "/action", *[f"/observations/images/{camera}" for camera in cameras]]
    if is_mobile:
        required.append("/base_action")
    missing = [key for key in required if key not in ep]
    if missing:
        raise ValueError(f"{ep_path}: missing required dataset(s): {missing}")

    qpos = ep["/observations/qpos"]
    action = ep["/action"]
    if qpos.ndim != 2 or qpos.shape[1] != 14:
        raise ValueError(f"{ep_path}: expected /observations/qpos [T,14], got {qpos.shape}")
    if action.ndim != 2 or action.shape[1] != 14:
        raise ValueError(f"{ep_path}: expected /action [T,14], got {action.shape}")

    length = qpos.shape[0]
    if length == 0:
        raise ValueError(f"{ep_path}: episode has no frames")
    time_series_keys = ["/action", *[f"/observations/images/{camera}" for camera in cameras]]
    for optional_key in ("/observations/qvel", "/observations/effort"):
        if optional_key in ep:
            time_series_keys.append(optional_key)
    if is_mobile:
        base_action = ep["/base_action"]
        if base_action.ndim != 2 or base_action.shape[1] != 2:
            raise ValueError(f"{ep_path}: expected /base_action [T,2], got {base_action.shape}")
        time_series_keys.append("/base_action")
    mismatched = {key: ep[key].shape[0] for key in time_series_keys if ep[key].shape[0] != length}
    if mismatched:
        raise ValueError(f"{ep_path}: expected all streams to have length {length}, got {mismatched}")

    finite_keys = ["/observations/qpos", "/action"]
    if is_mobile:
        finite_keys.append("/base_action")
    for key in finite_keys:
        if not np.isfinite(ep[key][:]).all():
            raise ValueError(f"{ep_path}: {key} contains NaN or Inf")
    return length


def _decode_image(image_dataset: h5py.Dataset, frame_index: int, *, ep_path: Path, camera: str) -> np.ndarray:
    if image_dataset.ndim == 4:
        image = np.asarray(image_dataset[frame_index])
    elif image_dataset.ndim == 2:
        from PIL import Image

        encoded = np.asarray(image_dataset[frame_index], dtype=np.uint8).tobytes()
        try:
            with Image.open(io.BytesIO(encoded)) as decoded:
                image = np.asarray(decoded.convert("RGB"))
        except Exception as exc:
            raise ValueError(f"{ep_path}: failed to decode {camera} frame {frame_index}") from exc
    else:
        raise ValueError(f"{ep_path}: unsupported image shape for {camera}: {image_dataset.shape}")

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{ep_path}: expected RGB image for {camera}, got {image.shape}")
    return image.astype(np.uint8, copy=False)


def iter_episode_frames(
    ep_path: Path,
    *,
    cameras: tuple[str, ...] | list[str],
    is_mobile: bool,
    task: str,
):
    """Yield validated frames while keeping at most one decoded image per camera in memory."""
    cameras = tuple(cameras)
    if not task.strip():
        raise ValueError("Task prompt must not be empty")

    with h5py.File(ep_path, "r") as ep:
        num_frames = _validate_episode(ep, ep_path, cameras, is_mobile=is_mobile)
        for frame_index in range(num_frames):
            arm_action = np.asarray(ep["/action"][frame_index], dtype=np.float32)
            action = (
                np.concatenate([arm_action, np.asarray(ep["/base_action"][frame_index], dtype=np.float32)])
                if is_mobile
                else arm_action
            )
            frame = {
                "observation.state": np.asarray(ep["/observations/qpos"][frame_index], dtype=np.float32),
                "action": action.astype(np.float32, copy=False),
                "task": task,
            }
            for camera in cameras:
                frame[f"observation.images.{camera}"] = _decode_image(
                    ep[f"/observations/images/{camera}"], frame_index, ep_path=ep_path, camera=camera
                )
            if "/observations/qvel" in ep:
                frame["observation.velocity"] = np.asarray(
                    ep["/observations/qvel"][frame_index], dtype=np.float32
                )
            if "/observations/effort" in ep:
                frame["observation.effort"] = np.asarray(ep["/observations/effort"][frame_index], dtype=np.float32)
            yield frame


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path] | list[EpisodeSpec],
    task: str | None = None,
    episodes: list[int] | None = None,
    *,
    cameras: tuple[str, ...] | list[str] = ALOHA_CAMERAS,
    is_mobile: bool = False,
) -> LeRobotDataset:
    specs = [
        item
        if isinstance(item, EpisodeSpec)
        else EpisodeSpec(path=item, task_directory=item.parent.name, prompt=task or "")
        for item in hdf5_files
    ]
    selected_indices = list(range(len(specs))) if episodes is None else episodes
    for ep_idx in tqdm.tqdm(selected_indices):
        spec = specs[ep_idx]
        for frame in iter_episode_frames(spec.path, cameras=cameras, is_mobile=is_mobile, task=spec.prompt):
            dataset.add_frame(frame)
        # In the pinned LeRobot API, the task is stored in each frame.
        dataset.save_episode()

    return dataset


def _channel_statistics(values: np.ndarray, names: tuple[str, ...]) -> dict[str, object]:
    quantile_levels = (0.01, 0.1, 0.5, 0.9, 0.99)
    quantiles = np.quantile(values, quantile_levels, axis=0)
    return {
        "names": list(names),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "quantiles": {
            str(level): quantiles[index].tolist() for index, level in enumerate(quantile_levels)
        },
    }


def _audit_episodes(
    episode_specs: list[EpisodeSpec], cameras: tuple[str, ...], *, is_mobile: bool
) -> dict[str, object]:
    total_frames = 0
    prompt_counts: dict[str, int] = {}
    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    decoded_shapes: dict[str, set[tuple[int, ...]]] = {camera: set() for camera in cameras}
    for spec in episode_specs:
        with h5py.File(spec.path, "r") as ep:
            total_frames += _validate_episode(ep, spec.path, cameras, is_mobile=is_mobile)
            state_chunks.append(np.asarray(ep["/observations/qpos"][:], dtype=np.float32))
            arm_action = np.asarray(ep["/action"][:], dtype=np.float32)
            action_chunks.append(
                np.concatenate([arm_action, np.asarray(ep["/base_action"][:], dtype=np.float32)], axis=1)
                if is_mobile
                else arm_action
            )
            for camera in cameras:
                image = _decode_image(ep[f"/observations/images/{camera}"], 0, ep_path=spec.path, camera=camera)
                decoded_shapes[camera].add(image.shape)
        prompt_counts[spec.prompt] = prompt_counts.get(spec.prompt, 0) + 1
    states = np.concatenate(state_chunks, axis=0)
    actions = np.concatenate(action_chunks, axis=0)
    return {
        "episode_count": len(episode_specs),
        "total_frames": total_frames,
        "fps": FPS,
        "state_dim": 14,
        "action_dim": 16 if is_mobile else 14,
        "cameras": list(cameras),
        "episodes_per_prompt": prompt_counts,
        "state_statistics": _channel_statistics(states, ALOHA_MOTOR_NAMES),
        "action_statistics": _channel_statistics(
            actions, MOBILE_ACTION_NAMES if is_mobile else ALOHA_MOTOR_NAMES
        ),
        "image_checks": {
            "decoded_first_frame_per_camera_per_episode": True,
            "decoded_frame_count": len(episode_specs) * len(cameras),
            "decoded_shapes": {
                camera: [list(shape) for shape in sorted(shapes)] for camera, shapes in decoded_shapes.items()
            },
            "decode_failures": 0,
        },
    }


def _assign_splits(episode_count: int, train_fraction: float, split_seed: int) -> dict[int, str]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")
    indices = list(range(episode_count))
    random.Random(split_seed).shuffle(indices)
    if episode_count <= 1:
        train_count = episode_count
    else:
        train_count = min(max(int(episode_count * train_fraction), 1), episode_count - 1)
    train_indices = set(indices[:train_count])
    return {index: "train" if index in train_indices else "val" for index in range(episode_count)}


def _write_conversion_metadata(
    dataset_root: Path,
    episode_specs: list[EpisodeSpec],
    audit: dict[str, object],
    *,
    train_fraction: float,
    split_seed: int,
) -> None:
    splits = _assign_splits(len(episode_specs), train_fraction, split_seed)
    records = [
        {
            "converted_episode_id": index,
            "source_task_directory": spec.task_directory,
            "source_episode_file": str(spec.path),
            "prompt": spec.prompt,
            "split": splits[index],
        }
        for index, spec in enumerate(episode_specs)
    ]
    prompt_split_counts: dict[str, dict[str, int]] = {}
    for record in records:
        counts = prompt_split_counts.setdefault(record["prompt"], {"train": 0, "val": 0})
        counts[record["split"]] += 1
    for prompt, counts in prompt_split_counts.items():
        if counts["val"] == 0:
            print(f"Warning: prompt {prompt!r} has no validation episodes after global shuffling")

    metadata_dir = dataset_root / "meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "seed": split_seed,
        "train_fraction": train_fraction,
        "episodes": records,
    }
    (metadata_dir / SPLIT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    train_ids = {record["converted_episode_id"] for record in records if record["split"] == "train"}
    val_ids = {record["converted_episode_id"] for record in records if record["split"] == "val"}
    audit_with_splits = {
        **audit,
        "episodes_per_prompt_and_split": prompt_split_counts,
        "split_summary": {
            "train_episode_count": len(train_ids),
            "val_episode_count": len(val_ids),
            "intersection": sorted(train_ids & val_ids),
        },
    }
    (metadata_dir / AUDIT_REPORT_NAME).write_text(
        json.dumps(audit_with_splits, indent=2, ensure_ascii=False) + "\n"
    )


def _remove_existing_dataset(dataset_root: Path, *, overwrite: bool) -> None:
    if not dataset_root.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Dataset already exists at {dataset_root}; pass --overwrite to replace it")
    lerobot_root = Path(HF_LEROBOT_HOME).resolve()
    resolved = dataset_root.resolve()
    if resolved == lerobot_root or lerobot_root not in resolved.parents:
        raise ValueError(f"Refusing to remove unsafe dataset path: {resolved}")
    shutil.rmtree(resolved)


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    task_prompts_path: Path | None = None,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    train_fraction: float = 0.7,
    split_seed: int = 42,
    audit_only: bool = False,
    overwrite: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        download_raw(raw_dir, repo_id=raw_repo_id)

    episode_specs = discover_episode_specs(raw_dir, task, task_prompts_path)
    if episodes is not None:
        episode_specs = [episode_specs[index] for index in episodes]
    hdf5_files = [spec.path for spec in episode_specs]
    cameras = MOBILE_CAMERAS if is_mobile else tuple(get_cameras(hdf5_files))
    if is_mobile:
        missing_cameras = sorted(set(MOBILE_CAMERAS) - set(get_cameras(hdf5_files)))
        if missing_cameras:
            raise ValueError(f"Mobile ALOHA episode is missing camera(s): {missing_cameras}")
    audit = _audit_episodes(episode_specs, cameras, is_mobile=is_mobile)
    if audit_only:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return None

    dataset_root = Path(HF_LEROBOT_HOME) / repo_id
    _remove_existing_dataset(dataset_root, overwrite=overwrite)

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        is_mobile=is_mobile,
        cameras=cameras,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        episode_specs,
        cameras=cameras,
        is_mobile=is_mobile,
    )
    _write_conversion_metadata(
        dataset_root,
        episode_specs,
        audit,
        train_fraction=train_fraction,
        split_seed=split_seed,
    )

    if push_to_hub:
        dataset.push_to_hub()
    return dataset


if __name__ == "__main__":
    tyro.cli(port_aloha)

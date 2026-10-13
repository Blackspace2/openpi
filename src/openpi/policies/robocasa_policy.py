import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

# Raw dataset layout, see examples/robocasa/merge_target_static.py:
#   observation.state (9): eef_pos_rel(3) + eef_quat_rel(4, xyzw) + gripper_qpos(2)
#   action            (7): eef_pos(3) + eef_rot(3, axis-angle) + gripper_close(1)
RAW_STATE_DIM = 9
ACTION_DIM = 7
# Model-facing state: the quaternion is replaced by a 6D rotation (first two rotation-matrix columns).
#   state (11): eef_pos_rel(3) + eef_rot6d(6) + gripper_qpos(2)
STATE_DIM = 11


def make_robocasa_example() -> dict:
    """Creates a random input example for the RoboCasa policy."""
    quat = np.random.randn(4)
    quat /= np.linalg.norm(quat)
    return {
        "state": np.concatenate([np.random.uniform(-1, 1, 3), quat, np.random.uniform(0, 0.04, 2)]),
        "images": {
            cam: np.random.randint(256, size=(3, 224, 224), dtype=np.uint8) for cam in RobocasaInputs.EXPECTED_CAMERAS
        },
        "actions": np.random.uniform(-1, 1, size=(50, ACTION_DIM)).astype(np.float32),
        "prompt": "pick the mug from the counter and place it under the coffee machine dispenser",
    }


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Converts xyzw quaternions (..., 4) to the 6D rotation representation (..., 6).

    Returns the first two columns of the rotation matrix, [r00, r10, r20, r01, r11, r21]. Unlike the
    quaternion it is continuous and has no q / -q ambiguity (both map to the same output), which
    matters here since ~74% of the episodes contain at least one quaternion sign flip.
    """
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(quat, -1, 0)
    col0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=-1)
    col1 = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=-1)
    return np.concatenate([col0, col1], axis=-1)


def convert_state(state: np.ndarray) -> np.ndarray:
    """Raw dataset state (..., 9) -> model-facing state (..., 11), quaternion replaced by 6D rotation."""
    state = np.asarray(state)
    if state.shape[-1] != RAW_STATE_DIM:
        raise ValueError(f"Expected state to have {RAW_STATE_DIM} dims, got {state.shape}")
    return np.concatenate(
        [state[..., 0:3], quat_xyzw_to_rot6d(state[..., 3:7]), state[..., 7:9]], axis=-1
    ).astype(np.float32)


def _convert_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    # Convert to uint8 if using float images.
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    # Convert from [channel, height, width] to [height, width, channel].
    return einops.rearrange(img, "c h w -> h w c")


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    """Inputs for the RoboCasa PandaOmron policy on base-static tasks (arm only, base never moves).

    State goes in with its quaternion converted to 6D rotation (9 -> 11 dims); actions are already
    normalized [-1, 1] eef deltas and are passed through as-is (no DeltaActions).
    """

    # Model slot <- dataset camera. The two third-person views fill base + one wrist slot since the
    # model's 3-slot layout has no dedicated second third-person slot.
    CAMERA_SLOTS: ClassVar[dict[str, str]] = {
        "base_0_rgb": "robot0_agentview_left",
        "left_wrist_0_rgb": "robot0_eye_in_hand",
        "right_wrist_0_rgb": "robot0_agentview_right",
    }
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = tuple(CAMERA_SLOTS.values())

    def __call__(self, data: dict) -> dict:
        state = convert_state(data["state"])

        in_images = data["images"]
        if set(in_images) != set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        inputs = {
            "image": {slot: _convert_image(in_images[cam]) for slot, cam in self.CAMERA_SLOTS.items()},
            "image_mask": {slot: np.True_ for slot in self.CAMERA_SLOTS},
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != ACTION_DIM:
                raise ValueError(f"Expected actions to have {ACTION_DIM} dims, got {actions.shape}")
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    """Outputs for the RoboCasa policy: the 7 env action dims, matching env.convert_action()[0:7]."""

    def __call__(self, data: dict) -> dict:
        # Only the first ACTION_DIM dims are meaningful; the rest is model padding.
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_forklift_example(state_dim: int = 2, action_dim: int = 3) -> dict:
    """Creates a random input example for the Forklift policy."""
    return {
        "state": np.ones((state_dim,)),
        "images": {
            "cam_front": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "actions": np.random.uniform(-1, 1, size=(action_dim,)).astype(np.float32),
        "prompt": "pick up the pallet with the forklift forks and raise them to carrying height",
    }


def _convert_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    # Convert to uint8 if using float images.
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    # Convert from [channel, height, width] to [height, width, channel].
    return einops.rearrange(img, "c h w -> h w c")


@dataclasses.dataclass(frozen=True)
class ForkliftInputs(transforms.DataTransformFn):
    """Inputs for the Ackermann forklift policy.

    Generic across state/action designs (see examples/forklift_sim_command/README.md and
    examples/forklift_sim_waypoints/README.md for what each design's dims actually mean) --
    `state_dim`/`action_dim` are the only thing that differs between designs; everything else
    (camera handling, image masking) is shared, since none of these dims are robot joint angles
    and no design needs a coordinate-space conversion (e.g. `adapt_to_pi`) -- state/actions are
    passed through as-is.

    If a future design needs real per-dim transform logic (not just a dim check), give it its own
    DataTransformFn instead of overloading this class with a transform callback -- see
    docs on this class's history for why a plain int-parameterized class was preferred over a
    speculative hook.
    """

    state_dim: int
    action_dim: int

    # If set, truncates the action chunk's horizon axis (axis -2) to this length. Only meaningful
    # for prechunked designs (e.g. examples/forklift_sim_waypoints, whose `action` feature is a full
    # [ACTION_HORIZON, action_dim] chunk per row) -- lets a dataset converted once at some max
    # horizon serve horizon-ablation TrainConfigs with a smaller `model.action_horizon` without
    # reconverting the data, since chunk step k's value doesn't depend on the horizon it was
    # converted with (see LeRobotForkliftDataConfig in src/openpi/training/config.py, which wires
    # this from `model_config.action_horizon`). None (the default, and always the case for
    # non-prechunked designs like forklift_sim_command) means no truncation.
    action_horizon: int | None = None

    # Only the forward RGB camera is used today. "cam_depth" is reserved for a future second
    # camera (depth expanded to 3 channels), see examples/forklift_sim_*/convert_forklift_data_to_lerobot.py's
    # --include-depth option.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_front",)

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state to have {self.state_dim} dims, got {state.shape}")

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = _convert_image(in_images["cam_front"])

        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        # The model expects a fixed 3-camera slot layout; mask out the wrist slots we don't have.
        for dest in ("left_wrist_0_rgb", "right_wrist_0_rgb"):
            images[dest] = np.zeros_like(base_image)
            image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != self.action_dim:
                raise ValueError(f"Expected actions to have {self.action_dim} dims, got {actions.shape}")
            if self.action_horizon is not None:
                # actions is [..., H_max, action_dim] here (prechunked design) -- truncate the
                # horizon axis so a dataset converted once at H_max can serve any smaller
                # model.action_horizon (see the field docstring above).
                actions = actions[..., : self.action_horizon, :]
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ForkliftOutputs(transforms.DataTransformFn):
    """Outputs for the Ackermann forklift policy. Generic across state/action designs -- see ForkliftInputs."""

    action_dim: int

    def __call__(self, data: dict) -> dict:
        # Only the first action_dim physical dims are meaningful; the rest is model padding.
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.policies import aloha_policy


def make_mobile_aloha_example(action_horizon: int = 50) -> dict:
    """Creates a random input example for the Mobile Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "actions": np.random.uniform(-1, 1, size=(action_horizon, 16)).astype(np.float32),
        "prompt": "do something",
    }


def _convert_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    # Convert to uint8 if using float images.
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    # Convert from [channel, height, width] to [height, width, channel].
    return einops.rearrange(img, "c h w -> h w c")


@dataclasses.dataclass(frozen=True)
class MobileAlohaInputs(transforms.DataTransformFn):
    """Inputs for the Mobile Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14] (left arm/gripper, right arm/gripper)
    - actions: [action_horizon, 16] (first 14 dims are arm/gripper, last 2 dims are base linear/angular velocity)
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # Mobile Aloha only has three cameras (no cam_low).
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape[-1] != 14:
            raise ValueError(f"Expected state to have 14 dims, got {state.shape}")

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = _convert_image(in_images["cam_high"])

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = _convert_image(in_images[source])
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": aloha_policy.aloha_state_to_pi(state, adapt_to_pi=self.adapt_to_pi),
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != 16:
                raise ValueError(f"Expected actions to have 16 dims, got {actions.shape}")
            # The base velocity dims (14:16) are not arm/gripper joints, so they are left untouched.
            arm_actions = aloha_policy.aloha_actions_to_pi(actions[..., :14], adapt_to_pi=self.adapt_to_pi)
            base_actions = actions[..., 14:16]
            inputs["actions"] = np.concatenate([arm_actions, base_actions], axis=-1)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class MobileAlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Mobile Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 16 semantic dims (14 arm/gripper + 2 base velocity); the rest is model padding.
        actions = np.asarray(data["actions"][..., :16])
        arm_actions = aloha_policy.pi_actions_to_aloha(actions[..., :14], adapt_to_pi=self.adapt_to_pi)
        base_actions = actions[..., 14:16]
        return {"actions": np.concatenate([arm_actions, base_actions], axis=-1)}

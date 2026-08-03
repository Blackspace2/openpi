import numpy as np

from openpi.policies import mobile_aloha_policy


def test_make_mobile_aloha_example():
    example = mobile_aloha_policy.make_mobile_aloha_example()
    assert example["state"].shape == (14,)
    assert example["actions"].shape[-1] == 16


def test_inputs_rejects_wrong_state_dim():
    example = mobile_aloha_policy.make_mobile_aloha_example()
    example["state"] = np.ones((16,))
    inputs = mobile_aloha_policy.MobileAlohaInputs()
    try:
        inputs(example)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for wrong state dim")


def test_inputs_rejects_wrong_action_dim():
    example = mobile_aloha_policy.make_mobile_aloha_example()
    example["actions"] = np.ones((50, 14))
    inputs = mobile_aloha_policy.MobileAlohaInputs()
    try:
        inputs(example)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for wrong action dim")


def test_inputs_maps_three_cameras():
    example = mobile_aloha_policy.make_mobile_aloha_example()
    inputs = mobile_aloha_policy.MobileAlohaInputs()(example)

    assert set(inputs["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert bool(inputs["image_mask"]["base_0_rgb"])
    assert bool(inputs["image_mask"]["left_wrist_0_rgb"])
    assert bool(inputs["image_mask"]["right_wrist_0_rgb"])
    assert inputs["state"].shape == (14,)
    assert inputs["actions"].shape == (50, 16)


def test_base_velocity_dims_untouched_by_aloha_transform():
    """The last two action dims (base linear/angular velocity) are not arm/gripper joints, so
    MobileAlohaInputs/Outputs must pass them through unchanged (no flip, no gripper remap)."""
    example = mobile_aloha_policy.make_mobile_aloha_example(action_horizon=1)
    base_velocity = np.array([0.37, -0.12])
    example["actions"][0, 14:16] = base_velocity

    inputs = mobile_aloha_policy.MobileAlohaInputs()(example)
    np.testing.assert_allclose(inputs["actions"][0, 14:16], base_velocity)

    outputs = mobile_aloha_policy.MobileAlohaOutputs()({"actions": inputs["actions"]})
    np.testing.assert_allclose(outputs["actions"][0, 14:16], base_velocity)


def test_output_restores_16d_action_and_ignores_padding():
    """Outputs must slice off model padding (e.g. 32D -> 16D) and only decode the real dims."""
    action_horizon = 4
    padded_actions = np.zeros((action_horizon, 32), dtype=np.float32)
    padded_actions[:, :16] = np.random.uniform(-1, 1, size=(action_horizon, 16))

    outputs = mobile_aloha_policy.MobileAlohaOutputs()({"actions": padded_actions})
    assert outputs["actions"].shape == (action_horizon, 16)


def test_input_output_roundtrip():
    """Round-tripping actions through MobileAlohaInputs then MobileAlohaOutputs should recover the
    original 16D action (up to the ALOHA gripper/joint transform's own numerical precision)."""
    example = mobile_aloha_policy.make_mobile_aloha_example(action_horizon=8)
    original_actions = example["actions"].copy()

    inputs = mobile_aloha_policy.MobileAlohaInputs()(example)
    outputs = mobile_aloha_policy.MobileAlohaOutputs()({"actions": inputs["actions"]})

    np.testing.assert_allclose(outputs["actions"], original_actions, atol=1e-5)


def test_inputs_does_not_mutate_caller_arrays():
    example = mobile_aloha_policy.make_mobile_aloha_example(action_horizon=2)
    state_before = example["state"].copy()
    actions_before = example["actions"].copy()

    mobile_aloha_policy.MobileAlohaInputs()(example)

    np.testing.assert_array_equal(example["state"], state_before)
    np.testing.assert_array_equal(example["actions"], actions_before)

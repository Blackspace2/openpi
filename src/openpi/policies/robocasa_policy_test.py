import numpy as np
import pytest

from openpi.policies import robocasa_policy


def _quat_xyzw_to_matrix_ref(q: np.ndarray) -> np.ndarray:
    # Independent reference via rotating the basis vectors: v' = v + 2w(u x v) + 2u x (u x v).
    u, w = q[:3], q[3]
    cols = []
    for v in np.eye(3):
        t = 2 * np.cross(u, v)
        cols.append(v + w * t + np.cross(u, t))
    return np.stack(cols, axis=1)


def test_rot6d_matches_rotation_matrix():
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        m = _quat_xyzw_to_matrix_ref(q)
        np.testing.assert_allclose(robocasa_policy.quat_xyzw_to_rot6d(q), m[:, :2].T.ravel(), atol=1e-12)


def test_rot6d_sign_invariant_and_batched():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(5, 7, 4))
    np.testing.assert_allclose(robocasa_policy.quat_xyzw_to_rot6d(q), robocasa_policy.quat_xyzw_to_rot6d(-q))
    assert robocasa_policy.quat_xyzw_to_rot6d(q).shape == (5, 7, 6)


def test_identity_quaternion():
    np.testing.assert_allclose(
        robocasa_policy.quat_xyzw_to_rot6d(np.array([0.0, 0.0, 0.0, 1.0])), [1, 0, 0, 0, 1, 0], atol=1e-12
    )


def test_inputs_layout():
    example = robocasa_policy.make_robocasa_example()
    inputs = robocasa_policy.RobocasaInputs()(example)

    assert set(inputs["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(bool(m) for m in inputs["image_mask"].values())
    assert inputs["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert inputs["state"].shape == (robocasa_policy.STATE_DIM,)
    assert inputs["state"].dtype == np.float32
    np.testing.assert_allclose(inputs["state"][:3], example["state"][:3], rtol=1e-6)
    np.testing.assert_allclose(inputs["state"][9:], example["state"][7:], rtol=1e-6)
    np.testing.assert_array_equal(inputs["actions"], example["actions"])


def test_inputs_rejects_wrong_dims():
    inputs = robocasa_policy.RobocasaInputs()
    example = robocasa_policy.make_robocasa_example()
    example["state"] = np.ones((16,))
    with pytest.raises(ValueError, match="state"):
        inputs(example)
    example = robocasa_policy.make_robocasa_example()
    example["actions"] = np.ones((50, 12))
    with pytest.raises(ValueError, match="actions"):
        inputs(example)


def test_outputs_truncates_padding():
    out = robocasa_policy.RobocasaOutputs()({"actions": np.ones((50, 32))})
    assert out["actions"].shape == (50, robocasa_policy.ACTION_DIM)

import openpi.training.config as _config
import openpi.transforms as _transforms


def test_pi05_mobile_aloha_config():
    config = _config.get_config("pi05_mobile_aloha")

    assert config.model.pi05 is True
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 50

    data = config.data
    assert isinstance(data, _config.LeRobotMobileAlohaDataConfig)
    assert data.repo_id == "mobile_aloha"
    assert data.base_config is not None
    assert data.base_config.episode_split == "train"
    assert data.base_config.prompt_from_task is True

    # Delta mask must stay 14-dim (12 arm joints + 2 grippers) so it never touches the 2 base
    # velocity action dims, which must remain absolute.
    mask = _transforms.make_bool_mask(6, -1, 6, -1)
    assert len(mask) == 14


def test_data_config_episode_split_defaults_to_all():
    assert _config.DataConfig().episode_split == "all"


def test_mobile_aloha_data_config_create_wires_14dim_delta_mask(tmp_path):
    config = _config.get_config("pi05_mobile_aloha")
    data_config = config.data.create(tmp_path, config.model)

    # DeltaActions/AbsoluteActions must be pushed with a 14-dim mask (12 arm joints + 2 grippers),
    # not expanded to 16 or 32 dims, so the base velocity and model-padding dims stay untouched.
    delta_actions = [t for t in data_config.data_transforms.inputs if isinstance(t, _transforms.DeltaActions)]
    assert len(delta_actions) == 1
    assert len(delta_actions[0].mask) == 14

    absolute_actions = [t for t in data_config.data_transforms.outputs if isinstance(t, _transforms.AbsoluteActions)]
    assert len(absolute_actions) == 1
    assert len(absolute_actions[0].mask) == 14

    assert data_config.action_sequence_keys == ("action",)


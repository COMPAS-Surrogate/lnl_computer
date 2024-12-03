from lnl_computer.observation import Observation, load_observation


def test_load_obs(tmpdir, mock_data):

    obs = load_observation("LVK")
    assert obs is not None
    assert isinstance(obs, Observation)

    obs = load_observation(mock_data.observations_filename)
    assert obs is not None
    assert isinstance(obs, Observation)

    data_dict = obs.__dict__()

    obs1 = Observation.from_dict(data_dict)
    obs2 = Observation.from_dict(data_dict)
    assert isinstance(obs, Observation)
    assert obs1.duration == obs2.duration
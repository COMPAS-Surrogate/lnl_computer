from lnl_computer.observation import Observation, load_observation


def test_load_obs(tmpdir, mock_data):

    obs = load_observation("LVK")
    assert obs is not None
    assert isinstance(obs, Observation)

    obs = load_observation(mock_data.observations_filename)
    assert obs is not None
    assert isinstance(obs, Observation)

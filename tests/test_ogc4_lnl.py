def test_ogc4_lnl(mock_data: MockData):
    lnl, unc = mock_data.mcz_grid.get_lnl(
        mock_data.observations.mcz, duration=1
    )
    assert lnl > -np.inf
    assert unc != np.nan

    # NO-BOOTSTRAPS LNL
    lnl, unc = McZGrid.lnl(
        mcz_obs=mock_data.observations.mcz,
        duration=1,
        compas_h5_path=mock_data.compas_filename,
        sf_sample=dict(aSF=0.01, dSF=4.70, mu_z=-0.23),
        n_bootstraps=0,
    )
    assert lnl > -np.inf
    assert np.isnan(unc)
    expected_lnl = 10642
    assert (
        np.abs(lnl - expected_lnl) < 5
    ), f"lnl={lnl:.1f} not close to the expected value of {expected_lnl}"

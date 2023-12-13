import os

from lnl_computer.data_generation.detection_matrix_generator import (
    generate_set_of_matrices,
)


def test_datagen(tmp_path, test_datapath):
    os.chdir(tmp_path)
    generate_set_of_matrices(
        compas_h5_path=test_datapath,
        n=1,
        save_images=False,
        outdir=tmp_path,
        parameters=["aSF"],
        save_h5_fname="det_matrix.h5",
    )
    assert os.path.exists("det_matrix.h5")

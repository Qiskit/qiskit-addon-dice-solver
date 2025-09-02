import os
import struct

import numpy as np

from qiskit_addon_dice_solver.dice_outfile_reader import from_bin_file_to_sci


def to_dice_det_str(alpha: str, beta: str):
    dice_det = ""

    for a, b in zip(alpha, beta):
        if a == "1" and b == "1":
            dice_det += "2"
        elif a == "1" and b == "0":
            dice_det += "a"
        elif a == "0" and b == "1":
            dice_det += "b"
        else:
            dice_det += "0"

    return dice_det


def test_from_bin_file_to_sci(tmp_path):
    num_spatial_orbs = 25

    max_int = 2**num_spatial_orbs - 1

    alphas = np.random.randint(0, max_int, size=300, dtype=np.uint64)
    betas = np.random.randint(0, max_int, size=500, dtype=np.uint64)

    alphas = np.unique(alphas)
    betas = np.unique(betas)

    fpath = os.path.join(tmp_path, "dets.bin")
    with open(fpath, 'wb') as tmp:
        tmp.write(
            (alphas.shape[0] * betas.shape[0]).to_bytes(length=4, byteorder="little")
        )
        tmp.write(num_spatial_orbs.to_bytes(length=4, byteorder="little"))

        amps = np.zeros((alphas.shape[0], betas.shape[0]), dtype=np.float64)
        for row, alpha in enumerate(alphas):
            alpha = f"{alpha:0{num_spatial_orbs}b}"[::-1]
            for col, beta in enumerate(betas):
                amp = np.random.uniform(-1, 1)
                beta = f"{beta:0{num_spatial_orbs}b}"[::-1]
                amps[row][col] = amp
                dice_det = to_dice_det_str(alpha=alpha, beta=beta)
                tmp.write(struct.pack("<d", amp))
                tmp.write(dice_det.encode())

    ci_vec, unique_a, unique_b = from_bin_file_to_sci(fpath)

    assert np.array_equal(amps, ci_vec)
    assert np.array_equal(alphas, unique_a)
    assert np.array_equal(betas, unique_b)

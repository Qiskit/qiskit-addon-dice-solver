# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os

from pyscf import ao2mo, tools
import numpy as np

from qiskit_addon_sqd.counts import generate_counts_uniform, counts_to_arrays
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.subsampling import postselect_and_subsample
from qiskit_addon_dice_solver import solve_fermion


# Specify molecule properties
num_orbitals = 16
num_elec_a = num_elec_b = 5
open_shell = False
spin_sq = 0

# Read in molecule from disk
active_space_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "molecules", "n2_fci.txt"
)
mf_as = tools.fcidump.to_scf(active_space_path)
hcore = mf_as.get_hcore()
eri = ao2mo.restore(1, mf_as._eri, num_orbitals)
nuclear_repulsion_energy = mf_as.mol.energy_nuc()

# Create a seed to control randomness throughout this workflow
rand_seed = 42

# Generate random samples
counts_dict = generate_counts_uniform(10_000, num_orbitals * 2, rand_seed=rand_seed)

# Convert counts into bitstring and probability arrays
bitstring_matrix_full, probs_arr_full = counts_to_arrays(counts_dict)

# SQSD options
iterations = 5

# Eigenstate solver options
n_batches = 5
samples_per_batch = 300
max_davidson_cycles = 200

# Self-consistent configuration recovery loop
avg_occupancy = None
e_hist = np.zeros((iterations, n_batches))
for i in range(iterations):
    print(f"Starting configuration recovery iteration {i}")
    # On the first iteration, we have no orbital occupancy information from the
    # solver, so we just post-select from the full bitstring set based on hamming weight.
    if avg_occupancy is None:
        bs_mat_tmp = bitstring_matrix_full
        probs_arr_tmp = probs_arr_full

    # In following iterations, we use both the occupancy info and the target hamming
    # weight to refine bitstrings.
    else:
        bs_mat_tmp, probs_arr_tmp = recover_configurations(
            bitstring_matrix_full,
            probs_arr_full,
            avg_occupancy,
            num_elec_a,
            num_elec_b,
        )

    # Throw out samples with incorrect hamming weight and create batches of subsamples.
    batches = postselect_and_subsample(
        bs_mat_tmp,
        probs_arr_tmp,
        hamming_right=num_elec_a,
        hamming_left=num_elec_b,
        samples_per_batch=samples_per_batch,
        num_batches=n_batches,
    )
    # Run eigenstate solvers in a loop. This loop should be parallelized for larger problems.
    int_e = np.zeros(n_batches)
    occs_tmp = []
    for j, batch in enumerate(batches):
        energy_sci, wf_mags, avg_occs = solve_fermion(
            batch,
            hcore,
            eri,
            mpirun_options=["-quiet", "-n", "8"],
        )
        energy_sci += nuclear_repulsion_energy
        int_e[j] = energy_sci
        occs_tmp.append(avg_occs)

    # Combine batch results
    avg_occupancy = tuple(np.mean(occs_tmp, axis=0))

    # Track optimization history
    e_hist[i, :] = int_e

print("Exact energy: -109.10288938")
print(f"Estimated energy: {np.min(e_hist[-1])}")

import os

from pyscf import ao2mo, tools
import numpy as np

from sqd.utils.counts import generate_counts_uniform, counts_to_arrays
from sqd.configuration_recovery import recover_configurations
from qiskit_addon_dice_solver import solve_dice
from sqd.subsampling import postselect_and_subsample
from sqd.utils.fermion import (
    bitstring_matrix_to_sorted_addresses,
    flip_orbital_occupancies,
)


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
occupancies_bitwise = None  # orbital i corresponds to column i in bitstring matrix
e_hist = np.zeros((iterations, n_batches))
for i in range(iterations):
    print(f"Starting configuration recovery iteration {i}")
    # On the first iteration, we have no orbital occupancy information from the
    # solver, so we just post-select from the full bitstring set based on hamming weight.
    if occupancies_bitwise is None:
        bs_mat_tmp = bitstring_matrix_full
        probs_arr_tmp = probs_arr_full

    # In following iterations, we use both the occupancy info and the target hamming
    # weight to refine bitstrings.
    else:
        bs_mat_tmp, probs_arr_tmp = recover_configurations(
            bitstring_matrix_full,
            probs_arr_full,
            occupancies_bitwise,
            num_elec_a,
            num_elec_b,
            # rand_seed=rand_seed,
        )

    # Throw out samples with incorrect hamming weight and create batches of subsamples.
    batches = postselect_and_subsample(
        bs_mat_tmp,
        probs_arr_tmp,
        num_elec_a,
        num_elec_b,
        samples_per_batch,
        n_batches,
        # rand_seed=rand_seed,
    )
    # Run eigenstate solvers in a loop. This loop should be parallelized for larger problems.
    int_e = np.zeros(n_batches)
    int_occs = np.zeros((n_batches, 2 * num_orbitals))
    for j in range(n_batches):
        addresses = bitstring_matrix_to_sorted_addresses(
            batches[j], open_shell=open_shell
        )
        energy_sci, wf_mags, avg_occs = solve_dice(
            addresses,
            active_space_path,
            os.path.abspath(os.path.dirname(__file__)),
            spin_sq=spin_sq,
            max_davidson=max_davidson_cycles,
            clean_working_dir=True,
            mpirun_options=["-n", "8"],
        )
        int_e[j] = energy_sci
        int_occs[j, :num_orbitals] = avg_occs[0]
        int_occs[j, num_orbitals:] = avg_occs[1]

    # Combine batch results
    avg_occupancy = np.mean(int_occs, axis=0)
    # The occupancies from the solver should be flipped to match the bits in the bitstring matrix.
    occupancies_bitwise = flip_orbital_occupancies(avg_occupancy)

    # Track optimization history
    e_hist[i, :] = int_e

print(f"Exact energy: -109.10288938")
print(f"Estimated energy: {np.min(e_hist[-1]) + nuclear_repulsion_energy}")

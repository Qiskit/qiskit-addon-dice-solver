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

import numpy as np
from pyscf import ao2mo, tools
from qiskit_addon_dice_solver import solve_sci_batch

from qiskit_addon_sqd.counts import generate_bit_array_uniform
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian

# Specify molecule properties
num_orbitals = 16
num_elec_a = num_elec_b = 5
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
rand_seed = np.random.default_rng(42)

# Generate random samples
bit_array = generate_bit_array_uniform(10_000, num_orbitals * 2, rand_seed=rand_seed)


# Run SQD
result_history = []


def callback(results: list[SCIResult]):
    result_history.append(results)
    iteration = len(result_history)
    print(f"Iteration {iteration}")
    for i, result in enumerate(results):
        print(f"\tSubsample {i}")
        print(f"\t\tEnergy: {result.energy + nuclear_repulsion_energy}")
        print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")


result = diagonalize_fermionic_hamiltonian(
    hcore,
    eri,
    bit_array,
    samples_per_batch=300,
    norb=num_orbitals,
    nelec=(num_elec_a, num_elec_b),
    num_batches=5,
    max_iterations=5,
    sci_solver=solve_sci_batch,
    symmetrize_spin=True,
    callback=callback,
    seed=rand_seed,
)

print("Exact energy: -109.10288938")
print(f"Estimated energy: {result.energy + nuclear_repulsion_energy}")

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

"""A wrapper for the DICE groundstate solver."""

from __future__ import annotations

import os
import tempfile
import struct
import subprocess
import shutil
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from pyscf import tools


# Ensure the runtime linker can find the local boost binaries at runtime
DICE_BIN = os.path.join(os.path.abspath(os.path.dirname(__file__)), "bin")
os.environ["LD_LIBRARY_PATH"] = f"{DICE_BIN}:{os.environ.get('LD_LIBRARY_PATH', '')}"


class DiceExecutionError(Exception):
    """Custom exception for Dice command line application execution errors."""

    def __init__(self, command, returncode, output, stderr, working_dir):
        """Initialize a ``DiceExecutionError`` instance."""
        self.command = command
        self.returncode = returncode
        self.output = output
        self.stderr = stderr
        self.working_dir = working_dir
        message = (
            f"Command '{command}' failed with return code {returncode}\n"
            f"Working directory: {working_dir}\n"
            f"Output: {output}\n"
            f"Error: {stderr}"
        )
        super().__init__(message)


def solve_dice(
    addresses: tuple[Sequence[int], Sequence[int]],
    active_space_path: str | Path,
    working_dir: str | Path,
    spin_sq: float = 0.0,
    max_davidson: int = 100,
    clean_working_dir: bool = True,
    mpirun_options: Sequence[str] | str | None = None,
) -> tuple[float, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Approximate the ground state given an active space and Slater determinant addresses.

    In order to leverage the multi-processing nature of this tool, the user must specify
    the CPU resources to use via the `mpirun_options` argument.

    For example, to use 8 CPU slots in parallel in quiet mode:

    .. code-block:: python

       # Run 8 parallel slots in quiet mode
       mpirun_opts = "-quiet -n 8"
       # OR
       mpirun_opts = ["-quiet", "-n", "8"]

       energy, sci_coeffs, avg_occs = solve_dice(..., mpirun_options=mpirun_opts)

    For more information on the ``mpirun`` command line options, refer to the `man page <https://www.open-mpi.org/doc/current/man1/mpirun.1.php>`_.

    .. note::

       Only closed-shell systems are currently supported. The particle number for both
       spin-up and spin-down determinants is expected to be equal.

    .. note::

       Determinant addresses are interpreted by the ``Dice`` command line application as 5-byte unsigned integers; therefore, only systems
       of ``40`` or fewer orbitals are supported.

    Args:
        addresses: A length-2 tuple of ``Sequence`` containing base-10, unsigned integer
            representations of bitstrings. The first ``Sequence`` represents configurations of the alpha
            particles, and the second ``Sequence`` represents that of the beta particles.
        active_space_path: An absolute path to an FCI dump -- a format partially defined in
            `Knowles and Handy 1989 <https://www.sciencedirect.com/science/article/abs/pii/0010465589900337?via%3Dihub>`_.
        working_dir: An absolute path to a directory in which intermediate files can be written to and read from.
        spin_sq: Target value for the total spin squared for the ground state. If ``None``, no spin will be imposed.
        max_davidson: The maximum number of cycles of Davidson's method to perform.
        clean_working_dir: A flag indicating whether to remove the intermediate files used by the ``Dice``
            command line application. If ``False``, the intermediate files will be left in a temporary directory in the
            ``working_dir``.
        mpirun_options: Options controlling the CPU resource allocation for the ``Dice`` command line application.
            These command-line options will be passed directly to the ``mpirun`` command line application during
            invocation of ``Dice``. These may be formatted as a ``Sequence`` of strings or a single string. If a ``Sequence``,
            the elements will be combined into a single, space-delimited string and passed to
            ``mpirun``. If the input is a single string, it will be passed to ``mpirun`` as-is. If no
            ``mpirun_options`` are provided by the user, ``Dice`` will run on a single MPI slot. For more
            information on the ``mpirun`` command line options, refer to the `man page <https://www.open-mpi.org/doc/current/man1/mpirun.1.php>`_.

    Returns:
        Minimum energy from SCI calculation, SCI coefficients, and average orbital occupancy for spin-up and spin-down orbitals
    """
    # Write Dice inputs to working dir
    num_configurations = len(addresses[0])
    num_up = bin(addresses[0][0])[2:].count("1")
    num_dn = bin(addresses[1][0])[2:].count("1")

    intermediate_dir = Path(tempfile.mkdtemp(prefix="dice_cli_files_", dir=working_dir))

    _write_input_files(
        addresses,
        active_space_path,
        num_up,
        num_dn,
        num_configurations,
        intermediate_dir,
        spin_sq,
        max_davidson,
    )

    # Navigate to working dir and call Dice
    _call_dice(intermediate_dir, mpirun_options)

    # Read outputs and convert outputs
    mf_as = tools.fcidump.to_scf(active_space_path)
    num_orbitals = mf_as.get_hcore().shape[0]
    e_dice, sci_coefficients, avg_occupancies = _read_dice_outputs(
        intermediate_dir, num_orbitals
    )
    e_dice -= mf_as.mol.energy_nuc()

    # Clean up the working directory of intermediate files, if desired
    if clean_working_dir:
        shutil.rmtree(intermediate_dir)

    return (
        e_dice,
        sci_coefficients,
        (avg_occupancies[:num_orbitals], avg_occupancies[num_orbitals:]),
    )


def _read_dice_outputs(
    working_dir: str | Path, num_orbitals: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Calculate the estimated ground state energy and average orbitals occupancies from Dice outputs."""
    # Read in the avg orbital occupancies
    spin1_rdm_dice = np.loadtxt(
        os.path.join(working_dir, "spin1RDM.0.0.txt"), skiprows=1
    )
    avg_occupancies = np.zeros(2 * num_orbitals)
    for i in range(spin1_rdm_dice.shape[0]):
        if spin1_rdm_dice[i, 0] == spin1_rdm_dice[i, 1]:
            orbital_id = spin1_rdm_dice[i, 0]
            parity = orbital_id % 2
            avg_occupancies[int(orbital_id // 2 + parity * num_orbitals)] = (
                spin1_rdm_dice[i, 2]
            )

    # Read in the estimated ground state energy
    file_energy = open(os.path.join(working_dir, "shci.e"), "rb")
    bytestring_energy = file_energy.read(8)
    energy_dice = struct.unpack("d", bytestring_energy)[0]

    # Construct the SCI wavefunction coefficients from Dice output dets.bin
    occs, amps = _read_wave_function_magnitudes(os.path.join(working_dir, "dets.bin"))
    addresses = _addresses_from_occupancies(occs)
    sci_coefficients = _construct_ci_vec_from_addresses_amplitudes(amps, addresses)

    return energy_dice, sci_coefficients, avg_occupancies


def _call_dice(working_dir: Path, mpirun_options: Sequence[str] | str | None) -> None:
    """Navigate to the working dir, invoke Dice, and navigate back."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dice_path = os.path.join(script_dir, "bin", "Dice")
    dice_log_path = os.path.join(working_dir, "dice_solver_logfile.log")
    if mpirun_options:
        if isinstance(mpirun_options, str):
            mpirun_options = [mpirun_options]
        dice_call = ["mpirun"] + list(mpirun_options) + [dice_path]
    else:
        dice_call = ["mpirun", dice_path]

    try:
        with open(dice_log_path, "w") as logfile:
            subprocess.run(dice_call, cwd=working_dir, stdout=logfile, stderr=logfile)
    except subprocess.CalledProcessError as e:
        raise DiceExecutionError(
            command=dice_call,
            returncode=e.returncode,
            output=e.stdout,
            stderr=e.stderr,
            working_dir=working_dir,
        ) from e


def _write_input_files(
    addresses: tuple[Sequence[int], Sequence[int]],
    active_space_path: str | Path,
    num_up: int,
    num_dn: int,
    num_configurations: int,
    working_dir: str | Path,
    spin_sq: float,
    max_davidson: int,
) -> None:
    """Prepare the Dice inputs in the working directory."""
    ### Move the FCI Dump to working dir ###
    shutil.copy(active_space_path, os.path.join(working_dir, "fcidump.txt"))

    ### Write the input.dat ###
    num_elec = num_up + num_dn
    # Return only the lowest-energy state
    nroots = "nroots 1\n"
    # Spin squared
    spin = f"spin {spin_sq}\n"
    # Path to active space dump
    orbitals = "orbitals fcidump.txt\n"
    # The Dice/Riken branch this package is built on modifies the normal behavior
    # of the SHCI application, so we hard-code this value to prevent unintended
    # behavior due to these modifications.
    schedule = "schedule\n0 1.e+12\nend\n"
    # Floating point tolerance for Davidson solver
    davidson_tol = "davidsonTol 1e-5\n"
    # Energy floating point tolerance
    de = "dE 1e-10\n"
    # The number of davidson cycles to perform during SHCI
    maxiter = f"maxiter {max_davidson}\n"
    # We don't want Dice to be noisyu for now so we hard code noio
    noio = "noio\n"
    # The number of determinants to write as output. We always want all of them.
    write_best_determinants = f"writeBestDeterminants {num_configurations}\n"
    # Number of perturbation theory parameters. Must be 0.
    n_pt_iter = "nPTiter 0\n"
    # Requested reduced density matrices
    rdms = "DoSpinRDM\nDoRDM\nDoSpinOneRDM\nDoOneRDM\n"
    # Number of electrons
    nocc = f"nocc {num_elec}\n"
    # An input which will be ignored by the modified Dice application.
    # This still needs to be present to prevent Dice from breaking.
    dummy_det = " ".join([str(i) for i in range(num_elec)]) + "\nend"
    input_list = [
        nroots,
        spin,
        orbitals,
        schedule,
        davidson_tol,
        de,
        maxiter,
        noio,
        write_best_determinants,
        n_pt_iter,
        rdms,
        nocc,
        dummy_det,
    ]
    file1 = open(os.path.join(working_dir, "input.dat"), "w")
    file1.writelines(input_list)
    file1.close()

    ### Write the determinants to working dir ###
    up_addr, dn_addr = addresses
    bytes_up = _address_list_to_bytes(up_addr)
    bytes_dn = _address_list_to_bytes(dn_addr)
    file1 = open(os.path.join(working_dir, "AlphaDets.bin"), "wb")  # type: ignore
    for bytestring in bytes_up:
        file1.write(bytestring)  # type: ignore
    file1.close()
    file1 = open(os.path.join(working_dir, "BetaDets.bin"), "wb")  # type: ignore
    for bytestring in bytes_dn:
        file1.write(bytestring)  # type: ignore
    file1.close()


def _integer_to_bytes(n: int) -> bytes:
    """
    Pack an integer into 5 bytes.

    The 5 is hard-coded because that is what the modified Dice branch
    expects currently.
    """
    return int(n).to_bytes(5, byteorder="big")


def _address_list_to_bytes(addresses: Sequence[int]) -> list[bytes]:
    """Convert a list of base-10 determinant addresses into a list of bytes."""
    byte_list = []
    for address in addresses:
        byte_list.append(_integer_to_bytes(address))
    return byte_list


def _read_wave_function_magnitudes(
    filename: str | Path,
) -> tuple[list[str], list[float]]:
    """Read the wavefunction magnitudes from binary file output from Dice."""
    file2 = open(filename, "rb")

    # Read 32 bit integers describing the # dets and # orbs
    det_bytes = file2.read(4)
    num_dets = struct.unpack("i", det_bytes)[0]
    norb_bytes = file2.read(4)
    num_orb = struct.unpack("i", norb_bytes)[0]

    occupancy_strs = []
    amplitudes = []
    for i in range(2 * num_dets):
        # read the wave function amplitude
        if i % 2 == 0:
            # Read the double-precision float describing the amplitude
            wf_bytes = file2.read(8)
            wf_amplitude = struct.unpack("d", wf_bytes)[0]
            amplitudes.append(wf_amplitude)
        else:
            b = file2.read(num_orb)
            occupancy_strs.append(str(b)[2:-1])

    return occupancy_strs, amplitudes


def _bitstring_from_occupancy_str(occupancy_str: str) -> np.ndarray:
    """Convert an occupancy string into a bit array."""
    norb = len(occupancy_str)
    bitstring = np.zeros(2 * norb)
    for i in range(len(occupancy_str)):
        if occupancy_str[i] == "2":
            bitstring[i] = 1
            bitstring[i + norb] = 1
        if occupancy_str[i] == "a":
            bitstring[i] = 1
        if occupancy_str[i] == "b":
            bitstring[i + norb] = 1

    return bitstring


def _addresses_from_occupancies(occupancy_strs: list[str]) -> list[list[int]]:
    """Convert occupancies to PySCF determinant addresses."""
    norb = len(occupancy_strs[0])
    addresses = []
    for occ in occupancy_strs:
        bitstring = _bitstring_from_occupancy_str(occ)
        bitstring_a = bitstring[:norb]
        bitstring_b = bitstring[norb:]
        address_a = int(
            np.array2string(np.flip(bitstring_a.astype("int")), separator="")[1:-1], 2
        )
        address_b = int(
            np.array2string(np.flip(bitstring_b.astype("int")), separator="")[1:-1], 2
        )
        address = [address_a, address_b]
        addresses.append(address)

    return addresses


def _construct_ci_vec_from_addresses_amplitudes(
    amps: list[float], addresses: list[list[int]]
) -> np.ndarray:
    """Construct wavefunction amplitudes from determinant addresses and their associated amplitudes."""
    uniques = np.unique(np.array(addresses))
    num_dets = len(uniques)
    ci_vec = np.zeros((num_dets, num_dets))

    for idx, address in enumerate(addresses):
        address_a = address[0]
        address_b = address[1]

        i = np.where(uniques == address_a)[0][0]
        j = np.where(uniques == address_b)[0][0]
        ci_vec[i, j] = amps[idx]

    return ci_vec

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

import math
import os
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pyscf import tools
from qiskit_addon_sqd.fermion import SCIState, bitstring_matrix_to_ci_strs

# Ensure the runtime linker can find the local boost binaries at runtime
DICE_BIN = os.path.join(os.path.abspath(os.path.dirname(__file__)), "bin")
os.environ["LD_LIBRARY_PATH"] = f"{DICE_BIN}:{os.environ.get('LD_LIBRARY_PATH', '')}"


class DiceExecutionError(Exception):
    """Custom exception for Dice command line application execution errors."""

    def __init__(self, command, returncode, log_path):
        """Initialize a ``DiceExecutionError`` instance."""
        self.command = command
        self.returncode = returncode
        self.log_path = log_path

        message = (
            f"Command '{command}' failed with return code {returncode}\n"
            f"See the log file at {log_path} for more details."
        )
        super().__init__(message)


def solve_hci(
    hcore: np.ndarray,
    eri: np.ndarray,
    *,
    norb: int,
    nelec: tuple[int, int],
    ci_strs: tuple[Sequence[int], Sequence[int]] | None = None,
    spin_sq: float = 0.0,
    select_cutoff: float = 5e-4,
    energy_tol: float = 1e-10,
    max_iter: int = 10,
    mpirun_options: Sequence[str] | str | None = None,
    temp_dir: str | Path | None = None,
    clean_temp_dir: bool = True,
) -> tuple[float, SCIState, tuple[np.ndarray, np.ndarray]]:
    """
    Approximate the ground state of a molecular Hamiltonian using the heat bath configuration interaction method.

    In order to leverage the multi-processing nature of this tool, the user must specify
    the CPU resources to use via the `mpirun_options` argument.

    For example, to use 8 CPU slots in parallel in quiet mode:

    .. code-block:: python

       # Run 8 parallel slots in quiet mode
       mpirun_opts = "-quiet -n 8"
       # OR
       mpirun_opts = ["-quiet", "-n", "8"]

       energy, sci_coeffs, sci_strings, avg_occs = solve_hci(..., mpirun_options=mpirun_opts)

    For more information on the ``mpirun`` command line options, refer to the `man page <https://www.open-mpi.org/doc/current/man1/mpirun.1.php>`_.

    .. note::

       Determinants are interpreted by the ``Dice`` command line application as 16-byte unsigned integers; therefore, only systems
       of ``128`` or fewer orbitals are supported.

    Args:
        hcore: Core Hamiltonian matrix representing single-electron integrals.
        eri: Electronic repulsion integrals representing two-electron integrals.
        norb: The number of spatial orbitals.
        nelec: The numbers of spin up and spin down electrons.
        ci_strs: CI strings specifying the subspace to use at the beginning of the first HCI iteration.
            Should be specified as a pair of lists, with the first list containing the alpha strings and the
            second list containing the beta strings. If not specified, only the Hartree-Fock string will be used.
            A CI string is specified as an integer whose binary expansion encodes the string. For example,
            the Hartree-Fock string with 3 electrons in 5 orbitals is `0b00111`.
        spin_sq: Target value for the total spin squared for the ground state. If ``None``, no spin will be imposed.
        select_cutoff: Cutoff threshold for retaining state vector coefficients.
        energy_tol: Energy floating point tolerance.
        max_iter: The maximum number of HCI iterations to perform.
        mpirun_options: Options controlling the CPU resource allocation for the ``Dice`` command line application.
            These command-line options will be passed directly to the ``mpirun`` command line application during
            invocation of ``Dice``. These may be formatted as a ``Sequence`` of strings or a single string. If a ``Sequence``,
            the elements will be combined into a single, space-delimited string and passed to
            ``mpirun``. If the input is a single string, it will be passed to ``mpirun`` as-is. If no
            ``mpirun_options`` are provided by the user, ``Dice`` will run on a single MPI slot. For more
            information on the ``mpirun`` command line options, refer to the `man page <https://www.open-mpi.org/doc/current/man1/mpirun.1.php>`_.
        temp_dir: An absolute path to a directory for storing temporary files. If not provided, the
            system temporary files directory will be used.
        clean_temp_dir: Whether to delete intermediate files generated by the ``Dice`` command line application.
            These files will be stored in a directory created inside ``temp_dir``. If ``False``, then
            this directory will be preserved.

    Returns:
        - Minimum energy from SCI calculation
        - Approximate ground state from SCI
        - Average orbital occupancy
    """
    n_alpha, n_beta = nelec

    if ci_strs is None:
        # If CI strings not specified, use the Hartree-Fock bitstring
        ci_strs = ([(1 << n_alpha) - 1], [(1 << n_beta) - 1])

    # Set up the temp directory
    temp_dir = temp_dir or tempfile.gettempdir()
    dice_dir = Path(tempfile.mkdtemp(prefix="dice_cli_files_", dir=temp_dir))

    # Write the integrals out as an FCI dump for Dice command line app
    active_space_path = dice_dir / "fcidump.txt"
    tools.fcidump.from_integrals(active_space_path, hcore, eri, norb, nelec)

    _write_input_files(
        ci_strs=ci_strs,
        active_space_path=active_space_path,
        norb=norb,
        num_up=n_alpha,
        num_dn=n_beta,
        dice_dir=dice_dir,
        spin_sq=spin_sq,
        select_cutoff=select_cutoff,
        energy_tol=energy_tol,
        max_iter=max_iter,
    )

    # Navigate to dice dir and call Dice
    _call_dice(dice_dir, mpirun_options)

    # Read and convert outputs
    e_dice, sci_state, avg_occupancies = _read_dice_outputs(dice_dir, norb)

    # Clean up the temp directory of intermediate files, if desired
    if clean_temp_dir:
        shutil.rmtree(dice_dir)

    return (
        e_dice,
        sci_state,
        (avg_occupancies[:norb], avg_occupancies[norb:]),
    )


def solve_fermion(
    bitstring_matrix: np.ndarray | tuple[Sequence[int], Sequence[int]],
    /,
    hcore: np.ndarray,
    eri: np.ndarray,
    *,
    open_shell: bool = False,
    mpirun_options: Sequence[str] | str | None = None,
    temp_dir: str | Path | None = None,
    clean_temp_dir: bool = True,
) -> tuple[float, SCIState, tuple[np.ndarray, np.ndarray]]:
    """
    Approximate the ground state of a molecular Hamiltonian given a bitstring matrix defining the Hilbert subspace.

    This solver is designed for compatibility with `qiskit-addon-sqd <https://qiskit.github.io/qiskit-addon-sqd/>`_ workflows.

    In order to leverage the multi-processing nature of this tool, the user must specify
    the CPU resources to use via the `mpirun_options` argument.

    For example, to use 8 CPU slots in parallel in quiet mode:

    .. code-block:: python

       # Run 8 parallel slots in quiet mode
       mpirun_opts = "-quiet -n 8"
       # OR
       mpirun_opts = ["-quiet", "-n", "8"]

       energy, sci_coeffs, avg_occs = solve_fermion(..., mpirun_options=mpirun_opts)

    For more information on the ``mpirun`` command line options, refer to the `man page <https://www.open-mpi.org/doc/current/man1/mpirun.1.php>`_.

    .. note::

       Determinants are interpreted by the ``Dice`` command line application as 16-byte unsigned integers; therefore, only systems
       of ``128`` or fewer orbitals are supported.

    Args:
        bitstring_matrix: A set of configurations defining the subspace onto which the Hamiltonian will be projected and diagonalized.
            This may be specified in two ways:

            Bitstring matrix: A 2D ``numpy.ndarray`` of ``bool`` representations of bit values such that each row represents a single bitstring. The spin-up
            configurations should be specified by column indices in range ``(N, N/2]``, and the spin-down configurations should be specified by column
            indices in range ``(N/2, 0]``, where ``N`` is the number of qubits.

            CI strings: A length-2 tuple of sequences containing integer representations of the spin-up and spin-down determinants, respectively.
        
            The expected ordering is ``([a_str_0, ..., a_str_N], [b_str_0, ..., b_str_M])``.
        hcore: Core Hamiltonian matrix representing single-electron integrals
        eri: Electronic repulsion integrals representing two-electron integrals
        open_shell: A flag specifying whether configurations from the left and right
            halves of the bitstrings should be kept separate. If ``False``, CI strings
            from the left and right halves of the bitstrings are combined into a single
            set of unique configurations and used for both the alpha and beta subspaces.
        mpirun_options: Options controlling the CPU resource allocation for the ``Dice`` command line application.
            These command-line options will be passed directly to the ``mpirun`` command line application during
            invocation of ``Dice``. These may be formatted as a ``Sequence`` of strings or a single string. If a ``Sequence``,
            the elements will be combined into a single, space-delimited string and passed to
            ``mpirun``. If the input is a single string, it will be passed to ``mpirun`` as-is. If no
            ``mpirun_options`` are provided by the user, ``Dice`` will run on a single MPI slot. For more
            information on the ``mpirun`` command line options, refer to the `man page <https://www.open-mpi.org/doc/current/man1/mpirun.1.php>`_.
        temp_dir: An absolute path to a directory for storing temporary files. If not provided, the
            system temporary files directory will be used.
        clean_temp_dir: Whether to delete intermediate files generated by the ``Dice`` command line application.
            These files will be stored in a directory created inside ``temp_dir``. If ``False``, then
            this directory will be preserved.

    Returns:
        - Minimum energy from SCI calculation
        - Approximate ground state from SCI
        - Average orbital occupancy
    """
    if isinstance(bitstring_matrix, Sequence):
        if len(bitstring_matrix) != 2:
            raise ValueError(
                "CI strings must be in form of a bitstring matrix or a length-2 tuple of sequences containing the spin-up and spin-down determinants, respectively."
            )
        ci_strs = bitstring_matrix
    else:
        ci_strs = bitstring_matrix_to_ci_strs(bitstring_matrix, open_shell=open_shell)
    num_up = format(ci_strs[0][0], "b").count("1")
    num_dn = format(ci_strs[1][0], "b").count("1")
    e_dice, sci_state, avg_occupancies = solve_hci(
        hcore=hcore,
        eri=eri,
        norb=hcore.shape[0],
        nelec=(num_up, num_dn),
        ci_strs=ci_strs,
        # Hard-code S^2 = 0 until other values are supported
        spin_sq=0.0,
        # Large select cutoff to prevent the addition of additional configurations
        select_cutoff=2147483647,
        energy_tol=1e-10,
        max_iter=1,
        mpirun_options=mpirun_options,
        temp_dir=temp_dir,
        clean_temp_dir=clean_temp_dir,
    )
    return e_dice, sci_state, avg_occupancies


def _read_dice_outputs(
    dice_dir: str | Path, num_orbitals: int
) -> tuple[float, SCIState, np.ndarray]:
    """Calculate the estimated ground state energy and average orbitals occupancies from Dice outputs."""
    # Read in the avg orbital occupancies
    spin1_rdm_dice = np.loadtxt(os.path.join(dice_dir, "spin1RDM.0.0.txt"), skiprows=1)
    avg_occupancies = np.zeros(2 * num_orbitals)
    for i in range(spin1_rdm_dice.shape[0]):
        if spin1_rdm_dice[i, 0] == spin1_rdm_dice[i, 1]:
            orbital_id = spin1_rdm_dice[i, 0]
            parity = orbital_id % 2
            avg_occupancies[int(orbital_id // 2 + parity * num_orbitals)] = (
                spin1_rdm_dice[i, 2]
            )

    # Read in the estimated ground state energy
    file_energy = open(os.path.join(dice_dir, "shci.e"), "rb")
    bytestring_energy = file_energy.read(8)
    energy_dice = struct.unpack("d", bytestring_energy)[0]

    # Construct the SCI wavefunction coefficients from Dice output dets.bin
    occs, amps = _read_wave_function_magnitudes(os.path.join(dice_dir, "dets.bin"))
    ci_strs = _ci_strs_from_occupancies(occs)
    sci_coefficients, ci_strs_a, ci_strs_b = _construct_ci_vec_from_amplitudes(
        amps, ci_strs
    )
    sci_state = SCIState(
        amplitudes=sci_coefficients, ci_strs_a=ci_strs_a, ci_strs_b=ci_strs_b
    )

    return energy_dice, sci_state, avg_occupancies


def _call_dice(dice_dir: Path, mpirun_options: Sequence[str] | str | None) -> None:
    """Navigate to the dice dir, invoke Dice, and navigate back."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dice_path = os.path.join(script_dir, "bin", "Dice")
    dice_log_path = os.path.join(dice_dir, "dice_solver_logfile.log")
    if mpirun_options:
        if isinstance(mpirun_options, str):
            mpirun_options = mpirun_options.split()
        dice_call = ["mpirun"] + list(mpirun_options) + [dice_path]
    else:
        dice_call = ["mpirun", dice_path]

    with open(dice_log_path, "w") as logfile:
        process = subprocess.run(
            dice_call, cwd=dice_dir, stdout=logfile, stderr=logfile
        )
    rdm_path = dice_dir / "spin1RDM.0.0.txt"
    # We check this manually because Dice is returning non-zero codes on successful executions,
    # so we can't rely on the status code to tell us whether Dice executed succesfully. Unclear
    # if this only happens on Dice/Riken branch.
    if not os.path.exists(rdm_path):
        raise DiceExecutionError(
            command=dice_call,
            returncode=process.returncode,
            log_path=dice_log_path,
        )


def _write_input_files(
    ci_strs: tuple[Sequence[int], Sequence[int]],
    active_space_path: str | Path,
    norb: int,
    num_up: int,
    num_dn: int,
    dice_dir: str | Path,
    spin_sq: float,
    select_cutoff: float,
    energy_tol: float,
    max_iter: int,
) -> None:
    """Prepare the Dice inputs in the specified directory."""
    ### Move the FCI Dump to dice dir if it's not already there. ###
    dice_fci_path = Path(dice_dir) / "fcidump.txt"
    if not os.path.exists(dice_fci_path):
        shutil.copy(active_space_path, dice_fci_path)
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
    schedule = f"schedule\n0 {select_cutoff}\nend\n"
    # Floating point tolerance for Davidson solver
    davidson_tol = "davidsonTol 1e-5\n"
    # Energy floating point tolerance
    de = f"dE {energy_tol}\n"
    # The maximum number of HCI iterations to perform
    maxiter = f"maxiter {max_iter}\n"
    # We don't want Dice to be noisyu for now so we hard code noio
    noio = "noio\n"
    # The number of determinants to write as output. We always want all of them.
    # Cap the number of determinants returned to the largest possible `int` value in C
    dim = min(2147483647, math.comb(norb, num_up) * math.comb(norb, num_dn))
    write_best_determinants = f"writeBestDeterminants {dim}\n"
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
    file1 = open(os.path.join(dice_dir, "input.dat"), "w")
    file1.writelines(input_list)
    file1.close()

    ### Write the determinants to dice dir ###
    str_a, str_b = ci_strs
    bytes_a = _ci_strs_to_bytes(str_a)
    bytes_b = _ci_strs_to_bytes(str_b)
    file1 = open(os.path.join(dice_dir, "AlphaDets.bin"), "wb")  # type: ignore
    for bytestring in bytes_a:
        file1.write(bytestring)  # type: ignore
    file1.close()
    file1 = open(os.path.join(dice_dir, "BetaDets.bin"), "wb")  # type: ignore
    for bytestring in bytes_b:
        file1.write(bytestring)  # type: ignore
    file1.close()


def _integer_to_bytes(n: int) -> bytes:
    """
    Pack an integer into 16 bytes.

    The 16 is hard-coded because that is what the modified Dice branch
    expects currently.
    """
    return int(n).to_bytes(16, byteorder="big")


def _ci_strs_to_bytes(ci_strs: Sequence[int]) -> list[bytes]:
    """Convert a list of CI strings into a list of bytes."""
    byte_list = []
    for ci_str in ci_strs:
        byte_list.append(_integer_to_bytes(ci_str))
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
    bitstring = np.zeros(2 * norb, dtype=bool)
    for i in range(len(occupancy_str)):
        if occupancy_str[i] == "2":
            bitstring[i] = 1
            bitstring[i + norb] = 1
        if occupancy_str[i] == "a":
            bitstring[i] = 1
        if occupancy_str[i] == "b":
            bitstring[i + norb] = 1

    return bitstring


def _ci_strs_from_occupancies(occupancy_strs: list[str]) -> list[list[int]]:
    """Convert occupancies to CI strings."""
    norb = len(occupancy_strs[0])
    ci_strs = []
    for occ in occupancy_strs:
        bitstring = _bitstring_from_occupancy_str(occ)
        bitstring_a = bitstring[:norb]
        bitstring_b = bitstring[norb:]
        ci_str_a = sum(b << i for i, b in enumerate(bitstring_a))
        ci_str_b = sum(b << i for i, b in enumerate(bitstring_b))
        ci_str = [ci_str_a, ci_str_b]
        ci_strs.append(ci_str)

    return ci_strs


def _construct_ci_vec_from_amplitudes(
    amps: list[float], ci_strs: list[list[int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct wavefunction amplitudes from CI strings and their associated amplitudes."""
    uniques = np.unique(np.array(ci_strs))
    num_dets = len(uniques)
    ci_vec = np.zeros((num_dets, num_dets))
    ci_strs_a = np.zeros(num_dets, dtype=np.int64)
    ci_strs_b = np.zeros(num_dets, dtype=np.int64)

    ci_str_map = {uni_str: i for i, uni_str in enumerate(uniques)}

    for amp, ci_str in zip(amps, ci_strs):
        ci_str_a, ci_str_b = ci_str
        i = ci_str_map[ci_str_a]
        j = ci_str_map[ci_str_b]

        ci_vec[i, j] = amp

        ci_strs_a[i] = uniques[i]
        ci_strs_b[j] = uniques[j]

    return ci_vec, ci_strs_a, ci_strs_b

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

from dataclasses import dataclass

import numpy as np
from pyscf import tools

from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs


@dataclass(frozen=True)
class experimental_SCIState:
    """The amplitudes and determinants describing a quantum state."""

    amplitudes: np.ndarray
    """Either an :math:`M \\times N` array where :math:`M =` len(``ci_strs_a``)
    and :math:`N` = len(``ci_strs_b``). ``amplitudes[i][j]`` is the
    amplitude of the determinant pair (``ci_strs_a[i]``, ``ci_strs_b[j]``).

    Or an array of length :math:`M` where :math:`M =` len(``ci_strs_a``) = len(``ci_strs_a``).
    ``amplitudes[i]`` is the amplitude of the determinant pair (``ci_strs_a[i]``, ``ci_strs_b[i]``).
    """

    ci_strs_a: np.ndarray
    """The alpha determinants."""

    ci_strs_b: np.ndarray
    """The beta determinants."""

    norb: int
    """The number of spatial orbitals."""

    nelec: tuple[int, int]
    """The numbers of alpha and beta electrons."""

    def __post_init__(self):
        """Validate dimensions of inputs."""
        object.__setattr__(
            self, "amplitudes", np.asarray(self.amplitudes)
        )  # Convert to ndarray if not already
        if len(self.amplitudes.shape) == 2 and self.amplitudes.shape != (
            len(self.ci_strs_a),
            len(self.ci_strs_b),
        ):
            raise ValueError(
                f"'amplitudes' shape must be ({len(self.ci_strs_a)}, {len(self.ci_strs_b)}) "
                f"but got {self.amplitudes.shape}"
            )

        if len(self.amplitudes.shape) == 1 and len(self.amplitudes) != len(
            self.ci_strs_a
        ):
            raise ValueError(
                f"'amplitudes' length must be ({len(self.ci_strs_a)}, ) in non Cartesian product mode"
                f"but got {len(self.amplitudes)}"
            )

        if len(self.amplitudes.shape) == 1 and len(self.ci_strs_b) != len(
            self.ci_strs_a
        ):
            raise ValueError(
                f"In non Cartesian product mode len(ci_strs_a) must be equal to len(ci_strs_b) "
                f"but got {len(self.ci_strs_a)} and {len(self.ci_strs_b)}"
            )

    @property
    def subspace_dimension(self):
        """Returns the size of the subspace where the state is supported."""
        return self.amplitudes.size

    @property
    def cartesian_product_structure(self):
        """Whether the state is specified by the cartesian product of two lists of alpha and beta determinants."""
        return len(self.amplitudes.shape) == 2

    def save(self, filename):
        """Save the SCIState object to an .npz file."""
        np.savez(
            filename,
            amplitudes=self.amplitudes,
            ci_strs_a=self.ci_strs_a,
            ci_strs_b=self.ci_strs_b,
            norb=self.norb,
            nelec=self.nelec,
        )

    @classmethod
    def load(cls, filename):
        """Load an SCIState object from an .npz file."""
        with np.load(filename) as data:
            return cls(
                data["amplitudes"],
                data["ci_strs_a"],
                data["ci_strs_b"],
                norb=data["norb"],
                nelec=tuple(data["nelec"]),
            )

    def orbital_occupancies(self) -> tuple[np.ndarray, np.ndarray]:
        """Average orbital occupancies."""
        raise NotImplementedError()

    def rdm(self, rank: int = 1, spin_summed: bool = False) -> np.ndarray:
        """Compute reduced density matrix."""
        # Reason for type: ignore: mypy can't tell the return type of the
        # PySCF functions
        raise NotImplementedError()

    def spin_square(self) -> float:
        """Return spin squared."""
        raise NotImplementedError()


@dataclass(frozen=True)
class experimental_SCIResult:
    """Result of an SCI calculation."""

    energy: float
    """The SCI energy."""

    sci_state: experimental_SCIState
    """The SCI state."""

    orbital_occupancies: tuple[np.ndarray, np.ndarray]
    """The average orbital occupancies."""

    rdm1: np.ndarray | None = None
    """Spin-summed 1-particle reduced density matrix."""

    rdm2: np.ndarray | None = None
    """Spin-summed 2-particle reduced density matrix."""


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


def solve_sci(
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    # TODO allow spin_sq to be None
    spin_sq: float | None = None,
    cartesian_product: bool = True,
    symmetrize_spin: bool = True,
    mpirun_options: Sequence[str] | str | None = None,
    temp_dir: str | Path | None = None,
    clean_temp_dir: bool = True,
) -> experimental_SCIResult:
    """Diagonalize Hamiltonian in subspace defined by CI strings.

    Args:
        ci_strings: Pair (strings_a, strings_b) of arrays of spin-alpha CI
            strings and spin-beta CI strings.  When 'cartesian_product=True'their Cartesian product give the basis of
            the subspace in which to perform a diagonalization. When 'cartesian_product=False'their concatenation give the basis of
            the subspace in which to perform a diagonalization. i.e. 'det[i] = concatenate(strings_a[i], strings_b[i])'.
        one_body_tensor: The one-body tensor of the Hamiltonian.
        two_body_tensor: The two-body tensor of the Hamiltonian.
        norb: The number of spatial orbitals.
        nelec: The numbers of alpha and beta electrons.
        spin_sq: Target value for the total spin squared for the ground state. If ``None``, no spin will be imposed.
        cartesian_product: Whether to take the Cartesian product of the unique alpha and beta configurations.
        symmetrize_spin: when `cartesian_product=Flase`, and when `nelec[0] = nelec[1]`, setting to `True` will enforce that each alpha-beta pair
            is also present in beta-alpha form.
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
        The diagonalization result.
    """
    energy, sci_state, occupancies = solve_hci(
        hcore=one_body_tensor,
        eri=two_body_tensor,
        norb=norb,
        nelec=nelec,
        ci_strs=ci_strings,
        spin_sq=spin_sq,
        cartesian_product=cartesian_product,
        symmetrize_spin=symmetrize_spin,
        # Large select cutoff to prevent the addition of additional configurations
        select_cutoff=2147483647,
        energy_tol=1e-10,
        max_iter=1,
        mpirun_options=mpirun_options,
        temp_dir=temp_dir,
        clean_temp_dir=clean_temp_dir,
    )
    return experimental_SCIResult(energy, sci_state, orbital_occupancies=occupancies)


def solve_sci_batch(
    ci_strings: list[tuple[np.ndarray, np.ndarray]],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    spin_sq: float | None = None,
    cartesian_product: bool = True,
    symmetrize_spin: bool = True,
    mpirun_options: Sequence[str] | str | None = None,
    temp_dir: str | Path | None = None,
    clean_temp_dir: bool = True,
) -> list[experimental_SCIResult]:
    """Diagonalize Hamiltonian in subspaces.

    Args:
        ci_strings: List of pairs (strings_a, strings_b) of arrays of spin-alpha CI
            strings and spin-beta CI strings. When 'cartesian_product=True'their Cartesian product give the basis of
            the subspace in which to perform a diagonalization. When 'cartesian_product=False'their concatenation give the basis of
            the subspace in which to perform a diagonalization. i.e. 'det[i] = concatenate(strings_a[i], strings_b[i])'.
        one_body_tensor: The one-body tensor of the Hamiltonian.
        two_body_tensor: The two-body tensor of the Hamiltonian.
        norb: The number of spatial orbitals.
        nelec: The numbers of alpha and beta electrons.
        spin_sq: Target value for the total spin squared for the ground state. If ``None``, no spin will be imposed.
        cartesian_product: Whether to take the Cartesian product of the unique alpha and beta configurations.
        symmetrize_spin: when `cartesian_product=Flase`, and when `nelec[0] = nelec[1]`, setting to `True` will enforce that each alpha-beta pair
            is also present in beta-alpha form.
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
        The results of the diagonalizations in the subspaces given by ci_strings.
    """
    return [
        solve_sci(
            ci_strs,
            one_body_tensor,
            two_body_tensor,
            norb=norb,
            nelec=nelec,
            spin_sq=spin_sq,
            cartesian_product=cartesian_product,
            symmetrize_spin=symmetrize_spin,
            mpirun_options=mpirun_options,
            temp_dir=temp_dir,
            clean_temp_dir=clean_temp_dir,
        )
        for ci_strs in ci_strings
    ]


def solve_hci(
    hcore: np.ndarray,
    eri: np.ndarray,
    *,
    norb: int,
    nelec: tuple[int, int],
    ci_strs: tuple[np.ndarray, np.ndarray] | None = None,
    spin_sq: float | None = None,
    cartesian_product: bool = True,
    symmetrize_spin: bool = True,
    select_cutoff: float = 5e-4,
    energy_tol: float = 1e-10,
    max_iter: int = 10,
    mpirun_options: Sequence[str] | str | None = None,
    temp_dir: str | Path | None = None,
    clean_temp_dir: bool = True,
) -> tuple[float, experimental_SCIState, tuple[np.ndarray, np.ndarray]]:
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
        cartesian_product: Whether to take the Cartesian product of the unique alpha and beta configurations.
        symmetrize_spin: when `cartesian_product=Flase`, and when `nelec[0] = nelec[1]`, setting to `True` will enforce that each alpha-beta pair
            is also present in beta-alpha form.
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
        ci_strs = (np.array([(1 << n_alpha) - 1]), np.array([(1 << n_beta) - 1]))
        if cartesian_product:
            print(
                "Overwrite 'cartesian_product' option to 'False'. Dice runs HCI without taking Cartesian products of alpha-beta pairs"
            )
            cartesian_product = False

    if select_cutoff < 2147483647 / 100 or max_iter > 1:
        print(
            "Assuming you want to run HCI... Overwrite 'cartesian_product' option to 'False'. Dice runs HCI without taking Cartesian products of alpha-beta pairs"
        )
        cartesian_product = False

    if not cartesian_product and len(ci_strs[0]) != len(ci_strs[1]):
        print(
            f"Incompatible input variables 'cartesian_product=False' with len(alpha_strings)={len(ci_strs[0])} and len(beta_strings)={len(ci_strs[1])}"
        )
        print("Overwrite 'cartesian_product' option to 'True'")
        cartesian_product = True

    if nelec[0] != nelec[1] and symmetrize_spin:
        symmetrize_spin = False
        print(
            "The value of `symmetrize_spin=True` is incompatible with different numbers of spin-up and spin-dwn electrons. Overwriting `symmetrize_spin=False`"
        )

    if cartesian_product and not symmetrize_spin:
        print(
            "`cartesian_product=True` and `symmetrize_spin=False` are incompatible inputs. Overwriting `symmetrize_spin=True`"
        )

    ci_strs = _cleanup_input_strings(ci_strs, cartesian_product, symmetrize_spin)

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
        cartesian_product=cartesian_product,
    )

    # Navigate to dice dir and call Dice
    _call_dice(dice_dir, mpirun_options)

    # Read and convert outputs
    e_dice, sci_state, avg_occupancies = _read_dice_outputs(
        dice_dir, norb, nelec, cartesian_product
    )

    # Clean up the temp directory of intermediate files, if desired
    if clean_temp_dir:
        shutil.rmtree(dice_dir)

    return (
        e_dice,
        sci_state,
        (avg_occupancies[:norb], avg_occupancies[norb:]),
    )


def solve_fermion(
    bitstring_matrix: np.ndarray | tuple[np.ndarray, np.ndarray],
    /,
    hcore: np.ndarray,
    eri: np.ndarray,
    *,
    open_shell: bool = False,
    cartesian_product: bool = True,
    mpirun_options: Sequence[str] | str | None = None,
    temp_dir: str | Path | None = None,
    clean_temp_dir: bool = True,
) -> tuple[float, experimental_SCIState, tuple[np.ndarray, np.ndarray]]:
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
        bitstring_matrix: A set of configurations defining the subspace onto which the Hamiltonian
            will be projected and diagonalized.

            This may be specified in two ways:

            A bitstring matrix: A 2D ``numpy.ndarray`` of ``bool`` representations of bit values such that each row represents a single bitstring. The spin-up
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
        cartesian_product: Whether to take the Cartesian product of the unique alpha and beta configurations.
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
    if isinstance(bitstring_matrix, tuple):
        if len(bitstring_matrix) != 2:
            raise ValueError(
                "CI strings must be in form of a bitstring matrix or a length-2 tuple of sequences containing the spin-up and spin-down determinants, respectively."
            )
        ci_strs = bitstring_matrix
    else:
        ci_strs = bitstring_matrix_to_ci_strs(bitstring_matrix, open_shell=open_shell)
    if (not open_shell) and (not cartesian_product):
        raise ValueError(
            f"Incompatible setting of arguments 'open_shell'({open_shell}) and 'cartesian_product' ({cartesian_product})"
            f"Comparible values are: (False, True), (True, False), (True, True)."
        )
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
        cartesian_product=cartesian_product,
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
    dice_dir: str | Path,
    norb: int,
    nelec: tuple[int, int],
    cartesian_product: bool,
) -> tuple[float, experimental_SCIState, np.ndarray]:
    """Calculate the estimated ground state energy and average orbitals occupancies from Dice outputs."""
    # Read in the avg orbital occupancies
    spin1_rdm_dice = np.loadtxt(os.path.join(dice_dir, "spin1RDM.0.0.txt"), skiprows=1)
    avg_occupancies = np.zeros(2 * norb)
    for i in range(spin1_rdm_dice.shape[0]):
        if spin1_rdm_dice[i, 0] == spin1_rdm_dice[i, 1]:
            orbital_id = spin1_rdm_dice[i, 0]
            parity = orbital_id % 2
            avg_occupancies[int(orbital_id // 2 + parity * norb)] = spin1_rdm_dice[i, 2]

    # Read in the estimated ground state energy
    file_energy = open(os.path.join(dice_dir, "shci.e"), "rb")
    bytestring_energy = file_energy.read(8)
    energy_dice = struct.unpack("d", bytestring_energy)[0]

    # Construct the SCI wavefunction coefficients from Dice output dets.bin
    occs, amps = _read_wave_function_magnitudes(os.path.join(dice_dir, "dets.bin"))
    ci_strs = _ci_strs_from_occupancies(occs)
    sci_coefficients, ci_strs_a, ci_strs_b = _construct_ci_vec_from_amplitudes(
        amps, ci_strs, cartesian_product
    )
    sci_state = experimental_SCIState(
        amplitudes=sci_coefficients,
        ci_strs_a=ci_strs_a,
        ci_strs_b=ci_strs_b,
        norb=norb,
        nelec=nelec,
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
    ci_strs: tuple[np.ndarray, np.ndarray],
    active_space_path: str | Path,
    norb: int,
    num_up: int,
    num_dn: int,
    dice_dir: str | Path,
    spin_sq: float | None,
    select_cutoff: float,
    energy_tol: float,
    max_iter: int,
    cartesian_product: bool,
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
    spin = f"spin {spin_sq}\n" if spin_sq is not None else ""
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
    dummy_det = " ".join([str(i) for i in range(num_elec)]) + "\nend\n"
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
    if cartesian_product:
        input_list.append("alpha_beta_cartesian_product\n")
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


def _ci_strs_to_bytes(ci_strs: np.ndarray) -> list[bytes]:
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
    amps: list[float],
    ci_strs: list[list[int]],
    cartesian_product: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct wavefunction amplitudes from CI strings and their associated amplitudes."""
    strs_a, strs_b = zip(*ci_strs)

    if cartesian_product:
        uniques_a = np.unique(strs_a)
        uniques_b = np.unique(strs_b)
        num_dets_a = len(uniques_a)
        num_dets_b = len(uniques_b)
        ci_vec = np.zeros((num_dets_a, num_dets_b))
        ci_strs_a = np.zeros(num_dets_a, dtype=np.int64)
        ci_strs_b = np.zeros(num_dets_b, dtype=np.int64)

        ci_str_map_a = {uni_str: i for i, uni_str in enumerate(uniques_a)}
        ci_str_map_b = {uni_str: i for i, uni_str in enumerate(uniques_b)}

        for amp, ci_str in zip(amps, ci_strs):
            ci_str_a, ci_str_b = ci_str
            i = ci_str_map_a[ci_str_a]
            j = ci_str_map_b[ci_str_b]

            ci_vec[i, j] = amp

            ci_strs_a[i] = uniques_a[i]
            ci_strs_b[j] = uniques_b[j]
    else:
        ci_vec = np.array(amps)
        ci_strs_a = np.array(strs_a)
        ci_strs_b = np.array(strs_b)

    return ci_vec, ci_strs_a, ci_strs_b


def _cleanup_input_strings(
    ci_strs: tuple[np.ndarray, np.ndarray] | None = None,
    cartesian_product: bool = True,
    symmetrize_spin: bool = True,
):
    if cartesian_product:
        ci_strs_clean = (np.unique(ci_strs[0]), np.unique(ci_strs[1]))
        subspace_dim = len(ci_strs_clean[0]) * len(ci_strs_clean[1])

    if not cartesian_product and not symmetrize_spin:
        pairs = list(zip(ci_strs[0], ci_strs[1]))
        unique_pairs = {tuple(p) for p in pairs}
        unique_pairs_array = np.array([list(p) for p in unique_pairs])
        ci_strs_clean = (unique_pairs_array[:, 0], unique_pairs_array[:, 1])
        subspace_dim = len(ci_strs_clean[0])

    if not cartesian_product and symmetrize_spin:
        pairs = list(zip(ci_strs[0], ci_strs[1])) + list(zip(ci_strs[1], ci_strs[0]))
        unique_pairs = {tuple(p) for p in pairs}
        unique_pairs_array = np.array([list(p) for p in unique_pairs])
        ci_strs_clean = (unique_pairs_array[:, 0], unique_pairs_array[:, 1])
        subspace_dim = len(ci_strs_clean[0])

    print(f"The subspace dimension for diagonalization is: {subspace_dim}")
    return ci_strs_clean

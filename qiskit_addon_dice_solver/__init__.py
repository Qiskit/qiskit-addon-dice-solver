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

"""Dice eigensolver Qiskit addon.

.. currentmodule:: qiskit_addon_dice_solver

Functions
=========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   solve_fermion
   solve_hci
"""

from .dice_solver import solve_fermion, solve_hci, solve_sci, solve_sci_batch

__all__ = [
    "solve_dice",
    "solve_fermion",
    "solve_hci",
    "solve_sci",
    "solve_sci_batch",
]

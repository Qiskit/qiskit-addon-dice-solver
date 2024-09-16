##############################
Qiskit addon: Dice eigensolver
##############################

`Qiskit addons <git@github.com:Qiskit/qiskit-addon-cutting.git>`_ are a collection of modular tools for building utility-scale workloads powered by Qiskit.

This package contains a `Dice-based eigensolver [1-2] <https://sanshar.github.io/Dice/overview.html>`_ that can be used to scale sample-based quantum diagonalization (SQD) [3] chemistry workflows past 30 orbitals. It is designed as a plugin to the `SQD Qiskit addon <https://qiskit.github.io/qiskit-addon-sqd/>`_. No ``Dice`` executable is included in this package, but a build script is provided to assist users in properly setting up the package for installation. For an example of integrating ``qiskit-addon-dice-solver`` into SQD workflows, check out the `how-to <https://qiskit.github.io/qiskit-addon-sqd/how_tos/integrate_dice_solver.html>`_.

This package uses the ``Dice`` command line application to perform the Davidson diagonalization method, which allows for diagonalization of systems of 30+ orbitals. The ``Dice`` application is designed to perform semistochastic heat-bath configuration interaction (SHCI) calculations that involves more than a single run of Davidson's method; however, this package restricts the inputs to ``Dice`` such that it is used only to perform a single diagonalization routine in the subspace defined by the input determinants.

We acknowledge Sandeep Sharma's support, suggestions, and conversations that made this package possible.

Documentation
-------------

All documentation is available `here <https://qiskit.github.io/qiskit-addon-dice-solver/>`_.

Supported Platforms
-------------------

Architectures:

- x86_64

Operating systems:

- Ubuntu 24.04 LTS - Noble Numbat
- Ubuntu 22.04 LTS - Jammy Jellyfish

Installation
------------

First, install some required libraries:

.. code-block:: bash

   sudo apt install build-essential libboost-all-dev libopenmpi-dev openmpi-bin libhdf5-openmpi-dev

To build the binaries required for this package:

.. code-block:: bash

   ./build.sh

And finally, to install the Python package:

.. code-block:: bash
   
   pip install -e .

Limitations
-----------

- The determinant addresses are interpreted by the ``Dice`` command line application to be 5-byte unsigned integers; therefore, only systems of 40 or fewer orbitals are supported.
- Only closed-shell systems are supported. The particle number of the spin-up and spin-down determinants are expected to be equal.

Deprecation Policy
------------------

We follow `semantic versioning <https://semver.org/>`_ and are guided by the principles in
`Qiskit's deprecation policy <https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md>`_.
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
release notes.

Contributing
------------

The source code is available `on GitHub <https://github.com/Qiskit/qiskit-addon-dice-solver>`_.

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/qiskit-addon-dice-solver/blob/main/CONTRIBUTING.md>`_
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/qiskit-addon-dice-solver/issues/new/choose>`_ for tracking requests and bugs.

License
-------

`Apache License 2.0 <https://github.com/Qiskit/qiskit-addon-dice-solver/blob/main/LICENSE.txt>`_

References
----------

[1] Sandeep Sharma, et al., `Semistochastic Heat-bath Configuration Interaction method: selected configuration interaction with semistochastic perturbation theory <https://arxiv.org/abs/1610.06660>`_, arXiv:1610.06660v2 [physics.chem-ph].

[2] Adam Holmes, et al., `Heat-bath Configuration Interaction: An efficient selected CI algorithm inspired by heat-bath sampling <https://arxiv.org/abs/1606.07453>`_, arXiv:1606.07453 [physics.chem-ph].

[3] Javier Robledo-Moreno, et al., `Chemistry Beyond Exact Solutions on a Quantum-Centric Supercomputer <https://arxiv.org/abs/2405.05068>`_, arXiv:2405.05068 [quant-ph].

.. toctree::
  :hidden:

   Documentation Home <self>
   API Reference <apidocs/qiskit_addon_dice_solver>
   GitHub <https://github.com/Qiskit/qiskit-addon-dice-solver>

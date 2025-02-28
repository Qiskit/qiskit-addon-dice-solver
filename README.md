# Qiskit addon: Dice eigensolver

### Table of Contents

* [About](#about)
* [Documentation](#documentation)
* [Supported Platforms](#supported-platforms)
* [Installation](#installation)
* [Limitations](#limitations)
* [Deprecation Policy](#deprecation-policy)
* [Contributing](#contributing)
* [License](#license)
* [References](#references)

----------------------------------------------------------------------------------------------------

### About

[Qiskit addons](https://docs.quantum.ibm.com/guides/addons) are a collection of modular tools for building utility-scale workloads powered by Qiskit.

This package contains a [Dice-based eigensolver [1-2]](https://sanshar.github.io/Dice/overview.html) that can be used to scale [sample-based quantum diagonalization (SQD) [3]](https://arxiv.org/abs/2405.05068) chemistry workflows past 30 orbitals. It is designed as a plugin to the [SQD Qiskit addon](https://qiskit.github.io/qiskit-addon-sqd/). No ``Dice`` executable is included in this package, but a build script is provided to assist users in properly setting up the package for installation. For an example of integrating ``qiskit-addon-dice-solver`` into SQD workflows, check out the [how-to](https://qiskit.github.io/qiskit-addon-sqd/how_tos/integrate_dice_solver.html).

This package uses the ``Dice`` command line application to perform the Davidson diagonalization method, which allows for diagonalization of systems of 30+ orbitals. The ``Dice`` application is designed to perform semistochastic heat-bath configuration interaction (SHCI) calculations that involves more than a single run of Davidson's method; however, this package restricts the inputs to ``Dice`` such that it is used only to perform a single diagonalization routine in the subspace defined by the input determinants.

We acknowledge Sandeep Sharma's support, suggestions, and conversations that made this package possible.

----------------------------------------------------------------------------------------------------

### Documentation:

All documentation is available at https://qiskit.github.io/qiskit-addon-dice-solver/.

----------------------------------------------------------------------------------------------------

### Supported Platforms:

Architectures:

- x86_64

Operating systems:

- Ubuntu 24.04 LTS - Noble Numbat
- Ubuntu 22.04 LTS - Jammy Jellyfish

----------------------------------------------------------------------------------------------------

### Requirements

#### Ubuntu

``sudo apt install build-essential libboost-all-dev libopenmpi-dev openmpi-bin libhdf5-openmpi-dev``

#### Fedora (not officially supported)

``sudo dnf install gcc-c++ make boost-devel openmpi-devel boost-openmpi-devel openmpi hdf5-openmpi-devel environment-modules``

On Fedora, installing OpenMPI does not automatically make it available in your PATH. To do that, you need to load the OpenMPI module, like this:

``module load mpi/openmpi-x86_64``

To make the effect permanent, consider creating a script in e.g. `/etc/profile.d` that runs this command.

----------------------------------------------------------------------------------------------------

### Installation

First, install the [requirements](#requirements).

Next, install from the most recent stable branch. Users who want to install from the `main` branch should
note that the hosted documentation may not accurately reflect the state of the API in the `main` branch.

``git checkout stable/X.Y``

Build the boost and Dice binaries required for this package:

``./build.sh``

And finally, to install the Python package:

``pip install -e .``

----------------------------------------------------------------------------------------------------

### Limitations

- The determinant addresses are interpreted by the ``Dice`` command line application to be 16-byte unsigned integers; therefore, only systems of 128 or fewer orbitals are supported.

----------------------------------------------------------------------------------------------------

### Deprecation Policy

We follow [semantic versioning](https://semver.org/) and are guided by the principles in
[Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md).
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.

----------------------------------------------------------------------------------------------------

### Contributing

The developer guide is located at [CONTRIBUTING.md](https://github.com/Qiskit/qiskit-addon-dice-solver/blob/main/CONTRIBUTING.md>)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

We use [GitHub issues](https://github.com/Qiskit/qiskit-addon-dice-solver/issues/new/choose) for tracking requests and bugs.

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)

----------------------------------------------------------------------------------------------------

### References

[1] Sandeep Sharma, et al., [Semistochastic Heat-bath Configuration Interaction method: selected configuration interaction with semistochastic perturbation theory](https://arxiv.org/abs/1610.06660), arXiv:1610.06660v2 [physics.chem-ph].

[2] Adam Holmes, et al., [Heat-bath Configuration Interaction: An efficient selected CI algorithm inspired by heat-bath sampling](https://arxiv.org/abs/1606.07453), arXiv:1606.07453 [physics.chem-ph].

[3] Javier Robledo-Moreno, et al., [Chemistry Beyond Exact Solutions on a Quantum-Centric Supercomputer](https://arxiv.org/abs/2405.05068), arXiv:2405.05068 [quant-ph].

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

from setuptools import setup, find_packages


setup(
    name="qiskit-addon-dice-solver",
    version="0.3.0",
    author="IBM Quantum Middleware",
    description="A Python wrapper for the Dice eigensolver.",
    packages=find_packages(),
    package_data={"dice_solver": ["bin/Dice", "bin/*.so*"]},
    include_package_data=True,
    install_requires=["numpy", "pyscf", "qiskit-addon-sqd>=0.8"],
    extras_require={
        "dev": ["tox>=4.0", "pytest>=8.0"],
        "docs": [
            "matplotlib",
            "qiskit-sphinx-theme~=2.0.0",
            "sphinx-design",
            "sphinx-autodoc-typehints",
            "sphinx-copybutton",
            "reno",
        ],
        "style": [
            "ruff==0.9.9",
        ],
        "lint": [
            "mypy==1.15.0",
            "pylint>=3.2.7",
            "pydocstyle==6.3",
            "reno",
        ],
    },
)

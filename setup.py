from setuptools import setup, find_packages


setup(
    name="qiskit-addon-dice-solver",
    version="0.0.1",
    author="IBM Quantum Middleware",
    description="A Python wrapper for the Dice eigensolver.",
    packages=find_packages(),
    package_data={"dice_solver": ["bin/Dice", "bin/*.so*"]},
    include_package_data=True,
    install_requires=["numpy", "pyscf"],
    extras_require={
        "dev": ["tox>=4.0", "pytest>=8.0"],
        "docs": [
            "qiskit-sphinx-theme~=2.0.0",
            "sphinx-design",
            "sphinx-autodoc-typehints",
            "sphinx-copybutton",
        ],
        "style": [
            "ruff==0.6.4",
        ],
        "lint": [
            "mypy==1.11.2",
            "pylint>=3.2.7",
            "pydocstyle==6.3",
        ],
    },
)

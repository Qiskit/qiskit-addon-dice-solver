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
)

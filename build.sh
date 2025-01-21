#!/bin/bash
set -euo pipefail

# Start with a fresh external dir
rm -rf external/ && mkdir external

# Get Boost
wget https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz \
    && tar -xzf boost_1_85_0.tar.gz -C $(pwd)/external \
    && rm boost_1_85_0.tar.gz

# Clone Dice default branch
git clone https://github.com/caleb-johnson/Dice.git $(pwd)/external/Dice

export BOOST_ROOT=$(pwd)/external/boost_1_85_0
export CURC_HDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/openmpi/
DICE_ROOT=$(pwd)/external/Dice
DICE_BIN_PATH=$(pwd)/qiskit_addon_dice_solver/bin

# Build boost
cd $BOOST_ROOT
./bootstrap.sh --with-libraries=serialization,mpi
./b2 -j$(nproc)

# Build DICE
cd $DICE_ROOT
make -j$(nproc) Dice

# Put the runtime libraries in the package
mkdir -p $DICE_BIN_PATH
cp $DICE_ROOT/bin/Dice $DICE_BIN_PATH
cp $BOOST_ROOT/stage/lib/*.so* $DICE_BIN_PATH

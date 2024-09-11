# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Install essential tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    vim \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    libboost-all-dev \
    libopenmpi-dev \
    openmpi-bin \
    libhdf5-openmpi-dev

# Set up a working directory
WORKDIR /workspace

# Create dice-solver dir
RUN mkdir -p /workspace/dice-solver

# Copy contents of repo into dice-solver
COPY . /workspace/dice-solver

# Run build script
WORKDIR /workspace/dice-solver
RUN ./build.sh

# Set the default command to start an interactive shell
CMD ["/bin/bash"]

#!/bin/bash

sudo apt-get update -y #Gets access to packages
sudo apt-get install -y libhdf5-mpi-dev #grabs hdf5 for parallel
h5pcc --showconfig
# sudo apt install python3-pip
# sudo apt-get install -y pkg-config
# export CC=mpicc
# export HDF5_MPI="ON"
HDF5_PATH=$(dirname $(which h5pcc))/../lib
CC=mpicc HDF5_MPI="ON" HDF5_DIR=$HDF5_PATH pip install --no-binary=h5py h5py
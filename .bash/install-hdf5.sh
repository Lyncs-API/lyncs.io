#!/bin/bash

sudo apt update #Gets access to packages
sudo apt install libhdf5-mpi-dev #grabs hdf5 for parallel
h5pcc - showconfig
export CC=mpicc
export HDF5_MPI="ON"
pip install --no-binary=h5py h5py
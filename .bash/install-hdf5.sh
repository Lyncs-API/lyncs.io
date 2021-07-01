#!/bin/bash

sudo apt-get update -y #Gets access to packages
sudo apt-get install -y libhdf5-mpi-dev #grabs hdf5 for parallel
h5pcc -showconfig
CC=mpicc HDF5_MPI="ON" HDF5_DIR="/usr/lib/x86_64-linux-gnu/hdf5/openmpi" pip install --no-binary=h5py h5py
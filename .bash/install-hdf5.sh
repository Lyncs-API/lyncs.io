#!/bin/bash

sudo apt-get update -y #Gets access to packages
sudo apt-get install -y libhdf5-mpi-dev #grabs hdf5 for parallel
h5pcc -showconfig

export $LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/openmpi/lib:/usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib"
export $INCLUDE_PATH="$INCLUDE_PATH:/usr/lib/x86_64-linux-gnu/openmpi/include:/usr/lib/x86_64-linux-gnu/hdf5/openmpi/include"
# CC=mpicc HDF5_MPI="ON" HDF5_DIR="/usr/lib/x86_64-linux-gnu/hdf5/openmpi" pip install --no-binary=h5py h5py
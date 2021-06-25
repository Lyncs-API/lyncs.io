#!/bin/bash

sudo apt-get update -y #Gets access to packages
sudo apt-get install -y libhdf5-mpi-dev #grabs hdf5 for parallel
h5pcc -showconfig
# var1=$( h5pcc -showconfig | grep "Installation point" )
# HDF5_PATH=$( echo $var1 | awk '{split($0,a," "); print a[3]}' )
# sudo apt install python3-pip
# sudo apt-get install -y pkg-config
CC=mpicc HDF5_MPI="ON" HDF5_DIR="/usr/lib/x86_64-linux-gnu" pip install --no-binary=h5py h5py
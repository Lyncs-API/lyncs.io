#!/bin/bash

sudo apt-get update -y #Gets access to packages
sudo apt-get install -y libhdf5-mpi-dev #grabs hdf5 for parallel
h5pcc -showconfig

if ! [ -x "$(command -v lyncs_find_package)" ]; then
  echo 'Error: lyncs_find_package is not installed.' >&2
  exit 1
fi


LIO_MPI_PATH=$(lyncs_find_package MPI | grep "include_path:" | sed 's/^.*: \(.*\)$/\1/')
LIO_MPI_PATH=$(echo $LIO_MPI_PATH | awk '{print $1}')
LIO_MPI_PATH=$(dirname $LIO_MPI_PATH)

LIO_HDF5_PATH=$(lyncs_find_package HDF5 | grep "include_dir:" | sed 's/^.*: \(.*\)$/\1/')
LIO_HDF5_PATH=$(echo $LIO_HDF5_PATH | awk '{print $1}')
LIO_HDF5_PATH=$(dirname $LIO_HDF5_PATH)

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${LIO_MPI_PATH}/lib:${LIO_HDF5_PATH}/lib"
export INCLUDE_PATH="$INCLUDE_PATH:${LIO_MPI_PATH}/include:${LIO_HDF5_PATH}/include"

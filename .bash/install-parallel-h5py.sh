#!/bin/bash -l

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

create_mpi_testfile()
{
    local tmp_file=$1
# Create and write the test file
cat << EOF > $tmp_file
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{

MPI_Init(&argc,&argv);

MPI_Finalize();
}
EOF
}

mpi_works(){

    if [ ! -z "${$TMP_PATH}" ]; then
        mkdir -p ${TMP_PATH}
        tmpfile="${TMP_PATH}/mpi-test"
    else
        tmpfile="mpi-test"
    fi
    create_mpi_testfile "$tmpfile.c"

    if [ -z "$CC" ];then
        echo "Prefered CC compiler did not found. Selecting default option (mpicc)." >&2
        CC=mpicc
    fi

    if $CC -c "$tmpfile.c" -o "$tmpfile.o" 2> /dev/null ; then
        return 0
    fi

    return 1

}

POSITIONAL=()
# parse arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --prefix)
        INSTALL_DIR="$2"
        shift
        shift
        ;;
        --enable-tmp)
        shift
        ;;
        --prefix=*)
        INSTALL_DIR="${key#*=}"
        shift
        ;;
        --hdf5)
        HDF5_PATH="$2"
        shift
        shift
        ;;
        --hdf5=*)
        HDF5_PATH="${key#*=}"
        shift
        ;;
        *)
        printf "Invalid arguments..\nExiting..\n"
        return
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

[[ ! -z "${$TMP_PATH}" ]] && TMP_PATH="$(dirname $(realpath "$0"))/tmp"

if $(mpi_works); then
    echo "MPI Works"
else
    echo "MPI does not work"
fi

# if HDF5_PATH is not set we need to download it and build it with MPI Support
if [ -z "${HDF5_PATH}" ];then
    [[ ! -z "${$TMP_PATH}" ]] && cd $TMP_PATH
    git clone https://github.com/HDFGroup/hdf5.git
    cd hdf5
    git checkout hdf5-1_10_7

    if [ -z "$INSTALL_DIR" ];then
        PREFIX=""
    else
        PREFIX="--prefix=$INSTALL_DIR"
    fi

    CC=$CC ./configure --enable-parallel --enable-shared $PREFIX && make -j && make install
    HDF5_PATH=$INSTALL_DIR
fi

CC=$CC HDF5_MPI="ON" HDF5_DIR=$HDF5_PATH pip install --no-binary=h5py h5py

# cleanup
[[ ! -z "${$TMP_PATH}" ]] && rm -rf $TMP_PATH
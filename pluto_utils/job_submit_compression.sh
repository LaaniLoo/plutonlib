#!/bin/bash -l
#PBS -lselect=8:ncpus=1:mpiprocs=1
#PBS -lwalltime=48:00:00
#PBS -m abe
#PBS -M alainm@utas.edu.au

cd $PBS_O_WORKDIR
run_dir="?"

if [[ -n "$RUN_DIR" ]]; then
    run_dir="$RUN_DIR"
    echo "run_dir = $run_dir"
fi

if [[ "$run_dir" == "?" ]]; then
    echo "run_dir = $run_dir"
    echo "run_dir not correctly entered, exiting..."
    exit 1
fi 

module load HDF5 OpenMPI/4.1.5-GCC-12.3.0-pbs
python "$run_dir/compression_script.py"

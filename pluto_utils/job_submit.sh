#!/bin/bash -l
#PBS -lselect=40:ncpus=28:mpiprocs=28
#PBS -lwalltime=48:00:00
#PBS -m abe
#PBS -M alainm@utas.edu.au

cd $PBS_O_WORKDIR

module load HDF5 OpenMPI/4.1.5-GCC-12.3.0-pbs
mpirun -mca io ^ompio ./pluto -i $ini_file -maxtime 47.5
# save_dir="$save_dir" job_info_dir="$job_info_dir" ./clean.sh
# mv "$PBS_O_WORKDIR"/job_submit.sh.e* "$PBS_O_WORKDIR"/"$job_info_dir"
# mv "$PBS_O_WORKDIR"/job_submit.sh.o* "$PBS_O_WORKDIR"/"$job_info_dir"

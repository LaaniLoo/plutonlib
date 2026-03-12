#!/bin/bash -l
#PBS -lselect=1:ncpus=28:mpiprocs=28
#PBS -l walltime=44:00:00
#PBS -N cavity_AIO_script
#PBS -m abe
#PBS -M alainm@utas.edu.au

cd $PBS_O_WORKDIR

python cavity_AIO.py



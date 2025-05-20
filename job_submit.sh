#!/bin/bash -l
#PBS -lselect=1:ncpus=28:mpiprocs=28
#PBS -lwalltime=1:00:00
#PBS -N pluto-simulation
#PBS -m abe
#PBS -M alainm@utas.edu.au

cd $PBS_O_WORKDIR

module load intel impi hdf5

mpiexec ./pluto -i pluto.ini -maxtime 19.5

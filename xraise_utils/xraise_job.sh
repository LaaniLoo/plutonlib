#!/bin/bash -l
#PBS -lselect=1:ncpus=28:mpiprocs=28
#PBS -l walltime=44:00:00
#PBS -N xraise_script
#PBS -q LARGE_MEM
#PBS -m abe
#PBS -M alainm@utas.edu.au

cd $PBS_O_WORKDIR

# conda activate analysis

sim_name='Q38_v98_a7.5'
emis_mode='basic_xray_all'
out_dir="./cavity_data/${sim_name}.hdf5"

python -X dev do_XRAiSE-inputlist.py -s ${sim_name} -m ${emis_mode} -o ${out_dir} -gr 15 35 55




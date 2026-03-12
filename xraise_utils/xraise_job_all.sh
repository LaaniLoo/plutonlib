#!/bin/bash -l
#PBS -lselect=1:ncpus=28:mpiprocs=28
#PBS -l walltime=44:00:00
#PBS -N xraise_all_sims
#PBS -q LARGE_MEM
#PBS -m abe
#PBS -M alainm@utas.edu.au

cd $PBS_O_WORKDIR

# conda activate analysis

# Create output directory
out_dir="./cavity_data_tlast"
mkdir -p ${out_dir}

emis_mode='basic_xray_all'

# Run each simulation with its specific grid range
echo "Running Q38_v98_a7.5"
python -X dev do_XRAiSE-inputlist.py -s Q38_v98_a7.5 -m ${emis_mode} -o ${out_dir}/Q38_v98_a7.5.hdf5 -gr 48 49 50 51 52

echo "Running Q38_v98_a25"
python -X dev do_XRAiSE-inputlist.py -s Q38_v98_a25 -m ${emis_mode} -o ${out_dir}/Q38_v98_a25.hdf5 -gr 52 53 54 55 56

echo "Running Q36_v3_a25"
python -X dev do_XRAiSE-inputlist.py -s Q36_v3_a25 -m ${emis_mode} -o ${out_dir}/Q36_v3_a25.hdf5 -gr 45 46 47 48 49

echo "Running Q38_v99_a7.5"
python -X dev do_XRAiSE-inputlist.py -s Q38_v99_a7.5 -m ${emis_mode} -o ${out_dir}/Q38_v99_a7.5.hdf5 -gr 37 38 39 40 41

echo "Running Q36_v3_a25_G"
python -X dev do_XRAiSE-inputlist.py -s Q36_v3_a25_G -m ${emis_mode} -o ${out_dir}/Q36_v3_a25_G.hdf5 -gr 34 35 36 37 38

echo "Running Q38_v98_a7-5_G"
python -X dev do_XRAiSE-inputlist.py -s Q38_v98_a7-5_G -m ${emis_mode} -o ${out_dir}/Q38_v98_a7.5_G.hdf5 -gr 51 52 53 54 55

echo "Running Q38_v98_a25_G"
python -X dev do_XRAiSE-inputlist.py -s Q38_v98_a25_G -m ${emis_mode} -o ${out_dir}/Q38_v98_a25_G.hdf5 -gr 47 48 49 50 51
print("Importing libraries...") #TODO add a verbose tag to see the import time

import plutonlib.simulations as ps
# import plutonlib.analysis as pa
import plutonlib.surface_brightness as pl_sb
from plutonlib.pbs_job import create_run_script, submit_run_script

import yaml
import os

import itertools
import numpy as np 

import argparse
import sys
import logging

print("Import complete")

def_yml_path = "/u/alainm/plutonlib/scripts/praise/praise_setup.yml"
def_redshift = 0.02
nlfreqs = 8
plane = 'xz'

def run_sb_calc(args):
    if not os.path.isfile(args.yml):
        raise FileNotFoundError(f"yaml file '{args.yml}' does not exist, specify yaml file with --yml arg")

    with open(args.yml,'r') as file:
        setup_yml = yaml.safe_load(file)
    configs = setup_yml['configs']

    for sim_path in setup_yml["simulations"].keys(): #this will do sim by sim rather than constructing a sim dict
        sim = ps.SimulationData(ini_file="jet_units",rel_path=sim_path)
        sim_params = setup_yml["simulations"][sim_path]

        if sim_params['config'] not in configs:
            raise KeyError(f"Config '{sim_params['config']}' not found for sim '{sim_path}', available configs: {list(configs.keys())}")
        config = configs[sim_params['config']]

        freqs = config["freqs"]
        log_space = config["log_space"]
        angles = config["angles"]
        outputs = sim_params["outputs"]

        if isinstance(freqs,list) and log_space:
            if len(freqs) > 2:
                raise ValueError(f"Log frequency scale needs two values, start and end. Frequencies given: {freqs}")
            
            freqs = np.geomspace(freqs[0], freqs[1], num=nlfreqs).round(2)
            print(f"Using logarithmically spaced frequencies between {freqs[0]} and {freqs[-1]}\nfrequencies = {freqs}\n")

        if isinstance(angles, str): #angles = all config
            if angles == "all":
                angles = [list(combo) for combo in itertools.product(list(range(0, 100, 10)), repeat=3)]
            else:
                raise ValueError(f"'angles' string value '{angles}' not recognized, only 'all' is supported.")
        elif not isinstance(angles, list):
            raise TypeError(f"'angles' is type = {type(angles)}, either use list input or 'all' for a generated list.")

        # file_path = os.path.join(args.wdir,f"sbdata_{sim_params['config']}.h5")
        # pa.save_sb_hdf5(sim_dict={sim:outputs},angle_dict={sim:angles},freqs=freqs,redshift=args.redshift,plane=plane,alt_filepath = file_path,memory = args.memory)
        pl_sb.save_sb_hdf5(sim=sim,outputs=outputs,angles=angles,freqs=freqs,redshift=args.redshift,plane=plane,memory = args.memory)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml",type=str,help=f"Path to praise setup yaml defaults to {def_yml_path}",default=def_yml_path)
    parser.add_argument("-r","--redshift",type=float,help=f"Redshift used to calculate surface brightness, defaults to {def_redshift}",default=def_redshift)
    # parser.add_argument("-d","--wdir",type=str,help=f"Path to working dir (save location for h5 file), defaults to {"./"}",default="./")
    parser.add_argument("-c","--cluster",help="Submits this script as a PBS job to kunanyi",action="store_true")

    parser.add_argument("--job-name", default="sb_calc")
    parser.add_argument("--job-length", type=int, default=24)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=28)
    parser.add_argument("--memory", type=int, default=840) #NOTE if using a full node worth of cpus for each 30gb job -> 840Gb of mem

    args = parser.parse_args()

    if args.cluster:
        script_path = os.path.abspath(__file__)
        cmd = f'python3 -u "{script_path}" --yml "{args.yml}" --redshift {args.redshift} --memory {args.memory}'
        script_info = create_run_script(
            cmd=cmd,
            job_name=args.job_name,
            job_length=args.job_length,
            nodes=args.nodes,
            cpus=args.cpus,
            memory=args.memory,
        )
        job_id = submit_run_script(script_info=script_info)
        print(f"Submitted job {job_id}")
    else:
        run_sb_calc(args)

if __name__ == "__main__":
    main()
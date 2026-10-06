print("Importing libraries...") #TODO add a verbose tag to see the import time

from pathlib import Path
import os
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import plutonlib.simulations as ps
# import plutonlib.analysis as pa
import plutonlib.surface_brightness as pl_sb
from plutonlib.pbs_job import init_script_dirs,create_run_script, submit_run_script

import yaml

import itertools
import numpy as np 

import argparse
# import sys
# import logging

print("Import complete")

def_yml_path = "/home/laani/plutonlib/scripts/praise/praise_setup.yml"
def_redshift = 0.02
nlfreqs = 8
plane = 'xz'

def dry_run(sim,grid_output,angles,freqs,redshift):
    print("Executing a dry run to gauge memory usage...\n")
    task = pl_sb._compute_sb_task(sim=sim,grid_output=grid_output,angles=angles,freqs=freqs,redshift=redshift)
    print("Dry run complete!\n")
    return task["peak_gb"]

def run_sb_calc(args):
    # if not os.path.isfile(args.yml):
    if not Path(args.yml).exists():
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

        if args.no_dry_run: #do a dry run to see the memory usage, else use 30gb per worker
            req_mem = 30 
        else:
            req_mem = dry_run(sim,outputs[-1],angles[0],freqs,args.redshift)

        pl_sb.save_sb_hdf5(
            sim=sim,
            grid_outputs=outputs,
            angles=angles,
            freqs=freqs,
            redshift=args.redshift,
            plane=plane,
            memory = args.memory,
            task_req_mem=req_mem
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml",type=str,help=f"Path to praise setup yaml defaults to {def_yml_path}",default=def_yml_path)
    parser.add_argument("-r","--redshift",type=float,help=f"Redshift used to calculate surface brightness, defaults to {def_redshift}",default=def_redshift)
    parser.add_argument("-c","--cluster",help="Submits this script as a PBS job to kunanyi",action="store_true")
    parser.add_argument("-d","--no_dry_run",help="Dont pre-run a SB calculation to see memory usage and allocate workers",action="store_true")
    parser.add_argument("--job-name", default="sb_calc")
    parser.add_argument("--job-length", type=int, default=24)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=28)
    parser.add_argument("--memory", type=int, default=128) #NOTE if using a full node worth of cpus for each 30gb job -> 840Gb of mem
    args = parser.parse_args()

    if args.cluster:
        output_dir = init_script_dirs("~",files=[Path(args.yml).expanduser()],cluster=True) #NOTE set output dir to user home
        script_cluster = output_dir / Path(__file__).name #use script file in output dir
        yml_cluster = output_dir /Path(args.yml).name #use yml file in output dir

        cmd = ( #NOTE removes script files after job finishes 
            f'python3 -u "{str(script_cluster)}" --yml "{str(yml_cluster)}" '
            f'--redshift {args.redshift} --memory {args.memory} '
            f'&& rm -rf "{output_dir}"'
        )

        if args.no_dry_run:
            cmd += " --no_dry_run"
        script_info = create_run_script(
            cmd=cmd,
            job_name=args.job_name,
            job_length=args.job_length,
            nodes=args.nodes,
            cpus=args.cpus,
            memory=args.memory,
        )
        submit_run_script(script_info=script_info)
    else:
        run_sb_calc(args)

if __name__ == "__main__":
    main()
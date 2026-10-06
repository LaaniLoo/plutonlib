import yaml
import os
from pathlib import Path
import argparse
import gc 

import itertools
import numpy as np 
import math

import resource

import plutonlib.simulations as ps
# import plutonlib.read_write as prw
import plutonlib.splines as pl_splines
import plutonlib.fancy_plot as fancypl
from plutonlib.pbs_job import init_script_dirs,create_run_script, submit_run_script

def_yml_path = "/home/laani/plutonlib/scripts/praise/praise_setup.yml" #NOTE local yml path
def_redshift = 0.02
nlfreqs = 8
plane = 'xz'
chunk_size = 100

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def create_hybrid_plot(args):
    with open(args.yml,'r') as file:
        setup_yml = yaml.safe_load(file)
    configs = setup_yml['configs']

    base_dir = os.path.join(args.wdir,"hybrids_grid")  
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

        for output in outputs: #sim/output/freq/500_1_grid1.pdf -> 500Myr,1Ghz file 1 (of 10 for 100 angles) #NOTE each grid file corresponds to an angle in x
            for freq in freqs:
                grid_dir = os.path.join(base_dir,f"{sim.run_name}/{output}/{freq}")
                if not os.path.isdir(grid_dir):
                    os.makedirs(grid_dir)
                else:
                    print(f"Found {grid_dir}, skipping directory creation")  

                sim_dict = {sim:[output]}
                # sb_file = os.path.join(args.wdir,f"sbdata_{sim_params['config']}.h5")
                for itr, angle_chunk in enumerate(chunked(angles, chunk_size), start=0): #group into plots of 100 angles, each file +10 deg in x 
                    print(f"Plotting {sim.run_name} at output = {output}, freq = {freq} angles = \n{angle_chunk}")
                    angle_dict = {sim: angle_chunk}

                    # query_points = {}
                    # query_points[sim]={ #TODO fix this based on which way the inj is going
                    #     "x": [sim.get_injection_region(50)[0].value,sim.get_injection_region(50)[2].value],
                    # }

                    grid_file = os.path.join(grid_dir,f"{output}_{freq}_grid{itr}")

                    if not os.path.isfile(f"{grid_file}.pdf"):
                        #TODO change vmin vmax and xlim ylim etc
                        plot_kwargs = {
                            "fig_size": 8,
                            "xlim": (-70, 70),
                            "ylim": (-45, 70),
                            # "query_points": query_points,
                            "label_all_axes": True,
                            "row_len": math.ceil(math.sqrt(len(angle_chunk)))
                            # "vmin": 0,
                            # "vmax": 0.75,
                        }

                        params = pl_splines.SplineParams(max_itr = 0,percentile = 50,sb_weight = 10,smoothing=0.1)
                        fancypl.surf_brightness_splines(sim_dict=sim_dict,angle_dict=angle_dict,freq=freq,redshift=args.redshift,params=params,fname=grid_file,**plot_kwargs) #NOTE loads the sb data in house  

                    else:
                        print(f"File {grid_file}.pdf already exists, skipping...")
                    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)  # KB -> GB
                    print(f"Memory used: {peak_gb:.2f} GB")  
                # del 
                    gc.collect()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml",type=str,help=f"Path to praise setup yaml defaults to {def_yml_path}",default=def_yml_path)
    parser.add_argument("-r","--redshift",type=float,help=f"Redshift used to calculate surface brightness, defaults to {def_redshift}",default=def_redshift)
    parser.add_argument("-d","--wdir",type=str,help=f"Path to working dir (save location for h5 file), defaults to {'./'}",default="./")
    parser.add_argument("-c","--cluster",help="Submits this script as a PBS job to kunanyi",action="store_true")
    parser.add_argument("--job-name", default="hybrids_grid")
    parser.add_argument("--job-length", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--memory", type=int, default=400)
    args = parser.parse_args()

    if args.cluster:
        output_dir = init_script_dirs("~",files=[Path(args.yml).expanduser()],cluster=True) #NOTE set output dir to user home
        script_cluster = output_dir / Path(__file__).name #use script file in output dir
        yml_cluster = output_dir /Path(args.yml).name #use yml file in output dir

        cmd = ( #NOTE removes script files after job finishes 
            f'python3 -u "{script_cluster}" --yml "{yml_cluster}" '
            f'--redshift {args.redshift} --memory {args.memory} '
            f'&& rm -rf "{output_dir}"'
        )

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
        create_hybrid_plot(args)

if __name__ == "__main__":
    main()
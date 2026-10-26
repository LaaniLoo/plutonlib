#! /usr/bin/env python3
from pathlib import Path
import argparse
from pdb import run
import yaml
import subprocess
import sys
from datetime import datetime

sys.dont_write_bytecode = True #stops __pycache__ being written
import pluto_setup as ps
from plutonlib.pbs_job import create_run_script, submit_run_script,check_job

def write_run_info(info_dir,build_info,job_id,job_info,ssh=True):
    info = {
        "submitted": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "job_info": job_info,
        "build_info": build_info,
    }
    text = yaml.dump(info, default_flow_style=False, sort_keys=False)
    path = info_dir / f"{job_id.split('.')[0]}.yml"
    ps.write_file(path,text,ssh)

def read_build_info(sim_dir, ssh=True):
    path = sim_dir / "build_info.yml"
    if not ps.path_exists(path, ssh=ssh):
        return None
    if ssh:
        text = subprocess.run(["ssh", ps.HOST, f'cat "{path}"'], capture_output=True, text=True, check=True).stdout
    else:
        text = path.read_text()
    return yaml.safe_load(text)

def chk_update_ini(sim_root,sim_type,run_config,run_name,dirs,ssh):
    """Checks if the right pluto ini is in run_dir and updates log and output entries"""
    info_dir = dirs["remnant_info"] if run_config["remnant"] else dirs["run_info"]
    log_dir = dirs["remnant_log"] if run_config["remnant"] else dirs["run_log"]
    exec_dir = dirs["remnant"] if run_config["remnant"] else dirs["run_dir"]
    ini_path = info_dir / f"{run_name}.ini" #either run_dir/run_info or run_dir/remnant/remnant_info

    if not ps.path_exists(ini_path,ssh=ssh):
        ps.setup_run(sim_root, sim_type, run_name, run_config, ssh=ssh)
        if not ps.path_exists(ini_path, ssh):        # still missing: no ini_file in the yml, you make it
            raise FileNotFoundError(f"no ini_file entry yml and '{ini_path}' not found. Create it, or copy one and rename to {run_name}.ini")

    # ini_txt = ini_path.expanduser().read_text()
    ini_txt = ps.edit_ini(ps.read_file(ini_path, ssh), {"log_dir": log_dir, "output_dir": exec_dir})
    ps.write_file(ini_path, ini_txt, ssh=ssh)

    return ini_path,ini_txt

def chk_requires_setup(dirs,ssh):
    """Checks if sim/run dir is the correct structure returns True if setup_run is required"""
    src_dir = dirs["sim_dir"] / "src"
    if not ps.path_exists(src_dir, ssh=ssh): #check if jet-setup has been run
        raise RuntimeError(f"jet-setup source not found in '{src_dir}', please run pluto_setup.py")

    if not ps.path_exists(dirs["run_dir"], ssh=ssh):
        print(f"Run directory '{dirs['run_dir']}' not found, creating it from the yml entry")
        return True
    return False

def verify_build(build_info,dirs,pluto_config,modules,ssh):
    """Verifies PLUTO setup and build with the build_info yaml constructed at setup, if they dont match -> rebuild PLUTO"""
    jet_setup_path = ps.expand_path(pluto_config['jet_setup_path'],ssh=ssh) #path to jet-setup (specific to this config)
    build_script = jet_setup_path / "tools" / "create-build-script"

    defs_config = pluto_config['defs'] #definitions.h entries
    defs_usr_config = pluto_config['defs_usr'] #definitions_usr.h entries

    if build_info is not None:
            build_info.pop("build_time",None)
    if build_info == {**pluto_config,"modules": modules}:
        print("Build is up to date, skipping update_defs and build")
    else:        
        ps.update_defs(sim_dir=dirs["sim_dir"],defs_config=defs_config,defs_usr_config=defs_usr_config,ssh=ssh)
        ps.build_pluto(sim_dir=dirs["sim_dir"],build_script=build_script,modules=modules,ssh=ssh)
        ps.write_build_info(sim_dir=dirs["sim_dir"],pluto_config=pluto_config,modules=modules,ssh=ssh)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--local",help="Use to run setup locally (Create dirs and run simulation on local machine)",action="store_true")
    parser.add_argument("-y", "--yml",help=f"path to simulation config YAML (default: yml next to the script, else {ps.YML_PATH})",type=Path,default=None)
    parser.add_argument("--sim-type", help="simulation key in the yml, e.g. MHD_jet",default=None)
    parser.add_argument("--run-name", help="REQUIRED: simulation run name in the yml, e.g. Q36_test",required=True)
    parser.add_argument("-r","--restart",help="Override the restart variable in sim_config.yml",type=int,default=None)
    args = parser.parse_args()
    ssh = not (args.local or ps.ON_CLUSTER)

    ps.script_mode(args) #print the mode the script is running in
    script_dir = Path(__file__).resolve().parent
    local_yml = script_dir / "sim_config.yml"
    yml_path = args.yml.expanduser() if args.yml else (local_yml if local_yml.exists() else ps.YML_PATH) #in the case of ssh, local yml and YML_PATH are the same    
    with open(yml_path,'r') as f:
        setup_yml = yaml.safe_load(f)

    sim_type = args.sim_type or (script_dir.name if script_dir.name in setup_yml["simulations"] else None)
    if sim_type is None:
        if ssh:
            raise SystemExit(f"In ssh mode use --sim-type arg to point to simulation group. Available: {', '.join(setup_yml['simulations'])}")
        else:
            raise SystemExit(f"--sim-type needed. Available: {', '.join(setup_yml['simulations'])}")

    sim_type_config = setup_yml["simulations"].get(sim_type)
    if sim_type_config is None:
        raise SystemExit(f"sim_type '{sim_type}' not in {yml_path}. Available: {', '.join(setup_yml['simulations'])}")
    run_config = sim_type_config["runs"].get(args.run_name)
    if run_config is None:
        raise SystemExit(f"run '{args.run_name}' not in {sim_type}. Available: {', '.join(sim_type_config['runs'])}")
    pluto_config = setup_yml['pluto_config'][sim_type_config['pluto_config']] #get the pluto config for this simulation type e.g pluto_config: 'cluster-hd'
    job_config = setup_yml['job_config'][run_config['job_config']] #get the pbs job config for this simulation run e.g. job_config: 'normal'

    sim_root = ps.expand_path(setup_yml["simulations_root"], ssh=ssh)
    dirs = ps.get_simulation_dirs(sim_root, sim_type, args.run_name)
    info_dir = dirs["remnant_info"] if run_config["remnant"] else dirs["run_info"]
    exec_dir = dirs["remnant"] if run_config["remnant"] else dirs["run_dir"]

    req_setup = chk_requires_setup(dirs=dirs,ssh=ssh) #verify paths, return true if setup is required for run dirs
    if req_setup:
        ps.setup_run(sim_root=sim_root,sim_type=sim_type,run_name=args.run_name,run_config=run_config,ssh=ssh)

    #check PLUTO build against yaml
    build_info = read_build_info(dirs["sim_dir"],ssh=ssh) 
    verify_build(build_info=build_info,dirs=dirs,pluto_config=pluto_config,modules=setup_yml["modules"],ssh=ssh)

    #check ini after re-running build/defs in case you want custom ini not to be overwritten
    ini_path,ini_txt = chk_update_ini(sim_root=sim_root,sim_type=sim_type,run_config=run_config,run_name=args.run_name,dirs=dirs,ssh=ssh)

    if not args.local:
        is_restart = run_config["restart"][0] or args.restart is not None
        compression = job_config["compression"]
        restart = run_config["restart"][1] if args.restart is None else args.restart
        modules = " ".join(setup_yml["modules"])
        maxtime = job_config['job_length'] - 1.5 if job_config['job_length'] > 1.5 else 1
        pluto_str = f"mpirun -mca io ^ompio ./pluto -i {ini_path} -maxtime {maxtime}"

        if is_restart:
            pluto_str += f" -h5restart {restart}"

        lines = [f"cd {dirs['sim_dir']}", f"module load {modules}"]
        if compression:
            ps.copy_file(src=(ps.PLUTO_SCRIPTS / "compression_script.py"),dest=exec_dir/"compression_script.py",ssh=ssh) #copy compression script to current run
            lines += [f"python {exec_dir}/compression_script.py &", "COMP_PID=$!"] #run compression and pluto at the same time
        lines.append(pluto_str)
        if compression:
            lines.append("wait $COMP_PID")
        cmd = "\n".join(lines)

        script_info = create_run_script(
            cmd=cmd,
            job_name=args.run_name,
            job_length=job_config["job_length"],
            nodes=job_config["nodes"],
            cpus=job_config["cpus"],
            log_dir=str(info_dir),
            # memory=128,
        )
        job_id,job_info = submit_run_script(script_info=script_info,workdir=info_dir)
        ps.write_file(info_dir / f"{job_id.split('.')[0]}.ini", ini_txt, ssh) #copy the ini used and append job id
        write_run_info(info_dir=info_dir,build_info=build_info,job_id=job_id,job_info=job_info,ssh=ssh)
        
if __name__ == "__main__":
    main()

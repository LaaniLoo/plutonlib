#! /usr/bin/env python3

import shutil
import yaml
import subprocess
import re
import os
from pathlib import Path,PurePosixPath
from functools import lru_cache
import socket

import argparse
from datetime import datetime

#---Global vars and paths---#
#NOTE these paths are local unless script is run from cluster
try:
    PLUTO_DIR = Path(os.environ["PLUTO_DIR"])
except KeyError: #if no environment var set to home dir
    PLUTO_DIR = Path("~/pluto-master").expanduser()
try:
    PLUTONLIB = Path(os.environ["PLUTONLIB"])
except KeyError:
    PLUTONLIB = Path("~/plutonlib").expanduser()

PLUTO_UTILS = PLUTONLIB / "pluto_utils"
PLUTO_SCRIPTS = PLUTONLIB / "scripts"
YML_PATH = PLUTO_UTILS / "sim_config.yml"
HOST = "kunanyi"

ON_CLUSTER = HOST in socket.gethostname()

#---#

#---Notes---#
# If running via ssh, local files will be copied to the cluster, can edit setup locally and 'push' changes to cluster 
# If running directly on the cluster, cluster setup files (yml, ini) will be used for the simulation
#---#

#---helpers---#
@lru_cache(maxsize=None)  #Cache ssh home dir
def remote_home():
    out = subprocess.run(["ssh", HOST, "echo $HOME"],
            stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
    return PurePosixPath(out)

def script_mode(args):
    if ON_CLUSTER:
        print("Script mode: Cluster (direct, no SSH)")
    elif args.local:
        print("Script mode: local machine")
    else:
        print(f"Script mode: local machine -> {HOST} (SSH)")

def path_exists(path, ssh=True):
    if not ssh:
        return Path(path).exists()
    r = subprocess.run(["ssh", HOST, f'test -e "{path}"'])
    if r.returncode == 255:                      # ssh itself failed, not "file missing"
        raise ConnectionError(f"ssh to {HOST} failed")
    return r.returncode == 0

def expand_path(path, ssh=True):
    if not ssh:
        return Path(path).expanduser().resolve()
    p = str(path)
    if p == "~" or p.startswith("~/"):
        return remote_home() / p[2:]
    return PurePosixPath(p)

def read_file(path,ssh=True):
    """Determines method to read file, either via ssh or via Pathlib"""
    path = expand_path(path,ssh)
    if ssh:
        return subprocess.run(["ssh", HOST, f'cat "{path}"'], stdout=subprocess.PIPE, text=True, check=True).stdout
    else:
        return path.read_text()

def write_file(path, text, ssh=True):
    """Determines method to write file, either via ssh or via Pathlib"""
    path = expand_path(path, ssh)
    if ssh:
        subprocess.run(["ssh", HOST, f'cat > "{path}"'], input=text, text=True, check=True)
    else:
        path.write_text(text)

def copy_scripts_files(sim_dir,files=None,ssh=True):
    """If ssh=True, transfer required local files/scripts -> cluster """
    sim_dir = expand_path(sim_dir, ssh)
    # if not ssh and not sim_dir.exists():          # only check locally when local
    #     raise FileNotFoundError(f"Simulation directory '{sim_dir}' does not exist, please run `setup_sim()`")

    base_files = [PLUTO_UTILS / f for f in ["pluto_setup.py","pluto_run.py","pluto_template.ini","job_submit.sh"]]
    base_files.append((PLUTO_SCRIPTS / "compression_script.py"))
    files = [Path(f).expanduser() for f in (files or [])] + base_files

    if ssh:  #transfer files from local -> cluster
        print(f"\nTransferring {[f.name for f in files]} to {HOST}:{sim_dir}")
        subprocess.run(["scp", *map(str, files), f"{HOST}:{sim_dir}/"], check=True)
    else:
        for f in files:
            dest = sim_dir / f.name
            if dest.exists() and (dest.samefile(f) or f.stat().st_mtime <= dest.stat().st_mtime):
                continue
            shutil.copy(f, dest)

def copy_file(src,dest,ssh=True):
    "Copy additional files to dest, e.g ini to run_dir or remnant_dir"
    src = Path(src).expanduser()
    dest = expand_path(dest,ssh)
    if ssh:
        subprocess.run(["scp", str(src), f"{HOST}:{dest}"], check=True)
    else:
        shutil.copy(src, dest)
    
def setup_remote_paths(paths, ssh=True):
    paths = [expand_path(p, ssh) for p in paths]

    if ssh:
            dirs = " ".join(f'"{p}"' for p in paths)
            subprocess.run(["ssh", HOST, f'mkdir -p {dirs}'], check=True)

    else:
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    for path in paths:
        print(f"\nCreated directory '{path}'")

    return paths        

#---Update defs/ini---#
def edit_defs(defs_txt,defs_dict):
    for key, value in defs_dict.items():
        if isinstance(value, bool):
            raise ValueError(f'{key}: unquoted YAML boolean, write "YES"/"NO" in quotes')
        pattern = rf"^(#define\s+{key}\s+)\S+"
        defs_txt, n = re.subn(pattern, rf"\g<1>{value}", defs_txt, flags=re.M)
        if n == 0:
            raise KeyError(f"#define {key} not found")
    return defs_txt

def update_defs(sim_dir,defs_config,defs_usr_config,ssh=True):
    src_path = sim_dir / "src"
    for filename, config in [("definitions.h", defs_config),("definitions_usr.h", defs_usr_config)]:
        path = src_path / filename
        text = read_file(path,ssh)
        write_file(path,edit_defs(text,config),ssh)

def edit_ini(ini_txt, ini_dict):
    lines = ini_txt.splitlines()
    found = set()
    for i, line in enumerate(lines):
        parts = line.split(None, 1)               # "log_dir    ../temp/log" -> ["log_dir", "../temp/log"]
        if parts and parts[0] in ini_dict:
            key = parts[0]
            lines[i] = f"{key:<10} {ini_dict[key]}"
            found.add(key)
    missing = set(ini_dict) - found
    if missing:
        raise KeyError(f"not found in ini: {sorted(missing)}")
    return "\n".join(lines) + "\n"

#---setup dirs and pluto src---#
def jet_setup(sim_dir,setup_script,makefile_name,ssh=True):
    # if not mkf_path.exists():
    #     raise FileNotFoundError(f"Cannot find makefile.defs (`{mkf_path}`) in PLUTO_DIR/Config.")

    #run jet-setup setup problem
    print("\nRunning setup-problem...\n")
    if ssh:
        sed_cmd = f'sed -i "s/^ARCH .*/ARCH         = {makefile_name}/" src/makefile'
        subprocess.run(["ssh", HOST, f"cd {sim_dir} && bash {setup_script} && {sed_cmd}"], check=True)
    else:
        subprocess.run(["bash", str(setup_script)], cwd=str(sim_dir), check=True)

        #update makefile to the one specified -> skips menu interaction with setup.py
        jet_src = sim_dir / "src"
        makefile = jet_src / "makefile"
        # if not makefile.exists():
        #     raise FileNotFoundError(f"Cannot find makefile in `{jet_src}`")
        mkf_str = f"ARCH         = {makefile_name}" #this is the line being replaced with make file_name in the makefile #NOTE spacing is important?
        makefile.write_text(re.sub(r"^ARCH .*$",mkf_str,makefile.read_text(), flags=re.M))

def get_simulation_dirs(sim_root, sim_type, run_name):
    """Build the dict of paths. Touches no filesystem."""
    sim_dir = sim_root / sim_type
    run_dir = sim_dir / run_name
    dirs = {
        "sim_dir": sim_dir,
        "run_dir": run_dir,
        "run_log": run_dir / "log",
        "run_info": run_dir / "run_info",
    }
    # if remnant:
    remnant_dir = run_dir / "remnant"
    dirs["remnant"] = remnant_dir
    dirs["remnant_log"] = remnant_dir / "log"
    dirs["remnant_info"] = remnant_dir / "run_info"
    return dirs

def setup_simulation_dirs(sim_root, sim_type, run_name, ssh=True):
    dirs = get_simulation_dirs(sim_root, sim_type, run_name)
    expanded = setup_remote_paths(list(dirs.values()), ssh=ssh)   # creates them
    return dict(zip(dirs.keys(), expanded))

def setup_run(sim_root, sim_type, run_name, run_config, ssh=True):
    dirs = setup_simulation_dirs(sim_root, sim_type, run_name, ssh=ssh)
    info_dir = dirs["remnant_info"] if run_config["remnant"] else dirs["run_info"]  #destination for ini file, either run or remnant dir
    dest = info_dir / f"{run_name}.ini"
    ini = run_config.get("ini_file")
    if ini is not None:
        ini_path = Path(ini).expanduser()
        if not ini_path.exists():
            raise FileNotFoundError(f"ini_file '{ini_path}' not found on this machine")
        if path_exists(dest, ssh):
            print(f"Using existing ini '{dest}' (delete it to re-copy from the yml)")
        else:
            copy_file(src=ini_path, dest=dest, ssh=ssh)
    return dirs

#---Building pluto---#
def build_pluto(sim_dir,build_script,modules,ssh=True):
    #run jet-setup create-build script
    if ssh:
        subprocess.run(["ssh", HOST, f"cd {sim_dir} && bash {build_script}"], check=True)
    else:
        subprocess.run(["bash", str(build_script)], cwd=str(sim_dir), check=True)

    mod_str = " ".join(modules)
    if ON_CLUSTER: #running the script directly on the cluster
        print(f"Loading modules: {modules}\n")
        subprocess.run(["bash", "-lc", f"module load {mod_str} && bash build"], cwd=str(sim_dir), check=True)
    elif ssh: #running the script remotely via ssh on the cluster
        print(f"Loading modules: {modules}\n")
        subprocess.run(["ssh", HOST, f'bash -lc "module load {mod_str} && cd {sim_dir} && bash build"'], check=True)
    else: #running it on a local machine
        subprocess.run(["bash", "build"], cwd=str(sim_dir), check=True)

def write_build_info(sim_dir, pluto_config, modules, ssh=True):
    info = {
        "build_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "makefile": pluto_config["makefile"],
        "jet_setup_path": pluto_config["jet_setup_path"],
        "modules": modules,
        "defs": pluto_config["defs"],
        "defs_usr": pluto_config["defs_usr"],
    }
    text = yaml.dump(info, default_flow_style=False, sort_keys=False)
    path = sim_dir / "build_info.yml"
    write_file(path,text,ssh)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--local",help="Use to run setup locally (Create dirs and run simulation on local machine)",action="store_true")
    parser.add_argument("--no-build",help="Set up the simulation but skip compiling PLUTO",dest="build",action="store_false") #store it to args.build, avoid double negatives
    parser.add_argument("--skip-update",help="Skips updating definitions.h files",dest="update",action="store_false")
    parser.add_argument("-y", "--yml",help=f"path to simulation config YAML (default: yml next to the script, else {YML_PATH})",type=Path,default=None)
    args = parser.parse_args()
    ssh = not (args.local or ON_CLUSTER) #ssh local -> cluster when running the script locally without --local
 
    script_mode(args) #print the mode the script is running in
    script_dir = Path(__file__).resolve().parent
    local_yml = script_dir / "sim_config.yml"
    yml_path = args.yml.expanduser() if args.yml else (local_yml if local_yml.exists() else YML_PATH) #in the case of ssh, local yml and YML_PATH are the same   
    with open(yml_path,'r') as f:
        setup_yml = yaml.safe_load(f)

    sim_root = expand_path(setup_yml["simulations_root"], ssh=ssh) #root dir of simulations e.g. ~/pluto-master/Simulations #NOTE helper expands user to local or cluster
    
    for sim_type, sim_type_config in setup_yml["simulations"].items():
        pluto_config = setup_yml['pluto_config'][sim_type_config['pluto_config']] #get the pluto config for this simulation e.g pluto_config: 'cluster-hd'
        defs_config = pluto_config['defs'] #definitions.h entries
        defs_usr_config = pluto_config['defs_usr'] #definitions_usr.h entries
        sim_dir = sim_root / sim_type            # sim_root is already expanded, so this is absolute

        jet_setup_path = expand_path(pluto_config['jet_setup_path'],ssh=ssh) #path to jet-setup (specific to this config)
        setup_script = jet_setup_path / "tools" / "setup-problem" #no extension?
        build_script = jet_setup_path / "tools" / "create-build-script"

        #per run commands: create directories and setup ini
        for run_name,run_config in sim_type_config["runs"].items():
            dirs = setup_run(sim_root=sim_root,sim_type=sim_type,run_name=run_name,run_config=run_config,ssh=ssh) #setup all dirs for runs

        copy_scripts_files(sim_dir=sim_dir,files=[yml_path],ssh=ssh) #copy files + yml to sim_dir

        if not path_exists(sim_dir / "src", ssh=ssh):
            jet_setup(sim_dir=sim_dir, setup_script=setup_script, makefile_name=pluto_config["makefile"], ssh=ssh)
        else:
            print(f"\n'{sim_dir / 'src'}' exists, skipping setup-problem")
        if args.update:
            update_defs(sim_dir=sim_dir, defs_config=defs_config, defs_usr_config=defs_usr_config, ssh=ssh)
        if args.build:
            build_pluto(sim_dir=sim_dir, build_script=build_script, modules=setup_yml["modules"], ssh=ssh)
        if args.update and args.build:
            write_build_info(sim_dir=sim_dir,pluto_config=pluto_config,modules=setup_yml["modules"],ssh=ssh)

if __name__ == "__main__":
    main()



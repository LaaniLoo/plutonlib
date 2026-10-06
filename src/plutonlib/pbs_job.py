import argparse

import os
import sys
from pathlib import Path, PurePosixPath
import glob
import shutil
import subprocess
import shutil

modules = ["HDF5","OpenMPI/4.1.5-GCC-12.3.0-pbs"]
environment = "analysis"

def init_script_dirs(wdir=None,files=None,cluster=False,host="kunanyi"):
    files = [] if files is None else files
    if not isinstance(files,list):
        raise TypeError(f"Files arg is of type = {type(files)}, please input files as list")

    script_path = getattr(sys.modules["__main__"],"__file__",None)
    if script_path is None:
        raise RuntimeError("No entry script found (running interactively?)") 
    script_path = Path(script_path).resolve() #path to the called script

    files = [Path(f).expanduser() for f in files + [script_path]]
    name = f"{script_path.stem}_files"
    
    if cluster:
        rel = PurePosixPath(wdir or ".") / name
        script_dir = PurePosixPath(subprocess.run(
            ["ssh", host, f"mkdir -p {rel} && cd {rel} && pwd"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        for f in files:
            print(f"Transferring {f.name} to {host}:{script_dir}")
            try:
                subprocess.run(["scp", str(f), f"{host}:{script_dir}/"],
                            capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print(e.stderr)
                raise

    else:
        wdir = Path.home() if wdir is None else Path(wdir).expanduser()
        if not wdir.exists():
            raise FileNotFoundError(f"Working directory {wdir.resolve()}, does not exist")

        script_dir = wdir / name #~/praise_hdf5_files
        script_dir.mkdir(parents=True,exist_ok=True)

        for f in files:
            shutil.copy(f,script_dir)

    return script_dir

def create_run_script(*,cmd,job_name,job_length,nodes,cpus=28, memory=None):
    mem_str = " " if memory is None else f":mem={memory}gb"
    large_mem = memory is not None and memory > 128
    high_mem_q = "#PBS -q LARGE_MEM" if large_mem else ""
    cpu_resource_str = f"#PBS -lselect={nodes}:ncpus={cpus}:mpiprocs={cpus}" + mem_str

    run_script = "\n".join(
        [
            "#!/bin/bash -l",
            cpu_resource_str,
            f"#PBS -lwalltime={job_length}:00:00",
            f"#PBS -N {job_name}",
            high_mem_q,
            f"#PBS -j eo",
            f"#PBS -k oe",
            f"#PBS -m abe", #email 
            f"#PBS -M alain.mackay@utas.edu.au",
            f"#PBS -e {job_name}_log",
            f"conda activate {environment}",
            f"module load {' '.join(modules)}",
            "export NUMBA_THREADING_LAYER=workqueue", #NOTE fixes some multiprocessing bug causing it to crash
            "export NUMBA_NUM_THREADS=1",
            cmd, #what actually is being run
        ]
    )

    script_info = {
        "run_script": run_script,
        "job_name": job_name,  
    }

    return script_info

def submit_run_script(*,script_info):
    print(f"Submitting {script_info['job_name']} job to kunanyi")

    try:
        job = subprocess.run(
            ["ssh","-vvv", "kunanyi", "qsub", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            input=script_info['run_script'],
            check=True,
        )

    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        raise e

    job_id = job.stdout.strip()
    print(f"Submitted {job_id}")
    check_job(job_id)

def check_job(job_id):
    try:
        job = subprocess.run(
            ["ssh","-vvv", "kunanyi", f"qstat -fxw {job_id}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        raise e

    job_info_list = job.stdout.split("\n    ")
    skip = ("Variable_List",)
    job_info = {}
    for item in job_info_list:
        item = item.strip()
        if item.startswith("Job Id:"):
            job_info["Job_Id"] = item.split(":", 1)[1].strip()
        elif " = " in item:
            k, v = item.split(" = ", 1)
            if k.strip() not in skip:
                job_info[k.strip()] = v.strip()

    keys = [
        "Job_Id",
        "job_state",
        "Job_Name",
        # "Job_Owner",
        "queue",
        "Resource_List.select",
        "Error_Path",
        "Output_Path",
        "comment",
    ]    
    info_str = [f"{k}: {job_info[k]}" for k in keys]
    print("\n".join(info_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--job-length", type=int, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--cpus", type=int, default=28)
    parser.add_argument("--memory", type=int, default=None)
    parser.add_argument("--large_mem", default=False,action = "store_true")

    args = parser.parse_args()

    script_info = create_run_script(cmd=args.cmd, job_name=args.job_name, job_length=args.job_length,
                                     nodes=args.nodes, cpus=args.cpus, memory=args.memory)
    job_id = submit_run_script(script_info=script_info)

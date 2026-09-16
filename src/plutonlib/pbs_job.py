import argparse
import os
import glob
import subprocess

modules = ["HDF5","OpenMPI/4.1.5-GCC-12.3.0-pbs"]
environment = "analysis"

def create_run_script(*,cmd,job_name,job_length,nodes,cpus=28, memory=None):
    mem_str = " " if memory is None else f":mem={memory}gb"
    large_mem = memory is not None and memory > 128
    high_mem_q = "#PBS -q LARGE_MEM" if large_mem else ""
    cpu_resource_str = f"#PBS -lselect={nodes}:ncpus={cpus}:mpiprocs={cpus}" + mem_str

    run_script = "\n".join(
        [
            "#!/bin/bash -l",
            # queue_resource_dict[site],
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

    return job.stdout.strip()

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
    print(f"Submitted {job_id}")
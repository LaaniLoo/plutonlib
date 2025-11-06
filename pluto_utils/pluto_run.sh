#!/bin/bash
ncores=6 #number of cpu cores to run pluto on
ini_file="pluto_template.ini"
save_dir="."
run_dir="$save_dir/temp"
cluster_run=false

script_dir="$PLUTONLIB/pluto_utils"

show_help() {
    echo "-h = show help"
    echo "-c = cluster, submits job to cluster to run pluto"
    echo "-i = ini_file, changes ini_file from $ini_file"
    echo "-r = run_dir, changes run_dir from $run_dir"
    echo "-n = ncores, number of mpi cpu cores to run with, defaults to $ncores"
}

#script options
while getopts "hci:r:n:" opt; do
    case $opt in 
        h) show_help
            exit 0
            ;;
        c) cluster_run=true 
            ;;
        i) ini_file="$OPTARG"
            ;;
        r) run_dir="$save_dir/$OPTARG"
            ;;
        n) ncores="$OPTARG"
            ;;
        *) echo "Usage: $0" 
            show_help
            exit 1 
            ;;
    esac
done

while [[ "$change_output" != "y" && "$change_output" != "n" ]]; do
    read -p "Change run name? (output_dir = $run_dir) [y/n]:" change_output
done

if [[ "$change_output" == "y" ]]; then
    read -p "Enter new run name:" new_run_dir
    run_dir="$save_dir/$new_run_dir"
elif [[ "$change_output" == "n" ]]; then
    echo "output_dir set to $run_dir"
fi

log_dir="$run_dir/log"
job_info_dir="$run_dir"/job_info
ini_dir="$job_info_dir/$ini_file"

# Update the 'output_dir/log_dir' in the ini file
sed -i "s|^\(output_dir\s*\).*|\1$run_dir|" $ini_file
sed -i "s|^\(log_dir\s*\).*|\1$run_dir/log|" $ini_file

#check if the run directory exists, if not, create
if [ ! -d "$run_dir" ]; then
    mkdir -p "$run_dir"
    mkdir -p "$job_info_dir"
    mkdir -p "$log_dir"
fi

echo "Saving $ini_file to $job_info_dir"
mv "$save_dir"/$ini_file "$job_info_dir"

#--Functions--#
check_log() { 
    printf "\n"
    read -p "Check pluto log? [y/n]: " check_log
    if [[ "$check_log" == "y" ]]; then
        log_file="$log_dir/pluto.1.log"
        echo "Waiting for log file..."
        while [ ! -f "$log_file" ]; do 
            sleep 5
        done
        sleep 2
        # Show first 200 lines, then a blank line, then follow
        ( head -n 200 "$log_file"; printf "\n"; tail -f "$log_file" )
    fi
}


pluto_local() {
    printf "\n"
    echo "Running pluto ($ini_file) executable with $ncores CPU cores..."
    mpirun --use-hwthread-cpus -np $ncores ./pluto -i $ini_dir &
    pluto_pid=$! #pluto process ID

    # If user presses Ctrl+C, kill Pluto too
    # trap "echo 'Stopping PLUTO...'; kill -INT $pluto_pid 2>/dev/null" INT
    trap "printf '\nStopping PLUTO...\n'; kill -INT $pluto_pid 2>/dev/null" INT
    sleep 1
    check_log

    printf "\n"
    echo "PLUTO running with PID $pluto_pid (Ctrl+C to stop)"
    wait $pluto_pid   # keeps script tied to Pluto, Ctrl+C cleans up properly
}

pluto_cluster() {
    printf "\n"
    echo "Submitting pluto job ($ini_file)..."
    jobid=$(qsub -v ini_dir="$ini_dir",save_dir="$save_dir",job_info_dir="$job_info_dir" "./job_submit.sh")
    echo "Submitted job $jobid"
    jobnum=$(echo "$jobid" | cut -d. -f1)
    touch "$job_info_dir/$jobnum"
    sleep 5 
    check_log
}

#-----#
if [[ "$cluster_run" == true ]]; then
    cp "$script_dir/clean.sh" "$job_info_dir/clean.sh"
    pluto_cluster

elif [[ "$cluster_run" == false ]]; then
    pluto_local
fi

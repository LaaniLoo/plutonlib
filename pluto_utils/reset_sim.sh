#!/bin/bash
save_dir="."
run_dir="?"

show_help() {
    echo "-h = show help"
    echo "-r = run_dir, simulation run directory to clear"
}

#script options, needs help
while getopts "hr:" opt; do
    case $opt in
        h) show_help
            exit 0
            ;; 
        r) run_dir="$save_dir/$OPTARG"
            ;;
        *) echo "Usage: $0" 
            show_help
            exit 1 
            ;;
    esac
done

job_info_dir="$run_dir/job_info"
log_dir="$run_dir/log"

if [[ "$run_dir" == "?" ]]; then
    echo "run_dir optarg not entered (-r), exiting..."
    exit 1
fi 

if [[ -d "$job_info_dir" && -d "$log_dir" ]]; then #check if the dir is a simulation dir or exit
    
    while [[ "$reset_run" != "y" && "$reset_run" != "n" ]]; do
        read -p "Reset run? (removes all data) [y/n]:" reset_run
    done

    if [[ "$reset_run" == "y" ]]; then
        read -p "Confirm? [y/n]:" confirm
        if [[ "$confirm" == "y" ]]; then
            cd "$run_dir/job_info"
            cp *.ini ../..
            ./clean.sh
            cd ../..
            
            echo "Removing all data and resetting directory..."
            rm -rf $run_dir
        fi

    elif [[ "$reset_run" == "n" ]]; then
        exit 0
    fi

else
    echo "$run_dir is not a simulation directory, exiting..."
    exit 1
fi
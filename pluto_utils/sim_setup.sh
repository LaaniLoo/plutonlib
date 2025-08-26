#!/bin/bash
script_dir="$PLUTONLIB/pluto_utils"
cluster_run=false

#script options, needs help
while getopts "c" opt; do
    case $opt in 
        c) cluster_run=true ;;
        *) echo "Usage: $0 [-c]" ;; #[-n cores] [-i ini_file] [-r run_dir]
    esac
done

echo "Moving to $SIM_DIR" #env_var SIM_DIR
printf "Current Structure: " && ls "$SIM_DIR"

printf "\n"
read -e -p "Enter Current Simulation Name: " cur_sim
save_dir="${SIM_DIR}/${cur_sim}"
mkdir -p "$save_dir" # e.g Make sure the /Jet_xx folder exists

echo "Copying .sh/.ini files..." 
cp "$script_dir/job_submit.sh" "$save_dir/job_submit.sh"
cp "$script_dir/pluto_template.ini" "$save_dir/pluto_template.ini"
cp "$script_dir/pluto_run.sh" "$save_dir/pluto_run.sh"
# cp "$script_dir/clean.sh" "$save_dir/clean.sh"
cp "$script_dir/sim_setup.sh" "$save_dir/sim_setup.sh"

printf "\n"
read -p "Run jet-setup? [y/n]: " setup
if [[ "$setup" == "y" ]]; then
    cd "$save_dir"
    setup-problem
    create-build-script

    printf "\n"
    read -p "Edit setup.py? [y/n]: " python_setup
    if [[ "$python_setup" == "y" ]]; then
        cd "./src"
        python "$PLUTO_DIR"/setup.py #env_var PLUTO_DIR
    fi
fi

build_pluto(){
    read -p "Build pluto? [y/n]: " build_pluto
    if [[ "$build_pluto" == "y" ]]; then
        cd "$save_dir"
        ./build
    fi
}

build_pluto_cluster(){
    read -p "Build pluto? [y/n]: " build_pluto
    if [[ "$build_pluto" == "y" ]]; then
        ml "HDF5"
        cd "$save_dir"
        ./build
    fi
}
#-----#

if [[ "$cluster_run" == true ]]; then
    build_pluto_cluster
    cp "$script_dir/clean.sh" "$job_info_dir/clean.sh"

elif [[ "$cluster_run" == false ]]; then
    build_pluto
fi

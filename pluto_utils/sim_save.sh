#!/bin/bash

script_dir="$PLUTONLIB/pluto_utils"

echo "Moving to $SIM_DIR"
printf "Current Structure: " && ls "$SIM_DIR"

printf "\n"
read -p "Enter Current Simulation Name: " CUR_SIM
SAVE_DIR="${SIM_DIR}/${CUR_SIM}"
mkdir -p "$SAVE_DIR" # e.g Make sure the /Jet_xx folder exists

echo "Copying .sh/.ini files..." 
cp "$script_dir/job_submit_template.sh" "$SAVE_DIR/job_submit_template.sh"
cp "$script_dir/pluto_template.ini" "$SAVE_DIR/pluto_template.ini"
cp "$script_dir/sim_save.sh" "$SAVE_DIR/sim_save.sh"


#paths
TEMP_DIR="$SAVE_DIR/temp"
LOG_DIR="$TEMP_DIR/log"

mkdir -p "$TEMP_DIR" "$LOG_DIR"

# Check if there are any files to copy
if [ -z "$(find "$TEMP_DIR" -maxdepth 1 -type f)" ]; then
    echo "No files to copy in $TEMP_DIR. Exiting."
    exit 1
fi

read -p "Enter Simulation Run Name: " RUN_NAME
RUN_DIR="$SAVE_DIR/$RUN_NAME" # Create new <run_name> directory based on the simulation run name
mkdir -p "$RUN_DIR"

job_info_dir="$RUN_DIR"/job_info

mkdir -p "$job_info_dir"
job_script="job_submit.sh"
ini_file=$(grep -oP '(?<=-i )\S+' "$job_script")
# Copy everything in /Simulations/Temp? except the simulation folder itself (e.g., Jet)
# rsync -av "$TEMP_DIR/" "$RUN_DIR/"
mv "$TEMP_DIR"/* "$RUN_DIR/"

#move the job error and output files to a dir
echo "Moving job error/output files..."
mv "$SAVE_DIR"/job_submit.sh.e* "$job_info_dir"
mv "$SAVE_DIR"/job_submit.sh.o* "$job_info_dir"

echo "Saving run .ini file..,"
cp "$SAVE_DIR"/$ini_file "$job_info_dir"

echo "Simulation copied to $RUN_DIR"

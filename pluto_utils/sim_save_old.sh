#!/bin/bash

# Paths
read -e -p "Enter desired simulation save location: " USER_INPUT
USER_INPUT=$(eval echo "$USER_INPUT")
SIM_LOC=$(realpath -m "$USER_INPUT")

SIM_DIR="$SIM_LOC/Simulations"
TEMP_DIR="$SIM_DIR/Temp"
LOG_DIR="$TEMP_DIR/Log"
BKP_DIR="$SIM_DIR/bkp" #backup sims 

mkdir -p "$SIM_DIR" "$TEMP_DIR" "$BKP_DIR" "$LOG_DIR"


echo "Moving to $SIM_DIR"
printf "Current Structure: " && ls "$SIM_DIR"

printf "\n"
read -p "Enter Current Simulation Name: " CUR_SIM
SAVE_DIR="${SIM_DIR}/${CUR_SIM}"
mkdir -p "$SAVE_DIR" # Make sure the /Jet folder exists
cp job_submit.sh "$SAVE_DIR/job_submit.sh"

# Check if there are any files to copy
if [ -z "$(find "$TEMP_DIR" -maxdepth 1 -type f)" ]; then
    echo "No files to copy in $TEMP_DIR. Exiting."
    exit 1
fi

read -p "Enter Simulation Run Name: " RUN_NAME
RUN_DIR="$SAVE_DIR/$RUN_NAME" # Create new <run_name> directory based on the simulation run name
mkdir -p "$RUN_DIR"

# Copy everything in /Simulations/Temp? except the simulation folder itself (e.g., Jet)
# rsync -av "$TEMP_DIR/" "$RUN_DIR/"
mv "$TEMP_DIR"/* "$RUN_DIR/"

# BACKUP DIR WHEN COPYING
# if [ -z "$( ls -A "$BKP_DIR" )" ]; then
#    rsync -a "$TEMP_DIR/" "$BKP_DIR/"

# else
#    echo "Overwriting existing backup"
#    find "$BKP_DIR" -maxdepth 1 -type f -exec rm {} +
#    rsync -a "$TEMP_DIR/" "$BKP_DIR/"

# fi

# # Delete only the files in /Simulations/
# # find "$TEMP_DIR" -maxdepth 1 -type d -exec rm {} +
# find "$TEMP_DIR" -mindepth 1 -not -path "$LOG_DIR" -exec rm -rf {} +



echo "Simulation copied to $RUN_DIR"

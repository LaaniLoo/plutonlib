#!/bin/bash

# Paths
read -p "Enter Current Simulation Name: " CUR_SIM
# CUR_SIM="Jet"
TEMP_DIR="/home/alain/pluto-master/Simulations/Temp"
SIM_DIR="/home/alain/pluto-master/Simulations"
SAVE_DIR="${SIM_DIR}/${CUR_SIM}"

# Check if there are any files to copy
if [ -z "$(find "$TEMP_DIR" -maxdepth 1 -type f)" ]; then
    echo "No files to copy in $TEMP_DIR. Exiting."
    exit 1
fi

read -p "Enter Simulation Run Name: " SIM_NAME

# Make sure the /Jet folder exists
mkdir -p "$SAVE_DIR"

# Create new Run_n directory based on the simulation run name
RUN_DIR="$SAVE_DIR/$SIM_NAME"
mkdir -p "$RUN_DIR"

# Copy everything in /Simulations/Temp? except the simulation folder itself (e.g., Jet)
# rsync -av --exclude="${CUR_SIM}/" "$SIM_DIR/" "$RUN_DIR/"
rsync -av "$TEMP_DIR/" "$RUN_DIR/"


# Delete only the files in /Simulations/
find "$TEMP_DIR" -maxdepth 1 -type f -exec rm {} +

echo "Simulation copied to $RUN_DIR"

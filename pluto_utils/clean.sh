#!/bin/bash
echo "Moving job error/output files..."

# find numeric job file in current dir
job_num=$(find . -maxdepth 1 -type f -printf "%f\n" | grep -Eo '^[0-9]+')

# move only the files for that job number from parent dir to current dir
mv "../../job_submit.sh.e${job_num}" .
mv "../../job_submit.sh.o${job_num}" .
rm "./$job_num"

#!/bin/bash
echo "Moving job error/output files..."

# find numeric job file in current dir
job_files=($(find ./ -name "job_*" -exec basename {} \;))
# move only the files for that job number from parent dir to current dir
for job_file in "${job_files[@]}"; do
        job_num=${job_file#job_}
        mv "../../job_submit.sh.e${job_num}" .
        mv "../../job_submit.sh.o${job_num}" .
        rm "./job_${job_num}"
done

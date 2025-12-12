import plutonlib.compression as pcomp
import plutonlib.load as pl
import plutonlib.utils as pu

import h5py
import os
import time
import logging
import traceback
import numpy as np

wdir = os.path.dirname(os.path.abspath(__file__)) #work from where the script is stored
logging.basicConfig(
    filename=os.path.join(wdir,'compression.log'),
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S',
    force=True
)

check_freq_io = 120 #check every 2 minutes 
check_freq_idle = 900 #slow down to 15 mins at idle
# check_freq_io = 2 #check every 2 minutes 
# check_freq_idle = 60 #slow down to 15 mins at idle
t_last_out = None
t_prev_out = None
t_next_out = None
previous_files = []
out_dt = []
check_state = "idle"

logging.info(f"Current working directory: {wdir}")
while True:
    try: #if file loading error etc, break out of loop
        # time.sleep(2) #sometimes when writing both dbl and flt, dbl gets written first want preferentially compress flt
        current_files = pl.get_all_written_files(wdir)
    except FileNotFoundError:
        current_files = []
        warn_message = f"No files to compress in {wdir}, waiting for first output..."
        logging.warning(warn_message)
        print(warn_message)
        time.sleep(check_freq_idle)
        continue
    except Exception as e:
        error_message = f"Compression script failed: {e}"
        logging.error(error_message)
        trace = traceback.format_exc()
        logging.error(trace)
        print(trace)
        break
    
    new_files = [f for f in current_files if f not in previous_files] #finding new files
    if new_files: # new file detected
        current_time = time.time()

        if t_prev_out is not None:
            t_last_out = current_time
            out_dt.append(t_last_out - t_prev_out)
            t_next_out = t_last_out + np.mean(out_dt) #predict the next output time
            logging.info(f"Next expected output at: {time.strftime('%H:%M:%S', time.localtime(t_next_out))}")
        t_prev_out = current_time

        for file_to_compress in new_files: #loop across detected files
            fsize_chk_time = 3*check_freq_io
            if pu.pluto_is_written_out(file_to_compress,fsize_chk_time): #if the file size hasn't changed in e.g. 2 mins
                pcomp.compress_simulation_chunked_single(file_to_compress,keep_original = False) #compression stage
            else:
                logging.warning(f"Change in file size detected for {os.path.basename(file_to_compress)}, waiting {fsize_chk_time/60:.2f} minutes")
                # time.sleep(check_freq_io) #NOTE I dont think this is needed as pluto_is_written_out should wait
                new_files.append(file_to_compress)
                continue
        previous_files = current_files #update previous files
        continue

    if t_next_out and len(out_dt) >1:
        check_window = t_next_out - (0.05*np.mean(out_dt)) #so increase check freq at 95% of time before output
        if time.time() >= check_window and time.time() < t_next_out:
            if check_state != "active":
                logging.info("Close to expected output time, increasing check frequency")
                check_state = "active" 
            logging.info("Checking for output...")
            time.sleep(check_freq_io)
            continue

    if check_state != "idle":
        logging.info("Decreasing check frequency")
        check_state = "idle"
    # logging.info("Waiting for output...")
    time.sleep(check_freq_idle)

    if t_last_out and (time.time() - t_last_out) > 6*check_freq_idle: #e.g. logs every 3hrs to show that script is still running
        logging.info("Still running, waiting on next output...")
    
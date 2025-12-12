# during simulation compression
# rename while compressing -> rename to original
import h5py
import os
import time
from datetime import datetime
import logging
from plutonlib.colours import pcolours
import psutil

def count_total_hdf5_items(file):
    """
    counts the total datasets,groups and bytes to help log compression progress
    """
    total_stats = {
    'total_datasets': 0, 
    'total_groups': 0, 
    'total_bytes': 0,
    # 'current_dataset': 0,
    # 'current_group': 0
    }

    with h5py.File(file, "r") as f_in:
        def count_items(name,obj):
            if isinstance(obj, h5py.Dataset):
                total_stats['total_datasets'] += 1
                total_stats['total_bytes'] += obj.nbytes
            elif isinstance(obj, h5py.Group):
                total_stats['total_groups'] += 1
        f_in.visititems(count_items)

    return total_stats

def log_raw(text):
    logger = logging.getLogger()  # get root logger
    for handler in logger.handlers:
        for line in text.splitlines():   # split at \n
            handler.stream.write(line + '\n')
            handler.flush()

def compress_simulation_chunked_single(file,keep_original = True,compression_type = "gzip",plane = "xz"):
    """
    Function to compress and chunk PLUTO simulations
    Outputs written as output.ext.compressed, with verbose logging output to compression.log
    """
    logging.basicConfig(
        filename=os.path.join(os.path.dirname(file),'compression.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S',
        force=True
    )

    if not os.path.isfile(file):
        raise FileNotFoundError(f"File {file} does not exist")
    
    if file.endswith(".compressed"):
        logging.info(f"Skipping {os.path.basename(file)}, already compressed")
        return
    
    stats = {'datasets': 0, 'groups': 0, 'bytes': 0,}
    nc_fsize = (round(os.path.getsize(file) / (1024**3),3)) 
    file_comp = file + ".compressed"

    try:
        header = (
            f"\n{'='*80}"
            f"\n Compressing: {os.path.basename(file)}"
            f"\n Path: {file}"
            f"\n{'='*80}")

        # logging.info(header)
        log_raw(header)
        start = time.time()
        total_stats = count_total_hdf5_items(file)
        with h5py.File(file, "r",rdcc_nbytes=1024**3) as f_in, h5py.File(file_comp, "w") as f_out:

            last_log = {'value': 0}  # track percentage loaded for whole dataset
            def copy_with_chunking(name, obj,compression_type = compression_type, plane = plane):
                if isinstance(obj, h5py.Dataset):
                    stats["datasets"] += 1
                    
                    # chunks_dict = {
                    #     'xy': (obj.shape[0], obj.shape[1], 1), # for slicing: [:, :, z_mid]
                    #     'xz': (obj.shape[0], 1, obj.shape[2]), # For slicing: [:, y_mid, :]
                    #     'yz': (1, obj.shape[1], obj.shape[2]), # For slicing: [x_mid, :, :]
                    # }

                    chunks_dict = { #PLUTO indexes as z,y,x so orders are reversed
                        'xy': (1, obj.shape[1], obj.shape[2]),   # for slicing: [z_mid, :, :]
                        'xz': (obj.shape[0], 1, obj.shape[2]),   # for slicing: [:, y_mid, :]
                        'yz': (obj.shape[0], obj.shape[1], 1),   # for slicing: [:, :, x_mid]
                    }

                    #setup chunking
                    if obj.ndim == 3:
                        total_slices = obj.shape[0]
                        chunks = chunks_dict[plane]  # chunk to make 2D midpoint slices
                    elif obj.ndim == 2:
                        total_slices = obj.shape[0]
                        chunks = obj.shape
                    else:
                        total_slices = 1
                        chunks = None  # for 1D/scalars

                    #setup empty dataset
                    new_ds = f_out.create_dataset(
                        name,
                        shape=obj.shape,
                        dtype=obj.dtype,
                        chunks=chunks,
                        compression=compression_type,
                        compression_opts=3 if compression_type == "gzip" else None, #NOTE 2-3 compression opts
                        shuffle=True
                    )

                    # Copy attributes for fu ll PLUTO compatibility
                    for attr_name, attr_value in obj.attrs.items():
                        new_ds.attrs[attr_name] = attr_value
                    
                    #load a slice for incremental loading, if dset can fit in ram, load the whole thing
                    avail_ram = psutil.virtual_memory().available
                    if avail_ram // obj.nbytes >=1:
                        slice_size = total_slices   
                        # logging.info(f"{name} has size {obj.nbytes/1e9:.2f}GB, loading entire dataset into memory")
                    else:
                        slice_size = max(1,total_slices // 5)
                        logging.info(f"{name} has size {obj.nbytes/1e9:.2f}GB, loading a slice of size {slice_size}")

                    for i in range(0, total_slices, slice_size):
                        i_end = min(i + slice_size, total_slices)
                        if obj.ndim == 3:
                            data_slice = obj[i:i_end, :, :]
                            new_ds[i:i_end, :, :] = data_slice
                        elif obj.ndim == 2:
                            data_slice = obj[i:i_end, :]
                            new_ds[i:i_end, :] = data_slice
                        else:
                            new_ds[...] = obj[()]
                            break
                    
                        stats['bytes'] += data_slice.nbytes
                        percent_loaded = round(stats['bytes']/total_stats['total_bytes']*100,1)
                        datasets_count = f"{stats['datasets']}/{total_stats['total_datasets']}"

                        if percent_loaded - last_log['value'] >= 1:
                            percent_display = f"{percent_loaded}%"
                            name_width = 25  # adjust as needed
                            count_width = 10
                            percent_width = 8
                            message = f"{name.ljust(name_width)} | {datasets_count.center(count_width)} | {percent_display.rjust(percent_width)}"
                            logging.info(message)
                            # print(message)
                            last_log['value'] = percent_loaded
                    
                elif isinstance(obj, h5py.Group):
                    new_group = f_out.create_group(name)
                    for attr_name, attr_value in obj.attrs.items():
                        new_group.attrs[attr_name] = attr_value

            f_in.visititems(copy_with_chunking)

        c_fsize = (round(os.path.getsize(file_comp) / (1024**3),3))
        ratio = round(nc_fsize / c_fsize,1)
        c_time = round((time.time() - start)/60,2)

        compression_info = {file: {
            "success": True,
            "nc_fsize": nc_fsize,
            "c_fsize": c_fsize,
            "ratio":  ratio,
            "c_time": c_time 
            }}
        
        lines = [
            f"Compression successful!",
            f"Original size: {nc_fsize} Gb",
            f"Compressed size: {c_fsize} Gb",
            f"Compression ratio: {ratio}x",
            f"Compression time {c_time} min",
        ]

        longest_line = max(len(line) for line in lines)
        boxed_lines = [f"| {line.ljust(longest_line)} |" for line in lines]
        border_top = "_" * (longest_line + 4)
        border_bottom = "-" * (longest_line + 4)
        log_summary = f"\n{border_top}\n" + "\n".join(boxed_lines) + f"\n{border_bottom}\n"
        log_raw(log_summary)

        if not keep_original: #delete original file
                del_warn = f"Deleting {file}"
                print(del_warn)
                logging.warning(del_warn)
                os.remove(file)
    
    #error handling
    except Exception as e:
        print(f"Compression failed for {file}: {e}")
        if os.path.exists(file_comp):
            os.remove(file_comp)

        compression_info = {file: {"success": False,"error": e,}}
        logging.error(f"Failed to compress {file}: {e}")

    return compression_info


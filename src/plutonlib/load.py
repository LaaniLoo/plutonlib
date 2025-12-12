import plutonlib.config as pc
import plutonlib.simulations as ps
import plutonlib.utils as pu
from plutonlib.colours import pcolours

coord_systems = pc.coord_systems
PLUTODIR = pc.plutodir

from pathlib import Path 
import os

from concurrent.futures import ThreadPoolExecutor, as_completed,ProcessPoolExecutor
from functools import lru_cache
from collections import defaultdict 
import h5py as h5py
import numpy as np

import time
import glob
import inspect

# ------------------------#
#       functions
# ------------------------#

# ---Loading Files---#

# def set_hdf5_grid_info(grid_object):
"""
Taken from plutokore
"""
    # # Set the 1D cell edge coordinate arrays
    # setattr(grid_object, "ex", grid_object.ncx[0, 0, :])
    # setattr(grid_object, "ey", grid_object.ncy[0, :, 0])
    # setattr(grid_object, "ez", grid_object.ncz[:, 0, 0])

    # Set the 1D cell midpoint cooordinate arrays
    # setattr(grid_object, "mx", grid_object.ccx[:, 0, 0])
    # setattr(grid_object, "my", grid_object.ccy[0, :, 0])
    # setattr(grid_object, "mz", grid_object.ccz[0, 0, :])

    # # Set the 1D cell edge coordinate arrays
    # setattr(grid_object, "ex", grid_object.ncx[:, 0, 0])
    # setattr(grid_object, "ey", grid_object.ncy[0, :, 0])
    # setattr(grid_object, "ez", grid_object.ncz[0, 0, :])


    # setattr(grid_object, "dx", np.diff(grid_object.ex))
    # setattr(grid_object, "dy", np.diff(grid_object.ey))
    # setattr(grid_object, "dz", np.diff(grid_object.ez))

    # setattr(grid_object, "dx1", grid_object.dx)
    # setattr(grid_object, "dx2", grid_object.dy)
    # setattr(grid_object, "dx3", grid_object.dz)

def get_file_outputs(wdir):
    """
    Gets number of simulation file outputs
    """
    is_dbl_h5 = os.path.isfile(os.path.join(wdir,r"dbl.h5.out"))
    is_flt_h5 = os.path.isfile(os.path.join(wdir,r"flt.h5.out"))
    is_dbl = os.path.isfile(os.path.join(wdir,r"dbl.out"))
    # print(is_dbl_h5,is_flt_h5,is_dbl)

    if is_dbl_h5 and is_flt_h5: #combination of float and double -> get max
        out_fnames = ["dbl.h5.out","flt.h5.out"]
        last_outputs = []
        for out_fname in out_fnames:
            file_path = os.path.join(wdir,out_fname)
            with open(file_path, "r") as f:
                last_outputs.append(int(f.readlines()[-1].split()[0]))
        last_output = max(last_outputs)

    elif is_dbl_h5 or is_flt_h5:
        out_fname = "dbl.h5.out" if is_dbl_h5 else "flt.h5.out" #assigns correct dtype for loading
        file_path = os.path.join(wdir,out_fname)
        with open(file_path, "r") as f:
            last_output = int(f.readlines()[-1].split()[0])

    elif is_dbl: #should be deprecated?
        out_fname = "dbl.out"
        file_path = os.path.join(wdir,out_fname)
        with open(file_path, "r") as f:
            last_output = int(f.readlines()[-1].split()[0])

    else:
        raise FileNotFoundError("Either .out is missing or file is not of type [flt.h5,dbl.h5,.dbl]")
    
    return last_output

def get_all_written_files(wdir):
    """
    gets the file names of all fully-written PLUTO outputs NOT including compressed files
    """
    written_outputs = get_file_outputs(wdir) + 1 #to fix indexing

    is_dbl_h5 = os.path.isfile(os.path.join(wdir,r"dbl.h5.out"))
    is_flt_h5 = os.path.isfile(os.path.join(wdir,r"flt.h5.out"))
    is_dbl = os.path.isfile(os.path.join(wdir,r"dbl.out"))

    # compressed_files = glob.glob(os.path.join(wdir,f"*.compressed")) #check if there exists a single compressed file
    # is_compressed = os.path.isfile(compressed_files[0]) if compressed_files else False

    if pu.is_dbl_and_flt(wdir): #combination of float and double -> get only float for analysis 
        dext = "flt.h5" 

        files = sorted(glob.glob(os.path.join(wdir,f'data.*.{dext}')))
    
    elif is_dbl_h5 or is_flt_h5:
        dext = "flt.h5" if is_flt_h5 else "dbl.h5" #assigns correct dtype for loading, preferentially load float

        files = sorted(glob.glob(os.path.join(wdir,f'data.*.{dext}')))
    
    elif is_dbl:
        dext = "dbl"         
        files = sorted(glob.glob(os.path.join(wdir,f'data.*.{dext}')))

    return files[:written_outputs]

def load_file_output(wdir,load_output,var_choice,arr_type=None,load_slice=None):
    f"""
    loads a single file output from a PLUTO simulation, automatically detects file extension     
    :param wdir: str
        working directory where simulation files are included
    :param load_output: int
        integer of which output you want to load
    :param var_choice: list
        list of variables from the simulation file you want to load, e.g. ['x1','x2','rho','prs'] 
    :param arr_type: str
        Different arrays for different cell/grid coordinates see {pc.arr_type_key}
    :param load_slice: tuple
        Array slice to load, instead of loading whole hdf5 array

    """
    if arr_type is None:
        raise ValueError(f"arr_type set to None, see {pc.arr_type_key}")
    is_dbl_h5 = os.path.isfile(os.path.join(wdir,f"data.{load_output:04d}.dbl.h5")) or os.path.isfile(os.path.join(wdir,f"data.{load_output:04d}.dbl.h5.compressed"))
    is_flt_h5 = os.path.isfile(os.path.join(wdir,f"data.{load_output:04d}.flt.h5")) or os.path.isfile(os.path.join(wdir,f"data.{load_output:04d}.flt.h5.compressed"))
    is_dbl = os.path.isfile(os.path.join(wdir,r"dbl.out"))
    # print(is_dbl_h5,is_flt_h5,is_dbl)
    dtype = []

    if is_dbl_h5 or is_flt_h5:
        # start = time.time()
        dext = "flt.h5" if is_flt_h5 else "dbl.h5" #assigns correct dtype for loading, preferentially load float

        dtype.append("hdf5_double" if is_dbl_h5 else "hdf5_float")

        file_path = os.path.join(wdir,f"data.{load_output:04d}.{dext}")
        compressed_file_path = os.path.join(wdir,f"data.{load_output:04d}.{dext}.compressed")
        is_compressed = os.path.isfile(compressed_file_path)

        if is_compressed: #use available chunked data
            data_file_path = compressed_file_path
            print("Using compressed data")
        else:
            data_file_path = file_path
            # print(f"Using {dext} extension for {load_output}")

        with h5py.File(data_file_path, "r", 
               rdcc_nbytes=32 * 1024 * 1024, #was 512
               rdcc_nslots=2_000_003,
               rdcc_w0=0.75) as data_file:
            
            # setattr(data_file, "sim_time", data_file[f"Timestep_{load_output}"].attrs["Time"])
            geometry = "CARTESIAN" #NOTE this should be able to be read from grid.out

            prof_vars = pc.profiles(arr_type = arr_type)["profiles"]["all"]
            var_map = {'x1': prof_vars[0], 'x2': prof_vars[1], 'x3': prof_vars[2]} # used to map any arr_type to x1,x2,x3
            reverse_var_map = {v: k for k, v in var_map.items()}

            file_data = {}
            file_data["sim_time"] = data_file[f"Timestep_{load_output}"].attrs["Time"]

            avail_vars = [v for v in list(data_file[f"Timestep_{load_output}/vars"].keys()) if v in [var_map.get(x, x) for x in var_choice]]
            
            for v in avail_vars: #only reads avail vars from var_choice
                dataset_path = f"Timestep_{load_output}/vars/{v}"

                if load_slice is not None: #load in chunks though slicing the data
                    reversed_slice = tuple(load_slice[::-1])
                    data_array = data_file[dataset_path][reversed_slice]
                else: #no slicing
                    data_array = data_file[dataset_path][()]

                if data_array.ndim == 3: #NOTE this transposes z,y,x arrays into x,y,z arrays
                    file_data[v] = (np.transpose(data_array, (2, 1, 0)))
                    
                elif data_array.ndim == 2:
                    file_data[v] = (np.transpose(data_array, (1, 0)))

                # setattr(data_file, v, data_array) #store sliced data in attributes
            
            coord_vars = [var_map.get(v, v) for v in var_choice if var_map.get(v, v) in ['ncx', 'ncy', 'ncz', 'ccx', 'ccy', 'ccz']]
            for coord_var in coord_vars: #only read required coords
                coord_type = "node_coords" if coord_var.startswith('nc') else "cell_coords"
                coord_dim = coord_var[2].upper()  # X, Y, Z
                dataset = data_file[f"/{coord_type}/{coord_dim}"]

                coord_key = reverse_var_map.get(coord_var,coord_var) #swaps ncx to x1 etc

                # Apply the same slicing to coordinates
                if load_slice is not None:
                    reversed_slice = tuple(load_slice[::-1])
                    data_array = np.asarray(dataset[reversed_slice])
                else:
                    data_array = np.asarray(dataset[()])
             
                if data_array.ndim == 3:
                    file_data[coord_key] = (np.transpose(data_array, (2, 1, 0)))
               
                elif data_array.ndim == 2:
                    # 2D slice: just swap the two axes [y,x] -> [x,y] or [z,y] -> [y,z] etc.
                    file_data[coord_key] = (np.transpose(data_array, (1, 0)))

                # setattr(data_file, coord_var, data_array)

            loaded_vars = [v for v in var_choice if v in file_data or var_map.get(v,v) in file_data] #avail vars after attr setting

    elif is_dbl: #should be deprecated?
        out_fname = "dbl.out"
        dtype.append("double")

        # data_file = pk_io.pload(load_output, w_dir=wdir)

        #only want to do this calculation once
        geometry = data_file.geometry
        loaded_vars = [v for v in var_choice if hasattr(data_file, v)]

        file_data = {v: getattr(data_file, v) for v in loaded_vars}

    else:
        raise FileNotFoundError("Either .out is missing or file is not of type [flt.h5,dbl.h5,.dbl]")
    
    returns = {"file_data":file_data,"loaded_vars":loaded_vars,"geometry":geometry,"dtype":dtype}

    # print(f"dbl loading: {(time.time() - start):.2f}s")
    return returns

def get_particle_outputs(wdir,load_outputs=None):
    '''
    Gets the number of simulation particle file outputs
    '''
    # part_files = []
    pattern = os.path.join(wdir, "particles.*")
    particle_paths = sorted(glob.glob(pattern), key=lambda f: int(f.split(".")[-2]))
    file_ext = particle_paths[0].split(".")[-1]
    n_outputs = int(particle_paths[-1].split(".")[-2])    

    if not particle_paths:
        raise FileNotFoundError(f"No files found that match `particles.` in {wdir}")

    max_output = max(load_outputs) if isinstance(load_outputs,tuple) else load_outputs
    if max_output > n_outputs:
        raise ValueError(f"Attempting to load output {load_outputs} when there are {n_outputs} outputs")

    if isinstance(load_outputs, tuple):
        loaded_files = [particle_paths[output_n] for output_n in load_outputs if output_n <= n_outputs]
    elif isinstance(load_outputs, int):
        loaded_files = [particle_paths[output_n] for output_n in range(min(load_outputs, n_outputs) + 1)]
    elif load_outputs == "last":
        load_outputs = (particle_paths[n_outputs],)
    else:  # Load all
        loaded_files = particle_paths

    part_files = [f"part_{loaded_file.split('.')[-2]}" for loaded_file in loaded_files]

    returns = {
        "n_outputs":n_outputs,
        "files":particle_paths,
        "loaded_files":loaded_files,
        "part_files": part_files,
    }
    return returns

def get_particle_file_header(line_):
    file_header = {}
    if line_.split()[1] != "PLUTO":
        hlist = line_.split()[1:]
        if hlist[0] == "field_names":
            vars_ = hlist[1:]
            t = tuple([hlist[0], vars_])
            file_header.update(dict([t]))
        elif hlist[0] == "field_dim":
            varsdim_ = [int(fd) for fd in hlist[1:]]
            t = tuple([hlist[0], varsdim_])
            file_header.update(dict([t]))
        elif hlist[0] == "shk_thresh":
            shk_thresh_list = [float(st) for st in hlist[1:]]
            t = tuple([hlist[0], shk_thresh_list])
            file_header.update(dict([t]))
        else:
            t = tuple(["".join(hlist[:-1]), hlist[-1]])
            file_header.update(dict([t]))

    return file_header

def read_particle_file(file_name):
    # print("Reading Particle Data file : %s"%self.fname)
    file_header = {}
    with open(file_name, "rb") as fp_:
        for l in fp_.readlines():
            try:
                ld = l.decode("ascii").strip()
                if ld.split()[0] == "#":
                    file_header.update(get_particle_file_header(ld))
                else:
                    break
            except UnicodeDecodeError:
                break
    tot_fdim = np.sum(np.array(file_header["field_dim"], dtype=int))
    HeadBytes_ = int(file_header["nparticles"]) * tot_fdim * 8
    with open(file_name, "rb") as fp_:
        scrh_ = fp_.read()
        data_str = scrh_[len(scrh_) - HeadBytes_ :]
    
    hdict = file_header
    returns = {"hdict":hdict,"data_str":data_str,"tot_fdim":tot_fdim}
    return returns

# @lru_cache(maxsize=32)  # This caches based on input arguments
def pluto_particles(sim_type,run_name,load_outputs=None):
    wdir =  os.path.join(pc.sim_dir, sim_type, run_name)
    particle_data = defaultdict(list)  # Stores variables for each particle file

    particle_outputs = get_particle_outputs(wdir,load_outputs)
    loaded_files = particle_outputs["loaded_files"]
    part_files = particle_outputs["part_files"]

    for file_idx,file_name in enumerate(loaded_files): #used to loop over file path and particle file string
        particle_str = part_files[file_idx]

        hdict = read_particle_file(file_name)['hdict']
        data_str  = read_particle_file(file_name)['data_str']
        tot_fdim = read_particle_file(file_name)['tot_fdim']
        vars_ = hdict["field_names"]
        endianess = "<"
        if hdict["endianity"] == "big":
            endianess = ">"
        dtyp_ = np.dtype(endianess + "dbl"[0])
        DataDict_ = hdict
        n_particles = int(DataDict_["nparticles"])
        data_ = np.frombuffer(data_str, dtype=dtyp_)


        fdims = np.array(hdict["field_dim"], dtype=int)
        indx = np.where(fdims == 1)[0]
        spl_cnt = len(indx)
        counter = 0

        if n_particles <= 0 and isinstance(load_outputs,tuple) and len(load_outputs) == 1: 
            print(DataDict_)
            raise AttributeError(f"Particle file {file_name} has nparticles = 0")
        
        elif n_particles <= 0:
            # print(f"Particle file {file_name} has nparticles = 0") #debug part files with nparticles = 0 
            continue

        reshaped_data = data_.reshape(n_particles, tot_fdim)
        tup_ = []

        ind = 0
        for c, v in enumerate(vars_):
            v_data = reshaped_data[:, ind : ind + fdims[c]]
            if fdims[c] == 1:
                v_data = v_data.reshape((n_particles))
            tup_.append((v, v_data))
            ind += fdims[c]
        DataDict_.update(dict(tup_))

        particle_data[particle_str] = DataDict_

    returns = {
        "particle_data":particle_data,
    }
    return returns

@lru_cache(maxsize=32)  # This caches based on input arguments
def pluto_loader(sim_type, run_name, profile_choice, load_outputs=None,load_slice=None, arr_type=None,ini_file=None):
    f"""
    Loads simulation data from a specified Pluto simulation.

    Parameters:
    -----------
    sim_type : str
        Type of simulation to load (e.g., "Jet", "Stellar_Wind") #NOTE see config for saving structure.
    run_name : str
        Name of the specific simulation file to load e.g. "default".
    profile_choice : str
        Index selecting a profile from predefined variable lists (#NOTE found in config.py):
        - "2d_rho_prs": ["x1", "x2", "rho", "prs"]
    load_outputs : tuple
        Tuple of outputs you wish to load e.g. (0,), (0,1,2,), or "last" for last output
    arr_type : str
        Different arrays for different cell/grid coordinates see {pc.arr_type_key}
    
    ini_file : str
        File name not inc extension to use specified ini file with units

    load_slide : tuple
        Array slice to load, instead of loading whole hdf5 array

    Returns:
    --------
    dict
        Dictionary containing:
        - vars: dictionary of order vars[d_file][var_name] e.g. vars["data_0"]["x1"]
        - var_choice: List of variable names corresponding to the selected profile.
        - vars_extra: contains the geometry of the sim
        - d_files: contains a list of the available data files for the sim
    """
    load_slice = pu._hashable_to_slice(load_slice)
    # print(load_slice)
    vars = defaultdict(list)  # Stores variables for each D_file
    vars_extra = []
    warnings = []
    var_choice = pc.profiles()["profiles"][profile_choice]

    # wdir = SimulationData(sim_type, run_name, profile_choice, load_outputs=load_outputs, arr_type=arr_type).wdir
    wdir =  os.path.join(PLUTODIR, "Simulations", sim_type, run_name)

    n_outputs = get_file_outputs(wdir)
    warnings.append(f"{pcolours.WARNING}Outputs: {n_outputs}")

    if load_outputs == None:
        load_outputs = n_outputs
    elif load_outputs == "last":
        load_outputs = (get_file_outputs(wdir),)
    if isinstance(load_outputs, int) and load_outputs > n_outputs:
        raise ValueError(f"Trying to load more outputs ({load_outputs}) than available ({n_outputs})")

    # Number of digits for zero-padding
    n_digits = 3  # scales automatically with number of outputs

    # Assign d_files for display only
    if isinstance(load_outputs, tuple):
        d_files = [f"data.{output_n:0{n_digits}}" for output_n in load_outputs if output_n <= n_outputs]
    elif isinstance(load_outputs, int):
        d_files = [f"data.{output_n:0{n_digits}}" for output_n in range(min(load_outputs, n_outputs) + 1)]
    else:  # Load all
        d_files = [f"data.{output_n:0{n_digits}}" for output_n in range(n_outputs + 1)]

    # Define function for parallel processing
    def load_single_output(output_n):
        loaded_file = load_file_output(
            wdir=wdir,
            load_output=output_n,
            var_choice=var_choice,
            arr_type=arr_type,
            load_slice=load_slice,
        )

        sim_time = load_file_output(
            wdir=wdir, 
            load_output=output_n, 
            var_choice=["sim_time"],
            load_slice=None, 
            arr_type=arr_type
            )["file_data"]["sim_time"]
        return output_n, loaded_file,sim_time

    # Process outputs in parallel
    with ThreadPoolExecutor() as executor:
        # Prepare tasks based on load_outputs type
        if isinstance(load_outputs, int):
            tasks = range(load_outputs + 1)
        elif isinstance(load_outputs, tuple):
            tasks = load_outputs
        else:
            tasks = range(n_outputs + 1)

        # Submit all tasks
        futures = {executor.submit(load_single_output, output_n): output_n for output_n in tasks}

        # Process results as they complete
        for future in as_completed(futures):
            output_n, loaded_file,sim_time = future.result()

            # update d_files with sim_time #TODO make sure unit is relevant
            time_val = pc.code_to_usr_units("sim_time",sim_time,ini_file="jet_units")["conv_data_uuv"]
            time_unit = str(pc.get_pluto_units("CARTESIAN",ini_file)["sim_time"]["usr_uv"])
            time_str = f"_{time_val:.0f}{time_unit}"

            d_file_str = f"data.{output_n:0{n_digits}}{time_str}"
            for i, df in enumerate(d_files):
                if df.startswith(f"data.{output_n:0{n_digits}}"):
                    d_files[i] = d_file_str
                    break

            vars[d_file_str] = loaded_file["file_data"]

            # Only need to set these once (from first file)
            if output_n == (0 if isinstance(load_outputs, int) else load_outputs[0]):
                warnings.append(f"{pcolours.WARNING}Data Type: {loaded_file['dtype'][0]}")
                vars_extra.append(loaded_file["geometry"])

                # Check for missing variables
                loaded_vars = loaded_file["loaded_vars"]
                non_vars = set(var_choice) - set(loaded_vars)
                if non_vars:
                    warnings.append(f"{pcolours.WARNING}Simulation {run_name} doesn't contain: {', '.join(non_vars)}")

            warnings.append(f"{pcolours.WARNING}loaded File/s: {d_file_str}")  # DEBUG

    var_choice = loaded_vars #[v for v in var_choice if v not in non_vars]  # reassigning var_choice with avail vars

    return {"vars": vars, "var_choice": var_choice, "vars_extra": vars_extra, "d_files": d_files, "warnings": warnings}

@lru_cache(maxsize=32)  # This caches based on input arguments
def pluto_conv(sim_type, run_name, profile_choice, load_outputs=None,load_slice=None, arr_type=None, ini_file=None):
    f"""
    Converts Pluto simulation variables from code units to cuv and uuv units.

    Parameters:
    -----------
    sim_type : str
        Type of simulation to load (e.g., "Jet", "Stellar_Wind") #NOTE see config for saving structure.
    run_name : str
        Name of the specific simulation file to load e.g. "default".
    profile_choice : str
        Index selecting a profile from predefined variable lists (#NOTE found in config.py):
        - "2d_rho_prs": ["x1", "x2", "rho", "prs"]
    load_outputs : tuple
        Tuple of outputs you wish to load e.g. (0,), (0,1,2,), or "last" for last output
    arr_type : str
        Different arrays for different cell/grid coordinates see {pc.arr_type_key}
    
    ini_file : str
        File name not inc extension to use specified ini file with units

    load_slide : tuple
        Array slice to load, instead of loading whole hdf5 array

    Returns:
    --------
    dict
        Dictionary containing:
        - vars_uuv: dictionary of order vars[d_file][var_name] e.g. vars["data_0"]["x1"]
        - var_choice: List of variable names corresponding to the selected profile.
        - d_files: contains a list of the available data files for the sim
    """
    start1 = time.time()

    loaded_data = pluto_loader(
        sim_type=sim_type,
        run_name=run_name,
        profile_choice=profile_choice,
        load_outputs=load_outputs,
        load_slice=load_slice,
        arr_type=arr_type,
        ini_file=ini_file,
    )
    d_files = loaded_data["d_files"]
    vars_dict = loaded_data["vars"]
    var_choice = loaded_data["var_choice"]  # chosen vars at the chosen profile
    sim_coord = loaded_data["vars_extra"][0]  # gets the coordinate sys of the current sim
    warnings = loaded_data["warnings"]  # loads any warning messages about vars
    vars_uuv = defaultdict(dict)
    # print(f"Reloaded Pluto_loader: {(time.time() - start1):.2f}s")

    # Process each file and variable
    def process_file(d_file):
        file_results = {}
        for var_name in var_choice:
            raw_data = vars_dict[d_file][var_name]

            if isinstance(raw_data, h5py.Dataset): #actually loads the dataset when required
                raw_data = raw_data[()]

            conv_array = pc.code_to_usr_units(var_name, raw_data, ini_file=ini_file) #d_files # converts the raw pluto array
            file_results[var_name] = conv_array["conv_data_uuv"]

        return d_file, file_results

    # Use ThreadPoolExecutor for parallel processing
    start3 = time.time()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, d_file) for d_file in d_files]
        for future in as_completed(futures):
            d_file, file_results = future.result()
            vars_uuv[d_file].update(file_results)

    return {"vars_uuv": vars_uuv, "var_choice": var_choice, "d_files": d_files, "sim_coord": sim_coord, "warnings": warnings}

def pluto_conv_from_raw(raw_data, profile_choice, ini_file=None):
    """
    Convert already-loaded raw data to user units without reloading files
    """
    d_files = raw_data["d_files"]
    vars_dict = raw_data["vars"]
    var_choice = raw_data["var_choice"]
    sim_coord = raw_data["vars_extra"][0]
    warnings = raw_data["warnings"]
    
    vars_uuv = defaultdict(dict)
    
    def process_file(d_file):
        file_results = {}
        for var_name in var_choice:
            raw_data_array = vars_dict[d_file][var_name]
            
            if isinstance(raw_data_array, h5py.Dataset):
                raw_data_array = raw_data_array[()]
                
            conv_array = pc.code_to_usr_units(var_name, raw_data_array, ini_file=ini_file)
            file_results[var_name] = conv_array["conv_data_uuv"]
        
        return d_file, file_results

    # Use ThreadPoolExecutor for parallel processing of conversion (not loading)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, d_file) for d_file in d_files]
        for future in as_completed(futures):
            d_file, file_results = future.result()
            vars_uuv[d_file].update(file_results)

    return {
        "vars_uuv": vars_uuv, 
        "var_choice": var_choice, 
        "d_files": d_files, 
        "sim_coord": sim_coord, 
        "warnings": warnings
    }
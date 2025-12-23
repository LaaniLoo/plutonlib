import plutonlib.config as pc
import plutonlib.simulations as ps
import plutonlib.utils as pu
from plutonlib.colours import pcolours

coord_systems = pc.coord_systems
PLUTODIR = pc.plutodir

from pathlib import Path 
import os

from concurrent.futures import ThreadPoolExecutor, as_completed,ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Any, List
# from functools import lru_cache
from collections import defaultdict 
import h5py as h5py
import numpy as np

import time
import glob
import inspect
@dataclass
class HDF5Metadata:
    file_path: str
    sim_time: float
    dataset_paths: Dict[str, str]
    load_output: int

    t_load: float = 0
    geometry: str = "CARTESIAN"
    dtype: str = "hdf5_float"
    is_compressed: bool = False   
    is_conv: bool = True
    time_str: str = "0 Myr"
    extra_info: Dict[str, Any] = field(default_factory=dict)

# ------------------------#
#       functions
# ------------------------#

# ---Loading Files---#

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
               rdcc_nbytes=512 * 1024 * 1024, #was 512
               rdcc_nslots=2_000_003,
               rdcc_w0=0.75) as data_file:
            
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

            loaded_vars = [v for v in var_choice if v in file_data or var_map.get(v,v) in file_data] #avail vars after attr setting

    elif is_dbl: #should be deprecated?
        out_fname = "dbl.out"
        dtype.append("double")

        #only want to do this calculation once
        geometry = data_file.geometry
        loaded_vars = [v for v in var_choice if hasattr(data_file, v)]

        file_data = {v: getattr(data_file, v) for v in loaded_vars}

    else:
        raise FileNotFoundError("Either .out is missing or file is not of type [flt.h5,dbl.h5,.dbl]")
    
    returns = {"file_data":file_data,"loaded_vars":loaded_vars,"geometry":geometry,"dtype":dtype}

    # print(f"dbl loading: {(time.time() - start):.2f}s")
    return returns

def load_hdf5_metadata(wdir: str,load_output: int) -> HDF5Metadata:
    """
    Loads certain metadata from a single HDF5 file with minimal performance and memory impact,
    stores information in the HDF5Metadata dataclass.

    Parameters:
    -----------
    wdir : str
        Working directory where simulation files are located.
    load_output : int
        Integer of which output file to load metadata from (e.g., 0 for data.0000.flt.h5).
    
    Returns:
    --------
    HDF5Metadata
        Dataclass containing:
        - file_path: Path to the HDF5 file (compressed or uncompressed)
        - sim_time: Simulation time at this output
        - dataset_paths: Dictionary mapping variable names to their HDF5 dataset paths
        - load_output: The output number loaded
        - geometry: Coordinate system geometry (default "CARTESIAN")
        - dtype: Data type string ("hdf5_float" or "hdf5_double")
        - is_compressed: Boolean indicating if file is compressed
    """
    
    is_dbl_h5 = os.path.isfile(os.path.join(wdir,f"data.{load_output:04d}.dbl.h5")) or os.path.isfile(os.path.join(wdir,f"data.{load_output:04d}.dbl.h5.compressed"))
    is_flt_h5 = os.path.isfile(os.path.join(wdir,f"data.{load_output:04d}.flt.h5")) or os.path.isfile(os.path.join(wdir,f"data.{load_output:04d}.flt.h5.compressed"))
    dext = "flt.h5" if is_flt_h5 else "dbl.h5" #assigns correct dtype for loading, preferentially load float
    dtype = ("hdf5_double" if is_dbl_h5 else "hdf5_float")

    if not is_dbl_h5 and not is_flt_h5:
        raise FileNotFoundError #quick error as metadata only works for hdf5

    file_path = os.path.join(wdir,f"data.{load_output:04d}.{dext}")
    compressed_file_path = os.path.join(wdir,f"data.{load_output:04d}.{dext}.compressed")
    is_compressed = os.path.isfile(compressed_file_path)

    if is_compressed: #use available chunked data
        file_path = compressed_file_path
        # print("Using compressed data")

    data_file = h5py.File(file_path, "r", 
        rdcc_nbytes=512 * 1024 * 1024, #was 512
        rdcc_nslots=2_000_003,
        rdcc_w0=0.75)

    try:
        geometry = "CARTESIAN" #NOTE this should be able to be read from grid.out
        sim_time = data_file[f"Timestep_{load_output}"].attrs["Time"]
        file_vars = [v for v in list(data_file[f"Timestep_{load_output}/vars"].keys())]

        dataset_paths = {}
        for var in file_vars:
            dataset_paths[var] = f"Timestep_{load_output}/vars/{var}"

        if "node_coords" in data_file:
            for dim in ['X', 'Y', 'Z']:
                if dim in data_file["node_coords"]:
                    coord_name = f"nc{dim.lower()}"  # ncx, ncy, ncz
                    dataset_paths[coord_name] = f"node_coords/{dim}"

        # Cell coordinates: cell_coords/X,Y,Z -> ccx, ccy, ccz
        if "cell_coords" in data_file:
            for dim in ['X', 'Y', 'Z']:
                if dim in data_file["cell_coords"]:
                    coord_name = f"cc{dim.lower()}"  # ccx, ccy, ccz
                    dataset_paths[coord_name] = f"cell_coords/{dim}"
    finally:
        data_file.close()

    return HDF5Metadata(
        file_path=file_path,
        sim_time=sim_time,
        dataset_paths=dataset_paths,
        load_output=load_output,
        geometry=geometry,
        dtype=dtype,
        is_compressed=is_compressed
    )

def load_hdf5_lazy(wdir,load_output,var_choice,load_slice = None):
    """
    Loads specific variables from a single HDF5 file output using metadata for efficient access.
    
    This function uses load_hdf5_metadata to locate datasets, then loads only the requested
    variables from the HDF5 file. Arrays are automatically transposed from PLUTO's [z,y,x]
    ordering to [x,y,z] ordering for compatibility.
    
    Parameters:
    -----------
    wdir : str
        Working directory where simulation files are located.
    load_output : int
        Integer of which output file to load (e.g., 0 for data.0000.flt.h5).
    var_choice : list or str
        List of variable names to load from the file (e.g., ['rho', 'prs', 'ncx', 'ncy']).
        Includes both field variables and coordinate arrays.
    load_slice : tuple, optional
        Array slice to load a subset of data instead of the full array (e.g., (slice(0,100), slice(0,50))).
        Slice is applied in [z,y,x] order before transposition.
    
    Returns:
    --------
    dict
        Dictionary containing:
        - sim_time: Simulation time at this output
        - {var_name}: Loaded and transposed arrays for each requested variable

    """
    
    metadata = load_hdf5_metadata(wdir=wdir,load_output=load_output)
    dataset_paths = metadata.dataset_paths

    file_data = {}
    file_data["sim_time"] = metadata.sim_time
    data_file = h5py.File(metadata.file_path, "r", 
            rdcc_nbytes=512 * 1024 * 1024, #was 512
            rdcc_nslots=2_000_003,
            rdcc_w0=0.75)
    
    try:
        vars_to_load = [v for v in metadata.dataset_paths if v in var_choice]
        for v in vars_to_load:
            if load_slice is not None: #load in chunks though slicing the data
                reversed_slice = tuple(load_slice[::-1])
                data_array = data_file[dataset_paths[v]][reversed_slice] # avail_vars[v] = 'rho': 'Timestep_16/vars/rho', load this dataset 
            else: #no slicing
                data_array = data_file[dataset_paths[v]][()]
            

            if data_array.ndim == 3: #NOTE this transposes z,y,x arrays into x,y,z arrays
                file_data[v] = (np.transpose(data_array, (2, 1, 0)))
                
            elif data_array.ndim == 2:
                file_data[v] = (np.transpose(data_array, (1, 0)))
            else:
                file_data[v] = data_array

    finally:
        data_file.close()

    return file_data

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

# @lru_cache(maxsize=32)  # This caches based on input arguments
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

def pluto_loader_hdf5(wdir,var_choice, load_outputs=None,load_slice=None,ini_file=None,conv=True):
    """
    Loads and optionally converts multiple HDF5 simulation outputs with metadata tracking.
    
    This is the main high-level loading function that combines metadata extraction, lazy loading,
    and optional unit conversion. It processes multiple outputs in sequence and attaches metadata
    to each output's data dictionary.
    
    Parameters:
    -----------
    wdir : str
        Working directory where simulation files are located.
    var_choice : list or str
        List (or single string) of variable names to load (e.g., ['rho', 'prs', 'ncx', 'ncy']).
    load_outputs : tuple, int, or "last", optional
        Which outputs to load:
        - tuple: Load specific outputs (e.g., (0, 5, 10))
        - int: Load outputs from 0 to load_outputs (inclusive)
        - "last": Load only the last available output
        - None: Load the last available output (default)
    load_slice : tuple, optional
        Array slice to load a subset of data (e.g., (slice(None, None, None), slice(None, None, None), 733)).
    ini_file : str, optional
        INI file name (without extension) to use for unit conversion e.g. jet_units.
    conv : bool, optional
        If True, convert data from code units to user units (default True).
        If False, return raw code unit data.
    
    Returns:
    --------
    dict
        Dictionary with structure pluto_data[output] containing:
        - {var_name}: Arrays for each requested variable (converted or raw)
        - metadata: HDF5Metadata dataclass with additional fields:
            - t_load: Time taken to load this output (seconds)
            - time_str: Formatted time string (e.g., "$1.5 \\; [Myr]$")
            - sim_time: Simulation time in user units
            - is_conv: Boolean indicating if data was converted
    
    """
    
    t_start = time.time()
    var_choice = [var_choice] if isinstance(var_choice,str) else var_choice
    pluto_data = defaultdict(list)  # Stores variables for each D_file
    warnings = []
    d_files = []
    tsteps = []
    
    n_outputs = get_file_outputs(wdir)
    warnings.append(f"{pcolours.WARNING}Outputs: {n_outputs}")
    if load_outputs == None:
        load_outputs = n_outputs
    elif load_outputs == "last":
        load_outputs = (get_file_outputs(wdir),)
    if isinstance(load_outputs, int) and load_outputs > n_outputs:
        raise ValueError(f"Trying to load more outputs ({load_outputs}) than available ({n_outputs})")
    
    for output in load_outputs:
        metadata = load_hdf5_metadata(wdir=wdir,load_output=output)
        time_val = pc.code_to_usr_units("sim_time",metadata.sim_time,ini_file="jet_units")["conv_data_uuv"]
        time_unit = str(pc.get_pluto_units("CARTESIAN",ini_file)["sim_time"]["usr_uv"])
        time_str = f"${time_val:.2f} \\; [{time_unit}]$"

        basename = (os.path.basename(metadata.file_path))
        d_files.append('.'.join(basename.split('.')[:2]))
        tsteps.append(time_str)

        if not conv:
            pluto_data[output] = load_hdf5_lazy(wdir=wdir,load_output=output,var_choice=var_choice,load_slice=load_slice)
            metadata.is_conv = False

        if conv:
            raw_data = load_hdf5_lazy(wdir=wdir,load_output=output,var_choice=var_choice,load_slice=load_slice)
            def convert_var(var_name):
                conv = pc.code_to_usr_units(
                    var_name=var_name,
                    raw_data=raw_data[var_name],
                    ini_file=ini_file
                )
                return var_name, conv["conv_data_uuv"]
            
            file_results = {}
            with ThreadPoolExecutor() as exe:
                futures = [exe.submit(convert_var, v) for v in var_choice]
                for f in as_completed(futures):
                    v, conv_arr = f.result()
                    file_results[v] = conv_arr
            pluto_data[output] = file_results
            metadata.is_conv = True
        
        metadata.t_load = (round(time.time()-t_start,2))
        metadata.time_str = time_str
        metadata.sim_time = time_val
        pluto_data[output]["metadata"] = metadata
    return pluto_data

# @lru_cache(maxsize=32)  # This caches based on input arguments
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

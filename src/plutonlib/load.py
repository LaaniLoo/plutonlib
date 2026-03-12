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
    dtype: str = "dbl.h5"
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
    dtype = dext

    if not is_dbl_h5 and not is_flt_h5:
        raise FileNotFoundError(f"output {load_output} not found in {wdir}") #quick error as metadata only works for hdf5

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

def pluto_loader_hdf5(wdir,var_choice, load_outputs=None,load_slice=None,ini_file=None,conv=True):
    """
    Loads and optionally converts units of multiple HDF5 simulation outputs with metadata.
        
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
    d_files = []
    tsteps = []
    
    n_outputs = get_file_outputs(wdir)
    if load_outputs == None or load_outputs == "last":
        load_outputs = (n_outputs,)

    if isinstance(load_outputs,int):
        load_outputs = tuple(i for i in range(1,load_outputs+1)) #create a tuple of outputs up to int e.g load all up to 10

    if isinstance(load_outputs, int) and load_outputs > n_outputs:
        raise ValueError(f"Trying to load more outputs ({load_outputs}) than available ({n_outputs})")
    
    if isinstance(load_outputs, tuple) and max(load_outputs) > n_outputs:
        raise ValueError(f"Trying to load more outputs ({max(load_outputs)}) than available ({n_outputs})")

    for output in load_outputs:
        pluto_units = pc.PlutoUnits.from_ini(ini_file=ini_file)
        metadata = load_hdf5_metadata(wdir=wdir,load_output=output)
        time_val = pc.code_to_usr_units("sim_time",metadata.sim_time,ini_file="jet_units")["conv_data_uuv"]
        time_unit = str(pluto_units.sim_time.usr_uv)
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

def get_particle_outputs(wdir,load_outputs=None):
    '''
    Gets the number of simulation particle file outputs
    
    :param wdir: working directory containing the particle files
    :param load_outputs: used to make a list of all particle file directories to load
    '''
    # part_files = []
    pattern = os.path.join(wdir, "particles.*.dbl")
    particle_paths = sorted(glob.glob(pattern), key=lambda f: int(f.split(".")[-2]))
    file_ext = particle_paths[0].split(".")[-1]
    n_outputs = int(particle_paths[-1].split(".")[-2])
    
    if load_outputs == None:
        return n_outputs

    if not particle_paths:
        raise FileNotFoundError(f"No files found that match `particles.` in {wdir}")

    if isinstance(load_outputs,int):
        load_outputs = tuple(i for i in range(1,load_outputs+1)) #create a tuple of outputs up to int e.g load all up to 10

    elif load_outputs == "last":
        load_outputs = (particle_paths[n_outputs],)

    max_output = max(load_outputs) if isinstance(load_outputs,tuple) else load_outputs
    if max_output > n_outputs:
        raise ValueError(f"Attempting to load output {max_output} when there are {n_outputs} outputs")

    loaded_files = [particle_paths[output_n] for output_n in load_outputs if output_n <= n_outputs]
    part_files = list(load_outputs)

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

def pluto_particles(wdir,load_outputs=None):
    particle_data = defaultdict(list)  # Stores variables for each particle file
    var_map = {"tracer":"tr1","density":"rho","pressure":"prs"}

    particle_outputs = get_particle_outputs(wdir,load_outputs)
    loaded_files = particle_outputs["loaded_files"]
    part_files = particle_outputs["part_files"]

    for file_idx,file_name in enumerate(loaded_files): #used to loop over file path and particle file string
        part_file = part_files[file_idx]

        hdict = read_particle_file(file_name)['hdict']
        data_str  = read_particle_file(file_name)['data_str']
        tot_fdim = read_particle_file(file_name)['tot_fdim']

        hdict["field_names"] = [var_map.get(v,v) for v in hdict["field_names"]]
        vars_ = hdict["field_names"]

        endianess = "<"
        if hdict["endianity"] == "big":
            endianess = ">"
        dtyp_ = np.dtype(endianess + "dbl"[0])
        DataDict_ = hdict
        n_particles = int(DataDict_["nparticles"])
        data_ = np.frombuffer(data_str, dtype=dtyp_)

        fdims = np.array(hdict["field_dim"], dtype=int)

        if n_particles <= 0 and isinstance(load_outputs,tuple) and len(load_outputs) == 1: 
            print(DataDict_)
            raise AttributeError(f"Particle file {file_name} has nparticles = 0")

        elif n_particles <= 0:
            # print(f"Particle file {file_name} has nparticles = 0") #debug part files with nparticles = 0
            continue

        reshaped_data = data_.reshape(n_particles, tot_fdim)
        tup_ = []
        ind = 0
        field_data = {}
        for c, v in enumerate(vars_):
            v_data = reshaped_data[:, ind : ind + fdims[c]]
            if fdims[c] == 1:
                v_data = v_data.reshape((n_particles))
            field_data[v] = v_data
            tup_.append((v, v_data))
            ind += fdims[c]
        DataDict_.update(dict(tup_))
        particle_data[part_file] = DataDict_

    return particle_data

def pluto_particles_hdf5(wdir,output = None):
    particle_data_path = os.path.join(wdir,"particles.hdf5")

    particle_dict = {}
    particle_times = None

    particle_data_file = h5py.File(particle_data_path, "r")
    # var_map = {"tracer":"tr1","density":"rho","pressure":"prs"}
    
    for k, v in particle_data_file["particle_data"].items():
        # k = var_map.get(k,k)
        particle_dict[k] = v[:,output] if output is not None else v
    particle_times = particle_data_file["time"]
    particle_dict["particle_times"] = particle_times

    return particle_dict
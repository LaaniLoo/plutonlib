from pathlib import Path

import plutonlib.utils as pu
# import plutonlib.plot as pp 
import plutonlib.config as pc
from plutonlib.colours import pcolours

# profiles = pc.profiles
coord_systems = pc.coord_systems
PLUTODIR = pc.plutodir

# Native plutokore support
# import plutokore.io as pk_io

# importing src files??
# from plutonlib.plutokore_src import plutokore_io as pk_io

import numpy as np
from astropy import units as u
from collections import defaultdict 
import h5py as h5py
from pathlib import Path 

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed,ProcessPoolExecutor
import inspect
import os
from functools import lru_cache

import glob

class SimulationSetup:
    """
    Class used to load and store any PLUTO output/input data, e.g. run_name names, save directories, simulation types, 
    converted/raw data, units and var info
    """

    _last_ini_file = None
    _last_arr_type = None

    def __init__(self, sim_type=None, run_name=None, profile_choice=None,subdir_name=None,
                 load_outputs=None,arr_type=None,ini_file=None,):

        self.sim_type = sim_type
        self.run_name = run_name

        self.load_outputs = load_outputs
        if isinstance(self.load_outputs, list):
            self.load_outputs = tuple(self.load_outputs)

        self.ini_file = ini_file

        self.arr_type = arr_type

        self.profile_choice = profile_choice 

        # Saving 
        self.subdir_name = subdir_name

        # self.alt_dir = os.path.join(pc.start_dir,subdir_name) if self.subdir_name else None #used if want to skip setup_dir
        self.alt_dir = None

        # self.save_dir = self._select_dir() #saves save_dir 
        self._save_dir = None

        # Files
        self.avail_sims = os.listdir(pc.sim_dir)
        self.avail_runs =  os.listdir(os.path.join(pc.sim_dir,self.sim_type)) if self.sim_type else None #print(f"{pcolours.WARNING}Skipping avail_runs")
        
        # Set wdir safely
        self.wdir = os.path.join(PLUTODIR, "Simulations", self.sim_type, self.run_name) if self.run_name else None
        # if self.wdir is None:
        #     print(f"{pcolours.WARNING}Skipping wdir because run_name is None")

        # Set start_dir only if wdir exists
        if self.wdir is not None:
            self.start_dir = pc.get_start_dir(self.wdir, run_name=self.run_name)["start_dir"]
        else:
            self.start_dir = None

        # Vars
        self._var_choice = None 
        SimulationSetup._last_arr_type = self.arr_type

    #---Other---#
    def _select_dir(self):
        """If no specified directory string (subdir_name) to join to start_dir -> run pc.setup_dir """

        if self.alt_dir is None: #not alt dir -> run setup
            return  pu.setup_dir(self.start_dir)
            # return pu.setup_dir(self.wdir)
        
        elif self.alt_dir: #alt dir is specified 
            if os.path.isdir(self.alt_dir): #valid dir -> assign 
                return self.alt_dir

            else: #isn't a valid dir -> run_name _create_dir()
                return self._create_dir()

    def _create_dir(self):
        new_dir = self.alt_dir
        sys.stdout.flush()
        print(f"{self.subdir_name} is not a valid folder in start_dir: Would you like to create the dir {new_dir}?")

        save = None 
        while save not in (0,1):
            try:
                save = int(input("Create directory? [1/0]"))
            except ValueError:
                print("Invalid input, please enter 1 (yes) or 0 (no).")  

        if save:
            print(f"Creating {new_dir}")
            os.makedirs(new_dir)
            return new_dir
        
        elif not save:
            print("Cancelling operation")
            raise(AttributeError(f"Please specify a directory in {self.start_dir}"))
        
    #---Properties---#
    @property
    def save_dir(self):
        if self._save_dir is None:
            self._save_dir = self._select_dir()
        return self._save_dir
    
    @property
    def coord_names(self):
        coord_names = []
        for var_name in self.var_choice:
            if var_name in ('x1','x2','x3'):
                coord_names.append(var_name)

        return coord_names

    @property
    def spare_coord(self):
        '''e.g. returns y if profile is xz etc...'''
        spare_coord = ({'x1','x2','x3'} - set(self.coord_names)).pop()
        return spare_coord

    @property
    def jet_info(self):
        ini_info = pc.pluto_ini_info(self.wdir)
        return ini_info["key_params"]

    @property
    def usr_params(self):
        ini_info = pc.pluto_ini_info(self.wdir)
        return ini_info["usr_params"]

    @property
    def grid_setup(self):
        ini_info = pc.pluto_ini_info(self.wdir)
        return ini_info["grid_setup"]

class SimulationData(SimulationSetup):
    """
    Class used to load and store any PLUTO output/input data, e.g. run_name names, save directories, simulation types, 
    converted/raw data, units and var info
    """
    def __init__(self, sim_type=None, run_name=None, profile_choice=None, subdir_name=None,
                 load_outputs=None, arr_type=None, ini_file=None, is_conv=1, setup=None):
        if setup is not None:
            # inherit defaults from setup
            sim_type = sim_type or setup.sim_type
            run_name = run_name or setup.run_name
            profile_choice = profile_choice or setup.profile_choice
            subdir_name = subdir_name or setup.subdir_name
            load_outputs = load_outputs or setup.load_outputs
            ini_file = ini_file or setup.ini_file
            arr_type = arr_type if arr_type is not None else setup.arr_type            


        # Initialize parent class first
        super().__init__(sim_type, run_name, profile_choice, subdir_name, load_outputs, arr_type, ini_file)
        # Data 
        self.is_conv = is_conv
        self._raw_data = None

        self._conv_data = None
        self._grid_data = None
        self._is_loaded = False #used to track loading state

        self.particle_data = None
        self.particle_files = None

        # Extra
        self._units = None
        self._geometry = None

        # Files
        self._d_files = None
        self._d_file = None
        
        self.load_time = None
        self.dir_log = None

    @classmethod
    def from_setup(cls, setup,**kwargs):
        """Create SimulationData from existing SimulationSetup"""
        return cls(setup=setup,**kwargs)

    #---Loading Data---#
    def load_all(self):
        if not self._is_loaded:  # Only load if not already loaded
            self.load_raw()
            self.load_conv()
            self.load_units()
            self._is_loaded = True
            print("Loaded all")
    
    def load_raw(self):
        start = time.time() #for load time
        if isinstance(self.load_outputs, list):
            raise TypeError(f"{pcolours.WARNING}Cannot use list with load_outputs, try tuple")
        
        self._raw_data = pluto_loader(self.sim_type,self.run_name,self.profile_choice,self.load_outputs,self.arr_type,self.ini_file)
        self._d_files = self._raw_data['d_files']
        self._var_choice = self._raw_data['var_choice']
        self._geometry = self._raw_data['vars_extra'][0]
        print(f"Pluto Loader: {(time.time() - start):.2f}s")

    def load_conv(self,profile=None):
        if self.is_conv:
            if self._raw_data is None: #NOTE not sure if this is req 
                self.load_raw()
            
            start = time.time() #for load time

            profile = profile or self.profile_choice
            pluto_conv_data =pluto_conv(self.sim_type, self.run_name,profile,self.load_outputs,self.arr_type,self.ini_file)

            if profile == "grid": #failsafe to load all data if req
                self._grid_data = pluto_conv_data
            else:
                self._conv_data = pluto_conv_data
            
            print(f"Pluto Conv ({profile}): {(time.time() - start):.2f}s")
        else:
            print(f"{pcolours.WARNING}is_conv = {self.is_conv}, Skipping convert")

    def load_particles(self,load_outputs):
        self.particle_data = pluto_particles(self.sim_type,self.run_name,load_outputs)['particle_data']
        self.particle_files = list(self.particle_data.keys())

    def load_units(self):
        if self._conv_data is None:
            self.load_conv()
        
        self._units = pc.get_pluto_units(self._geometry,self.ini_file) #,self._d_files
    
    def change_arr_type(self, new_arr_type = None):
        print(f"{pcolours.WARNING}array type is set to '{pc.arr_type_key[self.arr_type]}', changing it to '{pc.arr_type_key[new_arr_type]}'")

        self.arr_type = new_arr_type
        # print(f"Changing array type to {pc.arr_type_key[self.arr_type]}")
        self.del_cache

    def reload_outputs(self,load_outputs):
        self.load_outputs = load_outputs
        return self.reload
    
    #---Accessing SimulationData---#
    def check_d_file(self,d_file):
        if d_file not in self.d_files:
            raise ValueError(f"{pcolours.WARNING}Output {d_file} has not been loaded, current files are {self.d_files}")
        
    def get_vars(self,d_file=None,system = 'uuv'): #NOTE d_file was None not sure about that
        """Loads only arrays specified by vars in profile_choice"""
        target_file = d_file or self.d_last
        self.check_d_file(target_file)

        if system == 'uuv':
            return self.conv_data['vars_uuv'][target_file]
        else:
            raise ValueError("system must be in user or code unit values: 'uuv' or 'cuv'")

    def get_all_vars(self,d_file=None,system = "uuv"):
        """Loads all available arrays"""
        target_file = d_file or self.d_last
        self.check_d_file(target_file)
        # print(target_file) #debug above

        if system == 'uuv':
            d_file_info = {"d_file" : target_file}
            self.grid_data['vars_uuv'].update(d_file_info)
            return self.grid_data['vars_uuv'][target_file]

        else:
            raise ValueError("system must be in user or code unit values: 'uuv' or 'cuv'")
    
    def get_coords(self,d_file=None):
        """Just gets the x,y,z arrays as needed"""
        target_file = d_file or self.d_last
        self.check_d_file(target_file)

        conv_data = self.grid_data['vars_uuv'][target_file]

        coords = {
            "x1": conv_data["x1"],
            "x2": conv_data["x2"],
            "x3": conv_data["x3"]
        }

        return coords

    def get_var_info(self,var_name):
        """Gets coordinate name, unit, norm value etc"""
        var_info = self.units.get(var_name)
        # shp_info = {"shape" : self.get_all_vars()[var_name].shape} #NOTE was get_all_vars
        # dim_info = {"ndim" : self.get_all_vars()[var_name].ndim}
        shp_info = {"shape" : self.get_vars()[var_name].shape}
        dim_info = {"ndim" : self.get_vars()[var_name].ndim}
        var_info.update(shp_info)
        var_info.update(dim_info)

        if not var_info:
            raise KeyError(f"No unit info for variable {var_name}")
        

        return var_info
    
    def d_sel(self,slice,start = 0):
        """Slices d_files to the number specified -> e.g. give me first 3 elements of d_files"""
        return self.d_files[start:slice]
    

    #---Properties---#
    @property
    def get_warnings(self):
        """Prints any warnings from loading process"""
        #---General Warnings---#
        # print(f"{pcolours.WARNING}WARNING: run is now run_name") 

        #---self and file related warnings---#
        warn_sim = f"please select an available simulation type from \n{self.avail_sims}"
        warn_run = f"please select an available {self.sim_type} simulation from \n{self.avail_runs}"
        is_wdir = os.path.isdir(self.wdir)

        if not self.sim_type:
            raise ValueError(f"{pcolours.WARNING}Invalid sim_type, {warn_sim}")

        if not self.run_name:
            raise ValueError(f"{pcolours.WARNING}Invalid run_name, {warn_run}")
        
        if not is_wdir:
            raise ValueError(f"{pcolours.WARNING}{self.wdir} doesn't contain the run {self.run_name}, {warn_run}")

        if self.alt_dir:
            dir_log = f"Final selected save directory: {self.save_dir}"
            print(dir_log)

        #--directory warnings--# 
        print(f"{pcolours.WARNING}---SimulationData Info---")
        print("\n")
        print(f"{pcolours.WARNING}Current Working Directory:", self.wdir)
        print("Save Directory:",pc.get_start_dir(self.wdir,run_name = self.run_name)["warnings"][0])
        print(f"Units file: {pc.get_ini_file(self.ini_file)}") # Prints current units.ini file
        print("\n")
        print(f"{pcolours.WARNING}Array Type: {pc.arr_type_key[self.arr_type]}")

        #get loading warnings from pluto_loader 
        warnings = self.conv_data['warnings']
        for warning in warnings:
            print(warning)
        print(pcolours.ENDC) #ends yellow warning colour 

    @property
    def raw_data(self):
        if self._raw_data is None:
            self.load_raw()
        return self._raw_data

    @property
    def conv_data(self):
        if self._conv_data is None:
            self.load_conv()
        return self._conv_data

    @property
    def grid_data(self):
        if self._grid_data is None:
            
            self.load_conv(profile="grid")  # Loads all profile
        return self._grid_data
    
    @property
    def units(self):
        if self._units is None:
            self.load_units()
        return self._units

    @property
    def geometry(self):
        if self._geometry is None:
            self.load_raw()
            # raise ValueError("Missing geometry data")
        return self._geometry

    @property
    def d_files(self):
        if self._d_files is None:
            self.load_raw()
            # raise ValueError("Missing d_files data")
        return self._d_files
    
    @property
    def d_last(self):
        return self.d_files[-1]

    @property
    def var_choice(self):
        if self._var_choice is None:
            self.load_raw()
            # raise ValueError("Missing var_choice data")

        return self._var_choice
    
    @property
    def grid_ndim(self):
        if self._units is None:
            self.load_units()

        if 'rho' in self.var_choice:    
            grid_ndim = self.get_var_info("rho")["ndim"]

        elif self.arr_type in ('nc','cc'):
            grid_ndim = self.get_var_info("x1")["ndim"]
        
        else:
            raise ValueError("Cannot determine grid dimensions without nc/cc arrays or rho")

        return grid_ndim
    @property    
    def reload(self):
        """Force reload all data"""
        pluto_loader.cache_clear()
        pluto_conv.cache_clear()
        self._raw_data = None
        self._conv_data = None
        self._units = None
        self.load_raw()
    
        return self
    
#------------------------#
#       functions    
#------------------------#

#---Loading Files---#
def get_file_outputs(wdir):
    """
    Gets number of simulation file outputs
    """
    is_dbl_h5 = os.path.isfile(os.path.join(wdir,r"dbl.h5.out"))
    is_flt_h5 = os.path.isfile(os.path.join(wdir,r"flt.h5.out"))
    is_dbl = os.path.isfile(os.path.join(wdir,r"dbl.out"))

    if is_dbl_h5 or  is_flt_h5:
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

def set_hdf5_grid_info(grid_object):
    """
    Taken from plutokore
    """
    # # Set the 1D cell midpoint cooordinate arrays
    # setattr(grid_object, "mx", grid_object.ccx[0, 0, :])
    # setattr(grid_object, "my", grid_object.ccy[0, :, 0])
    # setattr(grid_object, "mz", grid_object.ccz[:, 0, 0])

    # # Set the 1D cell edge coordinate arrays
    # setattr(grid_object, "ex", grid_object.ncx[0, 0, :])
    # setattr(grid_object, "ey", grid_object.ncy[0, :, 0])
    # setattr(grid_object, "ez", grid_object.ncz[:, 0, 0])

    # Set the 1D cell midpoint cooordinate arrays
    setattr(grid_object, "mx", grid_object.ccx[:, 0, 0])
    setattr(grid_object, "my", grid_object.ccy[0, :, 0])
    setattr(grid_object, "mz", grid_object.ccz[0, 0, :])

    # Set the 1D cell edge coordinate arrays
    setattr(grid_object, "ex", grid_object.ncx[:, 0, 0])
    setattr(grid_object, "ey", grid_object.ncy[0, :, 0])
    setattr(grid_object, "ez", grid_object.ncz[0, 0, :])


    setattr(grid_object, "dx", np.diff(grid_object.ex))
    setattr(grid_object, "dy", np.diff(grid_object.ey))
    setattr(grid_object, "dz", np.diff(grid_object.ez))

    setattr(grid_object, "dx1", grid_object.dx)
    setattr(grid_object, "dx2", grid_object.dy)
    setattr(grid_object, "dx3", grid_object.dz)

    # Set midpoint index attributes
    setattr(grid_object, "mid_x", grid_object.mx.shape[0] // 2 - 1)
    setattr(grid_object, "mid_y", grid_object.my.shape[0] // 2 - 1)
    setattr(grid_object, "mid_z", grid_object.mz.shape[0] // 2 - 1)

def load_file_output(wdir,load_output,var_choice,arr_type=None):
    is_dbl_h5 = os.path.isfile(os.path.join(wdir,r"dbl.h5.out"))
    is_flt_h5 = os.path.isfile(os.path.join(wdir,r"flt.h5.out"))
    is_dbl = os.path.isfile(os.path.join(wdir,r"dbl.out"))
    dtype = []

    if is_dbl_h5 or  is_flt_h5:
        # start = time.time()
        dext = "dbl.h5" if is_dbl_h5 else "flt.h5" #assigns correct dtype for loading
        dtype.append("hdf5_double" if is_dbl_h5 else "hdf5_float")
        data_file_path = wdir / Path(f"data.{load_output:04d}.{dext}")

        data_file = h5py.File(
            data_file_path, 
            mode="r",
            rdcc_nbytes=256 * 1024 * 1024,  # 256 MB
            rdcc_nslots=1_000_003,          # prime number of slots
            rdcc_w0=0.75                    # cache eviction aggressiveness
            )

        # print(data_file.attrs.keys()) #debug attributes
        # print(list(data_file.keys()))  # top-level groups
        # print(list(data_file["Timestep_0"]['vars'].keys()))  # groups inside timestep
        # print(data_file["Timestep_0"].attrs.keys())  # attributes of that timestep

        setattr(data_file, "sim_time", data_file[f"Timestep_{load_output}"].attrs["Time"])
        setattr(data_file, "variable_path", f"Timestep_{load_output}/vars")
        setattr(data_file, "geometry", "CARTESIAN")
        
        # Set the variables
        variables = list(data_file[data_file.variable_path])
        for v in variables:
            setattr(data_file, v, data_file[f"{data_file.variable_path}/{v}"])
        setattr(data_file, "vars", variables)

        # Set the coords
        coords = ["X", "Y", "Z"]
        for c in coords:
            setattr(data_file, f"cc{c.lower()}", data_file[f"/cell_coords/{c}"])
            setattr(data_file, f"nc{c.lower()}", data_file[f"/node_coords/{c}"])

        set_hdf5_grid_info(data_file)

        # Set the tracer count
        setattr(data_file, "ntracers", len([t for t in variables if "tr" in t]))

        #LOADING 
        geometry = data_file.geometry

        prof_vars = pc.profiles(arr_type = arr_type)["profiles"]["all"]
        var_map = {'x1': prof_vars[0], 'x2': prof_vars[1], 'x3': prof_vars[2]} # used to map any arr_type to x1,x2,x3
        loaded_vars = [v for v in var_choice if hasattr(data_file, var_map.get(v, v))]

        file_data = {v: getattr(data_file, var_map.get(v, v)) for v in loaded_vars}

        # TODO TEMP FIX FOR LOADING ARR CORRECTLY 
        for var_name in file_data.keys():
            #Loads the hdf5 dataset into easily readable 3d array

            #NOTE this transposes z,y,x arrays into x,y,z arrays
            if var_name in ("x1", "x2", "x3",'rho','prs','vx1','vx2','vx3','tr1') and file_data[var_name].ndim == 3:
                    file_data[var_name] = np.transpose(file_data[var_name],(2,1,0))
            
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

def pluto_particles(sim_type,run_name,load_outputs=None):
    wdir =  os.path.join(PLUTODIR, "Simulations", sim_type, run_name)
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
        data_ = np.fromstring(data_str, dtype=dtyp_)

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
def pluto_loader(sim_type, run_name, profile_choice, load_outputs=None, arr_type=None,ini_file=None):
    """
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

    Returns:
    --------
    dict
        Dictionary containing:
        - vars: dictionary of order vars[d_file][var_name] e.g. vars["data_0"]["x1"]
        - var_choice: List of variable names corresponding to the selected profile.
        - vars_extra: contains the geometry of the sim
        - d_files: contains a list of the available data files for the sim
    """
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
        loaded_file = load_file_output(wdir=wdir, load_output=output_n, var_choice=var_choice, arr_type=arr_type)
        # sim_time = loaded_file["file_data"]["sim_time"]

        sim_time = load_file_output(wdir=wdir, load_output=output_n, var_choice=['sim_time'], arr_type=arr_type)['file_data']['sim_time']
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

            #update d_files with sim_time #TODO make sure unit is relevant 
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

    var_choice = [v for v in var_choice if v not in non_vars]  # reassigning var_choice with avail vars

    return {"vars": vars, "var_choice": var_choice, "vars_extra": vars_extra, "d_files": d_files, "warnings": warnings}

@lru_cache(maxsize=32)  # This caches based on input arguments
def pluto_conv(sim_type, run_name, profile_choice, load_outputs=None, arr_type=None, ini_file=None):
    """
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

    Returns:
    --------
    dict
        Dictionary containing:
        - vars_uuv: dictionary of order vars[d_file][var_name] e.g. vars["data_0"]["x1"]
        - var_choice: List of variable names corresponding to the selected profile.
        - d_files: contains a list of the available data files for the sim
    """
    start1 = time.time()
    loaded_data = pluto_loader(sim_type, run_name, profile_choice, load_outputs, arr_type,ini_file)
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

            # if var_name == "sim_time":
            #     # adds both time in years and seconds as keys, sim_time defaults to yr
            #     file_results[var_name] = conv_vals["cgs"]
            # else:
            #     file_results[var_name] = conv_vals["si"]

        return d_file, file_results

    # Use ThreadPoolExecutor for parallel processing
    start3 = time.time()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, d_file) for d_file in d_files]
        for future in as_completed(futures):
            d_file, file_results = future.result()
            vars_uuv[d_file].update(file_results)


    return {"vars_uuv": vars_uuv, "var_choice": var_choice, "d_files": d_files, "sim_coord": sim_coord, "warnings": warnings}


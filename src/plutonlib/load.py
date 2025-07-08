from pathlib import Path

import plutonlib.utils as pu
import plutonlib.plot as pp 
import plutonlib.config as pc
from plutonlib.colours import pcolours


profiles = pc.profiles
coord_systems = pc.coord_systems
PLUTODIR = pc.plutodir

# Native plutokore support
# import plutokore.io as pk_io

# importing src files??
from plutonlib.plutokore_src import plutokore_io as pk_io

#for dbl:
# from plutokore.simulations import get_output_count as pk_sim_count

#for hdf5:
# from plutokore.simulations import get_hdf5_output_count as pk_sim_count_h5
# from pk_sim import get_output_count as pk_sim_count




import numpy as np
from astropy import units as u
from collections import defaultdict 
import h5py as h5py
from pathlib import Path 

import sys
import time
from concurrent.futures import ThreadPoolExecutor
import inspect
import os
from functools import lru_cache


class SimulationData:
    """
    Class used to load and store any PLUTO output/input data, e.g. run_name names, save directories, simulation types, 
    converted/raw data, units and var info
    """
    def __init__(self, sim_type=None, run_name=None, profile_choice=None,subdir_name=None,auto_load=False,
                 load_outputs=None,arr_type=None,ini_file=None):
        self.sim_type = sim_type
        self.run_name = run_name
        self.load_outputs = load_outputs
        self.arr_type = arr_type
        self.ini_file = ini_file

        #Safety measure to ensure correct array is loaded
        if self.arr_type is None:
            arr_type_key = (
                "\nKey:\n"
                "  e[x,y,z]   - 1D cell edge coordinate arrays\n"
                "  m[x,y,z]   - 1D cell midpoint coordinate arrays\n"
                "  d[x,y,z]   - 1D cell delta arrays\n"
                # "  x[1,2,3]   - 1D logical coordinate arrays\n"
                "  nc[x,y,z]  - 3D cell edge arrays\n"
                "  cc[x,y,z]  - 3D cell midpoint arrays"
            )
            raise ValueError(
                f"{pcolours.WARNING}arr_type is None, please select a coord array of type:"
                f"\n{pc.profiles2()['coord_prefixes']}"
                f"\n {arr_type_key}"
                )

        self.profile_choice = profile_choice or self._select_profile() # arg or function to select

        # Saving 
        self.subdir_name = subdir_name
        self.alt_dir = os.path.join(pc.start_dir,subdir_name) if self.subdir_name else None #used if want to skip setup_dir
        # self.save_dir = self._select_dir() #saves save_dir 
        self._save_dir = None

        # Data 
        self._raw_data = None
        self._conv_data = None
        self._all_conv_data = None
        self._is_loaded = False #used to track loading state

        # Extra
        self._units = None
        self._geometry = None

        # Files
        self._d_files = None
        self._d_file = None
        self.avail_sims = os.listdir(pc.sim_dir)
        self.avail_runs =  os.listdir(os.path.join(pc.sim_dir,self.sim_type)) if self.sim_type else print(f"{pcolours.WARNING}Skipping avail_runs")
        self.wdir =  os.path.join(PLUTODIR, "Simulations", self.sim_type, self.run_name) if self.run_name else print(f"{pcolours.WARNING}Skipping wdir")
        # self.ini_path = os.path.join(pc.src_path,ini_name)

        # Vars
        self._var_choice = None 
        self.coord_names = ["x1","x2","x3"] #for convenience 
        # self._coords

        # Metadata
        #Print warnings only when assigning sdata?
        called_func = inspect.stack()[1].function
        if called_func == "<module>":
            self.get_warnings() 

        self.load_time = None
        # self.__dict__.update(kwargs)
        self.dir_log = None

        if auto_load:
            self.load_all()
            self._loaded = True

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
        
        self._raw_data = pluto_loader(self.sim_type,self.run_name,self.profile_choice,self.load_outputs,self.arr_type)
        self._d_files = self._raw_data['d_files']
        self._var_choice = self._raw_data['var_choice']
        self._geometry = self._raw_data['vars_extra'][0]
        # self.load_time = time.time() - start
        print(f"Pluto Loader: {(time.time() - start):.2f}s")

    def load_conv(self,profile=None):
        # if self._raw_data is None:
        #     self.load_raw()
        
        start = time.time() #for load time

        profile = profile or self.profile_choice
        loaded_data =pluto_conv(self.sim_type, self.run_name,profile,self.load_outputs,self.arr_type,self.ini_file)

        self._conv_data = loaded_data

        if profile == "all": #failsafe to load all data if req
            self._all_conv_data = loaded_data
        
        print(f"Pluto Conv: {(time.time() - start):.2f}s")

    def load_units(self):
        if self._conv_data is None:
            self.load_conv()
                
        self._units = pc.get_pluto_units(self._geometry,self._d_files)
    
    #---Accessing SimulationData---#
    def get_vars(self,d_file=None,system = 'si'): #NOTE d_file was None not sure about that
        """Loads only arrays specified by vars in profile_choice"""
        target_file = d_file or self.d_last
        # print(target_file) #debug above

        if system == 'si':
            return self.conv_data['vars_si'][target_file]
        else:
            raise ValueError("system must be 'si' or 'cgs'")
    # @property
    def get_all_vars(self,d_file=None,system = "si"):
        """Loads all available arrays"""
        target_file = d_file or self.d_last
        # print(target_file) #debug above

        if system == 'si':
            return self.all_conv_data['vars_si'][target_file]
        else:
            raise ValueError("system must be 'si' or 'cgs'")
    
    def get_coords(self,d_file=None):
        """Just gets the x,y,z arrays as needed"""
        target_file = d_file or self.d_last

        conv_data = self.all_conv_data['vars_si'][target_file]

        coords = {
            "x1": conv_data["x1"],
            "x2": conv_data["x2"],
            "x3": conv_data["x3"]
        }

        return coords

    def get_var_info(self,var_name):
        """Gets coordinate name, unit, norm value etc"""
        var_info = self.units.get(var_name)
        shp_info = {"shape" : self.get_all_vars()[var_name].shape}
        dim_info = {"ndim" : self.get_all_vars()[var_name].ndim}
        var_info.update(shp_info)
        var_info.update(dim_info)

        if not var_info:
            raise KeyError(f"No unit info for variable {var_name}")
        

        return var_info

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

        #--Other--# 
        print(f"{pcolours.WARNING}---SimulationData Info---","\n")
        warnings = self.conv_data['warnings']
        for warning in warnings:
            print(warning)
        print("\n",f"{pcolours.WARNING}Current Working Dir:", self.wdir)
        print("\n",f"Units file: {pc.get_ini_file(self.ini_file)}") # Prints current units.ini file
        print(pcolours.ENDC) #ends yellow warning colour 

    def d_sel(self,slice,start = 0):
        """Slices d_files to the number specified -> e.g. give me first 3 elements of d_files"""
        return self.d_files[start:slice]
    
    #---Other---#
    def reload(self):
        """Force reload all data"""
        self._raw_data = None
        self._conv_data = None
        self._units = None
        self.load_all()
        return self
    
    def _select_profile(self):
        """if None is used as a profile choice, will show available profiles etc..."""
        if self.run_name is None:
            raise ValueError("run_name and profile_choice are None, IMPLEMENT RUN_NAMES FROM p_l_f")
            
        print("profile_choice is None, using pluto_load_profile to select profile")
        run_data = pluto_load_profile(self.sim_type,self.run_name,None)
        return run_data['profile_choices'][self.run_name][0] #loads the run_name names and selected profiles for runs

    def _select_dir(self):
        """If no specified directory string (subdir_name) to join to start_dir -> run pc.setup_dir """

        if self.alt_dir is None: #not alt dir -> run setup
            return  pu.setup_dir(pc.start_dir)
        
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
            raise(AttributeError(f"Please specify a directory in {pc.start_dir}"))
        
    #---Properties---#
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
    def all_conv_data(self):
        if self._all_conv_data is None:
            self.load_conv(profile="all")  # Loads all profile
        return self._all_conv_data
    
    @property
    def save_dir(self):
        if self._save_dir is None:
            self._save_dir = self._select_dir()
        return self._save_dir

    @property
    def units(self):
        if self._units is None:
            self.load_units()
        return self._units

    @property
    def geometry(self):
        if self._geometry is None:
            self.load_raw()
        return self._geometry

    @property
    def d_files(self):
        if self._d_files is None:
            self.load_raw()
        return self._d_files
    
    @property
    def d_last(self):
        return self.d_files[-1]

    @property
    def var_choice(self):
        if self._var_choice is None:
            self.load_raw()
        return self._var_choice
    
    @property
    def grid_ndim(self):
        if self._units is None:
            self.load_units()
        return self.get_var_info("rho")["ndim"]

    @property    
    def del_cache (self): 
        print("Deleting Cache...")
        pluto_loader.cache_clear()
        pluto_conv.cache_clear()

#------------------------#
#       functions    
#------------------------#

#---Profile Loading---#
def get_profiles(sim_type,run_name,profiles):
    """
    Prints available profiles for a specific simulation
    """
    if isinstance(run_name, list):
        run_name = run_name[0]  # Force unwrap if somehow a list gets through

    data = pluto_loader(sim_type,run_name,"all",load_outputs=(0,),arr_type="m") #NOTE pl should be faster than pc #NOTE loads 0th output for speed?
    var_choice = data["var_choice"]
    vars = data["vars"]["data_0"]

    for var in var_choice[:-1]: # doesn't include sim_time as it has no size
        if vars[var].size == 1:
            avail_vars = var_choice
            avail_vars.remove(var) # removes e.g. x3 in "Jet" if its only len 1 so x3 profiles aren't included

    else:
        avail_vars = var_choice

    keys = list(profiles.keys())

    print("Available profiles:")
    for i, prof in enumerate(keys):
        vars_set = set(avail_vars)
        prof_set = set(profiles[prof])
        common = vars_set & prof_set

        if len(common) >=4: #since the profiles are usually 4 elements make sure at least 4 match
            print(f"{i}: {prof}, {profiles[prof]}")
            sys.stdout.flush()

    return keys

def select_profile(sim_type,run_name,profiles):
    """Uses user input to select a profile"""
    keys = get_profiles(sim_type,run_name,profiles)

    while True:
        choice = input("Enter the number of the profile you want to select (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            print("Selection cancelled.")
            return None  # Return None if the user quits

        if choice.isdigit():
            choice = int(choice)

            if 0 <= int(choice) < len(profiles):
                return keys[choice]
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(profiles) - 1}.")
        else:
            print("Invalid input. Please enter a valid number or 'q' to quit.")

# def pluto_load_profile_list(sim_type,sel_runs,sel_prof,all = 0):
#     """
#     Uses get_profiles and select_profiles to store the selected profile/s across run_name/s in the profile_choices variable
#     * Use if need to gather a dict of runs each with a specific profile
#     """

#     # sel_runs = [sel_runs] if sel_runs and not isinstance(sel_runs,list) else sel_runs
#     sel = 0 if sel_runs is None else 1

#     #TODO could be in config to load following save tree?
#     run_dirs = os.path.join(PLUTODIR, "Simulations", sim_type)
#     all_runs = [
#         d for d in os.listdir(run_dirs) if os.path.isdir(os.path.join(run_dirs, d))
#     ]

#     # used to selected if plotting all subdirs or select runs
#     if sel == 0:
#         run_names = all_runs
#         print(f"Subdirectories:, {run_names}")

#     elif sel == 1:
#         run_names = sel_runs

#     # Assign a profile number for each run_name (supports duplicates)
#     profile_choices = defaultdict(list)  # Change from a single variable to defaultdict(list)

#     if sel_prof is None: #use if usr input multiple profiles, diff per run_name
#         if not all:
#             for run_name in run_names:
#                 profile_choice = select_profile(sim_type,run_name,profiles)
#                 profile_choices[run_name].append(profile_choice)  # Appends profile to list
#                 print(f"Selected profile {profile_choice} for run_name {run_name}: {profiles[profile_choice]}")

#         elif all:
#                 profile_choice = "all"
#                 print(f"Selected profile {profiles[profile_choice]} for all runs")
#                 print("\n")
    
#     if sel_prof is not None: #use if want one profile across all runs
#         for run_name in run_names:
#             get_profiles(sim_type,run_name,profiles)
#             profile_choice = sel_prof
#             profile_choices[run_name].append(profile_choice)  # Appends profile to list

#             try:
#                 print(f"Selected profile {profile_choice} for run_name {run_name}: {profiles[profile_choice]}")
#                 print("\n")

#             except KeyError:
#                 raise KeyError(f"{profile_choice} is not an available profile")



#     return {'run_names':run_names,'profile_choices':profile_choices}

def pluto_load_profile(sim_type, run_name, sel_prof):
    """
    Single-run version that:
    - Uses get_profiles() exactly like original
    - Prints available/selected profiles like original
    - Works ONLY with single run_name (string)
    """
    # Input validation
    if not isinstance(run_name, str):
        raise TypeError("run_name must be a single string (no lists/None)")
    if sel_prof is None:
        raise ValueError("sel_prof cannot be None in single-run mode")

    # Verify run exists (original didn't have this but it's good practice)
    run_dir = os.path.join(PLUTODIR, "Simulations", sim_type, run_name)
    if not os.path.isdir(run_dir):
        raise ValueError(f"Run directory not found: {run_dir}")

    # Get available profiles (using original get_profiles())
    get_profiles(sim_type, run_name, profiles)

    # Handle profile selection (simplified from original)
    profile_choices = defaultdict(list)
    
    try:
        profile_choices[run_name].append(sel_prof)
        print(f"Selected profile {sel_prof} for run {run_name}: {profiles[sel_prof]}")
        print("\n")
    except KeyError:
        raise KeyError(f"{sel_prof} is not an available profile")

    return {
        'run_names': [run_name],  # Still a list for compatibility
        'profile_choices': profile_choices
    }

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
    # Set the 1D cell midpoint cooordinate arrays
    setattr(grid_object, "mx", grid_object.ccx[0, 0, :])
    setattr(grid_object, "my", grid_object.ccy[0, :, 0])
    setattr(grid_object, "mz", grid_object.ccz[:, 0, 0])

    # Set the 1D cell edge coordinate arrays
    setattr(grid_object, "ex", grid_object.ncx[0, 0, :])
    setattr(grid_object, "ey", grid_object.ncy[0, :, 0])
    setattr(grid_object, "ez", grid_object.ncz[:, 0, 0])

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
        dext = "dbl.h5" if is_dbl_h5 else "flt.h5" #assigns correct dtype for loading
        dtype.append("hdf5_double" if is_dbl_h5 else "hdf5_float")
        data_file_path = wdir / Path(f"data.{load_output:04d}.{dext}")
        data_file = h5py.File(data_file_path, mode="r")

        setattr(data_file, "sim_time", data_file[f"Timestep_{load_output}"].attrs["Time"])
        setattr(data_file, "variable_path", f"Timestep_{load_output}/vars")
        setattr(data_file, "geometry", "CARTESIAN")
        
        # TODO WORRY ABOUT UNIT UNITS????

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

        prof_vars = pc.profiles2(arr_type = arr_type)["profiles"]["all"]
        var_map = {'x1': prof_vars[0], 'x2': prof_vars[1], 'x3': prof_vars[2]} # used to map any arr_type to x1,x2,x3
        loaded_vars = [v for v in var_choice if hasattr(data_file, var_map.get(v, v))]

        file_data = {v: getattr(data_file, var_map.get(v, v)) for v in loaded_vars}

        # TODO TEMP FIX FOR LOADING ARR CORRECTLY 
        for var_name in file_data.keys():
            #Loads the hdf5 dataset into easily readable 3d array
            if var_name in ("x1", "x2", "x3",'rho','prs','vx1','vx2','vx3') and file_data[var_name].ndim == 3:
                    file_data[var_name] = file_data[var_name][slice(None),slice(None),slice(None)]

            #Loads the hdf5 dataset into easily readable 2d array
            elif var_name in ("x1", "x2", "x3",'rho','prs','vx1','vx2','vx3') and file_data[var_name].ndim == 2:
                    file_data[var_name] = file_data[var_name][slice(None),slice(None)]

                    

    elif is_dbl: #should be deprecated?
        out_fname = "dbl.out"
        dtype.append("double")

        data_file = pk_io.pload(load_output, w_dir=wdir)

        #only want to do this calculation once
        geometry = data_file.geometry
        loaded_vars = [v for v in var_choice if hasattr(data_file, v)]

        file_data = {v: getattr(data_file, v) for v in loaded_vars}

    else:
        raise FileNotFoundError("Either .out is missing or file is not of type [flt.h5,dbl.h5,.dbl]")
    
    returns = {"file_data":file_data,"loaded_vars":loaded_vars,"geometry":geometry,"dtype":dtype}

    return returns

@lru_cache(maxsize=None)  # This caches based on input arguments
def pluto_loader(sim_type, run_name, profile_choice,load_outputs=None,arr_type=None):
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
    vars = defaultdict(list) # Stores variables for each D_file
    vars_extra = []
    warnings = []
    var_choice = profiles[profile_choice]
    wdir = SimulationData(sim_type, run_name, profile_choice,load_outputs=load_outputs,arr_type=arr_type).wdir

    # n_outputs = pk_io.nlast_info(w_dir=wdir)["nlast"] #NOTE uses pk_io instead of simulations
    n_outputs = get_file_outputs(wdir)
    warnings.append(f"{pcolours.WARNING}Found {n_outputs} output files")

    if load_outputs == None:
        load_outputs = n_outputs

    if isinstance(load_outputs,int) and load_outputs > n_outputs:
        raise ValueError(f"Trying to load more outputs ({load_outputs}) than available ({n_outputs})")

    # Assigning the number of d_files
    if isinstance(load_outputs, tuple):
        d_files = [f"data_{i}" for i in load_outputs if i <= n_outputs]
        load_outputs = tuple(load_outputs)

    elif load_outputs is not None:  # Original behavior for integer
        d_files = [f"data_{i}" for i in range(min(load_outputs, n_outputs) + 1)]

    else:  # Load all
        d_files = [f"data_{i}" for i in range(n_outputs + 1)]

    #load a single output
    if isinstance(load_outputs,int):
        ocount = 0
        for output_n in range(load_outputs + 1):
            ocount +=1
            loaded_file = load_file_output(wdir=wdir,load_output=output_n,var_choice=var_choice,arr_type=arr_type)
            file_data = loaded_file["file_data"]
            vars[f"data_{output_n}"] = file_data
            warnings.append(f"{pcolours.WARNING}Data Type = {loaded_file['dtype'][0]}") if ocount == 1 else None
            warnings.append(f"{pcolours.WARNING}loaded data_{output_n}") #DEBUG

        #Checking and loading correct vars
        loaded_vars = loaded_file["loaded_vars"]
        vars_extra.append(loaded_file["geometry"]) # gets the geo of the sim, always loads first file
        non_vars = set(var_choice) - set(loaded_vars)
        if non_vars:
            warnings.append(f"{pcolours.WARNING}Simulation {run_name} doesn't contain: {', '.join(non_vars)}")
    
    #load multiple outputs
    elif isinstance(load_outputs,tuple):
        ocount = 0
        for output_n in load_outputs:
            ocount +=1
            loaded_file = load_file_output(wdir=wdir,load_output=output_n,var_choice=var_choice,arr_type=arr_type)
            file_data = loaded_file["file_data"]
            vars[f"data_{output_n}"] = file_data
            warnings.append(f"{pcolours.WARNING}Data Type = {loaded_file['dtype'][0]}") if ocount == 1 else None
            warnings.append(f"{pcolours.WARNING}loaded data_{output_n}") #DEBUGG

        #Checking and loading correct vars
        loaded_vars = loaded_file["loaded_vars"]
        vars_extra.append(loaded_file["geometry"]) # gets the geo of the sim, always loads first file
        non_vars = set(var_choice) - set(loaded_vars)
        if non_vars:
            warnings.append(f"{pcolours.WARNING}Simulation {run_name} doesn't contain: {', '.join(non_vars)}")
    
    var_choice = [v for v in var_choice if v not in non_vars] # reassigning var_choice with avail vars

    return {"vars": vars, "var_choice": var_choice, "vars_extra": vars_extra, "d_files": d_files, "warnings": warnings} #"nlinf": nlinf

@lru_cache(maxsize=None)  # This caches based on input arguments
def pluto_conv(sim_type, run_name, profile_choice,load_outputs=None,arr_type=None,ini_file=None,**kwargs):
    """
    Converts Pluto simulation variables from code units to CGS and SI units.

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
        - vars_si: dictionary of order vars[d_file][var_name] e.g. vars["data_0"]["x1"]
        - var_choice: List of variable names corresponding to the selected profile.
        - d_files: contains a list of the available data files for the sim
    """
    start1 = time.time()
    loaded_data = pluto_loader(sim_type, run_name, profile_choice,load_outputs,arr_type)
    d_files = loaded_data["d_files"]
    vars_dict = loaded_data["vars"]
    var_choice = loaded_data["var_choice"] # chosen vars at the chosen profile
    sim_coord = loaded_data["vars_extra"][0] #gets the coordinate sys of the current sim
    warnings = loaded_data["warnings"] #loads any warning messages about vars
    vars_si = defaultdict(dict)
    print(f"Reloaded Pluto_loader: {(time.time() - start1):.2f}s")

    # Process each file and variable
    start2 = time.time()
    for d_file in d_files:
        for var_name in var_choice:
            # if var_name not in vars_dict[d_file]: 
            #     continue  # Skip missing variables

            raw_data =  vars_dict[d_file][var_name]
            conv_vals = pc.value_norm_conv(var_name,d_files,raw_data,ini_file=ini_file) #converts the raw pluto array

            if var_name == "sim_time":
                # adds both time in years and seconds as keys, sim_time defaults to yr
                vars_si[d_file][var_name] = conv_vals["cgs"]
                vars_si[d_file]["sim_time_s"] = conv_vals["si"]

            else:
                vars_si[d_file][var_name] = conv_vals["si"]
    print(f"Converted Pluto_loader: {(time.time() - start2):.2f}s")

    return {"vars_si": vars_si, "var_choice": var_choice,"d_files": d_files,"sim_coord": sim_coord,"warnings": warnings}

#---Debug---#
def pluto_sim_info(sim_type,sel_runs = None): #NOTE NOT USED NEEDS UPDATING
    """WIP function to load and debug simulation info"""
    run_data = pluto_load_profile(sim_type, sel_runs,all = 1)
    run_names = run_data['run_names'] #loads the run_name names and selected profiles for runs
    coord_shape = defaultdict(list)
    # pluto_load_data = pluto_loader(sim_type,run_names[0],profile_choices[run_names[0]][0])
    # d_files = pluto_load_data["d_files"]


    for run_name in run_names:
        var_shape = []
        coord_join = []
        data = pluto_conv(sim_type,run_name,"all")
        d_files = data["d_files"]
        var_dict = data["vars_si"]
        var_choice = data["var_choice"]

        geo = data["sim_coord"]
        
        coords = var_choice[:3]
        vars = var_choice[3:]

        title_string = f"run_name {run_name}: geometry = {geo} "
        print(title_string)
        print("-"*len(title_string))
        print(f"Available data files for {run_name}: {d_files}")
        d = var_dict[d_files[0]]

        for i, var in enumerate(vars):

            var_shape.append(d[var].shape)

            if i < len(coords):
                coord_shape[run_name].append(d[coords[i]].shape)
                

            if coord_shape[run_name][-1:][0][0] == 1: #removes last if small
                coord_shape[run_name] = coord_shape[run_name][:-1]
                coords = coords[:-1]



        coord_join.append(tuple(cs[0] for cs in coord_shape[run_name]))
        # coord_join = [(coord_shape[run_name][0][0],coord_shape[run_name][1][0])]
        print(f"{coords} shape: {coord_join}")


        for i, shp in enumerate(var_shape):
            if i < len(vars) and shp == coord_join[0]:
                print(f"{vars[i]} is indexed {coords}, with shape: {shp}")
            elif i < len(vars) and shp != coord_join[0]:
                print(f"{vars[i]} has shape {shp}")
        print("\n")

    return {"coord_shape": coord_shape}




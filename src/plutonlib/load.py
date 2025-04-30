import plutonlib.utils as pu
import plutonlib.plot as pp 
import plutonlib.config as pc

profiles = pc.profiles
coord_systems = pc.coord_systems
plutodir = pc.plutodir

import plutokore.io as pk_io
from plutokore.simulations import get_output_count as pk_sim_count

import numpy as np
from astropy import units as u
from collections import defaultdict 

import sys
import time
from concurrent.futures import ThreadPoolExecutor
import inspect
import os


class SimulationData:
    """
    Class used to load and store any PLUTO output/input data, e.g. run_name names, save directories, simulation types, 
    converted/raw data, units and var info
    """
    def __init__(self, sim_type=None, run_name=None, profile_choice=None,subdir_name = None,auto_load = False, **kwargs):
        self.sim_type = sim_type
        self.run_name = run_name
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

        # Extra
        self._units = None
        self._geometry = None

        # Files
        self._d_files = None
        self._d_file = None
        self.avail_runs =  os.listdir(os.path.join(pc.sim_dir,self.sim_type))


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
        self.__dict__.update(kwargs)
        self.dir_log = None



        if auto_load:
            self.load_all()

    #---Loading Data---#
    def load_all(self):
        self.load_raw()
        self.load_conv()
        self.load_units()
    
    def load_raw(self):
        start = time.time() #for load time

        self._raw_data = pluto_loader(self.sim_type,self.run_name,self.profile_choice)
        self._d_files = self._raw_data['d_files']
        self._var_choice = self._raw_data['var_choice']
        self._geometry = self._raw_data['vars_extra'][0]
        self.load_time = time.time() - start

    def load_conv(self,profile=None):
        if self._raw_data is None:
            self.load_raw()
        
        profile = profile or self.profile_choice
        loaded_data =pluto_conv(self.sim_type, self.run_name,profile)

        self._conv_data = loaded_data

        if profile == "all": #failsafe to load all data if req
            self._all_conv_data = loaded_data

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

        if not var_info:
            raise KeyError(f"No unit info for variable {var_name}")
        
        return var_info

    def get_warnings(self):
        """Prints any warnings from loading process"""
        print(f"{pu.bcolors.WARNING}WARNING: run is now run_name and dir_str is subdir_name")
        warnings = self.conv_data['warnings']

        if self.alt_dir:
            dir_log = f"Final selected save directory: {self.save_dir}"
            print(dir_log)

        for warning in warnings:
            print(warning)

    
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
            raise ValueError("run_name name is None, IMPLEMENT RUN_NAMES FROM p_l_f")
            
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

#------------------------#
#       functions    
#------------------------#

#---Profile Loading---#
def get_profiles(sim_type,run_name,profiles):
    """
    Prints available profiles for a specific simulation
    """
    data = pluto_loader(sim_type,run_name,"all") #NOTE pl should be faster than pc
    var_choice = data["var_choice"]
    vars = data["vars"]["data_0"]

    for var in var_choice[:-1]: # doesn't include SimTime as it has no size
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

def pluto_load_profile(sim_type,sel_runs,sel_prof,all = 0):
    """
    Uses get_profiles and select_profiles to store the selected profile/s across run_name/s in the profile_choices variable
    * Use if need to gather a dict of runs each with a specific profile
    """
    sel_runs = [sel_runs] if sel_runs and not isinstance(sel_runs,list) else sel_runs
    sel = 0 if sel_runs is None else 1

    #TODO could be in config to load following save tree?
    run_dirs = os.path.join(plutodir, "Simulations", sim_type)
    all_runs = [
        d for d in os.listdir(run_dirs) if os.path.isdir(os.path.join(run_dirs, d))
    ]

    # used to selected if plotting all subdirs or select runs
    if sel == 0:
        run_names = all_runs
        print(f"Subdirectories:, {run_names}")

    elif sel == 1:
        run_names = sel_runs

    # Assign a profile number for each run_name (supports duplicates)
    profile_choices = defaultdict(list)  # Change from a single variable to defaultdict(list)

    if sel_prof is None: #use if usr input multiple profiles, diff per run_name
        if not all:
            for run_name in run_names:
                profile_choice = select_profile(sim_type,run_name,profiles)
                profile_choices[run_name].append(profile_choice)  # Appends profile to list
                print(f"Selected profile {profile_choice} for run_name {run_name}: {profiles[profile_choice]}")

        elif all:
                profile_choice = "all"
                print(f"Selected profile {profiles[profile_choice]} for all runs")
                print("\n")
    
    if sel_prof is not None: #use if want one profile across all runs
        for run_name in run_names:
            get_profiles(sim_type,run_name,profiles)
            profile_choice = sel_prof
            profile_choices[run_name].append(profile_choice)  # Appends profile to list

            try:
                print(f"Selected profile {profile_choice} for run_name {run_name}: {profiles[profile_choice]}")
                print("\n")

            except KeyError:
                raise KeyError(f"{profile_choice} is not an available profile")



    return {'run_names':run_names,'profile_choices':profile_choices}

#---Loading Files---#
def pluto_loader(sim_type, run_name, profile_choice,max_workers = None):
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
    # print("Var Choice:", var_choice)

    wdir = os.path.join(plutodir, "Simulations", sim_type, run_name)

    #NOTE USE FOR LAST OUTPUT ONLY
    # nlinf = pk_io.nlast_info(w_dir=wdir) #info dict about PLUTO outputs

    n_outputs = pk_sim_count(wdir) # grabs number of data output files, might need datatype
    d_files = [f"data_{i}" for i in range(n_outputs + 1)]

    data_0 = pk_io.pload(0,wdir)
    geometry = data_0.geometry #gets the geometry of the first file = fast

    loaded_vars = [v for v in var_choice if hasattr(data_0, v)]
    # print("Loaded Vars:", loaded_vars)

    non_vars = set(var_choice) - set(loaded_vars)


    if non_vars: 
        warnings.append(f"{pu.bcolors.WARNING}Simulation {run_name} doesn't contain: {', '.join(non_vars)}")

    def load_file(output_num):
        data = pk_io.pload(output_num, w_dir=wdir)
        return output_num, {v: getattr(data, v) for v in loaded_vars}

    # Parallel load all files
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(load_file, range(n_outputs + 1))
        
        # Process results in completion order
        for output_num, file_data in results:
            vars[f"data_{output_num}"] = file_data
    

    var_choice = [v for v in var_choice if v not in non_vars]
    vars_extra.append(geometry) # gets the geo of the sim, always loads first file

    return {"vars": vars, "var_choice": var_choice,"vars_extra": vars_extra,"d_files": d_files,"warnings": warnings} #"nlinf": nlinf

def pluto_conv(sim_type, run_name, profile_choice,**kwargs):
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
    loaded_data = pluto_loader(sim_type, run_name, profile_choice)
    d_files = loaded_data["d_files"]
    vars_dict = loaded_data["vars"]
    var_choice = loaded_data["var_choice"] # chosen vars at the chosen profile
    sim_coord = loaded_data["vars_extra"][0] #gets the coordinate sys of the current sim
    warnings = loaded_data["warnings"] #loads any warning messages about vars
    vars_si = defaultdict(dict)

    # Process each file and variable

    for d_file in d_files:
        for var_name in var_choice:
            # if var_name not in vars_dict[d_file]: 
            #     continue  # Skip missing variables

            raw_data =  vars_dict[d_file][var_name]
            conv_vals = pc.value_norm_conv(var_name,d_files,raw_data) #converts the raw pluto array

            if var_name == "SimTime":
                # adds both time in years and seconds as keys, SimTime defaults to yr
                vars_si[d_file][var_name] = conv_vals["cgs"]
                vars_si[d_file]["SimTime_s"] = conv_vals["si"]

            else:
                vars_si[d_file][var_name] = conv_vals["si"]

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




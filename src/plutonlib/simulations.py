import plutonlib.load as pl
import plutonlib.utils as pu
import plutonlib.analysis as pa
import plutonlib.config as pc
from plutonlib.colours import pcolours

import sys
import time
import os

coord_systems = pc.coord_systems
PLUTODIR = pc.plutodir


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
                 load_outputs=None,load_slice=None,slice_type=None, arr_type=None, ini_file=None, is_conv=1, setup=None):
        if setup is not None:
            # inherit defaults from setup
            sim_type = sim_type or setup.sim_type
            run_name = run_name or setup.run_name
            profile_choice = profile_choice or setup.profile_choice
            subdir_name = subdir_name or setup.subdir_name
            load_outputs = load_outputs or setup.load_outputs
            load_slice = load_slice 
            ini_file = ini_file or setup.ini_file
            arr_type = arr_type if arr_type is not None else setup.arr_type            
        # Initialize parent class first
        super().__init__(sim_type, run_name, profile_choice, subdir_name, load_outputs, arr_type, ini_file)

        self.slice_type = slice_type
        self.load_slice = load_slice

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

        if self.load_slice is not None and not isinstance(self.load_slice,tuple) :
            raise TypeError(
                f"""
                load_slice is set to '{self.load_slice}', it must be a tuple e.g. (slice(None, None, None), slice(None, None, None), 75). 
                Use slice_type for slice_1D/slice_2D
                """
            )

        if self.slice_type is not None:
            if self.slice_type == "slice_1D":
                raise NotImplementedError("Slice_1D depends not on spare coord but on sel_coord, need to fix")
                # NOTE can use sel_coord from functions like plotter pa.calc_var_prof(test,sel_coord)["slice_1D"]
                # this would require reloading sdata
                # self.load_slice = pa.calc_var_prof(self,self.spare_coord)["slice_1D"]
            elif self.slice_type == "slice_2D":
                self.load_slice = pa.calc_var_prof(self,self.spare_coord)["slice_2D"] 
            else:
                raise ValueError(f"slice_type is set to '{self.load_slice}', it must be slice_1D or slice_2D if str")

    @classmethod
    def from_setup(cls, setup,**kwargs):
        """
        Create SimulationData from existing SimulationSetup.
        Args:
            setup: The SimulationSetup object to inherit from
            **kwargs: Override specific parameters:
                - sim_type: str - Simulation type
                - run_name: str - Run name
                - profile_choice: str - Profile choice
                - subdir_name: str - Subdirectory name
                - load_outputs: tuple - Outputs to load
                - load_slice: slice - Slice to load
                - arr_type: str - Array type
                - ini_file: str - INI file
                - is_conv: bool - Whether to convert data  
        Returns:
            SimulationData: New SimulationData instance
        """
        return cls(setup=setup,**kwargs)

    # ---Loading Data---#
    def load_all(self):
        if not self._is_loaded:
            # Load everything in one go, explicitly
            self.load_raw()
            if self.is_conv:
                self.load_conv()
            self.load_units()
            self._is_loaded = True
            print("Loaded all")

    def load_raw(self):
        start = time.time() #for load time
        if isinstance(self.load_outputs, list):
            raise TypeError(f"{pcolours.WARNING}Cannot use list with load_outputs, try tuple")

        self._raw_data = pl.pluto_loader(
            self.sim_type,
            self.run_name,
            self.profile_choice,
            self.load_outputs,
            pu._slice_to_hashable(self.load_slice), #converts slice objects to hashable inputs for lru_cache
            self.arr_type,
            self.ini_file,
        )
        self._d_files = self._raw_data['d_files']
        # self._var_choice = self._raw_data['var_choice']
        self._geometry = self._raw_data['vars_extra'][0]
        print(f"Pluto Loader ({self.profile_choice}): {(time.time() - start):.2f}s")

    def load_conv_old(self,profile=None):
        if self.is_conv:
            if self._raw_data is None: #NOTE not sure if this is req 
                self.load_raw()

            start = time.time() #for load time

            profile = profile or self.profile_choice
            pluto_conv_data = pl.pluto_conv(
                self.sim_type,
                self.run_name,
                profile,
                self.load_outputs,
                pu._slice_to_hashable(self.load_slice), #converts slice objects to hashable inputs for lru_cache
                self.arr_type,
                self.ini_file,
            )

            if profile == "grid": #failsafe to load all data if req
                self._grid_data = pluto_conv_data
            else:
                self._conv_data = pluto_conv_data

            print(f"Pluto Conv ({profile}): {(time.time() - start):.2f}s")
        else:
            print(f"{pcolours.WARNING}is_conv = {self.is_conv}, Skipping convert")

    def load_conv(self,profile=None):
        if self._conv_data is not None:
            return #already loaded

        if not self.is_conv:
            print(f"{pcolours.WARNING}is_conv = {self.is_conv}, Skipping convert")
            return

        if self._raw_data is None:
            self.load_raw()

        start = time.time()
        profile = profile or self.profile_choice

        # Use the new helper function that works with pre-loaded data
        try:
            pluto_conv_data = pl.pluto_conv_from_raw(
                self._raw_data, profile, self.ini_file
            )
        except AttributeError:
            # Fallback to existing method if helper function doesn't exist
            print(f"{pcolours.WARNING}Falling back to pluto_conv (will reload files)")
            pluto_conv_data = pl.pluto_conv(
                self.sim_type,
                self.run_name,
                profile,
                self.load_outputs,
                pu._slice_to_hashable(self.load_slice),
                self.arr_type,
                self.ini_file,
            )

        # if profile == "grid":
        #     continue
        #     # self._grid_data = pluto_conv_data
        # else:
        self._conv_data = pluto_conv_data

        print(f"Pluto Conv ({profile}): {(time.time() - start):.2f}s")
    def load_particles(self,load_outputs):
        self.particle_data = pl.pluto_particles(self.sim_type,self.run_name,load_outputs)['particle_data']
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

    # ---Accessing SimulationData---#
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

    def get_grid_data(self,d_file=None,system = "uuv"):
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

    # ---Properties---#
    @property
    def get_warnings(self):
        """Prints any warnings from loading process"""
        # ---General Warnings---#
        # print(f"{pcolours.WARNING}WARNING: run is now run_name")

        # ---self and file related warnings---#
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

        # --directory warnings--#
        print(f"{pcolours.WARNING}---SimulationData Info---")
        print("\n")
        print(f"{pcolours.WARNING}Current Working Directory:", self.wdir)
        print("Save Directory:",pc.get_start_dir(self.wdir,run_name = self.run_name)["warnings"][0])
        print(f"Units file: {pc.get_ini_file(self.ini_file)}") # Prints current units.ini file
        print("\n")
        print(f"{pcolours.WARNING}Array Type: {pc.arr_type_key[self.arr_type]}")
        print(f"{pcolours.WARNING}Loading arrays with slice of shape: {self.load_slice}") if self.load_slice else None 
        print("\n")
        # get loading warnings from pluto_loader
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

            self.load_conv_old(profile="grid")  # Loads all profile
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
        return pc.profiles()["profiles"][self.profile_choice]
        # old method where pluto_loader var choice was different
        # if self._var_choice is None:
        #     self.load_raw()
        # raise ValueError("Missing var_choice data")
        # return self._var_choice

    @property
    def grid_ndim(self):
        grid_ndim = self.grid_setup["dimensions"]

        # NOTE Old method using vars doesn't work if loading slices
        # if self._units is None:
        #     self.load_units()

        # if 'rho' in self.var_choice:
        #     grid_ndim = self.get_var_info("rho")["ndim"]

        # elif self.arr_type in ('nc','cc'):
        #     grid_ndim = self.get_var_info("x1")["ndim"]

        # else:
        #     raise ValueError("Cannot determine grid dimensions without nc/cc arrays or rho")

        return grid_ndim

    @property
    def slice_shape(self):
        if self.load_slice is None:
            slice_shape = None
        else:
            is_2d = sum(i == slice(None) for i in self.load_slice) == 2
            is_1d = sum(i == slice(None) for i in self.load_slice) == 1

            if is_2d:
                slice_shape = "slice_2D"
            elif is_1d:
                slice_shape = "slice_1D"

        return slice_shape

    @property
    def reset(self):
        """Clears all cached data"""
        pl.pluto_loader.cache_clear()
        pl.pluto_conv.cache_clear()
        # pl.pluto_particles.cache_clear()
        self._raw_data = None
        self._conv_data = None
        self._units = None

    @property    
    def reload(self):
        """Force reload all data"""
        self.reset
        self.load_raw()

        return self

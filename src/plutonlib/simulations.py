import plutonlib.load as pl
import plutonlib.utils as pu
import plutonlib.analysis as pa
import plutonlib.config as pc
from plutonlib.colours import pcolours

import sys
import time
import os
from pathlib import Path

import plutokore.pluto_simulation as pk_sim

coord_systems = pc.coord_systems
PLUTODIR = pc.plutodir


class SimulationSetup:
    """
    Class used to initialise PLUTO simulation information, e.g. run_name names, save directories, simulation types, ini information etc.
    """

    _last_ini_file = None
    _last_arr_type = None

    def __init__(self, sim_type=None, run_name=None, profile_choice=None,
                 load_outputs=None,arr_type=None,ini_file=None,):

        self.sim_type = sim_type
        self.run_name = run_name


        self.load_outputs = load_outputs
        
        if isinstance(self.load_outputs, list):
            self.load_outputs = tuple(self.load_outputs)

        self.ini_file = ini_file

        self.arr_type = arr_type

        self.profile_choice = profile_choice 

        # Files
        self.wdir = os.path.join(pc.sim_dir, self.sim_type, self.run_name) if self.run_name else None
        # self.wdir = os.path.join(PLUTODIR, "Simulations", self.sim_type, self.run_name) if self.run_name else None
        
        if self.wdir and not os.path.isdir(self.wdir):
            raise FileNotFoundError(f"Working directory does not exist: {self.wdir}, current sims: {os.listdir(pc.sim_dir)}")
        
        self.avail_sims = os.listdir(pc.sim_dir)
        self.avail_runs =  os.listdir(os.path.join(pc.sim_dir,self.sim_type)) if self.sim_type else None 
        
        # Vars
        self._var_choice = None 
        SimulationSetup._last_arr_type = self.arr_type
        
    #---Properties---#
    @property
    def save_dir(self,start_dir = None):
        if not start_dir:
            output_dir = os.path.join(os.environ["HOME"],"plutonlib_output")
        else:
            output_dir = os.path.join(start_dir,"plutonlib_output")

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            print(f"Creating plutonlib_output directory: {output_dir}")

        if not self.sim_type or not self.run_name:
            raise ValueError("Either sim.sim_type or sim.run_name are not defined.")
        else:
            save_dir = os.path.join(output_dir,self.sim_type,self.run_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
                print(f"Creating save directory: {save_dir}")
            
            return save_dir
    
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
    
    @property
    def grid_ndim(self):
        grid_ndim = self.grid_setup["dimensions"]
        return grid_ndim

class SimulationData(SimulationSetup):
    """
    Class used to load/convert PLUTO simulations as well as containing from SimulationSetup 
    """
    def __init__(self, sim_type=None, run_name=None, profile_choice=None,
                 load_outputs=None,load_slice=None,slice_type=None, arr_type=None, ini_file=None, is_conv=1, setup=None):
        if setup is not None:
            # inherit defaults from setup
            sim_type = sim_type or setup.sim_type
            run_name = run_name or setup.run_name
            profile_choice = profile_choice or setup.profile_choice
            load_outputs = load_outputs or setup.load_outputs
            load_slice = load_slice 
            ini_file = ini_file or setup.ini_file
            arr_type = arr_type if arr_type is not None else setup.arr_type            
        # Initialize parent class first
        super().__init__(sim_type, run_name, profile_choice, load_outputs, arr_type, ini_file)

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

    #---Swap to plutokore sim---#
    def to_plutokore(self):
        """
        Converts SimulationData objet into plutokore PlutoSimulation object
        """
        sim = pk_sim.PlutoSimulation(
            simulation_name=self.run_name,                
            simulation_directory=Path(self.wdir),      
            simulation_description="",        
            datatype=self.dtype,                      
            dimensions=self.grid_ndim,                           
        )

        return sim
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



        # --directory warnings--#
        print(f"{pcolours.WARNING}---SimulationData Info---")
        print("\n")
        print(f"{pcolours.WARNING}Current Working Directory:", self.wdir)
        print("Save Directory:",self.save_dir)
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
    def dtype(self):
        is_dbl_h5 = os.path.isfile(os.path.join(self.wdir,r"dbl.h5.out"))
        is_flt_h5 = os.path.isfile(os.path.join(self.wdir,r"flt.h5.out"))
        is_dbl = os.path.isfile(os.path.join(self.wdir,r"dbl.out"))

        if pu.is_dbl_and_flt(self.wdir): #combination of float and double -> get only float for analysis 
            dext = "float" 
        
        elif is_dbl_h5 or is_flt_h5:
            dext = "float" if is_flt_h5 else "double" #assigns correct dtype for loading, preferentially load float
        
        elif is_dbl:
            dext = "double"         
        return dext
    
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

import plutonlib.load as pl
import plutonlib.utils as pu
import plutonlib.analysis as pa
import plutonlib.config as pc
import plutonlib.plot as pp
from plutonlib.colours import pcolours

import sys
import time
import os
import gc
import h5py
from pathlib import Path
from collections import defaultdict 

import plutokore.pluto_simulation as pk_sim

coord_systems = pc.coord_systems
PLUTODIR = pc.plutodir


class SimulationSetup:
    """
    Class used to initialise PLUTO simulation information, e.g. run_name names, save directories, simulation types, ini information etc.
    """

    _last_ini_file = None
    _last_arr_type = None

    def __init__(self, sim_type=None, run_name=None,ini_file=None,):

        self.sim_type = sim_type
        self.run_name = run_name




        self.ini_file = ini_file

        # Files
        self.wdir = os.path.join(pc.sim_dir, self.sim_type, self.run_name) if self.run_name else None
        # self.wdir = os.path.join(PLUTODIR, "Simulations", self.sim_type, self.run_name) if self.run_name else None
        
        if self.wdir and not os.path.isdir(self.wdir):
            raise FileNotFoundError(f"Working directory does not exist: {self.wdir}, current sims: {os.listdir(pc.sim_dir)}")
        
        self.avail_sims = os.listdir(pc.sim_dir)
        self.avail_runs =  os.listdir(os.path.join(pc.sim_dir,self.sim_type)) if self.sim_type else None 
                
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
    def __init__(self, sim_type=None, run_name=None,
                 load_outputs=None, ini_file=None,conv=True):

        # Initialize parent class first
        super().__init__(sim_type, run_name, ini_file)
        
        self.load_outputs = load_outputs
        
        if isinstance(self.load_outputs, list):
            self.load_outputs = tuple(self.load_outputs)

        if self.load_outputs == "last":
            self.load_outputs = (pl.get_file_outputs(self.wdir),)

        # Data
        self.conv = conv
        self._fluid_data_cache = {}
        self._metadata = {}

        self.particle_data = None
        self.particle_files = None

        # Extra
        self._units = None
        self._geometry = None

        # Files

    @classmethod
    def from_setup(cls, setup,**kwargs):
        """
        Create SimulationData from existing SimulationSetup.
        Args:
            setup: The SimulationSetup object to inherit from
            **kwargs: Override specific parameters:
                - sim_type: str - Simulation type
                - run_name: str - Run name
                - load_slice: slice - Slice to load
                - arr_type: str - Array type
                - ini_file: str - INI file
                - is_conv: bool - Whether to convert data  
        Returns:
            SimulationData: New SimulationData instance
        """
        defaults = {
            'sim_type': setup.sim_type,
            'run_name': setup.run_name,
            # 'arr_type': setup.arr_type,
            'ini_file': setup.ini_file,
        }
        # Allow kwargs to override defaults
        for key, val in defaults.items():
            if key not in kwargs:
                kwargs[key] = val
        return cls(**kwargs)

    def load_particles(self,load_outputs):
        self.particle_data = pl.pluto_particles(self.sim_type,self.run_name,load_outputs)['particle_data']
        self.particle_files = list(self.particle_data.keys())

    def load_units(self):
        self._geometry = "CARTESIAN"
        self._units = pc.get_pluto_units(self._geometry,self.ini_file) #,self._d_files

    def get_metadata(self,output=None):
        if not output:
            output = pl.get_file_outputs(self.wdir)
        return pl.load_hdf5_metadata(wdir=self.wdir,load_output=output)

    def fluid_data(self,var_choice,output=None,load_slice = None,conv=None):

        var_choice = [var_choice] if isinstance(var_choice,str) else var_choice
        output = pl.get_file_outputs(self.wdir) if not output else output
        conv = self.conv if conv is None else conv
        cache_key = (tuple(sorted(var_choice)),output,pu._slice_to_hashable(load_slice),conv)
        # print(f"DEBUG cache_key: {cache_key}")  # Add this

        if cache_key in self._fluid_data_cache:
            # print("using cache")
            return self._fluid_data_cache[cache_key]
        
        data = pl.pluto_loader_hdf5(
            wdir=self.wdir,
            load_outputs=(output,),
            var_choice=var_choice,
            load_slice=load_slice,
            ini_file=self.ini_file,
            conv=conv,
        )
        self._metadata[output] = data[output]["metadata"]
        self._fluid_data_cache[cache_key] = data[output]

        return data[output]
    
    def clear_fluid_data_cache(self):
        self._fluid_data_cache.clear()

        gc.collect()
        print(gc.collect())
        for obj in gc.get_objects():
            if isinstance(obj, h5py.File):
                try:
                    obj.close()
                except:
                    pass

    def get_var_info(self, var_name):
        """Gets coordinate name, unit, norm value etc"""
        var_info = self.units.get(pu.map_coord_name(var_name)) #unify all XYZ arrays to x1,x2,x3

        # if self.load_slice: #actual array dimensions using np are fast for slice loads
        #     shp_info = {"shape" : self.fluid_data(var_name)[var_name].shape}
        #     dim_info = {"ndim" : self.fluid_data(var_name)[var_name].ndim}
            
        # else: #if not loading slice (3D), large sim will increase memory usage, use pluto.ini
        shp_info = {"shape" : self.grid_setup["arr_shape"]}
        dim_info = {"ndim" : len(self.grid_setup["arr_shape"])}


        var_info.update(shp_info)
        var_info.update(dim_info)

        if not var_info:
            raise KeyError(f"No unit info for variable {var_name}")

        return var_info
    
    def get_injection_region(self,output=None):
        output = pl.get_file_outputs(self.wdir) if not output else output
        sim_time = self.fluid_data(var_choice="sim_time",output=output,conv=True)["sim_time"] #in Myr
        rho_0 = pu.gcm3_to_kgm3(self.usr_params['env_rho_0']) #central density from ini 
        T = self.usr_params['env_temp']
        wind_vxx = [self.usr_params["wind_vx1"],self.usr_params["wind_vx2"],self.usr_params["wind_vx3"]]
        return pa.locate_injection_region(rho_0=rho_0,T=T,wind_vxx=wind_vxx,sim_time=sim_time)

    # ---Swap to plutokore sim---#
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
    def metadata(self):
        if not self._metadata:
            raise ValueError(f"No metadata present, need to load fluid data to get metadata")
        return self._metadata
    


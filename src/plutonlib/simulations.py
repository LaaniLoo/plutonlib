import plutonlib.load as pl
import plutonlib.config as pc
import plutonlib.utils as pu
import plutonlib.analysis as pa
# import plutonlib.plot as pp
import plutonlib.simulation_info as psim_info
# from plutonlib.colours import pcolours

# import sys
# import time
import os
import gc
import h5py
from pathlib import Path
# from collections import defaultdict 
import plutokore.pluto_simulation as pk_sim
import plutokore.particles as pk_part

import warnings

coord_systems = pc.coord_systems
PLUTODIR = pc.plutodir

import h5py 
import hdf5plugin
import numpy as np

def save_particle_data_hdf5(
    sim, particle_data_dict, particle_times, particle_data_path=None
):


    zfp_kwargs = {"reversible": True}
    if particle_data_path is None:
        particle_data_path = f"{sim.processed_data_path}.particles.hdf5"
    with h5py.File(particle_data_path, "a") as particle_data_file:
        data_group = particle_data_file.require_group("particle_data")
        for k, v in particle_data_dict.items():

            #chunk sizes
            if v.ndim == 2:
                chunks = (min(v.shape[0], 100_000), 1)
            elif v.ndim == 3:
                chunks = (min(v.shape[0], 100_000), 1, v.shape[2])
            else:
                chunks = None

            ds_maxshape = [None] * len(v.shape)
            # We create a dataset if it doesn't already exist
            if k in data_group:
                ds = data_group[k]
                # resize dataset if necessary
                ds.resize(v.shape)
            else:
                ds = data_group.create_dataset(
                    k,
                    shape=v.shape,
                    dtype=np.float32,
                    maxshape=ds_maxshape,
                    chunks = chunks,
                    **hdf5plugin.Zfp(**zfp_kwargs),
                )
            ds[...] = v.astype(np.float32)
        # get time dataset
        if "time" in particle_data_file:
            ds = particle_data_file["time"]
            # resize if necessary
            ds.resize(particle_times.shape)
        else:
            # create it if it doesn't already exist
            ds = particle_data_file.create_dataset(
                "time",
                shape=particle_times.shape,
                dtype=np.float32,
                maxshape=[None] * len(particle_times.shape),
                chunks = None,
                **hdf5plugin.Zfp(**zfp_kwargs),
            )
        ds[...] = particle_times.astype(np.float32)

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
        _sim_dir = os.path.join(pc.sim_dir,self.sim_type) #Simulation types in /pluto-master/Simulations
        if self.sim_type and not os.path.isdir(_sim_dir): #error if sim_type not found
            raise FileNotFoundError(f"Simulation type '{self.sim_type}' not found in sim dir '{pc.sim_dir}', available sim types: {os.listdir(pc.sim_dir)}")
        
        if self.run_name:
            self.wdir = os.path.join(pc.sim_dir, self.sim_type, self.run_name) if self.run_name else None
            if os.path.isdir(_sim_dir) and not os.path.isdir(self.wdir): #error if run_name not found
                raise FileNotFoundError(f"Simulation run '{self.run_name}' does not exist for simulation type '{self.sim_type}', current simulation runs in {_sim_dir}: {os.listdir(_sim_dir)}")
        else:
            warnings.warn(f"No simulation run specified, continuing setup without simulation directory")

        self.avail_sims = os.listdir(pc.sim_dir)
        self.avail_runs =  os.listdir(_sim_dir) if self.sim_type else None 

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
    def usr_params(self):
        ini_info = pc.pluto_ini_info(sim_dir=self.wdir)
        return ini_info["usr_params"]

    @property
    def grid_setup(self):
        ini_info = pc.pluto_ini_info(sim_dir=self.wdir)
        return ini_info["grid_setup"]
    
    @property
    def grid_ndim(self):
        grid_ndim = self.grid_setup["dimensions"]
        return grid_ndim
    
    @property
    def grid_output(self):
        ini_info = pc.pluto_ini_info(sim_dir=self.wdir)
        return ini_info["grid_output"]
    
    @property
    def part_output(self):
        ini_info = pc.pluto_ini_info(sim_dir=self.wdir)
        return ini_info["part_output"]

class SimulationData(SimulationSetup):
    """
    Class used to load/convert PLUTO simulations as well as containing from SimulationSetup 
    """
    def __init__(self, sim_type=None, run_name=None,
                 load_outputs=None, ini_file=None,conv=True):

        # Initialize parent class first
        super().__init__(sim_type, run_name, ini_file)

        self.load_outputs = load_outputs

        # Data
        self.conv = conv
        self._fluid_data_cache = {}
        self._metadata = {}
        
        # Extra
        self._units = None
        self._geometry = None

        # Files
        _sim_dir = os.path.join(pc.sim_dir,self.sim_type) #Simulation types in /pluto-master/Simulations
        if self.sim_type and not os.path.isdir(_sim_dir): #error if sim_type not found
            raise FileNotFoundError(f"Simulation type '{self.sim_type}' not found in sim dir '{pc.sim_dir}', available sim types: {os.listdir(pc.sim_dir)}")
        
        if self.run_name:
            self.wdir = os.path.join(pc.sim_dir, self.sim_type, self.run_name) if self.run_name else None
            if os.path.isdir(_sim_dir) and not os.path.isdir(self.wdir): #error if run_name not found
                raise FileNotFoundError(f"Simulation run '{self.run_name}' does not exist for simulation type '{self.sim_type}', current simulation runs in {_sim_dir}: {os.listdir(_sim_dir)}")
        else:
            raise ValueError(f"run_name is set to None, please specify a simulation run to inspect simulation data")
        
        self.avail_sims = os.listdir(pc.sim_dir)
        self.avail_runs =  os.listdir(_sim_dir) if self.sim_type else None 

        if isinstance(self.load_outputs, list):
            self.load_outputs = tuple(self.load_outputs)

        if self.load_outputs == "last":
            self.load_outputs = (pl.get_file_outputs(self.wdir),)

    @classmethod
    def from_setup(cls, setup,**kwargs):
        """
        Create SimulationData from existing SimulationSetup.
        Args:
            setup: The SimulationSetup object to inherit from
            **kwargs: Override specific parameters:
                - sim_type: str - Simulation type
                - run_name: str - Run name
                - ini_file: str - INI file
                - conv: bool - Whether to convert data  
        Returns:
            SimulationData: New SimulationData instance
        """
        defaults = {
            'sim_type': setup.sim_type,
            'run_name': setup.run_name,
            'ini_file': setup.ini_file,
        }
        # Allow kwargs to override defaults
        for key, val in defaults.items():
            if key not in kwargs:
                kwargs[key] = val
        return cls(**kwargs)

    def load_units(self):
        self._geometry = "CARTESIAN"
        self._units = pc.PlutoUnits.from_ini(ini_file=self.ini_file)
    def get_metadata(self,output=None):
        if not output:
            output = pl.get_file_outputs(self.wdir)
        
        self._metadata[output] = pl.load_hdf5_metadata(wdir=self.wdir,load_output=output)
        if self._metadata[output].time_str == '0 Myr':
            time_unit = str(self.units.sim_time.usr_uv)
            time_val = pc.code_to_usr_units("sim_time",self.metadata[output].sim_time,ini_file="jet_units")["conv_data_uuv"]
            time_str = f"${time_val:.2f} \\; [{time_unit}]$"

            self._metadata[output].time_str = time_str
            self._metadata[output].sim_time = time_val

        return self._metadata[output]

    def load_fluid_data(self,var_choice,output=None,load_slice = None,conv=None):

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
    
    def load_jet_spline_data(self,var_choice,output,conv=None):
        """Loads simulation fluid quantities along the arc length of the jet

        Args:
            var_choice (list): list of variables to load e.g. ['ncx','rho']
            output (int): PLUTO output file number
            conv (bool, optional): to convert the data to user specified units or not. Defaults to None.

        Returns:
            fluid_data (dict): dict of fluid data per output for jet arc length, see load_fluid_data
        """
        
        var_choice = [var_choice] if isinstance(var_choice,str) else var_choice
        output = pl.get_file_outputs(self.wdir) if not output else output
        conv = self.conv if conv is None else conv
        fluid_data = {}

        spline_data = pa.get_jet_splines(self,output,None)
        spline_slice_map = spline_data["spline_slice_map"]

        temp_data = self.load_fluid_data(var_choice,output=output,load_slice=self.quick_slice_2D('xz'))
        for var in var_choice:
            fluid_data[var] = temp_data[var][spline_slice_map]

        return fluid_data
    
    def save_particles_hdf5(self):
        sim = self.to_plutokore()
        file_path = os.path.join(self.wdir,"particles.hdf5")

        if os.path.isfile(file_path):
            with h5py.File(file_path,"r") as f:
                n_outpus = f["time"][:].shape
                part_outputs = pl.get_particle_outputs(wdir=self.wdir)

                if part_outputs >= n_outpus[-1]:
                    print("New particle files found, resaving particles.hdf5...")
                else:
                    raise FileExistsError(f"{file_path} allready exists")
    
        particle_data_dict, particle_times = pk_part.load_all_particles(sim)
        # pk_part.save_particle_data_hdf5(
        #     sim=sim,
        #     particle_data_dict=particle_data_dict,
        #     particle_times=particle_times,
        #     particle_data_path=file_path,
        # )
        save_particle_data_hdf5(
            sim=sim,
            particle_data_dict=particle_data_dict,
            particle_times=particle_times,
            particle_data_path=file_path,
        )
        print(f"File saved to {file_path}")

    def load_particle_data(self,output=None,tr_cut = None):
        file_path = os.path.join(self.wdir,"particles.hdf5")
        if not os.path.isfile(file_path):
            print("particles.hdf5 not found, creating HDF5 dataset...")
            self.save_particles_hdf5()

        # output = pl.get_particle_outputs(self.wdir) if not output else output
        output = pl.get_particle_outputs(self.wdir) if output == "last" else output
        data = pl.pluto_particles_hdf5(self.wdir,output=output,tr_cut=tr_cut)
        return data

    def part_to_simtime(self,output):
        """
        Converts particle output to grid simtime
        """
        dtype = self.get_metadata().dtype
        grid_out_freq = self.grid_output[dtype+"_freq"]
        part_out_freq = self.part_output["particles_dbl_freq"] #assuming only ever dbl
        output_ratio = part_out_freq/grid_out_freq
        self.part_output["particle_spacing"] = output_ratio
        return output * output_ratio

    def simtime_to_part(self, simtime):
        """
        Converts grid simtime to particle output number
        """
        dtype = self.get_metadata().dtype
        grid_out_freq = self.grid_output[dtype + "_freq"]
        part_out_freq = self.part_output["particles_dbl_freq"]
        output_ratio = part_out_freq / grid_out_freq
        self.part_output["particle_spacing"] = output_ratio
        return simtime / output_ratio

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
        var_name = pu.map_coord_name(var_name) #unify all XYZ arrays to x1,x2,x3
        var_info = getattr(self.units,var_name)

        shp_info = {"shape" : self.grid_setup["arr_shape"]}
        dim_info = {"ndim" : len(self.grid_setup["arr_shape"])}
        
        setattr(var_info,"shp",shp_info)
        setattr(var_info,"ndim",dim_info)

        if not var_info:
            raise KeyError(f"No unit info for variable {var_name}")

        return var_info

    def get_injection_region(self,output=None):
        """Uses pa.locate_injection_region to find x,y,z location for a moving injection region"""

        output = pl.get_file_outputs(self.wdir) if not output else output
        # sim_time = self.load_fluid_data(var_choice="sim_time",output=output,conv=True)["sim_time"] #in Myr 
        # NOTE having simtime as the real simulation time caused a bug in ofset btwn output and simtime value -> keep as file output
        sim_time = output
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
    
    def quick_slice_1D(self,plane = "xz"):
        plane_map = {"xy": "x3", "xz": "x2", "yz": "x1"}

        if plane not in plane_map:
            raise KeyError(f"plane = {plane}, needs to be xy,xz or yz")

        qslice = pa.calc_var_prof(self,plane_map[plane])
        return qslice["slice_1D"]

    def quick_slice_2D(self,plane = "xz"):
        plane_map = {"xy": "x3", "xz": "x2", "yz": "x1"}

        if plane not in plane_map:
            raise KeyError(f"plane = {plane}, needs to be xy,xz or yz")

        qslice = pa.calc_var_prof(self,plane_map[plane])
        return qslice["slice_2D"]
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
        """dict of {output:HDF5Metadata}, contains dataclass of simulation HDF5 metadata"""
        if not self._metadata:
            raise ValueError(f"No metadata present, need to load fluid data to get metadata")
        return self._metadata
    
    @property 
    def jet(self):
        """Dataclass containing all jet parameters"""
        jet_info = psim_info.JetInfo.from_usr_params(self.usr_params, env=self.env)
        return jet_info
    
    @property
    def env(self):
        """Dataclass contining all simulation env parameters"""
        env_info = psim_info.EnvInfo.from_usr_params(self.usr_params)
        return env_info
    
    @property
    def sim_times(self):
        sim_times,_ = pl.get_sim_times(self.wdir)
        return sim_times
    
    @property
    def sim_times_matched(self):
        _,sim_times_matched = pl.get_sim_times(self.wdir)
        return sim_times_matched
    

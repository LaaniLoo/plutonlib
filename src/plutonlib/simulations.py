import plutonlib.read_write as prw
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
# import plutokore.particles as pk_part

import warnings

coord_systems = pc.coord_systems
PLUTODIR = pc.plutodir

import h5py 
import numpy as np

class SimulationSetup:
    """
    Class used to initialise PLUTO simulation information, e.g. run_name names, save directories, simulation types, ini information etc.
    """

    def __init__(self, rel_path = None,ini_file=None,):
        self.ini_file = ini_file

        if self.ini_file is None:
            raise ValueError("ini_file must be specified, for pluto defaults use 'pluto_units'")

        if self.ini_file is not None:
            ini_dir = os.path.join(os.environ["HOME"], 'plutonlib', 'units')
            ini_path = os.path.join(ini_dir, f"{self.ini_file}.ini")
            if not os.path.isfile(ini_path):
                raise FileNotFoundError(f"ini file {ini_path} not found, available units files: {os.listdir(ini_dir)}")
        
        if rel_path is None: #NOTE maybe change to an error?
            warnings.warn("No simulation run specified, continuing setup without simulation directory")
            rel_path = pc.sim_dir #NOTE not sure if this will cause errors
            # self.wdir = self.sim_type = self.run_name = self.rel_path = self.avail_sim_types = None
            # return

        if rel_path is not None and pc.sim_dir in rel_path: #if relative path is actually full sim path
            warnings.warn("Simulation relative path appears to contain pluto simulation directory, truncating to relative path...")
            rel_path = os.path.relpath(rel_path,start = pc.sim_dir)

        self.rel_path = rel_path
        self.wdir = os.path.join(pc.sim_dir,rel_path)
        path_parts = rel_path.split(os.sep)
        self.sim_type = path_parts[0]
        self.run_name = path_parts[-1]

        avail_sim_types = [d for d in os.listdir(pc.sim_dir) if os.path.isdir(os.path.join(pc.sim_dir, d))]
        self.avail_sim_types = avail_sim_types

        if not os.path.isdir(self.wdir): #check if its an actual dir
            print(f"Available simulation types in {pc.sim_dir}: {avail_sim_types}")
            raise FileNotFoundError(f"Simulation directory {self.wdir} does not exist\nLikely a directory issue check that sim_type: '{self.sim_type}' and run_name: '{self.run_name}' match")

        run_dir = os.path.join(pc.sim_dir,self.sim_type) #if wdir is path to sim, avail runs will show dirs inside current sim not level up
        avail_runs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
        self.avail_runs = avail_runs

    #---Properties---#
    @property
    def save_dir(self,start_dir = None): #NOTE not sure if this is needed
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

    def __eq__(self, other): #objects are equal when they share the same relative path
        return isinstance(other, SimulationData) and self.rel_path == other.rel_path

    def __hash__(self): #generate the hash based on its relative path
        return hash(self.rel_path)

    def __init__(self, rel_path=None,load_outputs=None, ini_file=None,conv=True):

        # Initialize parent class first
        super().__init__(rel_path, ini_file)
        self.load_outputs = load_outputs

        # Data
        self.conv = conv
        self._fluid_data_cache = {}
        self._metadata = {}
        
        # Extra
        self._units = None
        self._geometry = None

        if self.rel_path is None:
            raise ValueError("rel_path is set to None, please specify a simulation run to inspect simulation data")

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
            'rel_path': setup.rel_path,
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
        if output is None:
            output = prw.get_file_outputs(self.wdir)
        
        if output not in self._metadata:
            self._metadata[output] = prw.load_hdf5_metadata(wdir=self.wdir, load_output=output)

        if self._metadata[output].time_str == '0 Myr':
            time_unit = str(self.units.sim_time.usr_uv)
            time_val = pc.code_to_usr_units("sim_time",self.metadata[output].sim_time,ini_file="jet_units")["conv_data_uuv"]
            time_str = f"${time_val:.2f} \\; [{time_unit}]$"

            self._metadata[output].time_str = time_str
            self._metadata[output].sim_time = time_val

        return self._metadata[output]

    def load_fluid_data(self,var_choice,output=None,load_slice = None,conv=None):

        var_choice = [var_choice] if isinstance(var_choice,str) else var_choice
        output = prw.get_file_outputs(self.wdir) if not output else output
        conv = self.conv if conv is None else conv
        cache_key = (tuple(sorted(var_choice)),output,pu._slice_to_hashable(load_slice),conv)
        # print(f"DEBUG cache_key: {cache_key}")  # Add this

        if cache_key in self._fluid_data_cache:
            # print("using cache")
            return self._fluid_data_cache[cache_key]

        data = prw.load_fluid_hdf5(
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
    
    def load_jet_spline_data(self,var_choice,output,tr_stop = 0.2,conv=None):
        """Loads simulation fluid quantities along the arc length of the jet

        Args:
            var_choice (list): list of variables to load e.g. ['ncx','rho']
            output (int): PLUTO output file number
            conv (bool, optional): to convert the data to user specified units or not. Defaults to None.

        Returns:
            fluid_data (dict): dict of fluid data per output for jet arc length, see load_fluid_data
        """
        
        var_choice = [var_choice] if isinstance(var_choice,str) else var_choice
        output = prw.get_file_outputs(self.wdir) if not output else output
        conv = self.conv if conv is None else conv
        fluid_data_splines = {}

        spline_data = pa.get_jet_splines(sdata=self,output=output,tr_stop = tr_stop)
        spline_slice_map = spline_data["spline_slice_map"]

        temp_data = self.load_fluid_data(var_choice,output=output,load_slice=self.quick_slice_2D('xz'))
        for var in var_choice:
            fluid_data_splines[var] = temp_data[var][spline_slice_map]
        fluid_data_splines['inj_idx'] = spline_data['inj_idx']
        fluid_data_splines['L_jet'] = spline_data['jet_length'] #TODO add to jet class pls
        return fluid_data_splines
    
    def load_particle_data(self,output=None,tr_cut = None,force_check=True):
        prw.save_particle_data_hdf5(wdir=self.wdir,force_check=force_check)        
        output = prw.get_particle_outputs(self.wdir) if output == "last" else output
        data = prw.load_particles_hdf5(self.wdir,output=output,tr_cut=tr_cut)

        alias_map = {"rho": "density", "prs": "pressure", "tr1": "tracer"}
        for short_key, long_key in alias_map.items():
            if long_key not in data and short_key in data:
                data[long_key] = data[short_key]

        return data

    def part_to_simtime(self,output):
        """
        Converts particle output to grid simtime
        """
        dtype = self.get_metadata(output=1).dtype #NOTE assumes that there is at least 1 output, prevents r/w errors with running sims
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

        output = prw.get_file_outputs(self.wdir) if not output else output
        sim_time = self.load_fluid_data(var_choice="sim_time",output=output,conv=True)["sim_time"] #in Myr 
        # NOTE having simtime as the real simulation time caused a bug in ofset btwn output and simtime value -> keep as file output
        # sim_time = output
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
    # @property
    # def dtype(self):
    #     is_dbl_h5 = os.path.isfile(os.path.join(self.wdir,r"dbl.h5.out"))
    #     is_flt_h5 = os.path.isfile(os.path.join(self.wdir,r"flt.h5.out"))
    #     is_dbl = os.path.isfile(os.path.join(self.wdir,r"dbl.out"))

    #     if pu.is_dbl_and_flt(self.wdir): #combination of float and double -> get only float for analysis 
    #         dext = "float" 

    #     elif is_dbl_h5 or is_flt_h5:
    #         dext = "float" if is_flt_h5 else "double" #assigns correct dtype for loading, preferentially load float

    #     elif is_dbl:
    #         dext = "double"         
    #     return dext

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
        sim_times,_ = prw.get_sim_times(self.wdir)
        return sim_times
    
    @property
    def sim_times_matched(self):
        _,sim_times_matched = prw.get_sim_times(self.wdir)
        return sim_times_matched

    @property
    def xyz_lim(self):
        """Gets xyz axis limits based on pluto.ini grid setup"""
        xlim = (self.grid_setup["x1-grid"]['start'][0],self.grid_setup["x1-grid"]['end'][-1])
        ylim = (self.grid_setup["x2-grid"]['start'][0],self.grid_setup["x2-grid"]['end'][-1])
        zlim = (self.grid_setup["x3-grid"]['start'][0],self.grid_setup["x3-grid"]['end'][-1])

        return [xlim,ylim,zlim]

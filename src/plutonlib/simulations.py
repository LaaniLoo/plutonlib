import plutonlib.read_write as prw
import plutonlib.config as pc
import plutonlib.utils as pu
import plutonlib.analysis as pa
import plutonlib.splines as ps
import plutonlib.simulation_info as psim_info
import plutonlib.simulation_report as pl_report
import plutonlib.fancy_plot as pl_fancypl


import os
import gc
import h5py
from pathlib import Path
import warnings
import difflib


coord_systems = pc.coord_systems

import h5py 

class SimulationSetup:
    """
    Class used to initialise PLUTO simulation information, e.g. run_name names, save directories, simulation types, ini information etc.
    """

    def __init__(self, rel_path = None,ini_file=None,):
        self.ini_file = ini_file

        if self.ini_file is None:
            raise ValueError("ini_file must be specified, for pluto defaults use 'pluto_units'")

        if self.ini_file is not None:
            ini_dir = Path(Path.home()) / 'plutonlib' / 'units'
            ini_path = ini_dir / f"{ini_file}.ini"

            if not ini_path.is_file():
                raise FileNotFoundError(f"ini file {ini_path} not found, available units files: {[f.name for f in ini_dir.iterdir()]}")

        #Check that path naming and pluto sim directories are in order
        rel = Path(rel_path)
        if rel.is_relative_to(pc.SIM_PATH):
            warnings.warn("Simulation relative path appears to contain pluto simulation directory, truncating to relative path...")
            rel = rel.relative_to(pc.SIM_PATH)

        elif rel.parts[:len(pc.SIM_PATH.parts) - 1] == pc.SIM_PATH.parts[1:]:    # full path but missing the root
            raise ValueError(f"'{rel_path}' looks like a full path but is missing the leading '{pc.SIM_PATH.anchor}', "
                            f"try '{pc.SIM_PATH.anchor}{rel_path}'")
        
        elif rel.is_absolute():
            raise ValueError(f"'{rel_path}' is an absolute path outside the simulation directory ({pc.SIM_PATH}), "
                            f"use a path relative to it, e.g. 'MHD_jet/run_name'")

        if not rel.parts:
            raise ValueError("rel_path points at the simulation directory itself, include a sim type and run name")

        self.rel_path = str(rel) #NOTE convert back to passable str
        self.wdir = str(pc.SIM_PATH / rel)
        self.sim_type = rel.parts[0]
        self.run_name = rel.parts[-1]

        avail_sim_types = [d.name for d in pc.SIM_PATH.iterdir() if Path(d).is_dir()]
        self.avail_sim_types = avail_sim_types

        run_dir = pc.SIM_PATH / self.sim_type #if wdir is path to sim, avail runs will show dirs inside current sim not level up
        if not run_dir.is_dir(): #if path to sim_type doesn't exist
            closest_sims = difflib.get_close_matches(self.sim_type,self.avail_sim_types,n=3)
            if len(closest_sims) > 1:
                raise FileNotFoundError(f"Simulation sim type ('{self.sim_type}') does not exist in path '{pc.SIM_PATH}'\nPotential matching sim types: {closest_sims}")
            else:
                raise FileNotFoundError(f"Simulation sim type ('{self.sim_type}') does not exist in path '{pc.SIM_PATH}'\nAvailable simulation types in {pc.SIM_PATH}: {avail_sim_types}")
  
        avail_runs = [d.name for d in run_dir.iterdir() if Path(d).is_dir()]
        self.avail_runs = avail_runs

        if run_dir.is_dir() and not Path(self.wdir).is_dir(): #Path to sim_type: yes, path to run: no
            closest_runs = difflib.get_close_matches(self.run_name,self.avail_runs,n=3)
            if len(closest_runs) > 1:
                raise FileNotFoundError(f"Simulation run name ('{self.run_name}') does not exist in path '{run_dir}'\nPotential matching run names: {closest_runs}")
            else:
                raise FileNotFoundError(f"Simulation run name ('{self.run_name}') does not exist in path '{run_dir}'\nAvailable runs in {run_dir}: {avail_runs}")
        if not Path(self.wdir).is_dir(): #NO path at all
            raise FileNotFoundError(f"Simulation directory {self.wdir} does not exist\nNo match for sim_type: '{self.sim_type}' and run_name: '{self.run_name}'")



    #---Properties---#
    @property
    def save_dir(self):
        """Output directory for this run (~/plutonlib_output/<sim_type>/<run_name>), created if missing. Returns a str."""
        if not self.sim_type or not self.run_name:
            raise ValueError("Either sim.sim_type or sim.run_name are not defined.")

        output_dir = Path.home() / "plutonlib_output"
        save_path = output_dir / self.sim_type / self.run_name

        if not output_dir.is_dir():
            print(f"Creating plutonlib_output directory: {output_dir}")
        if not save_path.is_dir():
            print(f"Creating save directory: {save_path}")
            save_path.mkdir(parents=True, exist_ok=True)

        return str(save_path)
 
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
    def ini_grid_output(self):
        ini_info = pc.pluto_ini_info(sim_dir=self.wdir)
        return ini_info["ini_grid_output"]
    
    @property
    def ini_part_output(self):
        ini_info = pc.pluto_ini_info(sim_dir=self.wdir)
        return ini_info["ini_part_output"]

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

    def get_metadata(self,grid_output=None):
        if grid_output is None:
            grid_output = prw.get_file_outputs(self.wdir)
        
        if grid_output not in self._metadata:
            self._metadata[grid_output] = prw.load_hdf5_metadata(wdir=self.wdir, grid_output=grid_output)

        if self._metadata[grid_output].time_str == '0 Myr':
            time_unit = str(self.units.sim_time.usr_uv)
            time_val = pc.code_to_usr_units("sim_time",self.metadata[grid_output].sim_time,ini_file="jet_units")["conv_data_uuv"]
            time_str = f"${time_val:.2f} \\; [{time_unit}]$"

            self._metadata[grid_output].time_str = time_str
            self._metadata[grid_output].sim_time = time_val

        return self._metadata[grid_output]

    #---Load useful data---#
    def load_fluid_data(self,var_choice,grid_output=None,load_slice = None,conv=None):
        """Loads a singular output, can be iterated on"""
        if load_slice == "quick2D":
            load_slice = self.quick_slice_2D()
        if load_slice == "quick1D":
            load_slice = self.quick_slice_1D("xy")

        var_choice = [var_choice] if isinstance(var_choice,str) else var_choice
        grid_output = prw.get_file_outputs(self.wdir) if not grid_output else grid_output
        conv = self.conv if conv is None else conv
        cache_key = (tuple(sorted(var_choice)),grid_output,pu._slice_to_hashable(load_slice),conv)
        # print(f"DEBUG cache_key: {cache_key}")  # Add this

        if cache_key in self._fluid_data_cache:
            # print("using cache")
            return self._fluid_data_cache[cache_key]

        data = prw.load_fluid_hdf5(
            wdir=self.wdir,
            grid_outputs=(grid_output,),
            var_choice=var_choice,
            load_slice=load_slice,
            ini_file=self.ini_file,
            conv=conv,
        )
        self._metadata[grid_output] = data[grid_output]["metadata"]
        self._fluid_data_cache[cache_key] = data[grid_output]

        return data[grid_output]
    
    def load_jet_data_tr(self,var_choice,grid_output,tr_stop = 0.2,conv=None):
        """Loads simulation fluid quantities along the arc length of the jet

        Args:
            var_choice (list): list of variables to load e.g. ['ncx','rho']
            grid_output (int): PLUTO grid_output file number
            conv (bool, optional): to convert the data to user specified units or not. Defaults to None.

        Returns:
            fluid_data (dict): dict of fluid data per output for jet arc length, see load_fluid_data
        """
        
        var_choice = [var_choice] if isinstance(var_choice,str) else var_choice
        grid_output = prw.get_file_outputs(self.wdir) if not grid_output else grid_output
        conv = self.conv if conv is None else conv
        fluid_data_splines = {}

        spline_data = ps.get_jet_splines_tr(sdata=self,grid_output=grid_output,tr_stop = tr_stop)
        spline_slice_map = spline_data["spline_slice_map"]

        temp_data = self.load_fluid_data(var_choice,grid_output=grid_output,load_slice=self.quick_slice_2D('xz'))
        for var in var_choice:
            fluid_data_splines[var] = temp_data[var][spline_slice_map]
        fluid_data_splines['inj_idx'] = spline_data['inj_idx']
        fluid_data_splines['L_jet'] = spline_data['jet_length'] #TODO add to jet class pls
        return fluid_data_splines

    def load_jet_data_sb(self,freq,redshift,var_choice,grid_output):
        """Same functionality as load_jet_data_tr but path tracing via a skeletonised SB image"""
        data_splines = {}
        interp_data = ps.interp_splines_plutogrid(sim=self,grid_output=grid_output,freq=freq,redshift=redshift)
        temp_data = self.load_fluid_data(var_choice,grid_output=grid_output,load_slice=self.quick_slice_2D('xz'))
        for var in var_choice:
            data_splines[var] = temp_data[var][interp_data['spline_slice_map']]

        data_splines['inj_idx'] = interp_data["inj_idx"]
        data_splines['L_jet'] = interp_data["jet_length"]
        return data_splines
    
    def load_particle_data(self,grid_output=None,part_output=None,tr_cut = None,force_check=True):
        """if output = int, load all particles up to output, if output = tuple, load just that particle output"""
        prw.save_particle_data_hdf5(wdir=self.wdir, force_check=force_check)

        if grid_output is not None and part_output is not None:
            raise ValueError("Pass only one of `grid_output` or `part_output`, not both")

        if grid_output is not None and part_output is not None:
            raise ValueError("Pass only one of `grid_output` or `part_output`, not both")

        if part_output is not None:
            pass  # legacy path: use exactly as given — int loads cumulative, tuple/list loads specific outputs
        elif grid_output is None:
            part_output = None
        elif grid_output == "last":
            part_output = (prw.get_particle_outputs(self.wdir),)
        else:
            part_output = self.simtime_to_part(grid_output, round_val=True)

        data = prw.load_particles_hdf5(self.wdir,part_outputs=part_output,tr_cut=tr_cut)

        # alias_map = {"rho": "density", "prs": "pressure", "tr1": "tracer"}
        # for short_key, long_key in alias_map.items():
        #     if long_key not in data and short_key in data:
        #         data[long_key] = data[short_key]

        return data

    def load_sb_data(self,grid_output,angles,freq,redshift,plane = 'xz'):
        data = prw.load_sb_hdf5(
            sim=self,
            grid_outputs=[grid_output],
            angles=[angles],
            freqs="all", #NOTE this was changed from 'all'
            redshift=redshift,
            plane=plane)

        file_freqs = data['metadata']['freqs']
        if freq not in file_freqs:
            raise ValueError(f"Requested freq {freq} not found for redshift {redshift}. Available: {file_freqs}")

        entry = data[self][grid_output][tuple(angles)]
        obs = entry["obs_properties"]
        sb_data = {
            "freqs": data['metadata']['freqs'],
            "sb": entry["sb"], #full array with freq idxes
            "log_sb": entry["log_sb"][freq], #single array keyed by freq
            "alpha": entry["alpha"], #idx by freq groups e.g (0.15,0.25), (0.25,0.14)
            "grid_mx": obs["grid_mx"],
            "grid_x": obs["grid_x"],
            "grid_my": obs["grid_my"],
            "grid_y": obs["grid_y"],
            "rot_mat": obs["rot_mat"],
        }

        return sb_data
    #---#

    def _apply_ratio(self, value, ratio, round_val):
        """Applies a conversion ratio to a scalar, list, or tuple, preserving the input container type."""
        if isinstance(value, (list, tuple)):
            converted = [v * ratio for v in value]
            if round_val:
                converted = [round(v) for v in converted]
            return type(value)(converted)
        converted = value * ratio
        return round(converted) if round_val else converted

    def part_to_simtime(self, part_output, round_val=False):
        """
        Converts particle output to grid simtime. Accepts a scalar, list, or tuple.
        """
        dtype = self.get_metadata(grid_output=0).dtype  # NOTE assumes at least 1 output, prevents r/w errors with running sims
        grid_out_freq = self.ini_grid_output[dtype + "_freq"]
        part_out_freq = self.ini_part_output["particles_dbl_freq"]  # assuming only ever dbl
        output_ratio = part_out_freq / grid_out_freq
        self.ini_part_output["particle_spacing"] = output_ratio

        return self._apply_ratio(part_output, output_ratio, round_val)

    def simtime_to_part(self, simtime, round_val=False):
        """
        Converts grid simtime to particle output number. Accepts a scalar, list, or tuple.
        """
        dtype = self.get_metadata(grid_output=0).dtype  # NOTE assumes at least 1 output, prevents r/w errors with running sims
        grid_out_freq = self.ini_grid_output[dtype + "_freq"]
        part_out_freq = self.ini_part_output["particles_dbl_freq"]
        output_ratio = part_out_freq / grid_out_freq
        self.ini_part_output["particle_spacing"] = output_ratio

        return self._apply_ratio(simtime, 1 / output_ratio, round_val)

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

    def get_injection_region(self,grid_output=None):
        """Uses pa.locate_injection_region to find x,y,z location for a moving injection region"""

        grid_output = prw.get_file_outputs(self.wdir) if not grid_output else grid_output
        sim_time = self.load_fluid_data(var_choice="sim_time",grid_output=grid_output,conv=True)["sim_time"] #in Myr 
        # NOTE having simtime as the real simulation time caused a bug in ofset btwn output and simtime value -> keep as file output
        # sim_time = output
        rho_0 = pu.gcm3_to_kgm3(self.usr_params['env_rho_0']) #central density from ini 
        T = self.usr_params['env_temp']
        wind_vxx = [self.usr_params["wind_vx1"],self.usr_params["wind_vx2"],self.usr_params["wind_vx3"]]
        return pa.locate_injection_region(rho_0=rho_0,T=T,wind_vxx=wind_vxx,sim_time=sim_time)

    def quick_fig_fluid(self,var="rho",grid_output=None,**kwargs):
        grid_output = prw.get_file_outputs(self.wdir) if not grid_output else grid_output
        pdata = pl_fancypl.fluid(sim_dict={self:[grid_output]},var=var,**kwargs)

        return pdata.fig

    # ---Swap to plutokore sim---#
    def to_plutokore(self):
        """
        Converts SimulationData objet into plutokore PlutoSimulation object
        """
        print("Importing plutokore...")
        import plutokore.pluto_simulation as pk_sim
        # self.get_metadata().dtype.split(".")[0] #flt or dbl
        sim = pk_sim.PlutoSimulation(
            simulation_name=self.run_name,                
            simulation_directory=Path(self.wdir),      
            simulation_description="",        
            datatype="double", #NOTE plutokore doesnt like flt or dbl -> hardcode to double'                   
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

    @property 
    def report(self):
        """Generates a simulation report using SimulationReport"""
        return pl_report.SimulationReport.from_sim(self)

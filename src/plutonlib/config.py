import plutonlib.utils as pu
from plutonlib.colours import pcolours
import plutonlib.analysis as pa

from astropy import units as u
from astropy import constants as const 

from IPython.display import display, Latex
import numpy as np

import os
import configparser
import glob 
import re

from collections import defaultdict
from dataclasses import dataclass

@dataclass
class VarInfo:
    code_uv:    u.Unit      # code unit value  e.g. 1.0023678e-24 g/cm³
    usr_uv:     u.Unit      # user unit value  e.g. 1 kg/m³
    code_unit:  u.Unit      # just the unit         e.g. g/cm³
    var_name:   str         # latex string          e.g. r"$\rho$"
    coord_name: str = None  # only for x1, x2, x3
    shp: tuple = None
    ndim: tuple = None

@dataclass
class PlutoUnits:
    rho:       VarInfo
    prs:       VarInfo
    vx1:       VarInfo
    vx2:       VarInfo
    vx3:       VarInfo
    x1:        VarInfo
    x2:        VarInfo
    x3:        VarInfo
    T:         VarInfo
    Q:         VarInfo
    sim_time:  VarInfo
    tr1:       VarInfo

    @classmethod
    def from_ini(cls, ini_file=None):
        if ini_file is None: #gets raise error safer than null assignment
            ini_path = get_ini_file(ini_file=None)
            raise ValueError(f"{pcolours.WARNING}ini_file is None, please load defaults from {ini_path}")

        ini_path = get_ini_file(ini_file=ini_file)
        sel_coords = coord_systems["CARTESIAN"] #gets the coord vars for the specific coord sys #NOTE replace if not cartesian

        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(ini_path)

        c_unit = u.def_unit("c",const.c) #adds 1c (speed of light) as constant
        u.add_enabled_units([c_unit])
        code_unit_values = {k: u.Unit(v) for k, v in config["code_unit_values"].items()}
        code_units = {k: u.Unit(v.split("*", 1)[-1]) for k, v in config["code_unit_values"].items()}
        usr_unit_values = {k: u.Unit(v) for k, v in config["usr_unit_values"].items()}

        var_metadata = { #NOTE can probably depreciate coord_name
            "rho":      {"var_name": r"$\rho$"},
            "prs":      {"var_name": r"$P$"},
            "vx1":      {"var_name": f"$V_{{{sel_coords[0]}}}$"},
            "vx2":      {"var_name": f"$V_{{{sel_coords[1]}}}$"},
            "vx3":      {"var_name": f"$V_{{{sel_coords[2]}}}$"},
            "x1":       {"var_name": f"${sel_coords[0]}$", "coord_name": f"${sel_coords[0]}$"},
            "x2":       {"var_name": f"${sel_coords[1]}$", "coord_name": f"${sel_coords[1]}$"},
            "x3":       {"var_name": f"${sel_coords[2]}$", "coord_name": f"${sel_coords[2]}$"},
            "tr1":      {"var_name": f"Tracer"},
            "T":        {"var_name": r"$T$"},
            "Q":        {"var_name": r"$Q_{\rm{jet}}$"},
            "sim_time": {"var_name": r"$t$"},
        }

        var_infos = {
            v: VarInfo(
                code_uv=code_unit_values[v],
                usr_uv=usr_unit_values[v],
                code_unit=code_units[v],
                **metadata
            ) for v, metadata in var_metadata.items()
        }
        return cls(**var_infos)

# start_dir = r"/mnt/g/My Drive/Honours S4E (2025)/Notebooks/" #starting directory, used to save files starting in this dir

src_path = os.path.join(os.path.expanduser('~'),'plutonlib/src/plutonlib')
main_path = os.path.join(os.path.expanduser('~'),'plutonlib/')
try: #Checks for PLUTO_DIR env var
    plutodir = os.environ["PLUTO_DIR"]
except KeyError:
    print(f'{pcolours.WARNING}PLUTO_DIR env var not found, please set the location of the PLUTO code')


if os.path.isdir(os.path.join(plutodir, "Simulations")): #if simulation dir doesn't exist 
    sim_dir = os.path.join(plutodir, "Simulations")
else:
    raise FileNotFoundError(f"{pcolours.WARNING} Simulation directory not found, needs to be in PLUTO_DIR ({plutodir}), see sim_save.sh")

arr_type_key = {
    "e": "1D cell edge coordinate arrays [x, y, z]",
    "m": "1D cell midpoint coordinate arrays [x, y, z]",
    "d": "1D cell delta arrays [x, y, z]",
    "nc": "3D cell edge arrays [x, y, z]",
    "cc": "3D cell midpoint arrays [x, y, z]"
    }

patch_type_key = {
    "u":"uniform",
    "s":"stretched"
}

coord_systems = {
    "CYLINDRICAL": ['r','z','theta'],
    "CARTESIAN": ['x','y','z'],
    "SPHERICAL": ['r','theta','phi']  
}

# Read norm values from ini file
def get_ini_file(ini_file = None):
    """
    Gets pluto_units.ini file from plutonlib src directory,
    should follow naming convention: name_units.ini 
    """
    if ini_file is None:
        ini_path = os.path.join(main_path,"units","pluto_units" + ".ini")

    else:
        ini_path = os.path.join(main_path,"units",ini_file + ".ini")
    
    is_file = os.path.isfile(ini_path)
    if not is_file:
        raise FileNotFoundError(f"{pcolours.WARNING}{ini_path} Not found")
    
    return ini_path 

def get_grid_dimensions(grid_setup):
    dim_info = {}
    
    for coord in ['x1-grid', 'x2-grid', 'x3-grid']:
        if coord in grid_setup:
            total_cells = sum(grid_setup[coord]['patch_cells'])
            dim_info[coord] = {
                'total_cells': total_cells,
                'is_active': total_cells > 1
            }
    
    # Count active dimensions (dimensions with more than 1 cell)
    active_dims = sum(1 for coord_info in dim_info.values() if coord_info['is_active'])
     
    return active_dims

def pluto_ini_info(sim_dir=None,ini_file = None):

    if ini_file is not None: #direct ini file input
            latest_ini = ini_file
    else:
        job_info_dir = os.path.join(sim_dir,"job_info")

        if os.path.isdir(job_info_dir):
            job_dir_files = glob.glob(f"{job_info_dir}/*.ini")
            latest_ini = max(job_dir_files,key=os.path.getctime)

        else:
            dir_files = glob.glob(f"{sim_dir}/*.ini")
            latest_ini = max(dir_files,key=os.path.getctime)

    ini_name = latest_ini.split("/")[-1]

    config = configparser.ConfigParser(allow_no_value=True)
    config.read(latest_ini)

    raw_grids = config.options("Grid")
    grid_setup = defaultdict(dict)
    for grid in raw_grids: #loop across all ini grids
        grid_raw = grid.split(" ") #raw grid data
        grid_contents = list(filter(None,grid_raw))
        grid_coord = grid_contents[0] #current grid coord
        n_patches = int(grid_contents[1]) #number of grid patches
        grid_patches = grid_contents[2:] #list of all grid patches

        # idxs for grid patch start and end, there are 4 elements -> Patch Start	| Grid Cells | Patch Type | Patch End
        patch_start_idx = np.arange(0,len(grid_patches)-3,3) 
        patch_end_idx = np.arange(4,len(grid_patches)+1,3)

        grid_setup[grid_coord]["n_patches"] = n_patches
        starts,ends,n_cells,types = [],[],[],[]
        for i in range(0,n_patches):
            patch = grid_patches[patch_start_idx[i]:patch_end_idx[i]]
            starts.append(float(patch[0]))
            ends.append(float(patch[-1]))
            n_cells.append(float(patch[1]))
            types.append(patch_type_key[patch[2]])

        grid_setup[grid_coord]["start"] =  starts
        grid_setup[grid_coord]["end"] = ends
        grid_setup[grid_coord]["patch_cells"] = n_cells

        grid_setup[grid_coord]["type"] = types

    all_cells = []
    for grid_coord in grid_setup.keys():

        grid = grid_setup[grid_coord]
        starts = grid["start"]
        ends = grid["end"]
        grid["grid_extent"] = (grid["start"][0],grid["end"][-1])
        patch_cells = grid["patch_cells"]
        all_cells.append(int(sum(patch_cells)))
        for patch in range(grid["n_patches"]):
            if starts[patch] <= 0 <= ends[patch]:
                if grid["type"][patch] == "uniform":
                    patch_length = ends[patch] - starts[patch]                    
                    position_from_start = 0 - starts[patch]              
                    dx = position_from_start / patch_length  
                    patch_idx = int(dx * patch_cells[patch]) #'midpoint' of patch containing x/y/z = 0

                    origin_idx = int(sum(patch_cells[:patch])+patch_idx) # offset by previous grid cells

                else:
                    raise NotImplementedError("Finding origin idx only working with uniform patches")

        grid["origin_idx"] = origin_idx
        grid["dx"] =  np.sum(np.abs(starts+ends)) / np.sum(patch_cells)
    grid_setup["dimensions"] = get_grid_dimensions(grid_setup)
    grid_setup["arr_shape"] = tuple((cells) for cells in all_cells)

    raw_usr_params = config.options("Parameters")
    usr_params = {
        k: float(v.split(";",1)[0].strip())
        for line in raw_usr_params if " " in line
        for k,v in [line.split(None,1)]
    }

    raw_grid_output = config.options("Static Grid Output")
    grid_output = {
        # k: str(v.split(";",1)[0].strip())
        k: [pu.is_num_or_str(x) for x in v.split() ]
        for line in raw_grid_output if " " in line
        for k,v in [line.split(None,1)]
    }

    static_output_dict = {}
    for entry in config.options("Static Grid Output"):
        parts = entry.split()
        key, values = parts[0], parts[1:]

        if key in ("dbl","flt","dbl.h5","flt.h5"):
            value = float(values[0])
            static_output_dict[f"{key + '_freq'}"] = value if value > 0 else 0

        if key == "log":
            static_output_dict["log_freq"] = float(values[0])

        if key in ("log_dir","output_dir"):
            static_output_dict[key] = str(values[0])

    particle_output_dict = {}
    for entry in config.options("Particles"):
        parts = entry.split()
        key, values = parts[0], parts[1:]

        if key in ("particles_dbl","particles_flt","particles_vtk","particles_tab"):
            value = float(values[0])
            particle_output_dict[f"{key + '_freq'}"] = value if value > 0 else 0

        if key == "Nparticles":
            particle_output_dict[key] = values[1]

    # ini_goutput = {}
    # ini_goutput["static_grid"] = static_output_dict
    # ini_part_output["particles"] = particle_output_dict
    key_params = {key: usr_params[key] for key in ['jet_pwr','jet_spd','jet_chi','env_rho_0','env_temp','wind_vx1','wind_vx2','wind_vx3']}

    returns = {
        "grid_setup": grid_setup,
        "grid_output": grid_output,
        "usr_params": usr_params,
        "grid_output": static_output_dict,
        "part_output": particle_output_dict,
        "key_params": key_params,
        "ini_name": ini_name,
    }

    return returns

def code_to_usr_units(var_name,raw_data = None, self = 0,ini_file = None):
    """
    gets unit value from PlutoUnits to convert from code units to the user specified units in the ini file.
    """
    mapped_var_name = pu.map_coord_name(var_name) #makes sure to convert diff XYZ arrays to x1,x2,x3
    can_convert = True
    
    try:
        pluto_units = getattr(PlutoUnits.from_ini(ini_file=ini_file),mapped_var_name)
    except AttributeError:
        print(f"PlutoUnits doesn't have attribute '{mapped_var_name}', may be unitless")
        can_convert = False
    
    # code_uv = pluto_units[mapped_var_name]["code_uv"]
    code_uv = pluto_units.code_uv if can_convert else None
    usr_uv = pluto_units.usr_uv if can_convert else None
    np.asarray(raw_data) if np.any(raw_data) and not isinstance(raw_data,np.ndarray) else raw_data #calc only works if raw_data is numpy array 

    #convert raw_data
    if code_uv is None or usr_uv is None:  #skips if it doesn't need converting
        conv_data_uuv = raw_data
        uv_usr = None
    else:
        uv_usr = (1*code_uv).to(usr_uv).value
        raw_data *= uv_usr
        conv_data_uuv = raw_data

    returns = {
        "uv_usr":uv_usr, #like a scale factor??
        "conv_data_uuv":conv_data_uuv 
    }

    return returns

# ---Make changes to pluto.ini---#
def update_ini_value(filepath, section, key, new_value):
    """
    Update pluto.ini values
    """
    with open(filepath, 'r') as f:
        content = f.read()

    in_section = False
    lines = content.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('['):
            in_section = stripped.lower() == f'[{section.lower()}]'
        elif in_section and re.match(rf'^{key}\s+', stripped, re.IGNORECASE):
            # Preserve original spacing, just replace the value (before any comment)
            lines[i] = re.sub(
                r'(?i)(' + key + r'\s+)\S+',
                lambda m: m.group(1) + new_value,
                line,
                count=1
            )
            break

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

def calc_chi_from_ini(ini_file):
    """
    Calculates the chi parameter required to pressure match jet in both rel and non-rel cases
    """
    usr_params = pluto_ini_info(sim_dir=os.path.dirname(ini_file))['usr_params']

    jet_pwr = usr_params['jet_pwr'] * u.erg / u.s 
    jet_angle = usr_params['jet_oa_primary'] * u.deg
    inj_rad = usr_params['jet_initial_radius'] * u.kpc
    jet_spd = usr_params['jet_spd'] * const.c

    env_rho = usr_params['env_rho_0'] * u.g / u.cm**3
    env_temp = usr_params['env_temp'] * u.K
    env_prs = (env_rho * env_temp * (const.k_B / (0.60364 * const.u))).si

    if usr_params['jet_spd'] >= 0.1: #relativistic case
        jet_rho = pa.calc_jet_density_rel(jet_pwr,jet_spd,jet_angle,inj_rad,prs=env_prs)
    elif usr_params['jet_spd'] < 0.1: #regular case
        jet_rho = pa.calc_jet_density(jet_pwr,jet_spd,jet_angle,inj_rad)

    chi = pa.calc_chi(jet_rho,env_prs,5/3).si
    display(chi)

    return chi

def update_chi_pluto_ini(ini_file):
    """
    Updates pluto.ini with correct JET_CHI parameter to pressure match jet
    """
    chi = calc_chi_from_ini(ini_file)
    update_ini_value(
        ini_file,
        section='Parameters',
        key='JET_CHI',
        new_value=f'{chi.value:.1f}'
    )

#---Getting PLUTO Params from other files---#
def get_geometry_gridout(wdir):
    """Gets the simulation geometry from PLUTO's grid.out file

    Args:
        wdir (str): working directory of the simulation

    Raises:
        ValueError: if GEOMETRY string is not present

    Returns:
        geometry (str): e.g. "CARTESIAN"
    """
    gridout_path = os.path.join(wdir,"grid.out")
    with open(gridout_path, "r") as f:
        for line in f:
            if "GEOMETRY:" in line:
                return line.split(":")[-1].strip()
            
    raise ValueError(f"GEOMETRY not found in {gridout_path}")
# ---Sim data tree structure---#
def plutonlib_tree_helper():
    tree = """pluto_master/
└── Simulations/
    └── sim_type/
        └── run_name/
            ├── data.0000.dbl.h5
            ├── data.0000.dbl.xmf
            ├── dbl.h5.out
            ├── grid.out
            ├── restart.out
            ├── job_info
            │   └── pluto_template.ini
            ├── log
            │   ├── pluto.0.log
            │   ├── pluto.1.log
            │   ├── pluto.2.log
            │   ├── pluto.3.log
            │   ├── pluto.4.log
            │   └── pluto.5.log
            └── run_name_plutonlib_output
                └── Jet_wind_test_temp_xz_vel_plot.png"""
    print(tree)

import plutonlib.config as pc
import plutonlib.load as pl

import os
import time
import sys
from time import sleep
import importlib
from glob import glob 
import psutil
import h5py

from scipy import constants
from astropy import units as u

def py_reload(module):
    if isinstance(module,str):
        module_name = module
    else:
        module_name = module.__name__

    module = importlib.import_module(module_name) #, package=None
    importlib.reload(module)

    print(f"{module_name} Last Saved:",time.ctime(os.path.getmtime(module.__file__))) # Checks last modification time

def py_reload_all():
    is_path = os.path.isdir(pc.src_path)
    if not is_path:
        raise FileNotFoundError(f"Cannot locate src path, currently set to: {pc.src_path}")
    
    for f in glob(pc.src_path + '/*.py'):
        py_file = f.split("/")[6]
        module_name = "plutonlib." + py_file.split(".py")[0]

        module = importlib.import_module(module_name) #, package=None
        importlib.reload(module)

        print(f"{module_name} Last Saved:",time.ctime(os.path.getmtime(module.__file__))) # Checks last modification time

def _slice_to_hashable(slice_obj):
    """Convert slice object to hashable tuple representation"""
    if slice_obj is None:
        return None
    if isinstance(slice_obj, slice):
        return ('slice', slice_obj.start, slice_obj.stop, slice_obj.step)
    if isinstance(slice_obj, tuple):
        return tuple(_slice_to_hashable(s) for s in slice_obj)
    return slice_obj

def _hashable_to_slice(hashable_obj):
    """Convert hashable tuple representation back to slice object"""
    if hashable_obj is None:
        return None
    if isinstance(hashable_obj, tuple) and len(hashable_obj) == 4 and hashable_obj[0] == 'slice':
        return slice(hashable_obj[1], hashable_obj[2], hashable_obj[3])
    if isinstance(hashable_obj, tuple):
        return tuple(_hashable_to_slice(s) for s in hashable_obj)
    return hashable_obj

def is_num_or_str(x):
    try:
        x = float(x)
        return int(x) if x.is_integer() else x
    except ValueError:
        return x

def is_dbl_and_flt(wdir):
    """
    Checks pluto.ini to see if sim outputs both dbl and flt h5 files
    """
    grid_output = pc.pluto_ini_info(wdir)["grid_output"]
    dbl = grid_output["dbl.h5"][0]
    flt = grid_output["flt.h5"][0]
    is_dbl = True if dbl != -1 else False
    is_flt = True if flt != -1 else False

    if is_dbl and is_flt:
        return True
    else:
        return False

def guess_arr_type(coord_name):

    if not isinstance(coord_name, str):
        raise TypeError(f"coord_name must be string, got {type(coord_name)}")
    
    # Direct mapping: check if it matches any known pattern
    coord_patterns = {
        'nc': ['ncx', 'ncy', 'ncz'],
        'cc': ['ccx', 'ccy', 'ccz'],
        'e': ['ex', 'ey', 'ez'],
        'm': ['mx', 'my', 'mz'],
        'd': ['dx', 'dy', 'dz'],
    }
    
    for arr_type, coords in coord_patterns.items():
        if coord_name in coords:
            return arr_type
    
    # If it's x1/x2/x3 or unrecognized, return None (default)
    if coord_name in ['x1', 'x2', 'x3']:
        return None
    
    # If we get here, it's unrecognized
    raise ValueError(f"Unrecognized coordinate name: '{coord_name}'. "
                    f"Expected one of: {[c for coords in coord_patterns.values() for c in coords] + ['x1', 'x2', 'x3']}")

def get_coord_names(arr_type = "nc",coord = None):
    coord_prefixes = {
    "e":  ["ex",  "ey",  "ez"],
    "m":  ["mx",  "my",  "mz"],
    "d":  ["dx",  "dy",  "dz"],
    # "x":  ["x1",  "x2",  "x3"],
    "nc": ["ncx", "ncy", "ncz"],
    "cc": ["ccx", "ccy", "ccz"],
    None: ["x1",  "x2",  "x3"],  # default if None
    }

    if arr_type not in coord_prefixes:
        raise KeyError(f"{arr_type} not recognised array type, see {coord_prefixes}")

    # Get the correct coordinate labels, defaults to mx ...
    coords = coord_prefixes.get(arr_type, ["x1", "x2", "x3"])
    x, y, z = coords
    coord_idx = {"x1":0,"x2":1,"x3":2}

    if not coord:
        return x,y,z  
    else:
        return coords[coord_idx[coord]]



def map_coord_name(var):
    coord_map = {
    'ncx': 'x1', 'ncy': 'x2', 'ncz': 'x3',
    'ccx': 'x1', 'ccy': 'x2', 'ccz': 'x3',
    'ex': 'x1', 'ey': 'x2', 'ez': 'x3',
    'mx': 'x1', 'my': 'x2', 'mz': 'x3',
    'dx': 'x1', 'dy': 'x2', 'dz': 'x3'
    }
    
    return coord_map.get(var, var)

def unmap_coord_name(coord):
    arr_type = guess_arr_type(coord)
    if arr_type:
        if coord.startswith(arr_type) and coord[-1] in ('x', 'y', 'z'):
            return coord

        axis_map = {
            'x1': 'x',
            'x2': 'y',
            'x3': 'z',
        }

        return f"{arr_type}{axis_map.get(coord, coord)}"
    else:
        return coord    

def is_coord(var):
    mapped_var = map_coord_name(var)
    is_coord = True if mapped_var in ("x1","x2","x3") else False

    return is_coord

def pluto_is_written_out(file,chk_time):
    """
    Checks if PLUTO output is fully written as sometimes even if in .dbl.out the cluster hasn't fully written the file 
    """
    if not os.path.exists(file):
        return False
    
    s0 = os.path.getsize(file)
    time.sleep(chk_time)
    s1 = os.path.getsize(file)
    if s1 == s0:
        return True
    else:
        return False
    
def inspect_pluto_h5(file):
    """
    Displays shape and size (in GB) of PLUTO HDF5 file datasets
    """
    with h5py.File(file,"r") as f:
        def dset_info(name,obj):
            if isinstance(obj,h5py.Dataset):
                print(f"{name} size: {obj.nbytes / 1e9:.2f} GB, Shape: {obj.shape}")

        f.visititems(dset_info)

def ergs_to_watt(val):
    conv_unit = u.W
    original_unit = (u.erg / u.s)
    conv_value = original_unit.to(conv_unit)
    return val * conv_value

def gcm3_to_kgm3(val):
    conv_unit = (u.kg/u.m**3)
    original_unit = (u.g / u.cm**3)
    conv_value = original_unit.to(conv_unit)
    return (val * conv_value)

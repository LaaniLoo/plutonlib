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
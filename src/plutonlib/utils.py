import plutonlib.config as pc
import plutonlib.load as pl

import os
import time
import sys
from time import sleep
import importlib
from glob import glob 

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

def list_subdirectories(directory):
    """Return a list of subdirectories in the given directory."""
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def choose_directory(start_dir):
    """Allow the user to select a subdirectory and navigate deeper."""
    current_dir = start_dir

    while True:
        subdirectories = list_subdirectories(current_dir)

        if not subdirectories:
            print('\n')
            print(f"No subdirectories found in {current_dir}.")
            break

        # sleep(0.5)
        print("Running setup_dir...")
        print(f"Starting directory: {current_dir}")
        print('\n')
        sys.stdout.flush()
        print("Subdirectories:")

        for idx, subdir in enumerate(subdirectories, start=1):
            print(f"{idx}: {subdir}")
            sys.stdout.flush()
        
        # Prompt the user to select a subdirectory or exit
        choice = input(f"Select a subdirectory (1-{len(subdirectories)}), or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            print("Exiting...")
            break
        
        try:
            choice = int(choice)
            if 1 <= choice <= len(subdirectories):
                current_dir = os.path.join(current_dir, subdirectories[choice - 1])
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
    
    return current_dir

def setup_dir(start_dir):

    # Let the user select the directory (printing of subdirectories is handled inside the function)
    save_dir = choose_directory(start_dir)

    print(f"Final selected save directory: {save_dir}")

    return save_dir

def sim_type_match(sdata):
    is_jet_2d = sdata.sim_type.split("_")[0] in ("Jet") and sdata.grid_ndim == 2
    is_jet_3d = sdata.sim_type.split("_")[0] in ("Jet") and sdata.grid_ndim == 3
    is_stellar_wind = "_".join(sdata.sim_type.split("_",2)[:2]) in ("Stellar_Wind")
    
    returns = {
        "is_jet_2d":is_jet_2d,
        "is_jet_3d":is_jet_3d,
        "is_stellar_wind":is_stellar_wind,
    }

    return returns
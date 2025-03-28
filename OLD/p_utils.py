import os
import time
import sys
from time import sleep
import importlib


def py_reload(module):
    if isinstance(module,str):
        module_name = module
    else:
        module_name = module.__name__

    module = importlib.import_module(module_name) #, package=None
    importlib.reload(module)

    # print(time.ctime(os.path.getmtime(f"{module_name}.py"))) # Checks last modification time
    print(time.ctime(os.path.getmtime(module.__file__))) # Checks last modification time

def list_subdirectories(directory):
    """Return a list of subdirectories in the given directory."""
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def choose_directory(start_dir):
    """Allow the user to select a subdirectory and navigate deeper."""
    current_dir = start_dir

    while True:
        subdirectories = list_subdirectories(current_dir)

        if not subdirectories:
            print(f"No subdirectories found in {current_dir}.")
            break

        # sleep(0.5)
        print(f"\nCurrent directory: {current_dir}")
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

def select_profile(profiles):
    # sleep(0.5)
    print("Available profiles:")
    for i, profile in enumerate(profiles):
        print(f"{i}: {profile}")
        sys.stdout.flush()

    while True:
        choice = input("Enter the number of the profile you want to select (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            print("Selection cancelled.")
            return None  # Return None if the user quits

        if choice.isdigit():
            choice = int(choice)
            if 0 <= choice < len(profiles):
                # return profiles[choice]
                return choice
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(profiles) - 1}.")
        else:
            print("Invalid input. Please enter a valid number or 'q' to quit.")

def setup_dir():
    # SETTING DIR
    # Specify the initial directory to start navigation
    start_dir = r"/mnt/g/My Drive/Honours S4E (2025)/Notebooks/"

    # Let the user select the directory (printing of subdirectories is handled inside the function)
    save_dir = choose_directory(start_dir)
    print(f"Final selected directory: {save_dir}")

    return save_dir
save_dir = setup_dir()

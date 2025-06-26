
# Features:
* Get data from PLUTO simulations (through the use of [plutokore](https://github.com/pmyates/plutokore))
* Load and convert .dbl files to SI units
* Plot 2D and 3D colour maps, 1D slice plots across all simulation files
* Find peaks for a variable of interest

# Requirements
See `pyproject.toml`.

Some environment variables may be required to load data,
* `PLUTONLIB_START_DIR` (not required): used as starting directory for python output.
  * e.g. `PLUTONLIB_START_DIR='/mnt/g/My Drive/Honours S4E (2025)/Notebooks/'`
* `PLUTO_DIR` (required): master directory of the PLUTO code.
  * e.g. `PLUTO_DIR=/home/alain/pluto-master`

# Installing

## pip package manager (recommended)

Can be installed by cloning the git repo and navigating to it, then installed using the following command.

```
pip install . 
```

You can also use an "editable" install (any changes you make to files under `src/plutonlib` are visible without having to re-install the package) by passing the `-e` flag to `pip install`.

```
pip install -e .
```

# Module overview

* `utils` 
  * Simple setup functions to load/list directories for saving  
* `load`
  * loads all sim data, can convert all data to SI units, functions to load different variable profiles
  * Can output sim info, vars etc
* `config`
  * Used to store the `start_dir` variable which is the starting location for a save point, e.g. navigating outside WSL or to a different drive, 
  * load the PLUTO_DIR env var 
  * contains set profiles and coord systems used by `load`
  * contains all units and variables defined by PLUTO with relevant units and normalization 
* `plot`
  * can plot 2D/3D colour maps, 1D slices for any variable profile or .dbl data file, can also find max values/graphical peaks
* `analysis`
  * peak finding: graphical and numerical
  * plotting peaks
  * energy, density, velocity etc. calculations

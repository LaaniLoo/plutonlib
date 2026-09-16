<pre>
88""Yb  88      88   88 888888  dP"Yb  88b 88 88      88 88""Yb
88__dP  88      88   88   88    dP   Yb 88Yb88 88      88 88__dP
88"""   88  .o  Y8   8P   88    Yb   dP 88 Y88 88  .o 88 88""Yb
88      88ood8  `YbodP'   88     YbodP  88  Y8 88ood8 88 88oodP
</pre>

A passion project for plotting and data analysis of PLUTO simulations inspired by [plutokore](https://github.com/pmyates/plutokore).

# Features:
* Read, write and compress PLUTO HDF5 outputs, convert to si and load full or sliced data (without loading full array, minimising memory impact).
  * Compression and chunking of simulation outputs for faster loading times.
  * Automatic array profile calculations using the grid definitions in the pluto.ini
* Reading pluto.ini and populating dataclasses for quick access to input parameters
* Plot 2D and 3D colour maps, 1D slice plots across all simulation files and variables
  * Produce multi-sim grid plots for paper comparisons
* Surface brightness calculations using the PRAiSE analytic model, saving arrays to HDF5
* Analysis functions for equations of state, jet kinetic power, sound speed, jet angle and length evolution.

# Requirements
See `pyproject.toml`.

# Environment variables
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
* **`analysis`**
  * Grid indexing and slice calculation without loading full arrays
  * Peak finding: numerical maximums and scipy-based detection
  * Time progression tracking (jet radius, length evolution)
  * Physical calculations: energy, density, velocity, EOS, sound speed
  * Jet angle measurement
  * Injection region properties

* **`compression`**
  * Compresses HDF5 simulation outputs with chunking for faster 2D slice access
  * Supports gzip compression with configurable chunk planes (xy, xz, yz)
  * Incremental loading for memory-efficient compression of large files
  * Detailed logging with progress tracking and compression statistics
  * Optional deletion of original files after successful compression

* **`config`**
  * Manages `PLUTO_DIR` environment variable and simulation directory structure
  * Parses `pluto.ini` files for grid setup, user parameters, and output configuration
  * Handles unit definitions and conversions between code/user units for each variable
  * Provides coordinate system mappings for different array types (node/cell coords)

* **`fancy_plot`**
* Produce 'paper-ready' style plots with proper formatting and labels

* **`plot`**
  * **PlotData class:** Manages matplotlib figures, axes, and plotting state
  * Plots 2D/3D colormaps for fluid variables with automatic subplot layouts
  * Creates 1D slice plots across specified coordinate values
  * Generates animated GIFs of simulation evolution
  * Interactive save functionality with custom naming

* **`read_write`**
  * Loads HDF5 simulation data with metadata tracking (`load_fluid_hdf5`, `load_hdf5_metadata`, `load_hdf5_lazy`)
  * Automatic detection of float/double formats and compressed files
  * Converts data from code units to user-specified SI units
  * Handles particle data loading and file output detection

* **`simulation_info`**
  * Automatically initialise and setup `EnvInfo` and `JetInfo` dataclasses by reading the `pluto.ini` file
  * Uses same method to automatically calculate the jet length scales from Krause (2012)
  * Contains all useful units/values of Jet and Env params which can be accessed from `simulations.py` with `simulation.env` or `simulation.jet`

* **`simulations`**
  * **SimulationSetup:** Initializes simulation metadata, directories, and INI file parameters
  * **SimulationData:** Simulation module to quickly load and cache fluid/particle data for a specific simulation object
  * Methods for retrieving variable metadata, grid information, and injection regions
  * Conversion to plutokore simulation objects

* **`splines`**
  * Use image processing and vectors to trace the path along any jet to the edge of its lobe
  * PRAiSE SB array -> image skeletonisation -> initial path -> weighted SB path -> jet splines
  * see `get_jet_splines_sb()`

* **`surface_brightness`**
* Use PRAiSE analytic model and plutokore to generate and raytrace particles -> surface brightness
* Save these arrays to hdf5 (`save_sb_hdf5`) on a per-sim basis with structure /sim_dir/sbdata.h5
  * h5 structure: redshift/output/angles/sb e.g; `0.02/500/[0,0,0]/sb`

* **`utils`**
  * Module reloading utilities
  * Coordinate name mapping and conversion helpers (`map_coord_name`, `get_coord_names`, `guess_arr_type`)
  * Unit conversion helpers (ergs to watts, g/cm³ to kg/m³)

# Setting and converting PLUTO units (`plutonlib/units`)
* ini files are used to define a set of `code_unit_values` and `usr_unit_values`.
  * `code_unit_values` are the unit values that PLUTO outputs for each variable (as seen in the user guide)
  * `usr_unit_values` are the desired unit values that you wish to convert to later.

#### **Default pluto_units.ini**
```
[code_unit_values]
x1 = 1.496e13*cm
x2 = 1.496e13*cm
x3 = 1.496e13*cm
rho = 1.673e-24*g/cm**3
prs = 1.673e-14*dyne/cm**2
vx1 = 1.0e5*cm/s
vx2 = 1.0e5*cm/s
vx3 = 1.0e5*cm/s
sim_time = 4.744*yr
mass = 1.0*g 
energy = 1.0*erg  
T = 1.203e2*K  

[usr_unit_values]
x1 = 1*m
x2 = 1*m
x3 = 1*m
rho = 1*kg/m**3
prs = 1*Pa
vx1 = 1*m/s
vx2 = 1*m/s
vx3 = 1*m/s
sim_time = 1*yr
mass = 1*kg
energy = 1*J
T = 1*K
```

# Python scripts for analysis (`plutonlib/scripts`)
* **`compression_script.py`**
  * A python script that automatically compresses and chunks PLUTO HDF5 data files using the functions from `compression.py`
  * Can optionally delete uncompressed files
  * logs the compression process with timestamps

* **`praise/praise_hdf5.py`**
  * Run a pbs job on cluster to calculate multiple sets of PRAiSE calculations in parallel
    * Assuming each calculation takes ~30Gb of memory it can run 28 calculations at once with 840Gb of memory, this value can be lowered in `surface_brightness.save_sb_hdf5` -> `pu.setup_workers(task_req_mem=30)`
  * Requires a config setup using `praise_setup.yml`, using `angles` = "all" calculates a massive set of 100 angles

* **`praise/hybrids_grid.py`**
  * Generates a massive grid of surface brightness plots, useful when comparing orientation effects of PRAiSE calculations
  * Outputs into `hybrids_grid` directory with subdirs `output/frequency`, inside each freq is up to 10 pdfs containing 10 X 10 plots for each angle.

# Bash scripts/PLUTO tools (`plutonlib/pluto_utils`)
* **`sim_setup.sh`**
  * Automatically creates a PLUTO simulation directory copying a handful of useful scripts and ini file templates.
  * Runs a setup script that automatically copies the PLUTO source into the simulation directory creating a portable instance
    * PLUTO's `setup.py` can then be edited from here to change desired parameters.
  * Automatically builds/compiles PLUTO on local or PBS cluster setups.

* **`pluto_run.sh`** 
  * Script that creates and executes a simulation run within a simulation directory for a given ini file on local and PBS clusters
  * Automatically copies `compression_script.py` to relevant run directory
    * For a PBS cluster, the simulation will be submit though `job_submit.sh` passing in all relevant variables.
  * Creates a `run_dir` containing the pluto logs as well as a copy of the ini file and PBS job id stored in `run_dir/job_info`
  * Displays the first 200 lines and tail of the PLUTO logfile.
  * For a cluster with run name being set by the ini file (-a): `./pluto_run.sh -c -i ./Q36_v01_a25.ini -a`


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
* Plots are grouped by simulation, use `.fig` attribute to show plot
* file saving defaults to a rasterised pdf (`fname='name'` saves to `sim.save_dir` e.g. `$SAVE_DIR/name.pdf`)
* Common kwargs: `xlim`, `ylim`, `vmin`, `vmax`, `fig_size`, `plane`, `row_len`, `rotate_row`, `query_points`
* `fluid`: pcolormesh plot of simulation fluid variable e.g. 'rho'
  * `fluid(sim_dict={sim5:[50,75]},var='rho').fig`
* `jet_splines`: scatter plot of PLUTO particles for a variable with tracer spline fitting, at the particle output matching each sim time
  * `jet_splines(sim_dict={sim5:[50]},var='tr1').fig`
* `surf_brightness`: pcolormesh plot of PRAiSE particle emisivity
  * `surf_brightness({sim5:[50]},{sim5:angles},freq,0.02,).fig`
  * `freq` must be a single value (not a list), one subplot per output per angle set
* `surf_brightness_splines`: as `surf_brightness`, with the jet spline and injection point overlaid
  * `surf_brightness_splines({sim5:[50]},{sim5:angles},freq,0.02,params).fig`
* `spectral_idx`: pcolormesh plot of spectral index between a pair of frequencies
  * `spectral_idx({sim5:[50]},{sim5:angles},[freq1,freq2],0.02).fig`
* `fluid_1D_tr_spl`: 1D profile of a fluid variable along the jet using tracer splines
  * `fluid_1D_tr_spl(sim5,var='rho',grid_output=50,tick_axis='x').fig`
* `surface_brightnes_1D`: 1D surface brightness profile along the jet
  * `surface_brightnes_1D(sim5,grid_output=50,angles=angles,freq=freq,redshift=0.02).fig`
* `animation_fluid`: gif of a fluid variable over all valid outputs, saved to `sim.save_dir`
  * `animation_fluid(sim5,var_name='rho',fps=20)`

* **`plot_data`**
  * **PlotData class:** stores plotting state, used to track plotting variables across the plot functions
  * Holds the matplotlib `fig`, `axes` and `gs` (gridspec), plus the current `plot_idx`, `output`, `var_name`, `im` and `label`
  * Holds the plot settings from `setup_subplots`: `xlim`, `ylim`, `vmin`, `vmax`, `fig_size`, `plane`, `arr_type`, `rotate_row`
  * Returned by every `fancy_plot` function, so `.fig` shows the plot and the other attributes can be inspected or reused
  * Created with `PlotData(arr_type=arr_type, plane=plane)`, users don't normally need to build it directly

* **`read_write`**
  * Loads HDF5 simulation data with metadata tracking (`load_fluid_hdf5`, `load_hdf5_metadata`, `load_hdf5_lazy`)
  * load particle data from HDF5 using `load_particles_hdf5`
  * load surface brightness data from HDF5 using `load_sb_hdf5`
  * Automatic detection of float/double formats and compressed files
  * Converts data from code units to user-specified SI units
  * Handles particle data loading and file output detection

* **`simulations`**
  * **SimulationSetup:** Initializes simulation metadata, directories, and INI file parameters
  * **SimulationData:** Simulation module to quickly load and cache fluid/particle data for a specific simulation object
    * See `load_fluid_data()`, `load_particle_data()`, `load_sb_data()` etc...
  * Methods for retrieving variable metadata, grid information, and injection regions
  * Conversion to plutokore simulation objects
  * E.g. `sim5 = ps.SimulationData(ini_file="jet_units",rel_path="Jet_mvinj/Q36_v01_a25_wx064z064_eox58z58")`

* **`simulation_report`**
  * Quickly produce a report for a given simulation -> `SimulationReport.from_sim(sim)`
  * Prints simulation, grid, output file (including compression) and particle info
  * Currently WIP

* **`simulation_info`**
  * Automatically initialise and setup `EnvInfo` and `JetInfo` dataclasses by reading the `pluto.ini` file
  * Uses same method to automatically calculate the jet length scales from Krause (2012)
  * Contains all useful units/values of Jet and Env params which can be accessed from `simulations.py` with `simulation.env` or `simulation.jet`

* **`splines`**
  * Two methods of tracing jet path: tracer and surface brightness
  * Tracer:
    * Uses x,z vector weighted by tracer values within the jet radius
    * Accurate inner jet, cannot trace lobe
    * See `get_jet_splines_tr()`
  * Surface Brightness:
    * Use image processing and vectors to trace the path along any jet to the edge of its lobe
    * PRAiSE SB array -> image skeletonisation -> initial path -> weighted SB path -> jet splines
    * Less accurate inner jet, can trace lobe
    * See `get_jet_splines_sb()`

* **`surface_brightness`**
* Use PRAiSE analytic model and plutokore to generate and raytrace particles -> surface brightness
* Save these arrays to hdf5 (`save_sb_hdf5`) on a per-sim basis with structure /sim_dir/sbdata.h5
  * h5 structure: redshift/output/angles/sb e.g; `0.02/500/[0,0,0]/sb`

* **`utils`**
  * Module reloading utilities
  * Coordinate name mapping and conversion helpers (`map_coord_name`, `get_coord_names`, `guess_arr_type`)
  * Unit conversion helpers (ergs to watts, g/cm³ to kg/m³)

* **`pbs_job`**
  * Submit PBS jobs to the cluster (kunanyi) from a local computer via a python wrapper, e.g. running `praise_hdf5.py` to calculate surface brightness data
  * `python praise_hdf5.py -c --job_length 8 --cpus 28 --memory 128`
  * `init_script_dirs`: copies the calling script (plus any extra `files`) to `<wdir>/<script>_files` locally, or on the cluster over scp if `cluster=True`
  * `create_run_script`: builds the PBS script (resources, walltime, modules, conda env, `cmd`)
  * `submit_run_script`: submits the script over ssh with `qsub` and prints the job info via `check_job`
  * Input args:
    * `--cmd`: command to run on the cluster (required)
    * `--job-name`: PBS job name, also names the `<job-name>_log` error file (required)
    * `--job-length`: walltime in hours (required)
    * `--nodes`: number of nodes (required)
    * `--cpus`: cpus per node, defaults to 28
    * `--memory`: memory per node in GB, defaults to None; values above 128 automatically use the `LARGE_MEM` queue

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
  * Requires a config setup using `praise_setup.yml`, using `angles` = "all" calculates a massive set of 1000 angles

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


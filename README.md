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

* **`morphology`**
	* **WIP** functions to determine morphology of jets e.g FR morphology of jets using surface brightness splines in `get_fr_idx_spl`

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

* **`simulation_info`**
  * Automatically initialise and setup `EnvInfo` and `JetInfo` dataclasses by reading the `pluto.ini` file
  * Uses same method to automatically calculate the jet length scales from Krause (2012)
  * Contains all useful units/values of Jet and Env params which can be accessed from `simulations.py` with `simulation.env` or `simulation.jet`

* **`simulation_report`**
  * Quickly produce a report for a given simulation -> `SimulationReport.from_sim(sim)`
  * Prints simulation, grid, output file (including compression) and particle info
  * Currently WIP

* **`simulations`**
  * **SimulationSetup:** Initializes simulation metadata, directories, and INI file parameters
  * **SimulationData:** Simulation module to quickly load and cache fluid/particle data for a specific simulation object
    * See `load_fluid_data()`, `load_particle_data()`, `load_sb_data()` etc...
  * Methods for retrieving variable metadata, grid information, and injection regions
  * Units automatically converted to user units using `usr_units_ini`(see `user_units.ini`)
  * Conversion to plutokore simulation objects
  * E.g. `sim5 = ps.SimulationData(rel_path="Jet_mvinj/Q36_v01_a25_wx064z064_eox58z58")`

* **`splines`**
  * Two methods of tracing jet path: tracer and surface brightness
  * Tracer:
    * Uses x,z vector weighted by tracer values within the jet radius
    * Accurate inner jet, cannot trace lobe
    * See `get_jet_splines_tr()`
  * Surface Brightness:
    * Use a lowest cost map using `cost = length_penalty + (1-normalised_sb)^exponent` this gives the cost to each surface brightness value form injection point
    *  The first endpoint is selected from the `path_cost` map, as the furthest bright pixel away from the injection region. The lowest cost path is then drawn from injection to `ep1` 
    * The second endpoint is then selected using the same bright-far score, where it must also be separated from `ep1` by some `angle_cut` (reduces early path branching). 
    * A second cost map is defined not where path 1 is (plus some width) with a shared region around the injection zone. 
    * The second path is then traced from injection though a bright region (further reducing early branching), then from the bright region to the `ep2`.
    * Less accurate inner jet, can trace lobe
    * See `get_jet_splines_sb()`

* **`surface_brightness`**
	* Use PRAiSE analytic model and plutokore to generate and raytrace particles -> surface brightness
	* Save these arrays to hdf5 (`save_sb_hdf5`) on a per-sim basis with structure /sim_dir/sbdata.h5
	  * h5 structure: redshift/output/angles/sb e.g; `0.02/500/[0,0,0]/sb`

* **`units`**
	* Each unit e.g. time, density, pressure is either defined or derived in the `UnitDefinitions` class. Defaults to defining a unit `length` and `velocity` with `mu_norm = 0.60364`
	unit metadata now comes from `PlutoUnits.from_definitions` where individual units are fetched from `UnitDefinitions` or from `definitions.h` (NOT IMPLEMENTED), each var is then converted to the values in `user_ini_path` (defaults to `plutonlib/src/plutonlib/user_units.ini`)

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

# Setting and converting PLUTO units (`plutonlib/src/plutonlib/user_units.ini`)
* ini files are used to define a set of `usr_unit_values`.
  * `usr_unit_values` are the desired unit values that you wish to convert to later.

#### **Default user_units.ini**
```
[usr_unit_values]
x1 = 1*kpc
x2 = 1*kpc
x3 = 1*kpc
rho = 1*kg/m**3
prs = 1*Pa
vx1 = 1*c
vx2 = 1*c
vx3 = 1*c
sim_time = 1*Myr
mass = 1*kg
energy = 1*J
T = 1*K
Q = 1*watt
tr1 = 1
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

# Python scripts/PLUTO tools (`plutonlib/pluto_utils`)
* **`pluto_setup.py`**
  * Automatically creates PLUTO simulation directories from `sim_config.yml`, copying a handful of useful scripts and ini file templates
    * Each simulation type (`sim_type`) gets a directory in `simulations_root` holding one shared PLUTO source/build, plus a directory for each entry in its `runs`
  * Runs jet-setup's `setup-problem` which copies the PLUTO source into the simulation directory creating a portable instance
    * Skipped if `sim_type/src` already exists, so adding a run to an existing `sim_type` never overwrites the source or build
    * `defs` and `defs_usr` in `sim_config.yml` are written straight into `definitions.h` and `definitions_usr.h`, so no menu interaction with PLUTO's `setup.py` is needed
  * Copies each run's `ini_file` to `run_info/<run_name>.ini` (an existing ini is never overwritten, delete it to re-copy)
  * Automatically builds/compiles PLUTO using the yml `modules` and records the settings used in `sim_type/build_info.yml`
  * Where it runs is detected automatically:
    * From a local machine: everything is created/built on the cluster over `ssh`/`scp` (needs a `kunanyi` host in `~/.ssh/config`, set by `HOST` in `pluto_setup.py`)
    * On the cluster: everything is done directly, no ssh
    * `-l`/`--local`: create and build on the machine you are on, no ssh
  * Options: `--no-build` (skip compiling), `--skip-update` (skip editing the definitions files), `-y` (alternate yml)
  * Example: `python pluto_setup.py --no-build --skip-update`

* **`pluto_run.py`**
  * Script that prepares and submits a single run from `sim_config.yml` within a simulation directory, selected with `--sim-type` and `--run-name`
    * **When run from inside the `sim_type` directory**, the `--sim-type` and the yml fields are automatically filled with the directory name and its local `sim_config.yml`(input args overwrite this)
  * Checks the jet-setup source exists and creates the run directories/ini from the yml entry if they are missing
  * Compares `build_info.yml` with `pluto_config` and `modules` in the yml, if they differ the definitions are re-patched and PLUTO is rebuilt
  * Sets `log_dir` and `output_dir` in the run's ini to the run (or remnant) directories, all other ini lines can be edited by hand
  * Unless `--local`, creates and submits the PBS job through `plutonlib.pbs_job` and prints the job id and `qstat` summary
    * Runs `mpirun ... ./pluto -i <run_name>.ini -maxtime <job_length - 1.5>`, 1.5 hours before the walltime so PLUTO exits cleanly
    * If `compression: True` in the job config, `compression_script.py` is copied to the run directory and run alongside PLUTO
  * Stores `run_<job_id>.yml` in `run_info` containing the submit time, PBS job info and the build info the run was submitted with, the PBS `.o`/`.e` files are also written there
  * Restarts: set `restart: [True, n]` in the yml or pass `-r n` to restart from output `n` (`-h5restart n`)
  * Example: `python pluto_run.py --sim-type hd_jet --run-name Q36_v01_a25`
  * Example (inside `hd_jet`): `python pluto_run.py --run-name Q36_v03_a15 -r 5`

#### **Simulation directory structure**
```
simulations_root/
└── sim_type/                      # shared jet-setup + PLUTO build
    ├── src/, pluto, build_info.yml, sim_config.yml, pluto_setup.py, pluto_run.py ...
    └── run_name/
        ├── log/                   # PLUTO logs
        ├── run_info/              # <run_name>.ini, run_<job_id>.yml, PBS .o/.e files
        └── remnant/               # same layout, used when remnant: True
```

#### **Default sim_config.yml**
```yaml
simulations_root: "~/pluto-master/Simulations/" #root dir of all simulations, ~ expands to the cluster home when running via ssh
modules: ["HDF5","OpenMPI/4.1.5-GCC-12.3.0-pbs"] #modules loaded to build and run PLUTO

simulations:
  yml_test: #sim type directory, each with its own jet-setup and PLUTO build shared by all its runs e.g MHD_jet/run_name
    pluto_config: 'cluster_hd' #build config (below)
    runs:
      Q36_v01_a25_wx064z064_eox58z58: #run name, directory relative to simulations_root/sim_type
        ini_file: "./pluto_template.ini" #ini copied to run_info/run_name.ini, relative paths are from the yml's folder, null = supply your own
        job_config: 'normal' #pbs job config (below)
        restart: [False,0] #[restart the run?, output number to restart from]
        remnant: False #True = ini, logs and output go to run_name/remnant

job_config:
  normal:
    job_length: 100 #walltime in hours, PLUTO -maxtime is set 1.5 hours earlier
    nodes: 70
    cpus: 28 #cpus per node
    compression: True #run compression_script.py alongside PLUTO

pluto_config:
  cluster_hd:
    makefile: "Kunanyi.gcc.defs" #makefile.defs in PLUTO_DIR/Config
    jet_setup_path: '~/jet-setup'
    defs: #definitions.h entries, only list what you want to change
      PHYSICS: "HD"
      DIMENSIONS: 3
      GEOMETRY: "CARTESIAN"
      EOS: "IDEAL"
      TIME_STEPPING: "RK2"
      RECONSTRUCTION: "LINEAR"
      PARTICLES: "PARTICLES_LP"
    defs_usr: #definitions_usr.h entries, quote YES/NO so yaml doesn't read them as booleans
      DENSITY_PROFILE: "ENV_KING"
      UNIQUE_JETS: "YES"
      JET_INJECTION_ZONE: "SPHERICAL_INJECTION_ZONE"
      JET_INJECTION_CAP: "YES"
      JET_RELATIVISTIC_DENSITY: "NO"
      PARTICLES_LP_RAISE: "YES"
```


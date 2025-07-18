# Changelog

All notable changes to this project will be documented in this file.

## [0.5.1] - 2025-07-16

### 🐛 Bug Fixes


- *(load.py)* Propper hdf5 loading from pk, new arr_type for profiles, ini_file in sdata, ThreadPoolExecutor

* Added full hdf5 support, similar to plutokore, requires arr_type to load different grid array types (see SimulationData).
  * Can now add own .ini file for units into SimulationData as ini_file (without extension, needs to be in /src).
  * Added ThreadPoolExecutor in pluto_conv for faster converts.


### Other


- *(general)* update CHANGELOG.md


- *(general)* update CHANGELOG.md


- *(general)* test propper hdf5 loading with arr_type


- *(general)* try different pcmesh_3d_nc


- *(general)* try different pcmesh_3d_nc


- *(general)* try different pcmesh_3d_nc


- *(general)* try different pcmesh_3d_nc


- *(general)* try transpose xz_jet data


- *(general)* pre commit before load optimisation


## [v0.5.0] - 2025-07-02

### Documentation


- *(general)* update changelog



### 🚀 Features


- *(config.py)* added custom units as pluto_units.ini

default location is in config.py, will update to sdata attr ini_path_default = ...

- *(plot.py)* added 3D plotting, cleaned up plotting functions and use cases

added pcmesh 2/3D functions for wider use cases

- *(load.py)* fix load, can load individual d_files, added data chaching,  faster

can load only specific files with load_outputs (tuple), added lru_cache for caching pluto_loader data


### 🐛 Bug Fixes


- *(all)* fix circular import error

pu.bcolors was causing a circular error in config.py, created colours.py to fix

- *(general)* attempt to fix hdf5


- *(general)* hdf5 fix, plutkore edit


- *(general)* cluster-slowness


- *(general)* import select d_files with load_outputs


- *(general)* plot.py loading all d_files


- *(general)* plotting when loading select d_files


- *(general)* profile-loading functions load all d_files


- *(general)* attempt to fix 3D plotting of jet


- *(general)* plotting correct d_files


- *(general)* can load_outputs as list


- *(general)* added pluto_units.ini


- *(general)* cach loaded data for speedup


- *(general)* cache loaded data for speedup


- *(general)* cache loaded data for speedup


- *(general)* can now load both dbl and h5


- *(general)* try np.meshgrid() for cluster plot


- *(general)* try np.meshgrid() for cluster plot


- *(general)* simple 3d plot capability


- *(general)* simple 3d plot capability


- *(general)* added Jet_Cluster sim type


- *(general)* added Jet_Cluster sim type


- *(general)* removed Jet_Cluster, added dim check


- *(general)* diff'd plot cases


- *(general)* diff'd plot cases


- *(general)* diff'd plot cases


- *(general)* diff'd plot cases


- *(general)* diff'd plot cases


- *(general)* diff'd plot cases


- *(general)* diff'd plot cases


- *(general)* diff'd plot cases


- *(general)* diff'd plot cases


- *(general)* diff'd plot cases


- *(general)* diff'd plot cases


- *(general)* diff'd plot cases


- *(general)* fixed pu.py_reload_all()


- *(general)* subplot_base 3d case


- *(general)* subplot_base 3d case


- *(general)* attempt to fix plotting same var on both subplots



### Other


- *(general)* version 0.4.3 → 0.4.4


- *(general)* doc cleanup and misc changes


- *(general)* debug slow pluto_loader for cluster


- *(general)* attempt to get plot.py to work on cluster


- *(general)* test fix for 3D plots


- *(general)* slice for 3D plotting


- *(general)* plot jet with profile


- *(general)* diff slice


- *(general)* diff slice


- *(general)* diff slice


- *(general)* diff slice


- *(general)* diff slice


- *(general)* diff slice


- *(general)* diff slice


- *(general)* test commit before tag


- *(general)* meshgrid


- *(general)* doc cleanup and misc changes


- *(general)* version 0.4.4 → 0.5.0


## [v0.4.3] - 2025-05-19

### Documentation


- *(general)* update changelog



### 🐛 Bug Fixes


- *(config.py)* error handling

Added some error handling for env vars and required directories


### Other


- *(general)* doc cleanup and misc changes


- *(general)* version 0.4.2 → 0.4.3


## [v0.4.2] - 2025-05-13

### Documentation


- *(general)* update changelog


- *(general)* update changelog



### 🐛 Bug Fixes


- *(plot.py)* fixes and kwargs

Added vmin and vmax to fix plotting vxx, note cuttoffs are given (to be changed)  makes finer details easier to see.
  Added `fig_resize` and `save_ovr`  kwargs,`save_ovr = 3` skips the `plot_save` user input.

- *(analysis.py)* fixes, plot_troughs

Plotting `plot.sim()` with jet velocity now easier to see, can plot troughs from `plot.plotter` with `plot.plot_troughs`.
  Improved peak/trough values in `all_graph_peaks`, uses numpy filtering with thresh

- *(load.py)* added warnings, sdata.d_sel

Added `sdata.d_sel(slice,start)` in `SimulationData`, used to quickly slice d_files


### Other


- *(general)* doc cleanup and misc changes


- *(general)* version 0.4.1 → 0.4.2


## [v0.4.1] - 2025-04-30

### Documentation


- *(general)* update changelog


- *(general)* update changelog


- *(general)* update changelog



### 🚀 Features


- *(analysis.py)* log-log for plot_time_prog, fixes

* can now add type = "log" into plot_time_prog, will produce a log-log plot with the calculated slope.
  * cleaned up analysis functions, can now use Jet for calculations, removed pdata dependencies.

- *(load.py)* SimulationData added get_all_vars,get_coords,fixes/warnings

* get_all_vars: auto loads profile = "all"
  * get_coords: gets a dict of just x-arrays @d_last or d_file
  * original warnings from pluto_loader about non_vars now only prints for __init__


### 🐛 Bug Fixes


- *(config.py)* fixed start_dir

start_dir can be set as an env var or plutonlib_output folder will be created in current wd

- *(plot.py)* bug fixes and error handling

fixed bug where sel_prof would run profile select function. Added some basic error handling in plot helpers to check data is formatted correctly

- *(test)* test

test

- *(load.py)* added subdir_name, fixes

Added subdir_name in SimulationData,
  * looks for a subdir with name in pc.start dir, if it doesn't exist -> prompt to create a subdir with name. If subdir_name is none, run_name is used with pc.setup_dir.
   **Note SimulationData now uses run_name not run**

- *(load.py)* added custom subdir in SimulationData, fixes

Subdir_name in init, joins name with pc.start_dir, if not a valid subdir, will prompt to create.If none -> runs pc.setup_dir(pc.start_dir).
   **run is now run_name in SimulationData**

- *(plot.py)* plotter custom slice, save types

Can now plot at custom idx or at specific value -> `idx` and `value` are kwargs that tell pa.calc_var_prof to use the index value or find the index at where the array value occurs.

  plot_save now has `file_type` kwarg defaults to png.

   **Deprecated having mutliple sets of coords/vars**

- *(analysis.py)* added find_nearest() and custom var_prof slices

See plot.py commit, calc_var_prof now takes kwargs `idx` and `value` used to get array slice at specific idx or find closest idx for a given value


### Other


- *(general)* doc cleanup and misc changes


- *(general)* doc cleanup and misc changes


- *(general)* version 0.4.0 → 0.4.1


## [v0.3.0] - 2025-04-22

### 🚀 Features


- *(load.py)* add SimulationData class

added SimulationData class: used to load converted,raw and units data from pluto_loader, pluto_conv and get_pluto_units

- *(analysis.py)* added analysis.py

added an analysis.py, contains: calc_var_prof, all peak finding functions, plot_time_prog. Now with 4 new functions: calc_energy,calc_radius,calc_radial_vel and calc_density


### 🐛 Bug Fixes


- *(plot.py)* changed PlotData class

PlotData is now used for storing d_files from sdata.d_files, sel_var,sel_coord,plotting vars as pdata.vars (assigned from sdata.get_vars(d_file), and figs etc


### Other


- *(general)* doc cleanup and misc changes


- *(general)* version 0.2.5 → 0.3.0


## [v0.2.5] - 2025-04-11

### Documentation


- *(general)* update changelog for v0.2.4


- *(general)* update changelog for v0.2.4



### 🐛 Bug Fixes


- *(config.py/load.py)* changed normalisation and conversion of units

Added the two functions: get_pluto_units() and value_norm_conv() into config.py. This replaces CGS_code_units in load.py, get_pluto_units() returns a dict like CGS_.. but now uses keys instead of indexes e.g. `"x1": {"norm": 1.496e13, "cgs": u.cm, "si": u.m, "var_name": "x1", "coord_name": f"{sel_coords[0]}"}`. value_norm_conv() does all the conversion that pluto_conv did but can handle any np array as raw_data, and can convert the norm values into si or cgs to avoid unit-conversion errors.


### Other


- *(general)* version 0.2.4 → 0.2.5


## [v0.2.4] - 2025-04-08

### Documentation


- *(CHANGELOG.md)* fixed changelog



### 🐛 Bug Fixes


- *(plot.py/load.py)* bug fixes and optimisation

* used ThreadPoolExecutor for load.py, restructured how arrays are loaded for speed\n * fixed axis bugs in plot.py\n * all plot functions now use PlotData() class for loading and handling vars


### Other


- *(general)* version 0.2.3 → 0.2.4


## [v0.2.3] - 2025-04-07

### 🐛 Bug Fixes


- *(load.py)* added sel_prof in pluto_load_profile()

can load a specific profile rather than selecting via user input, now displays profiles that only contain plottable vars, e.g. no x3 for Jet in cyl


### Other


- *(general)* version 0.2.2 → 0.2.3


## [v0.2.2] - 2025-04-04

### 🐛 Bug Fixes


- *(Changelog)* using git-cliff



### Other


- *(general)* version 0.2.1 → 0.2.2


## [v0.2.1] - 2025-04-04

### 🐛 Bug Fixes


- *(Plotting)* fixed labeling of xy and titles

added different title assignment for sim_types in plot_extras(), "stellar_wind" has its title element as a list where: title_other = [[title_L,title_R]]


### Other


- *(general)* version 0.2.0 → 0.2.1


## [v0.2.0] - 2025-04-03

### 🚀 Features


- *(plotting)* All simulation plots can now be done via plutonlib.plot.plot_sim()

type of plot\colourmap is determined in cmap_base

- *(Plotting/Analysis)* added graph_peaks() to find graphical peaks with scipy

is buggy for stellar_wind, is used with peak_findr() to plot radius increase across time with plot_jet_prog()


### 🐛 Bug Fixes


- *(Plotting)* simplified plotting, all done with plot_sim() now

sim_type determines which branch of cmap_base() to add into plot_sim(), added subplot_base() (axis,fig), plot_label(), plot_save() as helper functions


### Other


- *(general)* version 0.1.1 → 0.2.0


## [v0.1.1] - 2025-04-01

### 🚀 Features


- *(general)* added  PlotData class for easier plotting, fix: removed some plotting functions, now plot_cmap_2d -> plot_jet_profile


## [v0.0.1] - 2025-03-28
<!-- generated by git-cliff -->

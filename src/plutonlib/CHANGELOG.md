## v0.3.0 (2025-04-22)

### BREAKING CHANGE

- replaces PlotData, all plots etc now take SimulationData as sdata instead of pdata

### Feat

- **analysis.py**: added analysis.py
- **load.py**: add SimulationData class

### Fix

- **plot.py**: changed PlotData class

## v0.2.5 (2025-04-11)

### Fix

- **config.py/load.py**: changed normalisation and conversion of units

## v0.2.4 (2025-04-08)

### Fix

- **plot.py/load.py**: bug fixes and optimisation

## v0.2.3 (2025-04-07)

### Fix

- **load.py**: added sel_prof in pluto_load_profile()

## v0.2.2 (2025-04-04)

### Fix

- **Changelog**: using git-cliff

## v0.2.1 (2025-04-04)

### Fix

- **Plotting**: fixed labeling of xy and titles

## v0.2.0 (2025-04-03)

### BREAKING CHANGE

- all previous plot_...() functions have been replaced by plot_sim()

### Feat

- **Plotting/Analysis**: added graph_peaks() to find graphical peaks with scipy
- **plotting**: All simulation plots can now be done via plutonlib.plot.plot_sim()
- find graphical peaks w graph_peaks(), used with peak_findr() to plot radius against time w plot_time_prog()

### Fix

- **Plotting**: simplified plotting, all done with plot_sim() now

## v0.1.1 (2025-04-01)

### Feat

- added  PlotData class for easier plotting, fix: removed some plotting functions, now plot_cmap_2d -> plot_jet_profile

## v0.1.0 (2025-03-30)

## v0.0.1 (2025-03-28)

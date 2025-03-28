
# Features:
* Get data from PLUTO simulations (through the use of plutokore)
* Plot 2D and 3D colour maps, 1D slice plots across all simulation files
* Find peaks for a variable of interest

# Requirements
See `pyproject.toml`.

# Installing

## pip package manager (recommended)

Can be installed by cloning the git repo and navigating to it, then installed using the following command.

```
pip install .
```

You can also use an "editable" install (any changes you make to files under `src/plutonlib` are visible without having to re-install the package) by passing the `-e` flag to `pip install`.

# Module overview

* `p_utils` - Simple setup functions to load the PLUTO_DIR var as well as set up a save environment
* `p_funcs` - loads all sim data, can plot and find peaks

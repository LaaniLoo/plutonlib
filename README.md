
<pre>
88""Yb  88      88   88 888888  dP"Yb  88b 88 88      88 88""Yb
88__dP  88      88   88   88    dP   Yb 88Yb88 88      88 88__dP
88"""   88  .o  Y8   8P   88    Yb   dP 88 Y88 88  .o 88 88""Yb
88      88ood8  `YbodP'   88     YbodP  88  Y8 88ood8 88 88oodP
</pre>

A passion project for plotting and data analysis of PLUTO simulations inspired by [plutokore](https://github.com/pmyates/plutokore).

# Features:
* Read and write PLUTO HDF5 outputs, convert and load their data with minimal memory impact.
  * Compression and chunking of simulation outputs for faster loading times.
  * Automatic array profile calculations using the grid definitions in the pluto.ini
* Reading pluto.ini for quick access to input parameters
* Plot 2D and 3D colour maps, 1D slice plots across all simulation files and variables
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
* **`utils`**
  * Module reloading utilities
  * Coordinate name mapping and conversion helpers (`map_coord_name`, `get_coord_names`, `guess_arr_type`)
  * Unit conversion helpers (ergs to watts, g/cm³ to kg/m³)

* **`load`**
  * Loads HDF5 simulation data with metadata tracking (`pluto_loader_hdf5`, `load_hdf5_metadata`, `load_hdf5_lazy`)
  * Automatic detection of float/double formats and compressed files
  * Converts data from code units to user-specified SI units
  * Handles particle data loading and file output detection

* **`config`**
  * Manages `PLUTO_DIR` environment variable and simulation directory structure
  * Parses `pluto.ini` files for grid setup, user parameters, and output configuration
  * Handles unit definitions and conversions between code/user units for each variable
  * Provides coordinate system mappings for different array types (node/cell coords)

* **`simulations`**
  * **SimulationSetup:** Initializes simulation metadata, directories, and INI file parameters
  * **SimulationData:** Simulation module to quickly load and cache fluid/particle data for a specific simulation object
  * Methods for retrieving variable metadata, grid information, and injection regions
  * Conversion to plutokore simulation objects

* **`plot`**
  * **PlotData class:** Manages matplotlib figures, axes, and plotting state
  * Plots 2D/3D colormaps for fluid variables with automatic subplot layouts
  * Creates 1D slice plots across specified coordinate values
  * Generates animated GIFs of simulation evolution
  * Interactive save functionality with custom naming

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

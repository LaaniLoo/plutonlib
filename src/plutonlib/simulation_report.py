import os
import glob 
import math 

from dataclasses import dataclass, field
from typing import List

import plutonlib.read_write as prw
import plutonlib.config as pc

# Coordinate arrays stored alongside the fluid variables in the HDF5 files
COORD_KEYS = {f"{prefix}{axis}" for prefix in ("nc", "cc") for axis in "xyz"}

def count_compressed(wdir) -> int: #TODO put in prw 
    return len(glob.glob(os.path.join(wdir, "data.*.compressed")))-1


#---Helpers (module level, each returns fields for the report)---#
def _grid_info(sim) -> dict:
    md = sim.get_metadata()                      # last output, cached on sim
    keys = list(md.dataset_paths)                # opens the file once, cached on md
    return {
        "n_grid_outputs": prw.get_file_outputs(sim.wdir),  
        "dtype": md.dtype,
        "geometry": md.geometry,
        "last_sim_time": md.sim_time,            # user units after get_metadata
        "time_unit": str(sim.units.sim_time.usr_uv),
        "variables": [k for k in keys if k not in COORD_KEYS],
        "coord_vars": [k for k in keys if k in COORD_KEYS],
        "is_conv": sim.conv,
    }

def _grid_summary(sim) -> dict:
    gs = sim.grid_setup
    md = sim.get_metadata()
    ndim = int(gs["dimensions"])

    axes = {}
    for i in range(1, ndim + 1):
        g = gs[f"x{i}-grid"]
        axes[f"x{i}-grid"] = {
            "extent": tuple(float(v) for v in g["grid_extent"]),
            "n_cells": int(sum(g["patch_cells"])),
            # one patch -> plain string, several patches -> keep the list
            "type": g["type"][0] if g["n_patches"] == 1 else list(g["type"]),
            "dx": float(g["dx"]) if "dx" in g else None,
        }

    shape = tuple(int(n) for n in gs["arr_shape"])
    n_cells = math.prod(shape)                           # plain ints, no overflow

    n_vars = len([k for k in md.dataset_paths if k not in COORD_KEYS]) + ndim #additional vars from each coordinate
    precision = 4 if md.dtype.startswith("flt") else 8   # bytes per value
    size_gib = n_cells * n_vars * precision / 1024**3    # coords are tiny 1D arrays, ignored

    return {
        "grid_ndim": ndim,
        "grid_shape": shape,
        "grid_n_cells": n_cells,
        "grid_axes": axes,
        "grid_dxyz": tuple(a["dx"] for a in axes.values()),
        "est_size_gib": size_gib,
        "size_n_vars": n_vars,
        "size_precision": precision,
    }

def _compression_info(sim) -> dict:
    md = sim.get_metadata()

    if md.is_compressed:
        comp_filepath = sorted(glob.glob(os.path.join(sim.wdir, f"data.*.{md.dtype}.compressed")))[1] #not the 0th file as it gets more compressed than others
        fsize_compressed = os.path.getsize(comp_filepath) / 1024**3
    else:
        fsize_compressed = 0

    return {
        "n_compressed": count_compressed(sim.wdir),
        "last_is_compressed": md.is_compressed,
        "fsize_compressed": fsize_compressed
    }

def _particle_info(sim) -> dict:
    try:
        n = prw.get_particle_outputs(sim.wdir)
    except (IndexError, FileNotFoundError):      # no particles.*.dbl files
        n = 0
    return {
        "n_part_outputs": n,
        "has_particles": n > 0,
        "has_particles_hdf5": os.path.isfile(os.path.join(sim.wdir, "particles.hdf5")),
    }

@dataclass
class SimulationReport:
    #---Simulation information---#
    wdir: str
    run_name: str
    n_grid_outputs: int
    dtype: str                       # e.g. "flt.h5"
    geometry: str                    # e.g. "CARTESIAN"
    time_unit: str
    last_sim_time: float             # user units
    is_conv: bool                    # whether data is converted to user units
    ini_file: str

    #---Compression information---#
    n_compressed: int                # outputs with a .compressed file
    last_is_compressed: bool
    fsize_compressed: float

    #---Grid information---#
    variables: List[str] = field(default_factory=list)
    coord_vars: List[str] = field(default_factory=list)

    grid_ndim: int = 0
    grid_shape: tuple = ()
    grid_n_cells: int = 0
    grid_axes: dict = field(default_factory=dict)
    grid_dxyz: tuple = ()
    est_size_gib: float = 0.0
    size_n_vars: int = 0
    size_precision: int = 8


    #---Particle information---#
    n_part_outputs: int = 0
    has_particles: bool = False
    has_particles_hdf5: bool = False

    @classmethod
    def from_sim(cls, sim):
        return cls(
            wdir=sim.wdir,
            run_name=sim.run_name,
            ini_file=pc.get_ini_file(sim.ini_file),
            **_grid_info(sim),
            **_grid_summary(sim),
            **_compression_info(sim),
            **_particle_info(sim),
        )

    # def __str__(self):
    #     return self.generate_report()

    def __post_init__(self):
        self.generate_report()

    def generate_report(self) -> str:
        path_str = f"Path: {self.wdir}"
        nchar = len(path_str) + 4

        def section(title):
            return f"\n{title:-^{nchar}}"

        def row(label, value):
            return f"{label:<24}{value}"


        lines = [
            "=" * nchar,
            f" Simulation: {self.run_name}",
            f" {path_str}",
            "=" * nchar,
        ]

        #---Simulation---#
        lines += [
            section(" SIMULATION INFO "),
            row("grid outputs:", self.n_grid_outputs),
            row("last sim time:", f"{self.last_sim_time:.2f} {self.time_unit}"),
            row("data type:", self.dtype),
            row("geometry:", self.geometry),
            row("units ini file path:",self.ini_file),
            row("convert to usr units:", "yes" if self.is_conv else "no (code units)"),
        ]

        #---Grid---#
        shape = " x ".join(str(n) for n in self.grid_shape)
        lines += [
            section(" GRID INFO "),
            row("dimensions:", f"{self.grid_ndim}D"),
            row("shape:", f"{shape} ({self.grid_n_cells:.2e} cells)"),
        ]
        for name, a in self.grid_axes.items():
            lo, hi = a["extent"]
            lines.append(row(f"{name}:", f"extent ({lo:g}, {hi:g}), type: {a['type']}"))
        dxyz = ", ".join("non-uniform" if d is None else f"{d:g}" for d in self.grid_dxyz)
        lines += [
            row("dx, dy, dz:", f"({dxyz})"),
            row("variables:", ", ".join(self.variables)),
            row("coordinates:", ", ".join(self.coord_vars)),
        ]

        #---Output files---#
        prec = "flt" if self.size_precision == 4 else "dbl"
        lines += [
            section(" OUTPUT FILES "),
            row("est. uncompressed:", f"{self.est_size_gib:.1f} GiB / output ({self.size_n_vars} arrays, {prec})"),
            row("compressed outputs:", self.n_compressed),
            row("last output:", "compressed" if self.last_is_compressed else "not compressed"),
        ]
        if self.fsize_compressed > 0:
            ratio = self.est_size_gib / self.fsize_compressed
            lines.append(row("compressed size:", f"{self.fsize_compressed:.2f} GiB / output (~{ratio:.1f}x smaller)"))

        #---Particles---#
        lines.append(section(" PARTICLES "))
        if self.has_particles:
            lines += [
                row("particle outputs:", self.n_part_outputs),
                row("particles.hdf5:", "yes" if self.has_particles_hdf5 else "no (not yet saved)"),
            ]
        else:
            lines.append(row("particles:", "none"))

        # return "\n".join(lines)
        print("\n".join(lines))
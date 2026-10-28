import plutonlib.utils as pu
import plutonlib.config as pc

from astropy import units as u
from astropy import constants as const 

import os
from pathlib import Path
import configparser

from dataclasses import dataclass
from functools import lru_cache

DEFAULT_USR_UNITS_INI = Path(__file__).parent / "user_units.ini"
C_UNIT = u.def_unit("c",const.c) #adds 1c (speed of light) as constant
u.add_enabled_units([C_UNIT])

@dataclass
class VarInfo:
    code_uv:    u.Unit      # code unit value  e.g. 1.0023678e-24 g/cm³
    usr_uv:     u.Unit      # user unit value  e.g. 1 kg/m³
    code_unit:  u.Unit      # just the unit         e.g. g/cm³
    var_name:   str         # latex string          e.g. r"$\rho$"
    # coord_name: str = None  # only for x1, x2, x3
    shp: tuple = None
    ndim: tuple = None

@dataclass
class UnitDefinitions:
    #---Constants/Base uv required to calc the rest---#
    mu_norm: float = 0.60364
    amu: u.Quantity = const.u.to(u.g)
    length: u.Quantity = 1 * u.kpc
    # velocity: u.Quantity = const.c.to(u.m / u.s)
    velocity: u.Quantity = 1 * C_UNIT
    unassigned: u.Quantity = 1 * u.dimensionless_unscaled
    
    #---derive code unit values---#
    @property
    def time(self):
        return (self.length / self.velocity).to(u.Myr)

    @property
    def density(self):
        return (self.mu_norm * self.amu / u.cm**3).to(u.g / u.cm**3)

    @property
    def pressure(self):
        return (self.density * self.velocity**2).to(u.Pa)

    @property
    def mass(self):
        return (self.density * self.length**3).to(u.kg)

    @property
    def energy(self):
        return (self.mass * self.velocity**2).to(u.J)

    @property
    def temperature(self):
        # PLUTO's KELVIN = UNIT_VELOCITY^2 * CONST_amu / CONST_kB
        # return (self.velocity**2 * self.amu / const.k_B).to(u.K)
        return (1 * u.K)

    @property
    def power(self):
        return (1 * u.erg / u.s)

    @property 
    def tracer(self):
        return self.unassigned

    def code_unit_values(self):
        """Dict keyed by PlutoUnits field names, values are Quantities."""
        return {
            "x1": self.length, "x2": self.length, "x3": self.length,
            "vx1": self.velocity, "vx2": self.velocity, "vx3": self.velocity,
            "rho": self.density,
            "prs": self.pressure,
            "sim_time": self.time,
            "T": self.temperature,
            "Q": self.power,
            "tr1": self.tracer,
        }

@dataclass
class PlutoUnits:
    rho:       VarInfo
    prs:       VarInfo
    vx1:       VarInfo
    vx2:       VarInfo
    vx3:       VarInfo
    x1:        VarInfo
    x2:        VarInfo
    x3:        VarInfo
    T:         VarInfo
    Q:         VarInfo
    sim_time:  VarInfo
    tr1:       VarInfo

    @staticmethod
    def read_usr_units(ini_path=None):
        """Read specified user units to convert to see plutonlib/units/user_units.ini"""
        ini_path = DEFAULT_USR_UNITS_INI if ini_path is None else ini_path

        if not ini_path.exists():
            raise FileNotFoundError(f"ini file: '{str(ini_path)}' not found")
        
        config = configparser.ConfigParser()
        config.optionxform = str
        # config.read(get_ini_file(ini_file=usr_ini))
        config.read(ini_path)
        return {k: u.Unit(v) for k, v in config["usr_unit_values"].items()}

    @classmethod
    def from_definitions(cls, defs=None, usr_ini_path=None):
        defs = defs if defs is not None else UnitDefinitions()
        quantities = defs.code_unit_values()
        usr_uvs = cls.read_usr_units(usr_ini_path)
        sel_coords = pc.coord_systems["CARTESIAN"]  # NOTE replace if not cartesian

        var_metadata = {  # NOTE can probably deprecate coord_name
            "rho":      {"var_name": r"$\rho$"},
            "prs":      {"var_name": r"$P$"},
            "vx1":      {"var_name": f"$V_{{{sel_coords[0]}}}$"},
            "vx2":      {"var_name": f"$V_{{{sel_coords[1]}}}$"},
            "vx3":      {"var_name": f"$V_{{{sel_coords[2]}}}$"},
            "x1":       {"var_name": f"${sel_coords[0]}$"},
            "x2":       {"var_name": f"${sel_coords[1]}$"},
            "x3":       {"var_name": f"${sel_coords[2]}$"},
            "tr1":      {"var_name": "Tracer"},
            "T":        {"var_name": r"$T$"},
            "Q":        {"var_name": r"$Q_{\rm{jet}}$"},
            "sim_time": {"var_name": r"$t$"},
        }

        var_infos = {}
        for v, metadata in var_metadata.items():
            q = quantities[v]
            code_uv, usr_uv = u.Unit(q), usr_uvs[v]
            if not code_uv.is_equivalent(usr_uv):
                raise ValueError(f"'{v}': can't convert code unit {code_uv} to user unit {usr_uv}")
            var_infos[v] = VarInfo(code_uv=code_uv, usr_uv=usr_uv, code_unit=q.unit, **metadata)
        return cls(**var_infos)

@lru_cache(maxsize=None) #cache the pluto units
def get_pluto_units(definitions_path=None, usr_ini_path=None):
    # defs = UnitDefinitions.from_definitions_h(definitions_path) if definitions_path else None
    return PlutoUnits.from_definitions(defs=None, usr_ini_path=usr_ini_path)

def code_to_usr_units(var_name, raw_data=None, usr_ini_path=None):
    """
    gets unit value from PlutoUnits to convert from code units to the user specified units in the ini file.
    NOTE: converts raw_data in place when it is a float ndarray.
    """
    mapped_var_name = pu.map_coord_name(var_name)  # makes sure to convert diff XYZ arrays to x1,x2,x3
    can_convert = True

    units = get_pluto_units(usr_ini_path=usr_ini_path)  # outside the try so real errors aren't hidden
    try:
        pluto_units = getattr(units, mapped_var_name)
    except AttributeError:
        print(f"PlutoUnits doesn't have attribute '{mapped_var_name}', may be unitless")
        can_convert = False

    code_uv = pluto_units.code_uv if can_convert else None
    usr_uv = pluto_units.usr_uv if can_convert else None

    # convert raw_data
    if code_uv is None or usr_uv is None:  # skips if it doesn't need converting
        conv_data_uuv = raw_data
        uv_usr = None
    else:
        uv_usr = (1 * code_uv).to(usr_uv).value
        raw_data *= uv_usr
        conv_data_uuv = raw_data

    returns = {
        "uv_usr": uv_usr,  # like a scale factor??
        "conv_data_uuv": conv_data_uuv,
    }

    return returns

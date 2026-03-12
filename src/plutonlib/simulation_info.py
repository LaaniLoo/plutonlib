import plutonlib.config as pc
import plutonlib.analysis as pa

from dataclasses import dataclass
from astropy import units as u
from astropy import constants as astro_const 
import scipy.constants as const

class SimInfoSetup:
    """Simple class to initialise units from plutonlib user unit ini file, with methods for fetching dataclass
    atributes from pluto.ini
    """
    _unit_map = {} #maps dataclass attrs to user specified unit values in plutonlib/units ini files
    _param_map = {} #maps dataclass attrs to user parameters present in pluto.ini setups
    ini_file = "jet_units"
    def __post_init__(self):
        units = pc.PlutoUnits.from_ini(ini_file=self.ini_file)
        for sim_v,map_v in self._unit_map.items():
            value = getattr(self,sim_v)
            if map_v is None:
                continue

            unit = getattr(units,map_v).usr_uv
            setattr(self,sim_v,self.assign_unit(value,unit))

            if sim_v == "v":
                setattr(self,sim_v,self.assign_unit(value,u.m / u.s))
                self.v_c = value


    @staticmethod
    def assign_unit(v,unit):
        """Assigns units to dataclass attrs, converts values input into dataclass to desired unit. 
        e.g JetInfo(Q=1e44 * u.erg / u.sec) or  JetInfo(Q=1e37) both give Q = 1e37 * u.W

        Args:
            v (float or astropy Quantity/Unit): dataclass attribute value e.g. jet power 'Q'
            unit (astropy unit): unit to convert to

        Returns:
            v: dataclass attr 'v' converted to unit
        """
        if isinstance(v,u.Quantity):
            return v.to(unit)
        else:
            return v * unit
        
    @classmethod
    def from_usr_params(cls,usr_params: dict,**kwargs):
        """Maps user params from pluto.ini to JetInfo and EnvInfo classes 

        Args:
            usr_params (dict): dict of usr_params from pluto.ini

        Returns:
            SimInfoSetup: _description_
        """
        units = pc.PlutoUnits.from_ini(ini_file=cls.ini_file)
        mapped_params = {}
        for usr_param, param in cls._param_map.items():
            unit_key = cls._unit_map.get(param)
            param_value = usr_params[usr_param]

            if param is None:
                continue

            if unit_key is None:                       # dimensionless, no unit
                mapped_params[param] = param_value
            else:
                unit = getattr(units, unit_key).code_unit
                mapped_params[param] = param_value * unit
        
        return cls(**mapped_params,**kwargs)

@dataclass
class EnvInfo(SimInfoSetup):
    """Dataclass for all simulation environment parameters such as environment density 'rho', etc..

    Args:
        SimInfoSetup (class): setup class that initialises units/parameter fetching from methods shared by dataclasses
    """
    rho:      u.Quantity = 0
    prs:      u.Quantity = 0
    T:        u.Quantity = 0
    r_scaling: u.Quantity = 0
    beta:     float = 0        # dimensionless ratio
    x1o:      u.Quantity = 0
    x2o:      u.Quantity = 0
    x3o:      u.Quantity = 0
    c_s:      u.Quantity = 0
    wx1: u.Quantity = 0
    wx2: u.Quantity = 0
    wx3: u.Quantity = 0
    # M_wind:   float = 0        # dimensionless mach number

    _unit_map = {
    "rho":      "rho",
    "prs":      "prs",
    "T":        "T",
    "r_scaling": "x1",
    "beta":     None,    # dimensionless
    "x1o":      "x1",
    "x2o":      "x2",
    "x3o":      "x3",
    "c_s":      None,   # sound speed, velocity unit
    "wx1": None,
    "wx2": None,
    "wx3": None,
    # "M_wind":   None,    # dimensionless
    }   
    
    _param_map = {
    "env_rho_0":            "rho",
    "env_temp":             "T",
    "env_r_scaling":        "r_scaling",
    "env_b_exponent":       "beta",
    "env_x1o":              "x1o",
    "env_x2o":              "x2o",
    "env_x3o":              "x3o",
    "wind_vx1":             "wx1",
    "wind_vx2":             "wx2",
    "wind_vx3":             "wx3",
    }

    def __post_init__(self):
        super().__post_init__()  # runs unit assignment first

        self.c_s = pa.calc_sound_speed(rho_0=self.rho.value,T=self.T.value) #env sound speed
        self.prs = pa.EOS(rho=self.rho.value,T=self.T.value) #env prs from rho and T
        for wxx in ["wx1", "wx2", "wx3"]:
            value = getattr(self, wxx)
            setattr(self, wxx, value * self.c_s)

@dataclass
class JetInfo(SimInfoSetup):
    """Dataclass for all simulation jet parameters such as jet power 'Q', jet density 'rho', etc..

    Args:
        SimInfoSetup (class): setup class that initialises units/parameter fetching from methods shared by dataclasses

    Raises:
        ValueError: raises value error if multiple wind components due to L_bend, #NOTE change to 3D vector
    """
    Q:      u.Quantity = 0
    v:      u.Quantity = 0
    v_c:    u.Quantity = 0
    rho:    u.Quantity = 0
    M:      float = 0       # dimensionless mach number
    radius: u.Quantity = 0
    theta:  u.Quantity = 0
    L1:     u.Quantity = 0
    L1a:    u.Quantity = 0
    L1b:    u.Quantity = 0
    L1c:    u.Quantity = 0
    L2:     u.Quantity = 0
    L_bend: u.Quantity = 0
    env: EnvInfo = None   # optional, only needed for lscales

    _unit_map = { #map info variables to pluto units
        "Q":     "Q",
        "v":     "vx3",
        "v_c":   "vx3",
        "rho":   "rho",
        "radius":     "x1",
        "L1":    "x1",
        "L1a":   "x1",
        "L1b":   "x1",
        "L1c":   "x1",
        "L2":    "x1",
        "L_bend":"x1",
        "M":     None,     # dimensionless
        "theta": None,     # degrees, handle separately
    }

    _param_map = {
        "jet_pwr":            "Q",
        "jet_spd":            "v",
        "jet_chi":            None,
        "jet_oa_primary":     "theta",
    }

    def __post_init__(self):
        super().__post_init__()  # runs unit assignment first
        

        if self.env is not None:
            wvx_counter = 0
            v_wind = 0
            for w in ["wx1", "wx2", "wx3"]:
                if getattr(self.env, w).value != 0:
                    v_wind = getattr(self.env, w)
                    wvx_counter += 1
                if wvx_counter > 1:
                    raise ValueError(f"Wind velocities have multiple components, cannot find L_bend")
    
            lscales = pa.calc_length_scales(
                Q       = self.Q,
                rho     = self.env.rho,
                v_jet   = self.v,
                theta   = self.theta,
                T       = self.env.T,
                v_wind  = v_wind
            )
            self.L1    = lscales["L1"]
            self.L1a   = lscales["L1a"]
            self.L1b   = lscales["L1b"]
            self.L1c   = lscales["L1c"]
            self.L2    = lscales["L2"]
            self.L_bend = lscales["L_bend"]
            self.radius = lscales["r_jet"]
            self.M = self.v / self.env.c_s

            self.rho = (pa.calc_jet_density(self.Q,self.v,self.theta,self.radius.to(u.m))).to(u.kg / u.m**3)

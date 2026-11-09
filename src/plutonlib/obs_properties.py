from plutonlib.simulation_info import SimInfoSetup

from dataclasses import dataclass,field,replace
from astropy import units as u
from astropy import cosmology as cosmo  # Astropy cosmology

from scipy.spatial.transform import Rotation

@dataclass
class ObsProperties:
    """Data class used to store frequently used radio observing properties 
    e.g. freqs, redshift etc, makes loading and plotting faster with a shared obs setup"""

    freqs: list[float] = field(default_factory=lambda:[0.15*u.GHz,1.4*u.GHz]) #list of frequencies to observe at
    redshift: float = 0.05 
    angle: list[float] = field(default_factory=lambda:[0,0,0]) #observing angle list x,y,z
    pixel_size: u.Quantity = 1*u.arcsec
    beam_fwhm: u.Quantity = 3*u.arcsec  # generally this should be ~3x bigger than pixel size
    delta_r: float = 0.3
    ray_depth_min: float = -200
    ray_depth_max: float = 200
    plane: str = "xz" #observing plane
    alpha_inj: float = 2.2 #The electron energy power law index, 2.2 gives a spectral index of \alpha = -0.6


    @property
    def arcsec2kpc(self):
        return cosmo.Planck15.kpc_proper_per_arcmin(self.redshift).to(u.kpc/u.arcsec)

    @property 
    def pixel_size_kpc(self):
        return (self.pixel_size * self.arcsec2kpc).to(u.kpc)

    @property
    def beam_kpc(self):
        return (self.beam_fwhm * self.arcsec2kpc).to(u.kpc)

    @property
    def grid_spacing(self):
        return self.pixel_size_kpc.value * 1.0 #0th redshift?

    @property 
    def rot_matrix(self):
        return Rotation.from_euler("xyz", self.angle, degrees=True).as_matrix()

    def replace(self,**changes):
        return replace(self,**changes)
    
    def __post_init__(self):
        setattr(self,"freqs",SimInfoSetup.assign_unit(getattr(self,"freqs"),u.GHz))
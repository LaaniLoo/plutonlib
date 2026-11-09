from cProfile import label

import plutonlib.analysis as pl_analysis
from plutonlib.simulation_info import JetInfo,SimInfoSetup
import plutonlib.splines as pl_splines
import plutonlib.peaks as pl_peaks
import plutonlib.morphology as pl_morph

import numpy as np 
from astropy import units as u

from dataclasses import dataclass,field,replace
import pandas as pd
from pathlib import Path


@dataclass
class JetMorphology():
    """Dataclass to automatically assign jet-morphology params/info from key attributes e.g. jet/lobe lengths"""
    theta_lobe: u.Quantity = 0 * u.deg
    l_lobes: list[u.Quantity] = field(default_factory=lambda: [0*u.kpc, 0*u.kpc])
    l_jets: list[u.Quantity] = field(default_factory=lambda: [0*u.kpc, 0*u.kpc])
    dist_to_brights: list[u.Quantity] = field(default_factory=lambda: [0*u.kpc, 0*u.kpc])

    # theta_jet: u.Quantity = 0 * u.deg
    # inj_xz: list[u.Quantity] = field(default_factory=lambda: [0*u.kpc, 0*u.kpc])

    @property
    def lobe_ratio(self):
        if 0 in self.l_lobes: #one of the lobes has 0 length
            return np.nan
        else:
            return np.array(self.l_lobes[0] / self.l_lobes[1]).round(3) 

    @property
    def jet_lobe_ratio(self):
        if 0 in (*self.l_lobes,*self.l_jets): #one of the jets or lobes have 0 length
            return np.nan
        else:
            return np.array([l / j for l, j in zip(self.l_lobes, self.l_jets)]).round(3)

    @property
    def fr_idx(self):
        """FR-index for each jet"""
        return [pl_morph.calc_fr_idx(x_br,ll)[0] for x_br,ll in zip(self.dist_to_brights,self.l_lobes)]

    @property
    def classification(self):
        """Morphological classification for lobes e.g. 'WAT+FR1'"""
        return pl_morph.classify_morphology(self.l_jets.value,self.fr_idx,self.theta_lobe.value)

    @classmethod
    def from_spline_data(cls,spline_data):
        """Creates a class instance by filling out parameters using spline data"""
        top,bot = spline_data["splines_top"],spline_data["splines_bot"]
        top_to_peak, bot_to_peak = pl_peaks.get_peak_spline_seg(spline_data)
        inj_xz = spline_data["inj_xz"]
        vtop, vbot = top[len(top)//2], bot[len(bot)//2] #jet vectors are at midpoints
        theta_lobe = pl_morph.calc_bending_angle(vtop,vbot,inj_xz)
        d_inj_peak = np.array([pl_analysis.calc_path_length(seg) for seg in [top_to_peak, bot_to_peak]])

        return cls(
            theta_lobe=theta_lobe.round(3),
            l_lobes= spline_data["lobe_length"].round(3),
            l_jets=spline_data["jet_length"].round(3), #NOTE need jet data somehow
            dist_to_brights=d_inj_peak.round(3),
        )

    @classmethod
    def from_obs_properties(cls,sdata,obs_props,grid_output,freq=None,path_params=None):
        """Creates a class instance from ObsProperties object using `from_spline_data` method

        Args:
            sdata (SimulationData): SimulationData object
            obs_props (ObsProperties): ObsProperties object
            grid_output (int): PLUTO grid output number
            freq (float, Optional): optional freq else ObsProperties.freqs[0]
            path_params (PathParams, Optional): optional PathParams object for path tracing, see defaults

        """
        #NOTE defaults to first element of ObsProperties.freqs --> dont think there is a big freq dependence
        spline_data = pl_splines.get_jet_splines_sb(
            sdata=sdata,obs_props=obs_props,
            grid_output=grid_output,freq=freq,
            path_params=path_params
            ) 

        return cls.from_spline_data(spline_data)

    def __post_init__(self):
        #run the assign unit method from SimInfoSetup to properly assign and convert units
        for name in ["l_lobes", "l_jets", "dist_to_brights"]:
            setattr(self, name, SimInfoSetup.assign_unit(getattr(self, name), u.kpc))
        # self.inj_xz = [SimInfoSetup.assign_unit(x, u.kpc) for x in self.inj_xz]
        for name in ["theta_lobe"]: #"theta_jet",
            setattr(self, name, SimInfoSetup.assign_unit(getattr(self, name), u.deg))

#--Equations---#
def calc_fr_idx(x_bright,x_length):
    """Calculates the FR index from Krause (2012)

    Args:
        x_bright (float): length of jet at brightest point at 150MHz 
        x_length (float): length of lobe at 150MHz

    Returns:
        tuple: index value (btwn 0.5,1.5 for FR1 or 1.5,2.5 for FR2), str of classification
    """
    fr_idx = 2*(x_bright/x_length) + 0.5
    fr_class = None

    if fr_idx > 0.5 and fr_idx < 1.5:
        fr_class = "FR1"
    elif fr_idx > 1.5 and fr_idx < 2.5:
        fr_class = "FR2"
    else:
        fr_class = "outside bounds"

    return fr_idx,fr_class

def calc_fr_idx_spl(spline_data):
    """Gets the FR index and classification for top and bottom jets using the skeleton spline data

    Args:
        spline_data (dict): dict returned by `plutonlib.splines.get_jet_splines_sb()`

    Returns:
        tuple: fr_info_top/bottom each is a tuple of fr_idx and classification, e.g. (0.5,"FR1"),(1.5,"FR2")
    """
    err_top,err_bot = pl_peaks.find_jets_without_peaks(spline_data)
    x_lengths = spline_data["lobe_length"] # L_top,L_bot

    inj_idx = spline_data["inj_idx"]
    idx_max_top,idx_max_bot = pl_peaks.get_sb_max_idx(spline_data)

    splines = spline_data["splines"]

    top_path = splines[inj_idx:idx_max_top + 1] if idx_max_top is not None else None
    bot_path = splines[idx_max_bot:inj_idx + 1][::-1] if idx_max_bot is not None else None

    def _fr_for_jet(path, x_length, err):
        if path is None or err:
            return (np.nan, "")
        x_bright = pl_analysis.calc_path_length(path)
        return calc_fr_idx(x_bright, x_length)

    fr_info_top = _fr_for_jet(top_path, x_lengths[0], err_top)
    fr_info_bot = _fr_for_jet(bot_path, x_lengths[1], err_bot)
    return fr_info_top, fr_info_bot

def calc_bending_angle(v1,v2,origin):
    """Calculates bending angle between two jet vectors WRT origin (jet injection x,z)

    Args:
        v1 (x,z array): first jet vector e.g top jet points
        v2 (x,z array): second jet vector e.g bottom jet points
        origin (x,z array): reference point of the angle, e.g. jet injection region

    Returns:
        float: bending angle at v1,v2 WRT origin
    """
    v1 = v1 - origin
    v2 = v2 - origin
    cos =(v1 @ v2)/(np.hypot(*v1) * np.hypot(*v2)) #dot product rule
    angle = np.degrees(np.arccos(np.clip(cos,-1,1))) #clip the cos value for edge cases
    return angle

def classify_morphology(lobe_lengths,fr_idxs,bending_angle):
    """Classifies jet morphology into some combination of 
    WAT,NAT,single-sided,head-tail,FR or hybrid morphology.

    Args:
        lobe_lengths (list): list of top/bottom jet/lobe lengths
        fr_idxs (list): list of top/bottom fr_idx for jets
        bending_angle (float): bending angle between lobes/jets -> used for WAT/NAT

    Returns:
        str: morphology string e.g. 'NAT+FR1', 'single-sided' or 'WAT+asymmetric-hybrid'
    """
    fr_idxs = np.array(fr_idxs)

    def single_or_head_tail():
        if 0 in lobe_lengths:
            return True,"single-sided"
        elif np.min(lobe_lengths) / np.max(lobe_lengths) < 0.2: #short jet is not 0 but 25% the length of long jet
            return True,"head-tail"
        else: 
            return False, None
    def wat_or_nat():
        if bending_angle >= 90 and bending_angle < 170:
            return "WAT"
        elif bending_angle < 90:
            return "NAT"
    def hybrid_or_fr(single_sided=False):
        if not single_sided:
            similar_lobes = True if np.min(lobe_lengths) / np.max(lobe_lengths) >= 0.45 else False
            is_hybrid  = True if sum(fr > 1.5 for fr in fr_idxs) == 1 else False

            if is_hybrid and similar_lobes: #exactly 1 jet is FR2 and have similar lobes
                return "symmetric-hybrid"
            elif is_hybrid and not similar_lobes:
                return "asymmetric-hybrid"
            elif np.all(fr_idxs < 1.5):
                return "FR1"
            elif np.all(fr_idxs > 1.5):
                return "FR2"
            else:
                return "non-FR"
        else:
            if np.any(fr_idxs < 1.5):
                return "FR1"
            elif np.any(fr_idxs > 1.5):
                return "FR2"
    single_sided,ss_or_ht = single_or_head_tail()
    morphologies = [ss_or_ht,wat_or_nat(),hybrid_or_fr(single_sided)]
    morph_str = "+".join([m for m in morphologies if m is not None])
    return morph_str

def classify_morphology_splines(spline_data):
    """Helper to run `classify_morphology` using just spline data"""
    lobe_lengths = spline_data["lobe_length"]
    top,bot = spline_data["splines_top"],spline_data["splines_bot"]

    v1 = top[len(top) // 2]
    v2 = bot[len(bot) // 2]
    angle = calc_bending_angle(v1,v2,spline_data["inj_xz"])
    print(f"Bending angle = {angle:.2f} deg")
    fr_labels = np.array(calc_fr_idx_spl(spline_data))
    fr_idxs = np.array([float(i[0]) for i in fr_labels])
    
    return classify_morphology(lobe_lengths,fr_idxs,angle)

def calc_morphology_params(sdata,grid_output,freq,redshift,angle,path_params=None,lobe_cut=50,jet_cut=98):
    path_params = pl_splines.PathParams() or path_params
    lobe_params = replace(path_params,sb_percentile=lobe_cut)
    jet_params = replace(path_params,sb_percentile=jet_cut)

    spline_args = {"sdata":sdata,"grid_output":grid_output,"freq":freq,"redshift":redshift,"angle":angle}
    params_dict = {"lobe":lobe_params,"jet":jet_params}
    spline_data = {}
    for name,params in params_dict.items():
        try:
            spline_data[name] = pl_splines.get_jet_splines_sb(**spline_args,path_params=params)
        except ValueError as e:
            if "no minimum-cost path" not in str(e):
                raise
            print(f"Failed to fetch {name} spline data for: {angle}")

    failed = {name: name not in spline_data for name in params_dict}

    sides, inj_xz = {}, {}
    vtop,vbot = {},{}
    for name, data in spline_data.items():
        top,bot = data["splines_top"],data["splines_bot"]
        sides[name] = (top,bot)
        vtop[name] = top[len(top) // 2] if name == "lobe" else top[-1] #jet top vector at midpoint of lobe and at jet
        vbot[name] = bot[len(bot) // 2] if name == "lobe" else bot[-1] #jet bot ""
        inj_xz[name] = data["inj_xz"]

    param_names = ["jet_morphology","lobe_morphology","theta_jet", "theta_lobe_mid", "l_lobe", "l_jet",
                   "d_inj_peak", "lobe_lobe_ratio", "lobe_jet_ratio", "fr_idx"]
    morph_params = {
        "angle": tuple(angle), "failed_lobe": failed['lobe'],
        "failed_jet": failed['jet'], **dict.fromkeys(param_names,np.nan)} #default = nan

    if not failed["lobe"]:
        lobe_spline_data = spline_data["lobe"]

        #---data---#
        top_seg, bot_seg = pl_peaks.get_peak_spline_seg(lobe_spline_data)
        fr_info_top, fr_info_bot = calc_fr_idx_spl(lobe_spline_data)

        #---params---#
        lobe_theta_mid = float(calc_bending_angle(vtop["lobe"], vbot["lobe"], inj_xz["lobe"]).round(3))
        lobe_lengths = tuple(lobe_spline_data["lobe_length"].round(3).tolist())
        d_inj_peak = tuple(np.array([pl_analysis.calc_path_length(seg) for seg in [top_seg, bot_seg]]).round(3).tolist())

        if 0 in lobe_lengths: #one of the lobes has 0 length
            ll_ratio = np.nan
        else:
            ll_ratio = np.array(lobe_lengths[0] / lobe_lengths[1]).round(3).tolist() #lobe-lobe ratio

        # fr_labels = tuple([
        #     f"{info[1]} ({info[0]:.2f})" if info[0] is not None else "Single-sided"
        #     for info in (fr_info_top, fr_info_bot)])
        fr_labels = tuple(np.array([fr_info_top,fr_info_bot]).round(3))
        lobe_morph = classify_morphology(lobe_lengths,np.array([i[0] for i in fr_labels]),lobe_theta_mid)
        morph_params.update({
            "lobe_morphology": lobe_morph,
            "theta_lobe_mid": lobe_theta_mid,
            "l_lobe": lobe_lengths,
            "d_inj_peak": d_inj_peak,
            "lobe_lobe_ratio": ll_ratio,
            "fr_idx": fr_labels,
        })

    if not failed["jet"]:
        jet_spline_data = spline_data["jet"]
        theta_jet = float(calc_bending_angle(vtop["jet"], vbot["jet"], inj_xz["jet"]).round(3))
        jet_lengths = tuple(jet_spline_data["jet_length"].round(3).tolist())

        morph_params.update({"theta_jet": theta_jet, "l_jet": jet_lengths})

    if not any(failed.values()):
        if 0 in (*lobe_lengths,*jet_lengths): #one of the jets or lobes have 0 length
            lj_ratios = np.nan
        else:
            lj_ratios = tuple(np.array([l / j for l, j in zip(lobe_lengths, jet_lengths)]).round(3).tolist())
        
        jet_morph = classify_morphology(jet_lengths,np.array([i[0] for i in fr_labels]),theta_jet)

        morph_params["jet_morphology"] = jet_morph
        morph_params["lobe_jet_ratio"] = lj_ratios

    return morph_params

def save_morphology_csv(sdata,grid_output,freq,redshift,angles,path_params=None,file_path="~/morphology.csv"):
    path_params = pl_splines.PathParams() or path_params

    morph_params = []
    for angle in angles:
        morph_params.append(calc_morphology_params(sdata,grid_output,freq,redshift,angle,path_params))

    csv_file = Path(file_path).expanduser()

    df = pd.DataFrame(morph_params)
    df.to_csv(csv_file,index=False)
    
    return df
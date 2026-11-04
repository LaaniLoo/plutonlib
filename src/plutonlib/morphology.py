import plutonlib.analysis as pl_analysis
from plutonlib.simulation_info import JetInfo,SimInfoSetup
import plutonlib.splines as pl_splines
import matplotlib.pyplot as plt

import numpy as np 
from scipy.signal import find_peaks
from astropy import units as u

from dataclasses import dataclass,field,replace
from collections import defaultdict
import pandas as pd
from pathlib import Path

@dataclass
class JetMorphology():
    theta_jet: u.Quantity = 0 * u.deg
    theta_lobe: u.Quantity = 0 * u.deg
    inj_xz: list[u.Quantity] = field(default_factory=lambda: [0*u.kpc, 0*u.kpc])
    l_lobe: u.Quantity = 0 * u.kpc
    l_jet: u.Quantity = 0 * u.kpc
    dist_peak: u.Quantity = 0 * u.kpc

    def __post_init__(self):
        #run the assign unit method from SimInfoSetup to properly assign and convert units
        for name in ["l_lobe", "l_jet", "dist_peak"]:
            setattr(self, name, SimInfoSetup.assign_unit(getattr(self, name), u.kpc))
        self.inj_xz = [SimInfoSetup.assign_unit(x, u.kpc) for x in self.inj_xz]
        for name in ["theta_jet", "theta_lobe"]:
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

#---SB spline calculations---#
def get_sb_peaks(spline_data,plot=False):
    """Use scipy peak finding to find index of surface brightness peaks along jet spline

    Args:
        spline_data (dict): dict returned by `plutonlib.splines.get_jet_splines_sb()`
        plot (bool, optional): Set True to show a diagnostic plot with peaks. Defaults to False.

    Returns:
        tuple: tuple of indices for top and bottom jets
    """
    inj_idx = spline_data["inj_idx"]
    jet_length = spline_data["jet_length"]
    sb_top = spline_data["log_sb_spline"][inj_idx:]
    sb_bot = spline_data["log_sb_spline"][:inj_idx]

    # arc_bot = np.linspace(0,jet_length[1],len(sb_bot))
    # arc_top = np.linspace(arc_bot[-1],np.sum(jet_length),len(sb_top))
    jet_splines = spline_data["jet_splines"]
    arc = np.r_[0, np.cumsum(np.hypot(*np.diff(jet_splines,axis=0).T))] #true distance along the full path
    arc -= arc[inj_idx] #0 at the injection point, negative on the bottom jet

    arc_top = arc[inj_idx:]
    arc_bot = arc[:inj_idx]

    prom = 0.03
    width = 1
    idx_top, _ = find_peaks(sb_top,prominence=prom,width=width) 
    idx_bot, _ = find_peaks(sb_bot,prominence=prom,width=width) 

    if plot:
        f, a = plt.subplots(figsize = (6,6))
        a.plot(arc_top,sb_top, color = 'k')
        a.plot(arc_bot,sb_bot, color = 'k')
        a.scatter(arc_top[idx_top],sb_top[idx_top],s = 4, color = 'red')
        a.scatter(arc_bot[idx_bot],sb_bot[idx_bot],s = 4, color = 'red')

        a.set_xlabel("Arc length [kpc]")
        a.set_ylabel("Log sb mJy/beam")
        plt.show()

    return idx_top,idx_bot

def get_max_peak_idx(sb,peak_idx):
    """Index of the brightest peak for one jet, or None if the jet has no peaks"""
    if len(peak_idx) == 0:
        return None
    return peak_idx[np.argmax(sb[peak_idx])]

def get_sb_max_idx(spline_data):
    """Get the index of the max sb value from the peak sb array

    Args:
        spline_data (dict): dict returned by `plutonlib.splines.get_jet_splines_sb()`

    Returns:
        tuple: max idx for top and bottom jets, None for a jet with no peaks
    """
    inj_idx = spline_data['inj_idx']
    sb_spl = spline_data['log_sb_spline']

    idx_top,idx_bot = get_sb_peaks(spline_data)
    top_slice  = np.arange(len(sb_spl))[inj_idx:][idx_top] #idx of top jet scipy peaks
    bot_slice  = np.arange(len(sb_spl))[:inj_idx][idx_bot]

    return get_max_peak_idx(sb_spl,top_slice), get_max_peak_idx(sb_spl,bot_slice)

def get_sb_max_vals(spline_data,vals_array):
    """Gets array values for max sb along jet spine

    Args:
        spline_data (dict): dict returned by `plutonlib.splines.get_jet_splines_sb()`
        vals_array (array): array to find points at max sb along spline 

    Returns:
        tuple: array value at max sb for top jet, array value at max sb for bottom
    """
    max_idx_top, max_idx_bot = get_sb_max_idx(spline_data)

    vals_top = np.nan if max_idx_top is None else vals_array[max_idx_top]
    vals_bot = np.nan if max_idx_bot is None else vals_array[max_idx_bot]

    return vals_top, vals_bot

def get_sb_max_xy(spline_data):
    """Gets the PRAiSE grid xy location at max sb

    Args:
        spline_data (dict): dict returned by `plutonlib.splines.get_jet_splines_sb()`

    Returns:
        tuple: values at max sb for top and bottom jets
    """
    grid_mx = spline_data['grid_mx']
    grid_my = spline_data['grid_my']

    peak_x = [*get_sb_max_vals(spline_data,vals_array=grid_mx)] #x peaks for top and bottom jet
    peak_y = [*get_sb_max_vals(spline_data,vals_array=grid_my)] #y peaks for top and bottom jet
    
    return peak_x,peak_y

def check_jets(spline_data):
    inj_idx = spline_data['inj_idx']
    sb_spl = spline_data['log_sb_spline']

    idx_top,idx_bot = get_sb_peaks(spline_data)
    top_slice  = np.arange(len(sb_spl))[inj_idx:][idx_top] #idx of top jet scipy peaks
    bot_slice  = np.arange(len(sb_spl))[:inj_idx][idx_bot]

    err_top = True if len(top_slice) == 0 else False
    err_bot = True if len(bot_slice) == 0 else False

    return err_top,err_bot

def _fr_for_jet(path, x_length, err):
    if path is None or err:
        return (np.nan, "")
    x_bright = pl_analysis.calc_path_length(path)
    return calc_fr_idx(x_bright, x_length)

def get_fr_idx_spl(spline_data):
    """Gets the FR index and classification for top and bottom jets using the skeleton spline data

    Args:
        spline_data (dict): dict returned by `plutonlib.splines.get_jet_splines_sb()`

    Returns:
        tuple: fr_info_top/bottom each is a tuple of fr_idx and classification, e.g. (0.5,"FR1"),(1.5,"FR2")
    """
    err_top,err_bot = check_jets(spline_data)
    x_lengths = spline_data["jet_length"] # L_top,L_bot

    inj_idx = spline_data["inj_idx"]
    idx_max_top,idx_max_bot = get_sb_max_idx(spline_data)

    splines = spline_data["jet_splines"]

    top_path = splines[inj_idx:idx_max_top + 1] if idx_max_top is not None else None
    bot_path = splines[idx_max_bot:inj_idx + 1][::-1] if idx_max_bot is not None else None

    fr_info_top = _fr_for_jet(top_path, x_lengths[0], err_top)
    fr_info_bot = _fr_for_jet(bot_path, x_lengths[1], err_bot)
    return fr_info_top, fr_info_bot

def get_peak_spline_seg(spline_data):
    """Gets the spline segments from injection to peak sb value for top/bottom jets

    Args:
        spline_data (dict): dict returned by `plutonlib.splines.get_jet_splines_sb()`

    Returns:
        top_segment,bot_segment: truncated jet_splines from inj->peak for top/bottom jets
    """

    top_idx,bot_idx = get_sb_max_idx(spline_data)
    inj_idx = spline_data["inj_idx"]
    
    top_segment = spline_data["jet_splines"][inj_idx:top_idx]
    bot_segment = spline_data["jet_splines"][bot_idx:inj_idx][::-1] #reversed, inj->peak  

    return top_segment,bot_segment

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

def classify_morphology(jet_lengths,fr_idxs,bending_angle):
    """Classifies jet morphology into some combination of 
    WAT,NAT,single-sided,head-tail,FR or hybrid morphology.

    Args:
        jet_lengths (list): list of top/bottom jet length
        fr_idxs (list): list of top/bottom fr_idx for jets
        bending_angle (float): bending angle between lobes/jets -> used for WAT/NAT

    Returns:
        str: morphology string e.g. 'NAT+FR1', 'single-sided' or 'WAT+asymmetric-hybrid'
    """
    fr_idxs = np.array(fr_idxs)

    def single_or_head_tail():
        if 0 in jet_lengths:
            return True,"single-sided"
        elif np.min(jet_lengths) / np.max(jet_lengths) < 0.2: #short jet is not 0 but 25% the length of long jet
            return True,"head-tail"
        else: 
            return False, None
    def wat_or_nat():
        if bending_angle >= 90 and bending_angle < 170:
            return "WAT"
        elif bending_angle < 90:
            return "NAT"
    def true_hybrid_or_fr(single_sided=False):
        if not single_sided:
            similar_lobes = True if np.min(jet_lengths) / np.max(jet_lengths) >= 0.45 else False
            is_hybrid  = True if sum(fr > 1.5 for fr in fr_idxs) == 1 else False

            if is_hybrid and similar_lobes: #exactly 1 jet is FR2 and have similar lobes
                return "true-hybrid"
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
    morphologies = [ss_or_ht,wat_or_nat(),true_hybrid_or_fr(single_sided)]
    morph_str = "+".join([m for m in morphologies if m is not None])
    return morph_str

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
        top_seg, bot_seg = get_peak_spline_seg(lobe_spline_data)
        fr_info_top, fr_info_bot = get_fr_idx_spl(lobe_spline_data)

        #---params---#
        lobe_theta_mid = float(calc_bending_angle(vtop["lobe"], vbot["lobe"], inj_xz["lobe"]).round(3))
        lobe_lengths = tuple(lobe_spline_data["jet_length"].round(3).tolist())
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
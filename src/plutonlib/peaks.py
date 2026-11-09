from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt 

#---SB spline calculations---#
def get_max_peak_idx(sb,peak_idx):
    """Index of the brightest peak for one jet, or None if the jet has no peaks"""
    if len(peak_idx) == 0:
        return None
    return peak_idx[np.argmax(sb[peak_idx])]

def find_sb_peaks(spline_data,plot=False):
    """Use scipy peak finding to find index of surface brightness peaks along jet spline

    Args:
        spline_data (dict): dict returned by `plutonlib.splines.get_jet_splines_sb()`
        plot (bool, optional): Set True to show a diagnostic plot with peaks. Defaults to False.

    Returns:
        tuple: tuple of indices for top and bottom jets
    """
    inj_idx = spline_data["inj_idx"]
    sb_top = spline_data["log_sb_spline"][inj_idx:]
    sb_bot = spline_data["log_sb_spline"][:inj_idx]

    # arc_bot = np.linspace(0,jet_length[1],len(sb_bot))
    # arc_top = np.linspace(arc_bot[-1],np.sum(jet_length),len(sb_top))
    splines = spline_data["splines"]
    arc = np.r_[0, np.cumsum(np.hypot(*np.diff(splines,axis=0).T))] #true distance along the full path
    arc -= arc[inj_idx] #0 at the injection point, negative on the bottom jet

    arc_top = arc[inj_idx:]
    arc_bot = arc[:inj_idx]

    prom = 0.03
    width = 1
    idx_top, _ = find_peaks(sb_top,prominence=prom,width=width) 
    idx_bot, _ = find_peaks(sb_bot,prominence=prom,width=width) 
    top_max, bot_max = get_max_peak_idx(sb_top,idx_top), get_max_peak_idx(sb_bot,idx_bot)

    if plot:
        f, a = plt.subplots(figsize = (6,6))
        a.plot(arc_top,sb_top, color = 'k')
        a.plot(arc_bot,sb_bot, color = 'k')
        a.scatter(arc_top[idx_top],sb_top[idx_top],s = 4, color = 'red')
        a.scatter(arc_bot[idx_bot],sb_bot[idx_bot],s = 4, color = 'red')

        a.scatter(arc_top[top_max],sb_top[top_max],s=15,color='blue',label="Peak sb (top)",marker="+")
        a.scatter(arc_bot[bot_max],sb_bot[bot_max],s=15,color='blue',label="Peak sb (bot)",marker="x")

        a.set_xlabel("Arc length [kpc]")
        a.set_ylabel("Log sb mJy/beam")
        a.legend()
        plt.show()

    return idx_top,idx_bot

def get_sb_max_idx(spline_data):
    """Get the index of the max sb value from the peak sb array

    Args:
        spline_data (dict): dict returned by `plutonlib.splines.get_jet_splines_sb()`

    Returns:
        tuple: max idx for top and bottom jets, None for a jet with no peaks
    """
    inj_idx = spline_data['inj_idx']
    sb_spl = spline_data['log_sb_spline']

    idx_top,idx_bot = find_sb_peaks(spline_data)
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

def find_jets_without_peaks(spline_data):
    inj_idx = spline_data['inj_idx']
    sb_spl = spline_data['log_sb_spline']

    idx_top,idx_bot = find_sb_peaks(spline_data)
    top_slice  = np.arange(len(sb_spl))[inj_idx:][idx_top] #idx of top jet scipy peaks
    bot_slice  = np.arange(len(sb_spl))[:inj_idx][idx_bot]

    err_top = True if len(top_slice) == 0 else False
    err_bot = True if len(bot_slice) == 0 else False

    return err_top,err_bot

def get_peak_spline_seg(spline_data):
    """Gets the spline segments from injection to peak sb value for top/bottom jets

    Args:
        spline_data (dict): dict returned by `plutonlib.splines.get_jet_splines_sb()`

    Returns:
        top_segment,bot_segment: truncated jet_splines from inj->peak for top/bottom jets
    """

    top_idx,bot_idx = get_sb_max_idx(spline_data)
    inj_idx = spline_data["inj_idx"]
    
    top_segment = spline_data["splines"][inj_idx:top_idx]
    bot_segment = spline_data["splines"][bot_idx:inj_idx][::-1] #reversed, inj->peak  

    return top_segment,bot_segment
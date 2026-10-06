

import plutonlib.analysis as pa
import plutonlib.plot_data as pp
import plutonlib.read_write as prw
import plutonlib.utils as pu
import plutonlib.splines as ps

import matplotlib.pyplot as plt
import scienceplots
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Circle
from matplotlib.collections import PathCollection
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
import matplotlib.animation as animation
from matplotlib.ticker import NullLocator

from contextlib import contextmanager

import math
import time
import numpy as np

def _get_fontsize(fig_size, ttype):
    if ttype == "text":
        return 3 * fig_size
    elif ttype == "ticks":
        return 3 * fig_size 
    elif ttype in ["subplots", "cbar"]:
        return 3 * fig_size
    else:
        return 3 * fig_size
    
def _get_ticksize(fig_size):
    return 0.8*fig_size

def _label_sim(sdata,pdata,is_1D = False):
    ax = pdata.axes[pdata.plot_idx]
    fontsize =  0.5*_get_fontsize(pdata.fig_size,"text") if is_1D else _get_fontsize(pdata.fig_size,"text")

    # if is_1D: #sim labels for 1D plots have a different location
    # ax.text(0.99, 0.10, sdata.get_metadata(pdata.output).time_str,
    #              ha="right", va="center", transform=ax.transAxes, fontsize=0.5*fontsize)
    # if pdata.label:
    #     ax.text(0.99, 0.05, sdata.run_name,
    #         ha="right", va="center", transform=ax.transAxes, fontsize=0.5*fontsize)
    # else:
    linegap = 1.3 * fontsize
    ax.annotate(sdata.get_metadata(pdata.output).time_str,
                    xy=(0.03, 0.98), xycoords=ax.transAxes,
                    xytext=(0, -linegap), textcoords="offset points",
                    ha="left", va="top", fontsize=fontsize)
    if pdata.label:
        ax.annotate(sdata.run_name,
                        xy=(0.03, 0.98), xycoords=ax.transAxes,
                        xytext=(0, -2*linegap - 1), textcoords="offset points",
                        ha="left", va="top", fontsize=fontsize)

    pdata.label = False

def _apply_limits(pdata):
    ax = pdata.axes[pdata.plot_idx]
    if pdata.rotate_row is None or pdata.plot_idx // pdata.ncols != pdata.rotate_row:
        if pdata.xlim:
            ax.set_xlim(pdata.xlim)
        if pdata.ylim:
            ax.set_ylim(pdata.ylim)

def _ticks_labels_limits(sdata, pdata, is_1D=False):
    ax = pdata.axes[pdata.plot_idx]
    tick_fontsize = _get_fontsize(pdata.fig_size, "ticks")
    major = _get_ticksize(pdata.fig_size)
    minor = major * 0.6

    if not is_1D:
        ax.set_aspect('equal', adjustable='datalim')

        ax.tick_params(which='major', direction='in', top=True, right=True,
                    length=major, width=major * 0.15, labelsize=tick_fontsize,
                    labelbottom=False, labelleft=False)
        ax.tick_params(which='minor', direction='in', top=True, right=True,
                    length=minor, width=minor * 0.15, labelsize=tick_fontsize)
        
    else: #different size ticks for 1D plots
        ax.tick_params(which='major', direction='in', length=major, width=major * 0.15,
            labelsize=0.6*tick_fontsize)
        ax.tick_params(which='minor', direction='in', length=minor, width=minor * 0.15,
            labelsize=0.6*tick_fontsize)
        
        if pdata.var_name in ('rho', 'prs'): #format log values for 1D plots, #NOTE sb is already logged
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'$10^{{{x:.0f}}}$'))

    _apply_limits(pdata)
    _label_sim(sdata, pdata, is_1D=is_1D)

def _get_cbar_label(sdata,pdata,**kwargs):
    """Builds colourbar label text from pdata.var_name. Shared by _colourbar and animation_fluid."""
    if pdata.var_name == 'sb':
        label_base = r"$\log_{10}$(SB [mJy beam$^{-1}$])"
        return label_base + f" @ {kwargs.get('freqs')[0]} GHz" if 'freqs' in kwargs else label_base

    elif pdata.var_name == 'alpha' and 'freqs' in kwargs:
        freq_lo = int(kwargs.get('freqs')[0] * 1000)
        freq_hi = int(kwargs.get('freqs')[1] * 1000)
        return f"$\\alpha^{{{freq_hi}}}_{{{freq_lo}}}$"

    try:
        var_label = getattr(sdata.get_var_info(pdata.var_name), "var_name")
        var_units = getattr(sdata.get_var_info(pdata.var_name), "usr_uv").to_string('latex')
    except AttributeError:
        var_units, var_label = "None", "None"
    except TypeError:
        print("pdata.var_choice elements are not of type(str), skipping units/labels")
        var_units, var_label = "None", "None"

    return var_label if var_units == '$\\mathrm{}$' else f"{var_label} [{var_units}]"

def _get_xy_labels(sdata,pdata,var_names):
    """
    Builds {name: label} for input var_names. Shared by _labels_2D and animation_fluid.
    """
    labels = {}
    for name in var_names:
        info = sdata.get_var_info(name)
        label = getattr(info,"var_name")
        units = getattr(info, "usr_uv").to_string('latex')
        labels[name] = label if units == '$\\mathrm{}$' else f"{label} [{units}]"
    return labels

def _colourbar(sdata,pdata,**kwargs):
    cbar_label = _get_cbar_label(sdata, pdata, **kwargs)
    cb = plt.colorbar(
        pdata.im, cax=pdata.cbar_ax, orientation='horizontal',
        label=cbar_label, ticklocation='top'        
    )
    tick_fontsize = _get_fontsize(pdata.fig_size, "cbar")
    cb.ax.tick_params(labelsize=tick_fontsize,
                  length=_get_ticksize(pdata.fig_size),
                  width=_get_ticksize(pdata.fig_size) * 0.15)
    pdata.cbar_ax.xaxis.labelpad = 0.6*tick_fontsize + _get_ticksize(pdata.fig_size)

    if pdata.var_name in ('rho', 'prs'):
        cb.formatter = FuncFormatter(lambda x, _: f'$10^{{{x:.0f}}}$')  # <- via cb, not cb.ax.xaxis
        cb.update_ticks()

@contextmanager
def _plot_style(fig_size):
    with plt.style.context(["science"]):
        plt.rcParams.update({'font.size': _get_fontsize(fig_size, "subplots"), 'text.usetex': False})
        yield

#---plotters for pcmesh,scatter etc---#
def _im_2d_fluid(sdata,pdata,**kwargs):
    ax = pdata.axes[pdata.plot_idx]
    pdata.value = kwargs.get('value') if 'value' in kwargs else 0
    slice_var = pdata.spare_coord #e.g. if plot xz -> profile in y
    slice_to_load = pa.calc_var_prof(sdata,slice_var,value_2D=pdata.value)["slice_2D"]
    fluid_data = sdata.load_fluid_data(
        grid_output=pdata.output,
        var_choice=pdata.coord_choice + [pdata.var_name],
        load_slice=slice_to_load,
    )

    is_log = pdata.var_name in ('rho', 'prs')
    vars_data = (
        np.log10(fluid_data[pdata.var_name])
        if is_log
        else fluid_data[pdata.var_name]
    )

    X = fluid_data[pdata.coord_choice[0]]
    Y = fluid_data[pdata.coord_choice[1]]
    pdata.im = ax.pcolormesh(X,Y,vars_data,cmap=pdata.get_colourmap(var_name = pdata.var_name),vmin=pdata.vmin,vmax=pdata.vmax)

def _im_2d_sb(entry,angles,pdata,freqs=None,**kwargs):
    ax = pdata.axes[pdata.plot_idx]

    if pdata.var_name == 'sb':
        pdata.im = ax.pcolormesh(
            entry["grid_x"], entry["grid_y"],
            entry["log_sb"],
            vmin=pdata.vmin, vmax=pdata.vmax,
            cmap="viridis",          # TODO: match cmap
        )

        contours = np.linspace(pdata.vmin,pdata.vmax,4)
        # contours = entry["contour_levels"]                    
        # print(f"Contours: {contours}")
        ax.contour(
            entry["grid_mx"], entry["grid_my"],
            entry["log_sb"],
            levels=contours, #or entry["contour_levels"]
            colors='white',
        )

    if pdata.var_name == 'alpha':
        if freqs is None:
            raise ValueError("freqs arg is set to none, for spectral index please input array of frequencies")
        pdata.im = ax.pcolormesh(
            entry["grid_x"], entry["grid_y"],
            entry["alpha"][tuple(freqs)],
            vmin=pdata.vmin, vmax=pdata.vmax,
            cmap="turbo",
        )

    ax.annotate(
        f"x={angles[0]}° y={angles[1]}° z={angles[2]}°",
        xy=(0.032, 0.98),
        xycoords=ax.transAxes,
        xytext=(0, 0),          # fixed offset from top
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=_get_fontsize(pdata.fig_size, "text"),
    )

def _im_scatter_particles(sdata,pdata,tr_cut=None,**kwargs):    
    plane_map = {"xy": ["x1","x2"], "xz": ["x1","x3"], "yz": ["x2","x3"]}
    # var_map = {"tr1":"tracer","rho":"density","prs":"pressure"}
    ax = pdata.axes[pdata.plot_idx]
    particle_var = pdata.var_name #var_map.get(pdata.var_name,pdata.var_name)
    particle_data = sdata.load_particle_data(part_output=(pdata.output,),tr_cut=tr_cut)

    is_log = pdata.var_name in ('rho', 'prs')
    vars_data = (
        np.log10(particle_data[particle_var])
        if is_log
        else particle_data[particle_var]
    )

    coord_choice = plane_map[pdata.plane]
    X = particle_data[coord_choice[0]]
    Y = particle_data[coord_choice[1]]
    pdata.im = ax.scatter(X,Y,c=vars_data,s=2.5,cmap = pdata.get_colourmap(var_name = pdata.var_name))

def _scatter_splines_1D(spline_data,sdata,pdata,tick_axis,query_points=None,use_sb =False):
    if use_sb:
        dx,dz = np.diff(spline_data["jet_splines"][:,0]), np.diff(spline_data["jet_splines"][:,1]) 
        var_data = spline_data["log_sb_spline"]
        tick_coord = f"grid_m{tick_axis}"
    else:
        dx,dz = np.diff(spline_data["ccx"]), np.diff(spline_data["ccz"]) 
        is_log = pdata.var_name in ('rho', 'prs') #not sb here, allready logged
        var_data = (
            np.log10(spline_data[pdata.var_name])
            if is_log
            else spline_data[pdata.var_name])
        tick_coord = f"cc{tick_axis}"        

    arc_length = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dz**2))]) #create arc length from jet spline data 
    inj_idx = spline_data["inj_idx"]  # exact, no argmin needed
    from_inj = arc_length - arc_length[inj_idx] #make inj region the 0 point

    ax    = pdata.axes[pdata.plot_idx]
    ax.plot(from_inj, var_data,color = '#d4b9da')

    # --- Top axis: (x, z) coords at evenly spaced arc-length ticks ---#
    if tick_axis:
        ax2 = ax.twiny()
        n_ticks = 8
        tick_idx = np.linspace(0, len(from_inj) - 1, n_ticks, dtype=int)
        tick_pos = from_inj[tick_idx]
        tick_labels = [
            f"{spline_data[tick_coord][i]:.0f}" #NOTE no decimals
            for i in tick_idx
        ]

        label = _get_xy_labels(sdata,pdata,[f"cc{tick_axis}"])[f"cc{tick_axis}"] #hardcoded to give x,y or z label
        ax2.set_xlabel(f"grid {label}", fontsize=0.5*_get_fontsize(pdata.fig_size, "text"))
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(tick_labels, fontsize=0.5*_get_fontsize(pdata.fig_size, "text"))
        ax2.xaxis.set_minor_locator(NullLocator())

    #---Injection region point---#
    ax.scatter(from_inj[inj_idx], var_data[inj_idx], label=f"Injection region", zorder=5,s=15,marker='x',color = 'k')

    colours = ["#222121", '#980043', '#e7298a', '#df65b0', '#c994c7']#taken from PuRd colmap
    col_idx = 0
    if query_points is not None:
        for label, offset in query_points.items():
            idx_plus  = np.argmin(np.abs(from_inj - offset))
            idx_minus = np.argmin(np.abs(from_inj + offset))
            idxs = [idx_minus, idx_plus]
            ax.scatter(from_inj[idxs], var_data[idxs], label=f"{label} = {offset:.2f} kpc", zorder=5,s=12,color = colours[col_idx])
            col_idx += 1

#---Labels/other---#
def _labels_2D(sdata,pdata):
    """Labels for 2D pcolormesh plots where axis are coordinates and not variables
    labels are automatically set by pdata.coord_choice"""
    xy_labels = _get_xy_labels(sdata, pdata,pdata.coord_choice)
    pdata.axes[-pdata.ncols].tick_params(labelbottom=True, labelleft=True, direction='in')

    #pdata.coord_choice is set by pdata.plane and pdata.arr_type e.g. plane = "xz", arr_type = "nc" -> ["ncx","ncz"]
    pdata.axes[-pdata.ncols].set_xlabel(xy_labels[pdata.coord_choice[0]])
    pdata.axes[-pdata.ncols].set_ylabel(xy_labels[pdata.coord_choice[1]])

def _vlim_rasterise(pdata, **kwargs):
    im = pdata.im
    vmin = pdata.vmin if pdata.vmin is not None else im.norm.vmin
    vmax = pdata.vmax if pdata.vmax is not None else im.norm.vmax

    for ax in pdata.axes:
        if 'bg_colour' in kwargs and kwargs.get('bg_colour') is not None:
            bg_colour = im.cmap(im.norm(kwargs.get('bg_colour')))
            ax.set_facecolor(bg_colour)

        for coll in ax.collections:
            if getattr(coll.draw, "_supports_rasterization", False):
                coll.set_rasterized(True)
            coll.set_clim(vmin=vmin, vmax=vmax)

        for artist in ax.get_children():
            if hasattr(artist, 'collections'):
                for coll in artist.collections:
                    if getattr(coll.draw, "_supports_rasterization", False):
                        coll.set_rasterized(True)

def _rotate_row(pdata):
    if pdata.rotate_row is not None:
        for ax in pdata.axes.reshape(pdata.nrows, pdata.ncols)[pdata.rotate_row]:
            xl = ax.get_xlim()
            yl = ax.get_ylim()
            swap = mtransforms.Affine2D([[0, 1, 0],[1, 0, 0],[0, 0, 1]])

            for coll in ax.collections:
                if isinstance(coll, PathCollection):
                    # Scatter: swap x/y offsets directly
                    offsets = coll.get_offsets().copy()
                    coll.set_offsets(offsets[:, [1, 0]])
                else:
                    # QuadMesh / pcolormesh: affine transform trick
                    coll.set_transform(swap + ax.transData)

            # Rotate spline lines
            if ax.lines:
                for line in ax.lines:
                    xdata = line.get_xdata().copy()
                    ydata = line.get_ydata().copy()
                    line.set_xdata(ydata)
                    line.set_ydata(xdata)

            # Rotate circle patches
            if ax.patches:
                for patch in ax.patches:
                    if isinstance(patch, Circle):
                        cx, cy = patch.center
                        patch.center = (cy, cx)

            ax.set_xlim(yl)
            ax.set_ylim(xl)

            if pdata.ylim:
                ax.set_ylim(pdata.ylim)
            if pdata.xlim:
                ax.set_xlim(pdata.xlim)     

def _query_points_2D(sdata,pdata,query_points,label_all_axes = False):
    # plot_idx is assigned as row_start * ncols for the first subplot of each sim
    if pdata.plot_idx != pdata.row_start * pdata.ncols and not label_all_axes: 
        return
    
    ax = pdata.axes[pdata.plot_idx]
    if query_points and query_points.get(sdata) is not None: #NOTE not sure what the diff btwn keying sim or not is 
        points = {k: v for k, v in query_points[sdata].items()}

        fig = ax.get_figure()
        fig_w, fig_h = fig.get_size_inches()
        s = 25 * pdata.fig_size  # scale relative to a 6-inch reference width

        for mkr, coords in points.items():
            # handle both single [x,z] and list of [x,z]
            if isinstance(coords[0], (int, float)):
                coords = [coords]
            for (x_val, z_val) in coords:
                ax.scatter([x_val], [z_val], s=s, zorder=5, color='r', marker=mkr)

def _rotate_query_points(sdata,query_points,rot_mat,plane='xz'):
    """Applies a rotation matrix to provided query points, use for rotated sb plots """
    #NOTE assumes a 2D array of query points
    if plane != 'xz':
        raise NotImplementedError(f"Plane = {plane}, need to implement other rotations")
    qp_rotated = {sdata: {}}
    for qp, (x,z) in query_points[sdata].items():
        rotated = np.array([x,0,z]) @ rot_mat.T

        qp_rotated[sdata][qp] = rotated[[0,2]] 
    return qp_rotated

def _hide_unused_tiles(pdata, n_items):
    """Each sim has ncols of images set by pdata.ncols, 
    this deletes unused tiles in the row and is used to advance the plot idx"""
    sim_rows = math.ceil(n_items / pdata.ncols)
    start = pdata.row_start * pdata.ncols + n_items
    end = (pdata.row_start + sim_rows) * pdata.ncols
    for ax in pdata.axes[start:end]:
        ax.set_visible(False)
    # return sim_rows
    pdata.row_start += sim_rows            # advance past this sim's rows for the next iteration

def _save(fname):
    if fname is not None:
        plt.savefig(f"./{fname}.pdf", bbox_inches='tight', dpi=300)

#---other helpers---#
def _alpha_percentiles(entries, freq_pair, lo=10, hi=90):
    """(lo, hi) percentiles of alpha for one freq pair, pooled over all entries."""
    vals = []
    for entry in entries.values():
        if entry is None or entry.get("alpha") is None:
            continue
        a = entry["alpha"].get(freq_pair)          # 2D array for this pair, or None
        if a is None:
            continue
        a = np.asarray(a).ravel()
        a = a[np.isfinite(a)]                      # drop NaN/inf up front
        if a.size:
            vals.append(a)
    if not vals:
        return None, None
    vals = np.concatenate(vals)
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))

#---Subplot setup---#
def setup_subplots(sim_dict,cbar=True,use_grid_limits=True,subplot_aspect=None,**kwargs):
    """Returns a PlotData obj"""
    plane = kwargs.get('plane') or "xz"
    row_len = kwargs.get('row_len',5) #TODO make pdata?
    arr_type = "nc" #hardcoded array type for 3D plots

    pdata = pp.PlotData(arr_type=arr_type, plane=plane)
    pdata.xlim = kwargs.get('xlim')
    pdata.ylim = kwargs.get('ylim')
    pdata.vmin = kwargs.get('vmin')
    pdata.vmax = kwargs.get('vmax')
    pdata.rotate_row = kwargs.get('rotate_row')
    pdata.fig_size = kwargs.get('fig_size',5)

    with _plot_style(pdata.fig_size):
        pdata.ncols = min(row_len, max(len(v) for v in sim_dict.values()))
        pdata.nrows = sum([math.ceil(len(v) / pdata.ncols) for v in sim_dict.values()]) #total nummber of subplot rows across all sims, some sims have multiple rows
        pdata.row_start = 0 #keep track of the row where the first subplot for this sim begins, first sim first plot lands in row 0
        pdata.plot_idx = pdata.row_start * pdata.ncols #to track index of each subplot, starts at the first row in the first col

        #use pluto.ini grid values, using the max grid span for sim in sim dict #NOTE this is for 2D pcmesh
        if use_grid_limits:
            if pdata.xlim is None: 
                pdata.xlim = max((sim.xyz_lim[0] for sim in sim_dict), key=lambda t: t[1])
            if pdata.ylim is None:
                pdata.ylim = max((sim.xyz_lim[2] for sim in sim_dict), key=lambda t: t[1]) #NOTE ylim is z axis

        #automatically set the aspect ratio of the subplot grid using xylim, arg, else: arbitrary value
        if subplot_aspect is None:
            if pdata.xlim is not None and pdata.ylim is not None:
                subplot_aspect = (pdata.ylim[1] - pdata.ylim[0]) / (pdata.xlim[1] - pdata.xlim[0])
            else:
                subplot_aspect = 0.6

        #set the figure dimensions
        cell_h = pdata.fig_size * subplot_aspect
        cbar_h = 0.125  #fixed cbar thickness
        fig_w = pdata.fig_size * pdata.ncols
        fig_h = cell_h * pdata.nrows + cbar_h
        pdata.fig = plt.figure(figsize=(fig_w, fig_h))

        # Dedicate the top row to the colorbar, rest to data
        pdata.gs = pdata.fig.add_gridspec(
            pdata.nrows + 1, pdata.ncols,
            height_ratios=[cbar_h / fig_h] + [cell_h / fig_h] * pdata.nrows,
            hspace=0, wspace=0,
            left=0.06, right=0.99, top=0.99, bottom=0.06,
        )

        pdata.cbar_ax = pdata.fig.add_subplot(pdata.gs[0, :]) if cbar else None 
        pdata.axes = np.array([[pdata.fig.add_subplot(pdata.gs[i + 1, j]) 
                                for j in range(pdata.ncols)] 
                                for i in range(pdata.nrows)]).flatten()  # flat, always

        return pdata

#---2D plotting functions---#
def fluid(sim_dict,var,fname=None,query_points=None,**kwargs):
    fig_size = kwargs.get('fig_size',5)
    with _plot_style(fig_size):
        pdata = setup_subplots(sim_dict=sim_dict, **kwargs)
        for sim in sim_dict.keys():
            pdata.label = True
            for output in sim_dict[sim]:
                pdata.var_name = var
                pdata.output = output

                _im_2d_fluid(sim,pdata)
                _query_points_2D(sdata=sim,pdata=pdata,query_points=query_points)
                _ticks_labels_limits(sim,pdata)
                pdata.plot_idx += 1

            _labels_2D(sim,pdata)
            _colourbar(sim,pdata,**kwargs)
            _hide_unused_tiles(pdata, len(sim_dict[sim]))

        _rotate_row(pdata)
        _vlim_rasterise(pdata,**kwargs)
        _save(fname)

        plt.close(pdata.fig) 
    return pdata

def jet_splines(sim_dict, var,tr_stop=0.2,fname=None,query_points=None,**kwargs):
    fig_size = kwargs.get('fig_size',5)
    with _plot_style(fig_size):
        pdata = setup_subplots(sim_dict=sim_dict, **kwargs)
        for sim in sim_dict.keys():
            pdata.label = True
            for output in sim_dict[sim]:
                pdata.var_name = var

                pdata.output = sim.simtime_to_part(output,round_val=True) # NOTE: particle time, not sim time, im scatter particles needs part_output
                _im_scatter_particles(sim,pdata,tr_cut=None,**kwargs) #base particles image
                pdata.output = output       # back to sim_time
                _ticks_labels_limits(sim, pdata)  # ticks, limits, time/name text

                # --- spline overlay --- (unique to this plotter) #TODO needs helper
                ax = pdata.axes[pdata.plot_idx]
                spline_data = ps.get_jet_splines_tr(sdata=sim, grid_output=output, tr_stop=tr_stop)
                spline_points = spline_data["spline_points"]
                ax.plot(spline_points[:, 0], spline_points[:, 1], color='k', linestyle='--')

                if query_points and query_points.get(sim) is not None:
                    points = {k: v for k, v in query_points[sim].items() if k != 'roc'}
                    roc = query_points[sim].get('roc')
                    colours = ['#67001f', '#980043', '#e7298a', '#df65b0', '#c994c7']
                    inj_pos = sim.get_injection_region(output)
                    inj_x, inj_z = inj_pos[0].value, inj_pos[2].value

                    dx = np.diff(spline_points[:, 0])
                    dz = np.diff(spline_points[:, 1])
                    arc_length = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dz**2))])
                    dist = np.sqrt((spline_points[:, 0] - inj_x)**2 + (spline_points[:, 1] - inj_z)**2)
                    from_inj = arc_length - arc_length[np.argmin(dist)]

                    for col_idx, (lbl, offset) in enumerate(points.items()):
                        idxs = [np.argmin(np.abs(from_inj + offset)), np.argmin(np.abs(from_inj - offset))]
                        ax.scatter(spline_points[idxs, 0], spline_points[idxs, 1],
                                   s=12, zorder=5, color=colours[col_idx],
                                   label=f"{lbl} = {offset:.2f} kpc")

                    if roc is not None and output == sim_dict[sim][-1]:
                        circle = Circle(xy=(inj_x + roc, 0), radius=roc,
                                        fill=False, edgecolor='k', linewidth=1.5, linestyle=':')
                        ax.add_patch(circle)
                # --- end spline overlay ---

                pdata.plot_idx += 1

            _labels_2D(sim,pdata)
            _colourbar(sim,pdata,**kwargs)
            _hide_unused_tiles(pdata, len(sim_dict[sim]))

        _rotate_row(pdata)
        _vlim_rasterise(pdata)
        _save(fname)

        plt.close(pdata.fig) 
    return pdata

def surf_brightness(sim_dict, angle_dict,freq,redshift,fname=None,query_points=None, **kwargs):
    if isinstance(freq,list):
        raise TypeError("Surface brightness plotting only supports single frequency values")

    entries = {}
    for sim, outputs in sim_dict.items():
        for output in outputs:
            for angles in angle_dict[sim]: #load all the entries first
                entries[(sim, output, tuple(angles))] = sim.load_sb_data(output, angles, freq, redshift)

    sim_dict_angles = { #subplot size should be output*angles
        sim: list(range(len(sim_dict[sim]) * len(angle_dict[sim]))) 
        for sim in sim_dict.keys()
    }
    label_all_axes = kwargs.get('label_all_axes',False)
    fig_size = kwargs.get('fig_size',5)
    with _plot_style(fig_size):
        pdata = setup_subplots(sim_dict=sim_dict_angles, **kwargs)

        if pdata.vmax is None:
            pdata.vmax = float(max(np.log10(np.nanpercentile(e["sb"], 99)) for e in entries.values())) #this should be across all freqs
        if pdata.vmin is None:
            pdata.vmin = pdata.vmax - 1


        for sim, outputs in sim_dict.items():
            pdata.label = True
            for output in outputs:
                for angles in angle_dict[sim]:
                    entry = entries[(sim, output, tuple(angles))]
                    pdata.var_name  = 'sb'
                    pdata.output    = output

                    _im_2d_sb(entry=entry,angles=angles,pdata=pdata)
                    
                    if query_points is not None:
                        query_points_rotated = _rotate_query_points(sdata=sim,query_points=query_points,rot_mat=entry['rot_mat'])
                        _query_points_2D(sdata=sim,pdata=pdata,query_points=query_points_rotated,label_all_axes=label_all_axes)
                    _ticks_labels_limits(sim,pdata)
                    pdata.plot_idx += 1
 
            _labels_2D(sim,pdata)
            _colourbar(sim,pdata,freqs = [freq], **kwargs)        # separate from _colourbar — label differs
            _hide_unused_tiles(pdata, len(outputs) * len(angle_dict[sim]))
 
        _rotate_row(pdata)
        _vlim_rasterise(pdata,**kwargs)
        _save(fname)
 
        plt.close(pdata.fig) 
    return pdata

def surf_brightness_splines(sim_dict,angle_dict,freq,redshift,params,fname=None,query_points=None,**kwargs):
    if isinstance(freq,list):
        raise TypeError("Surface brightness plotting only supports single frequency values")

    entries = {}
    for sim, outputs in sim_dict.items():
        for output in outputs:
            for angles in angle_dict[sim]: #load all the entries first
                entries[(sim, output, tuple(angles))] = sim.load_sb_data(output, angles, freq, redshift)

    sim_dict_angles = { #subplot size should be output*angles
        sim: list(range(len(sim_dict[sim]) * len(angle_dict[sim]))) 
        for sim in sim_dict.keys()
    }
    label_all_axes = kwargs.get('label_all_axes',False)
    fig_size = kwargs.get('fig_size',5)
    with _plot_style(fig_size):
        pdata = setup_subplots(sim_dict=sim_dict_angles, **kwargs)

        if pdata.vmax is None:
            pdata.vmax = float(max(np.log10(np.nanpercentile(e["sb"], 99)) for e in entries.values())) #this should be across all freqs
        if pdata.vmin is None:
            pdata.vmin = pdata.vmax - 1


        for sim, outputs in sim_dict.items():
            pdata.label = True
            for output in outputs:
                for angles in angle_dict[sim]:
                    entry = entries[(sim, output, tuple(angles))]
                    pdata.var_name  = 'sb'
                    pdata.output    = output

                    _im_2d_sb(entry=entry,angles=angles,pdata=pdata)
                    ax = pdata.axes[pdata.plot_idx]
                    spline_data = ps.get_jet_splines_sb(sim,output,angles,freq,redshift,params=params)
                    jet_splines = spline_data['jet_splines']
                    grid_my, grid_mx = spline_data["grid_my"],spline_data["grid_mx"]
                    inj_idx = spline_data['inj_idx']

                    ax.plot(jet_splines[:,0],jet_splines[:,1],color="k",linewidth = 1.5,linestyle = 'dashed')
                    ax.scatter(grid_mx[inj_idx],grid_my[inj_idx],marker="x",color='red')

                    if query_points is not None:
                        query_points_rotated = _rotate_query_points(sdata=sim,query_points=query_points,rot_mat=entry['rot_mat'])
                        _query_points_2D(sdata=sim,pdata=pdata,query_points=query_points_rotated,label_all_axes=label_all_axes)
                    _ticks_labels_limits(sim,pdata)
                    pdata.plot_idx += 1
 
            _labels_2D(sim,pdata)
            _colourbar(sim,pdata,freqs = [freq], **kwargs)        # separate from _colourbar — label differs
            _hide_unused_tiles(pdata, len(outputs) * len(angle_dict[sim]))
 
        _rotate_row(pdata)
        _vlim_rasterise(pdata,**kwargs)
        _save(fname)
 
        plt.close(pdata.fig) 
    return pdata

def spectral_idx(sim_dict, angle_dict,freqs,redshift,fname=None,query_points=None, **kwargs):
    entries = {}
    for sim, outputs in sim_dict.items():
        for output in outputs:
            for angles in angle_dict[sim]: #load all the entries first
                entries[(sim, output, tuple(angles))] = sim.load_sb_data(output, angles, freqs[0], redshift)

    sim_dict_angles = { #subplot size should be output*angles
        sim: list(range(len(sim_dict[sim]) * len(angle_dict[sim]))) 
        for sim in sim_dict.keys()
    }
    label_all_axes = kwargs.get('label_all_axes',False)
    fig_size = kwargs.get('fig_size',5)
    with _plot_style(fig_size):
        pdata = setup_subplots(sim_dict=sim_dict_angles, **kwargs)

        # --- compute global vmin/vmax from alpha percentiles ---
        if pdata.vmin is None or pdata.vmax is None:
            vmin, vmax = _alpha_percentiles(entries,freq_pair=tuple(freqs))
            if pdata.vmin is None:
                pdata.vmin = vmin
            if pdata.vmax is None:
                pdata.vmax = vmax

        for sim, outputs in sim_dict.items():
            pdata.label = True            
            for output in outputs:
                for angles in angle_dict[sim]:
                    entry = entries[(sim, output, tuple(angles))]
                    ax    = pdata.axes[pdata.plot_idx]
                    pdata.var_name   = 'alpha'
                    pdata.output = output

                    _im_2d_sb(entry=entry,angles=angles,pdata=pdata,freqs=freqs)

                    # ax.pcolormesh(
                    #     entry["grid_x"], entry["grid_y"],
                    #     entry["alpha"],
                    #     vmin=pdata.vmin, vmax=pdata.vmax,
                    #     cmap="turbo",
                    # )


                    # # pdata.output     = round(sim.part_to_simtime(output))

                    # ax.annotate(
                    #     f"x={angles[0]}° y={angles[1]}° z={angles[2]}°",
                    #     xy=(0.032, 0.98),
                    #     xycoords=ax.transAxes,
                    #     xytext=(0, 0),          # fixed offset from top
                    #     textcoords="offset points",
                    #     ha="left",
                    #     va="top",
                    #     fontsize=_get_fontsize(pdata.fig_size, "text"),
                    # )

                    if query_points is not None:
                        query_points_rotated = _rotate_query_points(sdata=sim,query_points=query_points,rot_mat=entry['rot_mat'])
                        _query_points_2D(sdata=sim,pdata=pdata,query_points=query_points_rotated,label_all_axes=label_all_axes)
                    _ticks_labels_limits(sim,pdata)
                    pdata.plot_idx += 1
 
            _labels_2D(sim,pdata)
            _colourbar(sim,pdata,freqs = freqs, **kwargs)        # separate from _colourbar — label differs
            _hide_unused_tiles(pdata, len(outputs) * len(angle_dict[sim]))
 
        _rotate_row(pdata)
        _vlim_rasterise(pdata,**kwargs)
        _save(fname)
 
        plt.close(pdata.fig) 
    return pdata

#---1D plotting functions---#
def fluid_1D_tr_spl(sdata,var,grid_output,tick_axis='x',tr_stop=0.2,fname=None,query_points=None,**kwargs):
    """Uses tracer splines to plot a 1D profile of a fluid variable along the jet"""
    fig_size = kwargs.get('fig_size',7)
    with _plot_style(fig_size):
        pdata = setup_subplots(sim_dict={sdata:[grid_output]},cbar=False,use_grid_limits=False,**kwargs)
        pdata.output = grid_output
        pdata.var_name = var
        pdata.label = True

        spline_data = sdata.load_jet_data_tr(["ccx", "ccz", pdata.var_name], grid_output=pdata.output,tr_stop=tr_stop)
        _scatter_splines_1D(spline_data,sdata,pdata,tick_axis,query_points,use_sb=False)

        _ticks_labels_limits(sdata, pdata, is_1D=True)
        label_fontsize = 0.7*_get_fontsize(pdata.fig_size, "text")
        y_label = _get_xy_labels(sdata, pdata, [pdata.var_name])[pdata.var_name]
        ax = pdata.axes[pdata.plot_idx]
        ax.set_ylabel(y_label, fontsize=label_fontsize)
        ax.set_xlabel("Arc length along jet [kpc]", fontsize=label_fontsize)
        ax.legend(fontsize = 0.5*_get_fontsize(pdata.fig_size, "text"),loc='lower left')

        _save(fname)
        plt.close(pdata.fig) 
        return pdata

def surface_brightnes_1D(sdata,grid_output,angles,freq,redshift,tick_axis='x',fname=None,query_points=None,**kwargs):
    fig_size = kwargs.get('fig_size',7)
    with _plot_style(fig_size):
        pdata = setup_subplots(sim_dict={sdata:[grid_output]},cbar=False,use_grid_limits=False,**kwargs)
        pdata.output = grid_output
        pdata.var_name = "sb"
        pdata.label = True

        spline_data = ps.get_jet_splines_sb(
            sim=sdata,
            grid_output=grid_output,
            angles=angles,
            freq=freq,
            redshift=redshift,
            params=None)

        _scatter_splines_1D(spline_data,sdata,pdata,tick_axis,query_points,use_sb=True)

        _ticks_labels_limits(sdata, pdata, is_1D=True)
        label_fontsize = 0.7*_get_fontsize(pdata.fig_size, "text")
        # y_label = r"$\log_{10}$(SB [mJy beam$^{-1}$])"
        y_label = _get_cbar_label(sdata,pdata,freqs=[freq]) #cbar label works fine
        ax = pdata.axes[pdata.plot_idx]
        ax.set_ylabel(y_label, fontsize=label_fontsize)
        ax.set_xlabel("Arc length along jet [kpc]", fontsize=label_fontsize)
        ax.legend(fontsize = 0.5*_get_fontsize(pdata.fig_size, "text"),loc='lower left')
        ax.annotate(
            f"x={angles[0]}° y={angles[1]}° z={angles[2]}°",
            xy=(0.032, 0.98),
            xycoords=ax.transAxes,
            xytext=(0, 0),          # fixed offset from top
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=0.5*_get_fontsize(pdata.fig_size, "text"),
        )
        _save(fname)
        plt.close(pdata.fig) 
        return pdata

#---Animations---#
def animation_fluid(sdata, var_name, load_outputs=None, fps=20, **kwargs):
    def _output_exists(sdata, output):
        try:
            sdata.get_metadata(output)
            return True
        except Exception:
            return False

    start = time.time()
    load_outputs = sdata.load_outputs if load_outputs is None else load_outputs
    load_outputs = [o for o in load_outputs if _output_exists(sdata, o)]
    total_outputs = len(load_outputs)
    print(f"({var_name}): {(time.time() - start):.2f}s - {total_outputs} valid outputs found")

    # fig_size = kwargs.get('fig_size', 7)
    # pdata = setup_subplots(
    #     sim_dict={sdata: [load_outputs[0]]},
    #     xlim=kwargs.get('xlim'), ylim=kwargs.get('ylim'),
    #     vmin=kwargs.get('vmin'), vmax=kwargs.get('vmax'),
    #     plane=kwargs.get('plane'), fig_size=fig_size,
    # )
    pdata = setup_subplots(sim_dict={sdata: [load_outputs[0]]}, **kwargs)

    # pdata.plot_idx = 0
    pdata.var_choice = [var_name]
    pdata.var_name = var_name
    pdata.label = True
    pdata.im = None
    pdata.output = load_outputs[0]

    _im_2d_fluid(sdata, pdata)           
    _ticks_labels_limits(sdata, pdata)   
    _labels_2D(sdata, pdata)
    _colourbar(sdata, pdata, **kwargs)   
    pdata.gs.update(top=0.92, left=0.16, right=0.93) #stops the figure getting cutoff when saving
    time_text = pdata.axes[pdata.plot_idx].texts[0]   

    def update_img(n):
        try:
            if pdata.im is not None:
                pdata.im.remove()
            pdata.output = load_outputs[n]
            _im_2d_fluid(sdata, pdata)
            time_text.set_text(sdata.get_metadata(pdata.output).time_str)
            print(f"({var_name}): {(time.time() - start):.2f}s - frame {n} done")
        except (OSError, KeyError, FileNotFoundError) as e:
            print(f"({var_name}): skipping output {pdata.output} — {type(e).__name__}: {e}")
        return pdata.im

    ani = animation.FuncAnimation(
        pdata.fig, update_img,
        frames=total_outputs, interval=30, save_count=total_outputs,
    )

    writergif = animation.PillowWriter(fps=fps)
    ani.save(f"{sdata.save_dir}/{sdata.run_name}.gif", writer=writergif, dpi=200)
    print(f"({var_name}): {(time.time() - start):.2f}s - animation saved")
    return ani

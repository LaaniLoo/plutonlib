

import plutonlib.analysis as pa
import plutonlib.plot as pp
import plutonlib.read_write as prw

import numpy as np
import matplotlib.pyplot as plt

import scienceplots
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Circle
from matplotlib.collections import PathCollection

import warnings
import math

import resource

def _get_fontsize(fig_size, ttype):
    if ttype == "text":
        return 3 * fig_size
    elif ttype in ["subplots", "cbar"]:
        return 3 * fig_size
    else:
        return 3 * fig_size
    
def _get_ticksize(fig_size):
    return 0.5*fig_size

def setup_subplots(sim_dict,xlim=None,ylim=None,vmin=None, vmax=None,fig_size = 6,row_len = 5):
    pdata = pp.PlotData(var_choice=[], plane="xz", show_cbar=False)
    pdata.xlim = xlim
    pdata.ylim = ylim
    pdata.vmin = vmin    
    pdata.vmax = vmax
    pdata.fig_size = fig_size

    with plt.style.context(["science"]):
        plt.rcParams.update({'font.size': _get_fontsize(fig_size,"subplots"), 'text.usetex': False})
        # ncols_list = [len(outputs) for outputs in sim_dict.values()]
        # ncols = ncols_list[0]
        # nrows = len(sim_dict.keys())
        # if len(set(ncols_list)) > 1:
        #     raise ValueError(f"Number of outputs per sim don't match: {ncols_list}")
        ncols = min(row_len, max(len(v) for v in sim_dict.values()))
        rows_per_sim = [math.ceil(len(v) / ncols) for v in sim_dict.values()]
        nrows = sum(rows_per_sim)

        cell_w = fig_size  # inches per column
        if xlim and ylim:
            data_aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            cell_h = cell_w * data_aspect
        else: 
            # cell_h = cell_w

            #use pluto.ini grid values, using the max grid span for sim in sim dict
            xlim = max((sim.xyz_lim[0] for sim in sim_dict), key=lambda t: t[1])
            # ylim = max((sim.xyz_lim[1] for sim in sim_dict), key=lambda t: t[1])
            ylim = max((sim.xyz_lim[2] for sim in sim_dict), key=lambda t: t[1]) #NOTE ylim is z axis
            data_aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            cell_h = cell_w * data_aspect

        cbar_h = 0.125  # fixed inches for colorbar row — not tied to data cells
        fig_w = cell_w * ncols
        fig_h = cell_h * nrows + cbar_h

        pdata.fig = plt.figure(figsize=(fig_w, fig_h))

        # Dedicate the top row to the colorbar, rest to data
        pdata.gs = pdata.fig.add_gridspec(
            nrows + 1, ncols,
            height_ratios=[cbar_h / fig_h] + [cell_h / fig_h] * nrows,
            hspace=0, wspace=0,
            left=0.06, right=0.99, top=0.99, bottom=0.06,
        )

        pdata.cbar_ax = pdata.fig.add_subplot(pdata.gs[0, :])  # spans all columns
        # pdata.axes = np.array([[fig.add_subplot(pdata.gs[i + 1, j]) for j in range(ncols)]
        #                 for i in range(nrows)])
        # pdata.axes_flat = pdata.axes.flatten()

        pdata.nrows = nrows
        pdata.ncols = ncols
        pdata.axes = np.array([[pdata.fig.add_subplot(pdata.gs[i + 1, j]) 
                                for j in range(ncols)] 
                                for i in range(nrows)]).flatten()  # flat, always
        # no axes_flat needed at all

        return pdata

def _ticks_labels_limits(sim,pdata):
    ax = pdata.axes[pdata.plot_idx]
    major = _get_ticksize(pdata.fig_size)
    minor = major * 0.6          # or 0.6 — whatever ratio looks right

    ax.tick_params(which='major', direction='in', top=True, right=True,
                   length=major, width=major * 0.15,
                   labelbottom=False, labelleft=False)
    ax.tick_params(which='minor', direction='in', top=True, right=True,
                   length=minor, width=minor * 0.15)

    if pdata.rotate_row is None or pdata.plot_idx // pdata.ncols != pdata.rotate_row:
        if pdata.xlim: # xlim kwarg to change x limits
            ax.set_xlim(pdata.xlim) 

        if pdata.ylim:
            ax.set_ylim(pdata.ylim) 

    linegap = 1.3 * _get_fontsize(pdata.fig_size,"text")
    ax.annotate(
        sim.get_metadata(pdata.output).time_str,
        xy=(0.03, 0.98),
        xycoords=ax.transAxes,
        xytext=(0, -linegap),          # fixed offset from top
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=_get_fontsize(pdata.fig_size, "text"),
    )

    if pdata.label:
        ax.annotate(
            sim.run_name,
            xy=(0.03, 0.98),
            xycoords=ax.transAxes,
            xytext=(0, -2*linegap - 1),          # fixed 10 point vertical offset
            textcoords="offset points",
            ha="left", va="top",
            fontsize=_get_fontsize(pdata.fig_size, "text"),
        )
    pdata.label = False

    # if pdata.xlim or pdata.ylim is None:
        # print("WARNING: xlim or ylim are None, aspect ratio will appear incorrect")

def _colourbar(pdata,**kwargs):
    if kwargs.get('hide_cbar', False):
        pdata.cbar_ax.set_visible(False)
        return

    im = pdata.axes[0].collections[0]
    if pdata.var_name == 'sb': #surface brightness
        label_base = r"$\log_{10}$(SB [mJy beam$^{-1}$])"
        cbar_label = label_base + f" @ {kwargs.get('freqs')[0]} GHz" if 'freqs' in kwargs else label_base
        cb = plt.colorbar(im, cax=pdata.cbar_ax, orientation='horizontal', label=cbar_label)

    elif pdata.var_name == 'alpha': #spectral index
        if 'freqs' in kwargs:
            freq_lo = int(kwargs.get('freqs')[0] * 1000) 
            freq_hi = int(kwargs.get('freqs')[1] * 1000) 
            cb = plt.colorbar(im, cax=pdata.cbar_ax, orientation='horizontal', label=f"$\\alpha^{{{freq_hi}}}_{{{freq_lo}}}$")
        else:
            cb = plt.colorbar(im, cax=pdata.cbar_ax, orientation='horizontal', label=f"None")

    else: #normal fluid var labels
        cb = plt.colorbar(im, cax=pdata.cbar_ax, orientation='horizontal', label=pdata.extras['cbar_labels'][0])

    # cb.ax.tick_params(labelsize=_get_fontsize(pdata.fig_size,"cbar"))     
    cb.ax.tick_params(labelsize=_get_fontsize(pdata.fig_size, "cbar"),
                  length=_get_ticksize(pdata.fig_size),
                  width=_get_ticksize(pdata.fig_size) * 0.15)
    pdata.cbar_ax.xaxis.set_label_position('top')

    is_log = pdata.var_name in ('rho', 'prs')
    if is_log:
        cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'$10^{{{x:.0f}}}$'))
        
    pdata.cbar_ax.xaxis.tick_top()

def _labels_simple(sdata,pdata):
    if pdata.extras is None:
        pdata.extras = pp.plot_extras(sdata,pdata)
    pdata.axes[-pdata.ncols].tick_params(labelbottom=True, labelleft=True, direction='in')
    pdata.axes[-pdata.ncols].set_xlabel(pdata.extras['xy_labels']['ncx'])
    pdata.axes[-pdata.ncols].set_ylabel(pdata.extras['xy_labels']['ncz'])

def _vlim_rasterise(pdata, **kwargs):
    im = pdata.axes[0].collections[0]
    vmin = pdata.vmin if pdata.vmin is not None else im.norm.vmin
    vmax = pdata.vmax if pdata.vmax is not None else im.norm.vmax

    for ax in pdata.axes:
        if 'bg_colour' in kwargs and kwargs.get('bg_colour') is not None:
            im = ax.collections[0]
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

def _query_points_simple(sim,pdata,query_points,label_all_axes = False):
    # plot_idx is assigned as row_start * ncols for the first subplot of each sim
    if pdata.plot_idx != pdata.row_start * pdata.ncols and not label_all_axes: 
        return
    
    ax = pdata.axes[pdata.plot_idx]
    if query_points and query_points.get(sim) is not None: #NOTE not sure what the diff btwn keying sim or not is 
        points = {k: v for k, v in query_points[sim].items()}

        fig = ax.get_figure()
        fig_w, fig_h = fig.get_size_inches()
        s = 25 * pdata.fig_size  # scale relative to a 6-inch reference width

        for mkr, coords in points.items():
            # handle both single [x,z] and list of [x,z]
            if isinstance(coords[0], (int, float)):
                coords = [coords]
            for (x_val, z_val) in coords:
                ax.scatter([x_val], [z_val], s=s, zorder=5, color='r', marker=mkr)

def _rotate_query_points(sim,query_points,rot_mat,plane='xz'):
    """Applies a rotation matrix to provided query points, use for rotated sb plots """
    #NOTE assumes a 2D array of query points
    if plane != 'xz':
        raise NotImplementedError(f"Plane = {plane}, need to implement other rotations")
    qp_rotated = {sim: {}}
    for qp, (x,z) in query_points[sim].items():
        rotated = np.array([x,0,z]) @ rot_mat.T

        qp_rotated[sim][qp] = rotated[[0,2]] 
    return qp_rotated

def _save(fname):
    if fname is not None:
        plt.savefig(f"./{fname}.pdf", bbox_inches='tight', dpi=300)
#---plotting functions---#

def fluid(sim_dict,var,fname=None,rotate_row = None,query_points=None, **kwargs):
    fig_size = kwargs.get('fig_size',7)
    with plt.style.context(["science"]):
        plt.rcParams.update({'font.size': _get_fontsize(fig_size,"subplots"), 'text.usetex': False})

        pdata = setup_subplots(
            sim_dict=sim_dict,
            xlim=kwargs.get('xlim'),
            ylim=kwargs.get('ylim'),
            vmin=kwargs.get('vmin'),   # pass through once here
            vmax=kwargs.get('vmax'),
            fig_size=fig_size
        )
        pdata.rotate_row = rotate_row

        # plot_idx = 0
        pdata.row_start = 0
        for sim in sim_dict.keys():
            pdata.label = True
            plot_idx = pdata.row_start *  pdata.ncols

            for output in sim_dict[sim]:
                pdata.var_choice = [var]
                pdata.var_name = var
                pdata.plot_idx = plot_idx
                pdata.output = output

                pp.pcmesh_3d_fluid(sim, pdata=pdata)

                _query_points_simple(sim=sim,pdata=pdata,query_points=query_points)
                _ticks_labels_limits(sim,pdata)

                plot_idx += 1

            _labels_simple(sim,pdata)

            n_items = len(sim_dict[sim])  
            sim_rows = math.ceil(n_items /  pdata.ncols) #number of rows for this simulation
            for ax in pdata.axes[pdata.row_start* pdata.ncols + n_items : (pdata.row_start + sim_rows)* pdata.ncols]:
                ax.set_visible(False)        # blank out unused trailing tiles in this sim's last row
            pdata.row_start += sim_rows            # advance past this sim's rows for the next iteration

        _rotate_row(pdata)
        _vlim_rasterise(pdata,**kwargs)
        _colourbar(pdata,**kwargs)
        _save(fname)

        plt.show()

def jet_splines(sim_dict, var, fname=None, tr_stop=0.2, query_points=None, rotate_row=None, **kwargs):
    fig_size = kwargs.get('fig_size',11)
    with plt.style.context(["science"]):
        plt.rcParams.update({'font.size': _get_fontsize(fig_size,"subplots"), 'text.usetex': False})

        pdata = setup_subplots(
            sim_dict=sim_dict,
            xlim=kwargs.get('xlim'),
            ylim=kwargs.get('ylim'),
            vmin=kwargs.get('vmin'),
            vmax=kwargs.get('vmax'),
            fig_size=fig_size, # wider cells for splines
        )
        pdata.rotate_row = rotate_row

        ncols = pdata.ncols
        row_start = 0
        for sim in sim_dict.keys():
            pdata.label = True
            plot_idx = row_start * ncols

            for output in sim_dict[sim]:
                metadata = sim.get_metadata(output)
                sim_time = output
                part_time = round(sim.simtime_to_part(sim_time))

                pdata.var_choice = [var]
                pdata.var_name = var
                pdata.plot_idx = plot_idx

                pdata.output = part_time       # NOTE: particle time, not sim time
                pp.scatter_3d_particles(sim, tr_cut=None, pdata=pdata, **kwargs)
                pdata.output = sim_time       # NOTE: particle time, not sim time

                _ticks_labels_limits(sim, pdata)  # ticks, limits, time/name text

                # --- spline overlay --- (unique to this plotter)
                ax = pdata.axes[plot_idx]
                spline_data = pa.get_jet_splines(sdata=sim, output=output, tr_stop=tr_stop)
                spline_points = spline_data["spline_points"]
                ax.plot(spline_points[:, 0], spline_points[:, 1], color='k', linestyle='--')

                if query_points and query_points.get(sim) is not None:
                    points = {k: v for k, v in query_points[sim].items() if k != 'roc'}
                    roc = query_points[sim].get('roc')
                    colours = ['#67001f', '#980043', '#e7298a', '#df65b0', '#c994c7']
                    inj_pos = sim.get_injection_region(sim_time)
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

                plot_idx += 1

            _labels_simple(sim,pdata)

            n_items = len(sim_dict[sim])  
            sim_rows = math.ceil(n_items / ncols) #number of rows for this simulation
            for ax in pdata.axes[row_start*ncols + n_items : (row_start + sim_rows)*ncols]:
                ax.set_visible(False)        # blank out unused trailing tiles in this sim's last row
            row_start += sim_rows            # advance past this sim's rows for the next iteration

        _rotate_row(pdata)
        _vlim_rasterise(pdata)
        _colourbar(pdata,**kwargs)
        _save(fname)

        plt.show()

def surface_brightness(sim_dict, angle_dict,freq,redshift, fname=None,rotate_row =None,query_points=None, **kwargs):
    #TODO most updated plotting syntax here, update all funcs to follow

    if isinstance(freq,list):
        raise TypeError("Surface brightness plotting only supports single frequency values")

    sb_data = {"metadata": {"angle_dict": angle_dict, "freqs": [freq]}}
    for sim, outputs in sim_dict.items():
        sim_result = prw.load_sb_hdf5(sim, outputs, angle_dict[sim], [freq], redshift, plane="xz")
        sb_data[sim] = sim_result[sim]

    fig_size = kwargs.get('fig_size',7)
    label_all_axes = kwargs.get('label_all_axes',False)
    row_len = kwargs.get('row_len',5)
    with plt.style.context(["science"]):
        plt.rcParams.update({'font.size':_get_fontsize(fig_size,"subplots"), 'text.usetex': False})
        
        angle_dict = sb_data["metadata"]["angle_dict"]
        # sim_dict_angles = {sim: angle_dict[sim] for sim in sim_dict.keys()}
        sim_dict_angles = { #subplot size should be output*angles
            sim: list(range(len(sim_dict[sim]) * len(angle_dict[sim]))) 
            for sim in sim_dict.keys()
        }


        pdata = setup_subplots(
            sim_dict=sim_dict_angles,
            xlim=kwargs.get('xlim'),
            ylim=kwargs.get('ylim'),
            vmin=kwargs.get('vmin'),   # pass through once here
            vmax=kwargs.get('vmax'),
            fig_size=fig_size,
            row_len=row_len,
        )
        pdata.rotate_row = rotate_row

        # --- compute global vmin/vmax from cache if not provided ---
        if pdata.vmin is None or pdata.vmax is None:
            all_vmax = [
                np.log10(np.nanpercentile(cache_entry["sb"], 99)) #.value
                for key, sim_cache in sb_data.items()
                if key != "metadata"
                for output_cache in sim_cache.values()
                for cache_entry in output_cache.values()
            ]
            pdata.vmax = pdata.vmax or float(np.nanmax(all_vmax))
            pdata.vmin = pdata.vmin or (pdata.vmax - 1)   

        # plot_idx = 0
        # ncols = pdata.ncols
        pdata.row_start = 0
        for sim, outputs in sim_dict.items():
            pdata.label = True
            plot_idx = pdata.row_start * pdata.ncols

            for output in outputs:
                for angles in angle_dict[sim]:
                    angle_key = tuple(angles)
                    entry = sb_data[sim][output][angle_key]   
                    obs   = entry["obs_properties"]
                    ax    = pdata.axes[plot_idx]

                    ax.pcolormesh(
                        obs["grid_x"], obs["grid_y"],
                        entry["log_sb"],
                        vmin=pdata.vmin, vmax=pdata.vmax,
                        cmap="viridis",          # TODO: match cmap
                    )
                    ax.contour(
                        obs["grid_mx"], obs["grid_my"],
                        entry["log_sb"],
                        levels=entry["contour_levels"], #TODO add contours as kwarg  
                        colors='white',
                    )

                    pdata.var_name  = 'sb'
                    pdata.var_choice = ['sb']
                    pdata.plot_idx  = plot_idx
                    pdata.output    = round(sim.part_to_simtime(output))
                    
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
                    if query_points is not None:
                        query_points_rotated = _rotate_query_points(sim=sim,query_points=query_points,rot_mat=obs['rot_mat'])
                        _query_points_simple(sim=sim,pdata=pdata,query_points=query_points_rotated,label_all_axes=label_all_axes)
                    _ticks_labels_limits(sim,pdata)
                    plot_idx += 1

            _labels_simple(sim,pdata)

            n_items = len(outputs) * len(angle_dict[sim]) #loops on outputs and angles
            sim_rows = math.ceil(n_items / pdata.ncols) #number of rows for this simulation
            for ax in pdata.axes[pdata.row_start*pdata.ncols + n_items : (pdata.row_start + sim_rows)*pdata.ncols]:
                ax.set_visible(False)        # blank out unused trailing tiles in this sim's last row
            pdata.row_start += sim_rows            # advance past this sim's rows for the next iteration

        _rotate_row(pdata)
        _vlim_rasterise(pdata,**kwargs)
        _colourbar(pdata,freqs = sb_data['metadata']['freqs'], **kwargs)        # separate from _colourbar — label differs
        _save(fname)

        # plt.show() #NOTE turn on if you dont wanna do .fig
        plt.close(pdata.fig) 
    return pdata

def spectral_idx(sim_dict, sb_data, fname=None, rotate_row=None, **kwargs):
    fig_size = kwargs.get('fig_size',7)
    with plt.style.context(["science"]):
        plt.rcParams.update({'font.size':_get_fontsize(fig_size,"subplots"), 'text.usetex': False})

        angle_dict = sb_data["metadata"]["angle_dict"]
        freqs = sb_data["metadata"]["freq"]
        # sim_dict_angles = {sim: angle_dict[sim] for sim in sim_dict.keys()}
        sim_dict_angles = { #subplot size should be output*angles
            sim: list(range(len(sim_dict[sim]) * len(angle_dict[sim]))) 
            for sim in sim_dict.keys()
        }

        pdata = setup_subplots(
            sim_dict=sim_dict_angles,
            xlim=kwargs.get('xlim'),
            ylim=kwargs.get('ylim'),
            vmin=kwargs.get('vmin'),
            vmax=kwargs.get('vmax'),
            fig_size=fig_size,
        )
        pdata.rotate_row = rotate_row

        # --- compute global vmin/vmax from alpha percentiles ---
        if pdata.vmin is None or pdata.vmax is None:
            all_alpha = [
                cache_entry["alpha"]
                for key, sim_cache in sb_data.items()
                if key != "metadata"
                for output_cache in sim_cache.values()
                for cache_entry in output_cache.values()
                if cache_entry["alpha"] is not None
            ]
            pdata.vmin = float(np.nanpercentile(np.concatenate([a.flatten() for a in all_alpha]), 10))
            pdata.vmax = float(np.nanpercentile(np.concatenate([a.flatten() for a in all_alpha]), 90))

        ncols = pdata.ncols
        row_start = 0
        for sim, outputs in sim_dict.items():
            pdata.label = True
            plot_idx = row_start * ncols
            for output in outputs:
                for angles in angle_dict[sim]:
                    angle_key = tuple(angles)
                    entry = sb_data[sim][output][angle_key]
                    obs   = entry["obs_properties"]
                    ax    = pdata.axes[plot_idx]

                    ax.pcolormesh(
                        obs["grid_x"], obs["grid_y"],
                        entry["alpha"],
                        vmin=pdata.vmin, vmax=pdata.vmax,
                        cmap="turbo",
                    )

                    pdata.var_name   = 'alpha'
                    pdata.var_choice = ['alpha']
                    pdata.plot_idx   = plot_idx
                    pdata.output     = round(sim.part_to_simtime(output))

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

                    # _query_points_simple(sim=sim,pdata=pdata,query_points=query_points)
                    _ticks_labels_limits(sim, pdata)

                    plot_idx += 1

            _labels_simple(sim, pdata)

            n_items = len(outputs) * len(angle_dict[sim]) #loops on outputs and angles
            sim_rows = math.ceil(n_items / ncols) #number of rows for this simulation
            for ax in pdata.axes[row_start*ncols + n_items : (row_start + sim_rows)*ncols]:
                ax.set_visible(False)        # blank out unused trailing tiles in this sim's last row
            row_start += sim_rows            # advance past this sim's rows for the next iteration

        _rotate_row(pdata)
        _vlim_rasterise(pdata,**kwargs)
        _colourbar(pdata,**kwargs)
        _save(fname)

        plt.show()
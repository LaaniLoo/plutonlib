import plutonlib.utils as pu
import matplotlib as mpl

class PlotData:
    """
    Class used to store data that needs to be accessed btwn multiple plotting functions e.g. 
    * matplotlib figures: fig
    * matplotlib axes: axes
    * plot_extras returns: extras
    * data files from SimulationData: load_outputs
    * function args for vars and coords: sel_var, sel_coord
    """
    def __init__(self,arr_type = "nc",plane = "xz",var_choice = None,output = None,show_cbar=True, **kwargs):
        self.output = output 

        self.sel_coord = None
        self.sel_var = None #NOTE not sure what this is for might be a duplicate of below
        self.var_name = None #keep track of var_name in loops across var_choice 

        self.load_outputs = None #used for sel load_outputs in plots

        self.fig = None
        self.fig_size = 7
        self.im = None #Used for storing a matplotlib pcolormesh image
        self.axes = None
        self.axes_flat = None
        self.gs = None
        self.cbar_ax = None 

        self.xlim = None
        self.ylim = None
        self.vmin = None
        self.vmax = None

        self.plot_idx = 0
        self.row_start = 0 
        self.rotate_row = None
        self.ncols = None
        self.nrows = None
        # self.extras = None #storing plot_extras() data
        self.value = 0
        self.__dict__.update(kwargs)

        self.var_choice = var_choice
        self.arr_type = arr_type
        self.plane = plane
        self.show_cbar = show_cbar
        self.cbar_label = None
        
        self.cmap_dict = {
            "vx1" : "hot",
            "vx2" : "hot",
            "vx3" : "hot",
            "rho": "inferno",
            "prs": "viridis"
        }

    def get_colourmap(self,var_name):  

        if var_name not in self.cmap_dict.keys():
            return mpl.colormaps["PuRd"]

        return mpl.colormaps[self.cmap_dict[var_name]]
        
    @property
    def spare_coord(self):
        '''e.g. returns y if profile is xz etc...'''
        un_mapped_coords = [pu.map_coord_name(c) for c in self.coord_choice]
        spare_coord = ({'x1','x2','x3'} - set(un_mapped_coords)).pop()
        
        return pu.unmap_coord_name(spare_coord)
    
    @property
    def coord_choice(self):
        x,y,z = pu.get_coord_names(arr_type=self.arr_type)
        plane_map = {
            "xy": [x,y],
            "xz": [x,z],
            "yz": [y,z],
        }

        if self.plane not in plane_map:
            raise KeyError(f"{self.plane} not recognised plane, see {plane_map}")

        return plane_map[self.plane]


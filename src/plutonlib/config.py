import os


start_dir = r"/mnt/g/My Drive/Honours S4E (2025)/Notebooks/" #starting directory, used to save files starting in this dir

plutodir = os.environ["PLUTO_DIR"]

profiles = {
    "all": ["x1", "x2", "x3", "rho", "prs", "vx1", "vx2", "vx3", "SimTime"],
    "2d_rho_prs": ["x1", "x2", "rho", "prs"],
    "2d_vel": ["x1", "x2", "vx1", "vx2"],

    "yz_rho_prs": ["x2", "x3", "rho", "prs"],
    "yz_vel": ["x2", "x3", 'vx2','vx3'],

    "xz_rho_prs": ['x1','x3','rho','prs'],
    "xz_vel": ['x1','x3','vx1','vx3'],
}

coord_systems = {
    "CYLINDRICAL": ['r','z','theta'],
    "CARTESIAN": ['x','y','z'],
    "SPHERICAL": ['r','theta','phi']  
}
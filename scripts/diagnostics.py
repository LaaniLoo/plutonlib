import plutonlib.simulations as ps
import plutonlib.analysis as pa
import plutonlib.plot as pp 
import plutokore as pk

import numpy as np
import matplotlib.pyplot as plt

import os

#NOTE script needs opt args eg for praise 
outputs = "last"

print(f"Current working dir: {os.getcwd()}")
sim_type = os.path.basename(os.path.dirname(os.getcwd()))
run_name = os.path.basename(os.getcwd())

sim = ps.SimulationData(sim_type=sim_type,run_name=run_name,ini_file="jet_units",load_outputs=outputs)
inj_loc = sim.get_injection_region(output=sim.load_outputs[-1])[0].value #location of the injection region for moving injection regions

diagnostic_path = os.path.join(sim.wdir,"diagnostics")
os.makedirs(diagnostic_path,exist_ok=True)

pp.plot_sim_fluid(sim,var_choice=["prs"],plane = "xz",save = 3,save_dir = diagnostic_path) #plot midplane density colourmap
pp.plot_1D_slice("x3","vx3",sim,value_dict={"x1":inj_loc},save=3,save_dir = diagnostic_path)
pa.plot_ram_pressure(sim,save =3, save_dir = diagnostic_path)
pp.plot_inj_region(sim,save =3, save_dir = diagnostic_path)
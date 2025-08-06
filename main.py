from grating import *
from sample import *
from propagator import *
import numpy as np
from parameters import *
from detector import *
from plotting import *
import time
import threading
import os
import csv
import matplotlib.pyplot as plt
"""
samp2d = Sample(t_samp_in_mm = t_samp_in_mm,
                d_sph_in_um = d_sph_in_um,
                f_sph = f_sph,
                mat_sph_type = mat_sph_type,
                mat_bkg_type = mat_bkg_type,
                mat_sph = mat_sph, 
                mat_bkg = mat_bkg,
                rho_sph_in_g_cm3 = rho_sph_in_g_cm3, 
                rho_bkg_in_g_cm3 = rho_bkg_in_g_cm3) 

prop = Propagator(grat = grat1d,
                     samp = samp2d)
"""
grat1d = Grating(px_in_um = px_in_um)


det = Detector(px_in_um= px_in_um)
bin_grat = grat1d.create_grating()
wavefld_bg = np.ones(img_size_in_pix)

slice_profiles_path =  "slices_data.npz"

def print_elapsed_time(start_time, stop_event):
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        print(f"\rElapsed time: {elapsed:.1f} seconds", end="")
        time.sleep(1)
    print()  # Move to next line after stopping

start = time.time()
stop_event = threading.Event()
timer_thread = threading.Thread(target=print_elapsed_time, args=(start, stop_event))
timer_thread.start()

try:
    """
    if os.path.exists(slice_profiles_path):
        # Load from file
        data = np.load(slice_profiles_path)
        slc2d_sph_padded = data['slc2d_sph_padded']
        slc2d_bkg_padded = data['slc2d_bkg_padded'] 
        print()
        print("Loaded slice profiles from file.")
    else:
        # Create and save
        slc2d_sph_padded, slc2d_bkg_padded = samp2d.create_projected_1d_slices(seed=0)
        np.savez(slice_profiles_path, slc2d_sph_padded=slc2d_sph_padded, slc2d_bkg_padded=slc2d_bkg_padded)
        print()
        print("Created and saved slice profiles.")
    """
    #Always create new sample

    results = []
    sphere_sizes_in_um = numbers = np.arange(2, 31)
    for sphere_size in sphere_sizes_in_um:
        samp2d = Sample(t_samp_in_mm = t_samp_in_mm,
                    d_sph_in_um = sphere_size,
                    f_sph = f_sph,
                    mat_sph_type = mat_sph_type,
                    mat_bkg_type = mat_bkg_type,
                    mat_sph = mat_sph, 
                    mat_bkg = mat_bkg,
                    rho_sph_in_g_cm3 = rho_sph_in_g_cm3, 
                    rho_bkg_in_g_cm3 = rho_bkg_in_g_cm3) 
        prop = Propagator(grat = grat1d,
                     samp = samp2d)
        slc2d_sph_padded, slc2d_bkg_padded = samp2d.create_projected_1d_slices(seed=0)
        np.savez(slice_profiles_path, slc2d_sph_padded=slc2d_sph_padded, slc2d_bkg_padded=slc2d_bkg_padded)
        visibility, epsilon = save_visibility_epsilon(det, prop,  wavefld_bg, bin_grat)
        print(f"Visibility with sample: {visibility.real:.3f} at Energy: {E_in_keV:.1f} keV at diameter: {sphere_size:.2f}")
        results.append([sphere_size, visibility.real, epsilon.real])
        os.remove(slice_profiles_path)
        del slc2d_sph_padded, slc2d_bkg_padded

    with open("visibility_results_40keV.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sphere size (um)", "Visibility", "Epsilon"])
        writer.writerows(results)
    
    #plot_single_slice_pair(slc2d_sph_padded, slc2d_bkg_padded, slice_idx=100, save_plot=True)

finally:

    end = time.time() 
    stop_event.set()
    timer_thread.join()
    print(f"Total simulation time: {end - start:.2f} seconds.")
    print()



# --- Save all the simulation parameters --------------------------------------

sim_param = {   
                "Energy in keV": E_in_keV, 
                "Simulated pixel size in m": sim_pix_size_in_m, 
                "Simulated image size in pix": img_size_in_pix,
                "Grating period in X in um": grat1d.px_in_um,                    
                "Sample size in pix": samp_size_in_pix,
                "Sample thickness in mm": samp2d.t_samp_in_mm,
                "Sphere diameter in um": samp2d.d_sph_in_um,
                "Packing fraction of the spheres": samp2d.f_sph, 
                "Sphere material": samp2d.mat_sph,
                "Background material": samp2d.mat_bkg,
                "Number of spheres": samp2d.num_sph_2dslice,
                "Number of slices": samp2d.num_slc,
                "Thickness of a sample slice in pix": t_slc_in_pix,
                "Talbot distance in m": round(prop.talbot_in_m, 1),
                "Grating-to-detector distance in cm": round(prop.grat2det_in_m * 100, 1),
                "Grating-to-sample distance in cm": round(prop.grat2samp_in_m * 100, 1),
                "Sample-to-detector distance in cm": round(prop.samp2det_in_m * 100, 1),
                "Propagation distance in cm": round(prop_in_m * 100, 1), 
                "Simulation time in min": round((end - start) / 60, 1)
            }

with open("sim_param.csv",
          "w", 
          newline = "") as file:
    w = csv.writer(file)
    w.writerows(sim_param.items())

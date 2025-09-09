from grating import *
from propagator import *
import numpy as np
from skull_parameter import *
from detector import *
from plotting import *
from skull_sample import *
import time
import threading
import os
import csv
import cv2  
import matplotlib.pyplot as plt

sampskull = Sample_Skull(mat_bone = mat_bone, mat_air = mat_air) 

grat1d = Grating(px_in_um = px_in_um)

det = Detector(px_in_um= px_in_um)

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
    
    if os.path.exists(slice_profiles_path):
        # Load from file
        print()
        print("Loaded slice profiles from file.")
    else:
        # Create and save
        #positive, negative = sampskull.create_slice2d()
        bone, pores = sampskull.create_projected_1d_slices()
        #np.savez(slice_profiles_path, slc2d_sph_padded=slc2d_sph_padded, slc2d_bkg_padded=slc2d_bkg_padded)
        np.savez(slice_profiles_path, slc2d_sph_padded=bone, slc2d_bkg_padded=pores)
        print()
        print("Created and saved slice profiles.")

    results = []
    #distances_samp_det = np.arange(0.01, 0.39, 0.02)   # in m
    distances = np.arange(0, 0.91, 0.02)
    for distance in distances:

        prop = Propagator(grat = grat1d,
                    samp = sampskull,
                    prop_in_m = distance)
        visibility, epsilon = save_visibility_epsilon(det, prop,  wavefld_bg, prop.bin_grat,thick_samp_mm=sampskull.thickness_in_mm)

        corr_length = (l_in_m*distance)/(px_in_um * 1e-6) 
        print(f"Visibility with sample: {visibility.real:.3f} at Energy: {E_in_keV:.1f} keV distance to detector: {distance*1e2:.2f} cm")
        print(f"Correlation length: {corr_length*1e6:.2f} um")
        particle_fraction = np.sum(prop.slc2d_sph_full)/(samp_size_in_pix*sampskull.thickness_in_mm * 1e-3/sim_pix_size_in_m)
        results.append([distance, corr_length, visibility.real, epsilon.real])



        with open("skull_test4_10-7.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Propagation distance","Correlation length", "Visibility", "Epsilon"])
            writer.writerows(results)
        

finally:

    end = time.time() 
    stop_event.set()
    timer_thread.join()
    print(f"Total simulation time: {end - start:.2f} seconds.")
    print()
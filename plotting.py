import matplotlib.pyplot as plt
from grating import *
from sample import *
from propagator import *
import numpy as np
from parameters import *
from detector import *
import os
import pandas as pd


def plot_intensity_withG2(det, prop,  wavefld_bg, bin_grat, save_plot=True):  
    
    Iref_large, Isamp_large = prop.obtain_Iref_Isamp(wavefld_bg, bin_grat)
    G2 = det.create_g2()
    Iref_stepped, Isamp_stepped = det.phasestepping_conv(Isamp_large, Iref_large, G2)

    I_max = np.max(Isamp_stepped)
    I_min = np.min(Isamp_stepped)
    visibility = (I_max - I_min) / (I_max + I_min)
    print(f"Visibility with sample: {visibility.real:.3f} at Energy: {E_in_keV:.1f} keV")
    print(f"Mean with sample: {np.mean(Isamp_stepped.real):.3f} at Energy: {E_in_keV:.1f} keV")
    plt.plot(Iref_stepped, label='Iref with G2', linewidth=0.5, color='red')
    plt.plot(Isamp_stepped, label='Isamp with G2', linewidth=0.5, color='blue')
    plt.title(f"Intensity Profile at {E_in_keV:.1f} keV | Visibility with sample: {visibility.real:.3f} \n Thickness of sample: {t_samp_in_mm:.1f} mm | Mean Intensity: {np.mean(Isamp_stepped.real):.3f}")
    #plt.title(f"Intensity Profile at {E_in_keV:.1f} keV no G2 \n Thickness of sample: {t_samp_in_mm:.1f} mm | Mean Intensity: {np.mean(Isamp_stepped.real):.3f}")
    plt.xlim(15000, 20000)
    plt.xlabel('Pixels')
    plt.ylabel('Intensity')
    plt.legend()

    if save_plot:
        path_image = os.path.join("images", "bone_like_1" ,f"intensity_withG2_with_absorb_2mat_{name_mat_sph}_{name_mat_bkg}_{E_in_keV:.1f}keV_{t_samp_in_mm:.1f}mm.pdf")
        plt.savefig(path_image, dpi=600, bbox_inches='tight')
    del Iref_stepped, Isamp_stepped

def save_visibility_epsilon(det, prop,  wavefld_bg, bin_grat):
    Iref_large, Isamp_large = prop.obtain_Iref_Isamp(wavefld_bg, bin_grat)
    G2 = det.create_g2()
    Iref_stepped, Isamp_stepped = det.phasestepping_conv(Isamp_large, Iref_large, G2)

    I_max = np.max(Isamp_stepped)
    I_min = np.min(Isamp_stepped)
    visibility = (I_max - I_min) / (I_max + I_min)
    epsilon = -np.log(visibility) / (t_samp_in_mm * 1e-3)
    print(f"Mean with sample: {np.mean(Isamp_stepped.real):.3f} at Energy: {E_in_keV:.1f} keV")
    return visibility, epsilon

def plot_epsilon_vs_d():
    df = pd.read_csv('visibility_results_2.csv')  # Replace with the actual path if needed

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['Sphere size (um)'], df['Epsilon'], marker='o', linestyle='-')

    # Labels and title
    plt.xlabel('Sphere Size (μm)')
    plt.ylabel('Epsilon')
    plt.title('Epsilon vs Sphere Diameter')

    # Grid and layout
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("epsilon_diam.pdf", dpi=300, bbox_inches='tight')

def plot_single_slice_pair(slc2d_sph_padded, slc2d_bkg_padded, slice_idx=0, save_plot=True):

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axs[0].plot(slc2d_sph_padded[slice_idx], color='blue')
    axs[0].set_title(f"Slice {slice_idx} (Spheres)")
    axs[0].set_xlabel("Pixels")
    axs[0].set_ylabel("Value")

    axs[1].plot(slc2d_bkg_padded[slice_idx], color='orange')
    axs[1].set_title(f"Slice {slice_idx} (Background)")
    axs[1].set_xlabel("Pixels")

    plt.tight_layout()
    if save_plot:
        plt.savefig(f"slice_pair_{slice_idx}.pdf", dpi=300, bbox_inches='tight')


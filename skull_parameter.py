import numpy as np
from scipy import constants as cte
import sys
from pathlib import Path


# --- Grating -----------------------------------------------------------------

type = "hex"
px_in_um = 10
dc = round(1 / 3, 1)
rot_angle = 0

ph_shift_in_rad = round(2 * np.pi / 3, 1)
t_grat_in_um = 500
mat_grat_type = "element"
mat_grat = "Si"

# --- Geometry ----------------------------------------------------------------

# Simulated pixel size in m
sim_pix_size_in_m = 1e-7
# Image size in pix
img_size_in_pix = 840 * int(px_in_um * 1e-6 / sim_pix_size_in_m) #decides the length in x direction of the setup 
# Sample size in pix in X direction
samp_size_in_pix = img_size_in_pix

samp_size_in_m = samp_size_in_pix * sim_pix_size_in_m
# Propagation distane in m. It is defined as the distance from the middle of

# the sample to the detector position
prop_in_m = 0.175

# --- Source ------------------------------------------------------------------

# Energy in keV
E_in_keV = 45
E_in_J = E_in_keV * 1e3 * cte.e
# Number of photons per pixel
num_ph = 1e5
# Wavelength in m 
l_in_m = (cte.physical_constants["Planck constant in eV s"][0] * cte.c) / \
         (E_in_keV * 1e3)  
# Wavevector magnitude in 1/m 
k_in_1_m = 2 * np.pi * (E_in_keV * 1e3) / \
           (cte.physical_constants["Planck constant in eV s"][0] * cte.c)

r_e = cte.physical_constants["classical electron radius"][0]  # in m
# --- Sample ------------------------------------------------------------------

sim_approx = "slice" #"slice"



name_air = "Air"
mat_air = "N0.78084O0.20946Ar0.00934C0.00036Ne0.000018He0.000005Kr0.000001" 
rho_air_in_g_cm3 = 0.001225 

name_bone = "bone"
mat_bone ="H0.39234C0.15008N0.03487O0.31620Na0.00051Mg0.00096P0.03867S0.00109Ca0.06529" 
rho_bone_in_g_cm3 = 1.92 

"""
mat_sph = "SiO2"
name_mat_sph = "glass"
rho_sph_in_g_cm3 = 2.196

mat_bkg = "C2H6O"
name_mat_bkg = "Ethanol"
rho_bkg_in_g_cm3 = 0.78945

mat_bkg = "H2O"
name_mat_bkg = "Water"
rho_bkg_in_g_cm3 = 0.998
"""

if(sim_approx == "slice"):
    t_slc_in_pix = int(2 * 1e-6 / sim_pix_size_in_m)
else:
    raise ValueError("Please provide a valid simulation approximation")


detector_pixel_size = 1*1e-6
binning_factor = int(detector_pixel_size/sim_pix_size_in_m)
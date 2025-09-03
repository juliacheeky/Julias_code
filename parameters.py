import numpy as np
from scipy import constants as cte
import sys
from pathlib import Path
import xraydb as xdb

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
sim_pix_size_in_m = 1e-6
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
E_in_keV = 50
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
t_samp_in_mm = 1
d_sph_in_um = 20
f_sph = 0.01

mat_sph_type = "compound"
mat_bkg_type = "compound"



mat_sph = "SiO2"
name_mat_sph = "glass"
rho_sph_in_g_cm3 = 2.196

mat_bkg = "C2H6O"
name_mat_bkg = "Ethanol"
rho_bkg_in_g_cm3 = 0.78945

"""
mat_bkg = "H2O"
name_mat_bkg = "Water"
rho_bkg_in_g_cm3 = 0.998
"""


# Thickness of a sample slice, in pix. 
# Note: - For the projection approximation, t_slc = t_samp.
#       - For the thin slice approximation, t_slc = 1.4 um (diameter of the 
#         greatest simulated spheres).
if (sim_approx == "proj"):
    t_slc_in_pix = int(t_samp_in_mm * 1e-3 / sim_pix_size_in_m)
elif(sim_approx == "slice"):
    t_slc_in_pix = int(2 * 1e-6 / sim_pix_size_in_m)
else:
    raise ValueError("Please provide a valid simulation approximation")

# --- Detector ----------------------------------------------------------------
binning_factor = 100


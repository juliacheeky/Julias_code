import numpy as np
from skull_parameter import *
from detector import *
from plotting import *
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy.optimize import curve_fit
import os

skull_2 = pd.read_csv('skull_test2.csv')
t_samp = 12e-3  

xdata = np.array(skull_2['Correlation length'])*1e6
ydata = np.array(skull_2['Visibility'])

def V_sphere(x, R, sigma):
    return np.exp(sigma*t_samp*(np.exp(-(9/8) * (x/R)**2)-1))

popt, pcov = curve_fit(V_sphere, xdata, ydata, p0=[1.0, 1.0])
R_fit, sigma_fit = popt

print(f"Best-fit sigma for skull = {sigma_fit}")
print(f"Best-fit D for skull = {R_fit*2}")

plt.figure(figsize=(8, 8))
plt.scatter(skull_2['Correlation length']*1e6, skull_2['Visibility'], label = 'E = 45 keV real skull', color = 'blue')
plt.plot(xdata, V_sphere(xdata, R_fit, sigma_fit), 'r-', label=f"Fit: D={np.abs(R_fit)*2:.4f}um, sigma={sigma_fit:.4f}")
plt.xlabel(r'Correlation length in $\mu$m', fontsize=16)
plt.ylabel('Visibility', fontsize=16)
plt.title(f"Visibility vs Correlation length at {E_in_keV_skull:.1f} keV ", fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=12, loc='upper right')
plt.grid(True)
plt.tight_layout()

#plt.savefig("skull_real_sim.pdf", dpi=300, bbox_inches='tight')

"""analytical solution with 3 layers"""

f_inner_table   = 0.191       #taken from Alexander et al. 2019
f_diploe        = 0.619       #taken from Alexander et al. 2019
f_outer_table   = 0.069       #taken from Alexander et al. 2019

t_inner_table   = 2.3*1e-3    #taken from Alexander et al. 2019
t_diploe        = 5.1*1e-3    #taken from Alexander et al. 2019
t_outer_table   = 2.3*1e-3    #taken from Alexander et al. 2019

d_pores_inner   = 200*1e-6
d_pores_diploe  = 600*1e-6
d_pores_outer   = 100*1e-6

sampskull = Sample_Skull(mat_bone = mat_bone, mat_air = mat_air) 

n_air   = 1 - sampskull.delta_air   + 1j * sampskull.mu_air_in_1_m/(2*k_in_1_m)
n_bone  = 1 - sampskull.delta_bone  + 1j * sampskull.mu_bone_in_1_m/(2*k_in_1_m)
delta_n = (n_air - n_bone) 

pre_ohne_corr_inner_table   = 3*np.pi**2/(l_in_m**2)*f_inner_table* np.abs(delta_n)**2
pre_ohne_corr_diploe        = 3*np.pi**2/(l_in_m**2)*f_diploe* np.abs(delta_n)**2
pre_ohne_corr_outer_table   = 3*np.pi**2/(l_in_m**2)*f_outer_table* np.abs(delta_n)**2

corr_lengths = np.array(skull_2["Correlation length"])

D_prime_corr_inner  = d_pores_inner   / corr_lengths
D_prime_corr_diploe = d_pores_diploe  / corr_lengths
D_prime_corr_outer  = d_pores_outer   / corr_lengths

def mu_d_normalisiert(D_prime):
    D_prime = np.asarray(D_prime)
    shape_factor = np.empty_like(D_prime)

    mask = D_prime > 1.13
    sqrt_term = np.sqrt(D_prime[mask]**2 - 1)

    shape_factor[mask] = (
        (D_prime[mask] - sqrt_term * (1 + 1 / (2 * D_prime[mask]**2)) +
         (1 / D_prime[mask] - 1 / (4 * D_prime[mask]**3)))
        * np.log((D_prime[mask] + sqrt_term) / (D_prime[mask] - sqrt_term))
    )

    shape_factor[~mask] = D_prime[~mask]
    return shape_factor

epsilon_inner   = mu_d_normalisiert(D_prime_corr_inner)*corr_lengths*pre_ohne_corr_inner_table
epsilon_diploe  = mu_d_normalisiert(D_prime_corr_diploe)*corr_lengths*pre_ohne_corr_diploe
epsilon_outer   = mu_d_normalisiert(D_prime_corr_outer)*corr_lengths*pre_ohne_corr_outer_table

epsilon_thick = epsilon_inner*t_inner_table + epsilon_diploe*t_diploe + epsilon_outer*t_outer_table
Visibility = np.exp(-epsilon_thick)

mask = ~np.isnan(Visibility)
Visibility_clean = Visibility[mask]
popt2, pcov2 = curve_fit(V_sphere,xdata[mask], Visibility_clean, p0=[1.0, 1.0])
R_fit2, sigma_fit2 = popt2

print(f"Best-fit sigma for skull from analyt = {sigma_fit2}")
print(f"Best-fit D for skull from analyt = {R_fit2*2}")

plt.plot(corr_lengths*1e6,Visibility, label = 'analytical with 3 layers', color = 'green' )
plt.xlabel(r'Correlation length in $\mu$m', fontsize=16)
plt.ylabel('Visibility', fontsize=16)
plt.title(f"Visibility vs Correlation length Analytical at {E_in_keV_skull:.1f} keV ", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=12, loc='upper right')
plt.grid(True)
plt.tight_layout()

plt.savefig("analytiiitsch.pdf", dpi=300, bbox_inches='tight')
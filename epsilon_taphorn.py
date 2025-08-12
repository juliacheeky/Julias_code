from grating import *
from sample import *
from propagator import *
import numpy as np
from parameters import *
from detector import *
from plotting import *
import matplotlib.pyplot as plt
import csv
import pandas as pd

samp2d = Sample(t_samp_in_mm = t_samp_in_mm,
                d_sph_in_um = d_sph_in_um,
                f_sph = f_sph,
                mat_sph_type = mat_sph_type,
                mat_bkg_type = mat_bkg_type,
                mat_sph = mat_sph, 
                mat_bkg = mat_bkg,
                rho_sph_in_g_cm3 = rho_sph_in_g_cm3, 
                rho_bkg_in_g_cm3 = rho_bkg_in_g_cm3) 

""" Something coppied and Adapted from taphorn 
plank_const = 6.62607*1e-34
light_v = 299792458
ev_to_j = 1.602176*1e-19
period_g2 = 7e-6
radius = 4e-4
energies = np.linspace(30,80,11)
ls = 0.175

lambdaas = plank_const*light_v /(energies*1000* ev_to_j)  # in m
corr_lengths = (ls *lambdaas)/(period_g2*1e-3)   # g2_pitch also in m so that corr lenghts is in m here
#corr_lengths *=1e6  # in um
sphere_size = radius 

G1 = np.exp(-9/8.* (corr_lengths/sphere_size )**2  )  # fritz phd thesis

dim_xi = corr_lengths/sphere_size

G2 = (1- (dim_xi/2)**2)**0.5 * (1 + 1/8. *dim_xi**2) + 0.5 * dim_xi**2 * (1 - (dim_xi/4)**2  ) * np.log( dim_xi / (2 + (4-dim_xi**2)**0.5))  # strobl paper
print(G2)
theo_df_signal2 = -0.5 * 50.**2/energies**2 * (G2   -1.) #wo kommt die 50 und die 0.5 her?

scatter = 3/2 * radius
"""

delta_delta = samp2d.rho_bkg_in_g_cm3 - samp2d.rho_sph_in_g_cm3
delta_mu = samp2d.mu_bkg_in_1_m - samp2d.mu_sph_in_1_m
delta_chi = - delta_delta - 1j * l_in_m * delta_mu / (2 * np.pi)

corr_length = (l_in_m*prop_in_m)/(px_in_um * 1e-6) 
def mu_d(lambda_, f, delta_chi, d, D):

    D_prime = D / d
    prefactor = (3 * np.pi**2 / lambda_**2) * f * abs(delta_chi)**2 * d
    print(f"D_prime: {D_prime:.3f}, prefactor: {prefactor:.3e}, delta_chi: {abs(delta_chi)**2:.3e}")
    if D > d:
        sqrt_term = np.sqrt(D_prime**2 - 1)
        shape_factor = (D_prime - sqrt_term * (1 + 1 / (2 * D_prime**2))+ (1 / D_prime - 1 / (4 * D_prime**3))) * np.log((D_prime + sqrt_term) / (D_prime - sqrt_term))
    else:
        shape_factor = D_prime
    print(f"Shape factor: {shape_factor:.3f}")
    return prefactor * shape_factor

#mu_d = mu_d(l_in_m, f_sph, delta_chi, corr_length, 10e-6)


def mu_d_normalisiert_alt(D_prime):

    if D_prime > 1:
        sqrt_term = np.sqrt(D_prime**2 - 1)
        shape_factor = (D_prime - sqrt_term * (1 + 1 / (2 * D_prime**2))+ (1 / D_prime - 1 / (4 * D_prime**3))) * np.log((D_prime + sqrt_term) / (D_prime - sqrt_term))
    if D_prime <= 1.:
        shape_factor = D_prime
    return shape_factor


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

E_keV_list = np.array([10, 20, 30, 40])
l_in_m_list = (cte.physical_constants["Planck constant in eV s"][0] * cte.c) / \
         (E_keV_list * 1e3) 
corr_lengths = (l_in_m_list * prop_in_m) / (px_in_um * 1e-6)

D = np.linspace(1e-6, 40e-6, 100)  # in m

D_prime_all = np.array([D / corr for corr in corr_lengths])
mu_d_all = np.array([mu_d_normalisiert(D_prime) for D_prime in D_prime_all])
"""
plt.figure(figsize=(8, 8))
plt.plot(D*1e6, mu_d_all[0,:], label='E = 10 keV', color = 'green')
plt.plot(D*1e6, mu_d_all[1,:], label='E = 20 keV', color = 'red')
plt.plot(D*1e6, mu_d_all[2,:], label='E = 30 keV', color = 'blue')
plt.plot(D*1e6, mu_d_all[3,:], label='E = 40 keV', color = 'orange')
plt.ylabel(r'$\frac{\varepsilon(D)}{\frac{3}{4} f k^2|\Delta n|^2 \zeta}$')
plt.xlabel("D [$\mu$m]")
plt.xlim(2, 30)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('shape_factor_epsilon_cropped.pdf')
"""
#Calculate prefactor 

n_bkg = 1 - samp2d.delta_bkg + 1j * samp2d.mu_bkg_in_1_m/(2*k_in_1_m)
n_sph = 1 - samp2d.delta_sph + 1j * samp2d.mu_sph_in_1_m/(2*k_in_1_m)
delta_n = (n_bkg - n_sph)

pre_20keV = 3*np.pi**2/(l_in_m_list[1]**2)*f_sph* np.abs(delta_n)**2*corr_lengths[1]
pre_10keV = 3*np.pi**2/(l_in_m_list[0]**2)*f_sph* np.abs(delta_n)**2*corr_lengths[0]
pre_30keV = 3*np.pi**2/(l_in_m_list[2]**2)*f_sph* np.abs(delta_n)**2*corr_lengths[2]
pre_40keV = 3*np.pi**2/(l_in_m_list[3]**2)*f_sph* np.abs(delta_n)**2*corr_lengths[3]
df_20keV = pd.read_csv('dark_field_SiO2_Ethanol_diameter/visibility_results_20keV.csv')  # Replace with the actual path if needed
df_10keV = pd.read_csv('dark_field_SiO2_Ethanol_diameter/visibility_results_10keV.csv')
df_30keV = pd.read_csv('dark_field_SiO2_Ethanol_diameter/visibility_results_30keV.csv')
df_40keV = pd.read_csv('dark_field_SiO2_Ethanol_diameter/visibility_results_40keV.csv')
df_lowres_30keV = pd.read_csv('visibility_results_30keV_lowres.csv')
df_highres_30keV = pd.read_csv('visibility_results_30keV_highres.csv')
df_10keV_fixed = pd.read_csv('double_corrected_visibility_results_10keV.csv')
df_10keV_noabsorb= pd.read_csv('corrected_visibility_results_10keV.csv')
df_30keV_fixed = pd.read_csv('double_corrected_visibility_results_30keV.csv')
df_20keV_fixed = pd.read_csv('double_corrected_visibility_results_20keV.csv')
df_40keV_fixed = pd.read_csv('double_corrected_visibility_results_40keV.csv')
df = pd.read_csv("scaled_up_epsilons.csv")
prefactors = df["prefactor"]


# Plot
plt.figure(figsize=(8, 8))
#plt.plot(D*1e6, mu_d_all[0,:]*prefactors[0], label='E = 10 keV analytical', color='green')
#plt.plot(df_10keV['Sphere size (um)'], df_10keV['Epsilon'], marker='o', linestyle='-', label = 'E = 10 keV simulated', color = 'green')
#plt.plot(df_10keV_fixed['Sphere size (um)'], df_10keV_fixed['Epsilon'], marker='o', linestyle='-', label = 'E = 10 keV simulated', color = 'green')
#plt.plot(df_10keV_noabsorb['Sphere size (um)'], df_10keV_noabsorb['Epsilon'], marker='o', linestyle='-', label = 'E = 10 keV simulated half fix', color = 'red')
plt.plot(D*1e6, mu_d_all[1,:]*prefactors[1], label='E = 20 keV analytical', color='red')
plt.plot(df_20keV_fixed['Sphere size (um)'], df_20keV_fixed['Epsilon'], marker='o', linestyle='-', label = 'E = 30 keV simulated', color = 'red')
#plt.plot(df_20keV['Sphere size (um)'], df_20keV['Epsilon'], marker='o', linestyle='-', label = 'E = 20 keV simulated', color = 'red')
plt.plot(D*1e6, mu_d_all[2,:]*prefactors[2], label='E = 30 keV analytical', color = 'blue')
plt.plot(df_30keV_fixed['Sphere size (um)'], df_30keV_fixed['Epsilon'], marker='o', linestyle='-', label = 'E = 30 keV simulated', color = 'blue')
#plt.plot(df_highres_30keV['Sphere size (um)'], df_highres_30keV['Epsilon'], marker='o', linestyle='-', label = 'E = 30 keV simulated high res', color = 'blue')
#plt.plot(df_30keV['Sphere size (um)'], df_30keV['Epsilon'], marker='o', linestyle='-', label = 'E = 30 keV simulated', color = 'red')
plt.plot(D*1e6, mu_d_all[3,:]*prefactors[3], label='E = 40 keV analytical', color = 'orange')
plt.plot(df_40keV_fixed['Sphere size (um)'], df_40keV_fixed['Epsilon'], marker='o', linestyle='-', label = 'E = 40 keV simulated', color = 'orange')

# Labels and title
plt.xlabel('Sphere Size (μm)')
plt.ylabel('Epsilon')
plt.title('Epsilon vs Sphere Diameter')
plt.xlim(2, 30)
#plt.ylim(0.25, 1.7)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("test5#_epsilon_diam_analyt.pdf", dpi=300, bbox_inches='tight')
"""
zeta10 = corr_lengths[0]/(D/2)
zeta20 = corr_lengths[1]/(D/2)
def G(zeta):
    term1 = np.sqrt(1 - (zeta / 2)**2) * (1 + (1/8) * zeta**2)
    term2 = 0.5 * zeta**2 * (1 - (zeta / 4)**2) * np.log(zeta / (2 + np.sqrt(4 - zeta**2)))
    return term1 + term2

def xray_sld(total_electrons, molar_mass, density):

    Calculate X-ray Scattering Length Density (SLD) in Å⁻²
    Inputs:
        total_electrons: total Z per molecule
        molar_mass: in g/mol
        density: in g/cm³

    N_A = 6.022e23            # mol⁻¹
    V_mol = (molar_mass / density) / N_A 
    return (r_e * total_electrons) / V_mol

sld_sph = xray_sld(total_electrons = 30, molar_mass = 60.08, density = samp2d.rho_sph_in_g_cm3 )
sld_bkg = xray_sld(total_electrons = 20, molar_mass = 46.07, density = samp2d.rho_bkg_in_g_cm3 )
delta_sld = sld_sph - sld_bkg
sigmas = 3/2 * l_in_m**2 * D/2 *f_sph * delta_sld**2
print(l_in_m**2)
print(delta_sld**2)

plt.plot(D*1e6,sigmas*(1-G(zeta10)), color = 'red')
plt.plot(D*1e6,sigmas*(1-G(zeta20)), color = 'red')
#plt.plot(D*1e6, mu_d_all[0,:], label='E = 10 keV analytical', color='green')
plt.savefig("test_test.pdf", dpi=300, bbox_inches='tight')
"""


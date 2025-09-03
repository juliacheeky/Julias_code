from parameters import *
import numpy as np
import xraydb as xdb
from xraydb import xray_delta_beta, material_mu, get_material


def get_material_data(material, E_keV):
    # Physical constants
    h = 6.626e-34          # Planck's constant (J·s)
    c = 3.0e8              # Speed of light (m/s)
    eV_to_J = 1.602e-19    # eV to Joule conversion
    r_e = 2.8179403227e-15 # Classical electron radius (m)
    
    # Convert energy to Joules for wavelength calculation
    E_J = E_keV * 1e3 * eV_to_J
    wavelength = (h * c) / E_J  
    k = 2 * np.pi / wavelength
    
    # Get material data
    formula, density = xdb.get_material(material)  # density in g/cm^3
    
    # Check if xdb functions expect energy in eV (not keV)
    E_eV = E_keV * 1e3
    
    # Get delta, beta and attenuation coefficient
    delta, beta_from_xdb, att_cm = xdb.xray_delta_beta(formula, density, E_eV)
    
    # Get total attenuation coefficient and convert from cm^-1 to m^-1
    mu_m = xdb.material_mu(formula, E_eV, kind='total') * 1e2  # Note: *1e2 for cm^-1 to m^-1
    
    # Calculate beta from mu
    beta = mu_m / (2*k)
    
    # Calculate electron density
    rho_m3 = k**2 * delta / (2 * np.pi * r_e)
    
    return delta, beta, mu_m, rho_m3



bone        = "H0.39234C0.15008N0.03487O0.31620Na0.00051Mg0.00096P0.03867S0.00109Ca0.06529" #bone from XrayDB
delta_bone, beta_bone, mu_bone, rho_bone = get_material_data(bone, E_keV)



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

n_bkg = 1 - samp2d.delta_bkg + 1j * samp2d.mu_bkg_in_1_m/(2*k_in_1_m)
n_sph = 1 - samp2d.delta_sph + 1j * samp2d.mu_sph_in_1_m/(2*k_in_1_m)
delta_n = (n_bkg - n_sph)
pres = 3*np.pi**2/(l_in_m**2)*f_sph* np.abs(delta_n)**2*corr_lengths
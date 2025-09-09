import numpy as np
from scipy.signal import correlate2d
from scipy.signal import correlate
import xraylib as xrl
import xraydb as xdb
from scipy import constants as cte
import cv2 
from scipy.signal import fftconvolve

E_in_keV = 20

prop_in_m = 0.175

pixel_size_skull = 6e-6
sim_pix_size_in_m = 1e-6

l_in_m = (cte.physical_constants["Planck constant in eV s"][0] * cte.c) / \
         (E_in_keV * 1e3)  
r_e = cte.physical_constants['classical electron radius'][0]


px_in_um = 10

distances = np.arange(0, 0.91, 0.02)
corr_lengths = (l_in_m*distances)/(px_in_um * 1e-6) 

name_air = "Air"
mat_air = "N0.78084O0.20946Ar0.00934C0.00036Ne0.000018He0.000005Kr0.000001" 
rho_air_in_g_cm3 = 0.001225 

name_bone = "bone"
mat_bone ="H0.39234C0.15008N0.03487O0.31620Na0.00051Mg0.00096P0.03867S0.00109Ca0.06529" 
rho_bone_in_g_cm3 = 1.92 

def get_delta_rho_array():
    image = cv2.imread("04-02__rec_Sag1236.bmp", cv2.IMREAD_GRAYSCALE)
    
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (binary > 0).astype(np.uint8) 
    binary_cropped = binary[900:2300,600:2600]
    scale_factor = pixel_size_skull / sim_pix_size_in_m

    binary_scaled = cv2.resize(
            binary_cropped,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_NEAREST
        )
    rho = np.zeros(binary_scaled.shape)
    delta_bone = xrl.Refractive_Index_Re(mat_bone,E_in_keV,rho_bone_in_g_cm3)
    delta_air = xrl.Refractive_Index_Re(mat_air,E_in_keV,rho_air_in_g_cm3)
    rho[binary_scaled == 1] = delta_bone
    rho[binary_scaled == 0] = delta_air
    rho_array=rho*2*np.pi/(r_e*l_in_m**2)
    rho_mean = np.mean(rho_array)
    delta_rho = rho_array - rho_mean
    return delta_rho

delta_rho = get_delta_rho_array()

def compute_gamma_alt(delta_rho):
    numerator = correlate2d(delta_rho, delta_rho, mode='full', boundary='fill', fillvalue=0)
    denominator = np.sum(delta_rho**2)
    gamma = numerator / denominator
    return gamma

def compute_gamma_conv(delta_rho):
    autocorr = fftconvolve(delta_rho, delta_rho, mode='same')
    #autocorr = correlate(delta_rho, delta_rho, mode='same', method='fft') isn't it a correlation?
    center = tuple(s // 2 for s in autocorr.shape)
    gamma = autocorr / autocorr[center]
    return gamma

def compute_gamma_fft(delta_rho):

    F = np.fft.fftn(delta_rho)
    
    power_spectrum = F * np.conj(F)
    
    autocorr = np.fft.ifftn(power_spectrum).real
    
    gamma = autocorr / autocorr.flat[0] # do we really need this at this point? It makes the first element turn 1

    return gamma

#gamma1 = compute_gamma_conv(delta_rho)
#gamma2 = compute_gamma_fft(delta_rho)
#print(gamma1)
#print(gamma2)
#print(np.allclose(gamma1, gamma2))

def compute_G(gamma, xi_corr):
    num = []
    for x in xi_corr: # xi corr sind viel kleiner als das ganze sample (die indexe bewegen sich kaum bei dieser resolution)
        #Wenn ich ich hoch gehe mit der resolution dauerts ewig. Sollte x in der mitte anfangen? Aber dann tummeln wir uns nur in der mitte
        x_centre = gamma.shape[0]//2
        xi_corr_pix = int(x/sim_pix_size_in_m)
        line_num = gamma[x_centre+xi_corr_pix, :]
        #do these two do the same: which to pick?
        G_single = np.sum(line_num)
        #G_single = np.trapz(line_num, dz)

        num.append(G_single)
    num = np.array(num)
    #What to pick here?
    #line_den = gamma[0, :] 
    line_den = gamma[x_centre, :] 
    
    den = np.sum(line_den) 
    #den = np.trapz(line_den, dz)
    return num / den

#G1 = compute_G(gamma2, corr_lengths)


Avogadro = 6.022e23


def macroscopic_rayleigh_from_compound(compound, energy_keV, density_g_cm3):
 
    sigma_atom = xrl.CS_Rayl_CP(compound, energy_keV)

    # Step 2: molar mass of the compound (g/mol)
    molar_mass = xrl.AtomicWeight(compound)

    # Step 3: mass attenuation coefficient (cm^2/g)
    mu_over_rho = (Avogadro / molar_mass) * sigma_atom

    # Step 4: macroscopic cross section Σ (cm^-1)
    Sigma = density_g_cm3 * mu_over_rho

    return Sigma

def macroscopic_from_CS_Rayl_CP(compound, E_keV, density_g_cm3):
    """
    If you used CS_Rayl_CP (mass coeff, cm^2/g) — simplest path.
    Returns mu_over_rho (cm^2/g) and Sigma (cm^-1).
    """
    mu_over_rho = xrl.CS_Rayl_CP(compound, float(E_keV))  #mu duch rho 
    Sigma = density_g_cm3 * mu_over_rho         # sollte mu sein          
    return {'mu_over_rho_cm2_per_g': mu_over_rho,
            'Sigma_cm_inv': Sigma}

def number_density(rho_g_cm3, mass_fraction_wi, A_g_mol):
    """
    Number density of element i (atoms/cm^3)
    """
    return (rho_g_cm3 * Avogadro * mass_fraction_wi) / A_g_mol



sigma_atom = xrl.CS_Rayl_CP(mat_bone, E_in_keV)
cross_xdb = xdb.coherent_cross_section_elam("H2O", E_in_keV)

print(sigma_atom)

def calc_N_element():
    N = rho_elem/()
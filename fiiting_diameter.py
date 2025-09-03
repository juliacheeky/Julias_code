import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd


# data for sample thickness of 0.1 mm and sphere diameter of 0.966 um
prade_45keV_corrlength_0966 = pd.read_csv('dark_field_parde_correlation_change/Prade_remake_10um_results_45keV_corr_lengths_0966diam.csv')
prade_45keV_corrlength_279 = pd.read_csv('dark_field_parde_correlation_change/Prade_remake_10um_results_45keV_corr_lengths_v2.csv')

xdata = np.array(prade_45keV_corrlength_0966['Correlation length'])*1e6
ydata_0966 = np.array(prade_45keV_corrlength_0966['Visibility'])
ydata_279 = np.array(prade_45keV_corrlength_279['Visibility'])
t_samp_01mm = 1e-3

def V_sphere(x, R, sigma):
    return np.exp(sigma*t_samp_01mm*(np.exp(-(9/8) * (x/R)**2)-1))


def V_sphere_lynch(xi_corr, D, prefactor):
    # Ensure no division by zero or sqrt of negative
    xi_corr = np.array(xi_corr)
    D_prime = D / xi_corr
    shape_factor = np.empty_like(D_prime)
    mask = D_prime > 1
    # Only calculate sqrt for valid values
    sqrt_term = np.zeros_like(D_prime)
    sqrt_term[mask] = np.sqrt(D_prime[mask]**2 - 1)
    shape_factor[mask] = (
        (D_prime[mask] - sqrt_term[mask] * (1 + 1 / (2 * D_prime[mask]**2))
         + (1 / D_prime[mask] - 1 / (4 * D_prime[mask]**3)))
        * np.log((D_prime[mask] + sqrt_term[mask]) / (D_prime[mask] - sqrt_term[mask]))
    )
    shape_factor[~mask] = D_prime[~mask]
    return np.exp(-prefactor * shape_factor * t_samp_01mm)

popt, pcov = curve_fit(V_sphere, xdata, ydata_279, p0=[1.0, 1.0])  
#popt2, pcov2 = curve_fit(V_sphere_lynch, xdata, ydata, p0=[1.0, 1.0])
# p0 = initial guesses for [R, sigma]

R_fit, sigma_fit = popt
#D_fit_lynch, prefactor_fit_lynch = popt2
R_err, sigma_err = np.sqrt(np.diag(pcov))

print(f"Best-fit R Prade = {R_fit} ± {R_err:.4f}")
print(f"Best-fit sigma Prade = {sigma_fit} ± {sigma_err:.4f}")
print(f"Best-fit D Prade = {R_fit*2} ± {R_err:.4f}")

plt.scatter(xdata, ydata_279, label="Simulated data 2.79 um", color='blue')
plt.plot(xdata, V_sphere(xdata, R_fit, sigma_fit), 'r-', label=f"Fit: D={np.abs(R_fit)*2:.4f}um, sigma={sigma_fit:.4f}")

plt.xlabel("correlation length in um")
plt.ylabel("Visibility")
plt.legend()
plt.savefig("fit_results_2.pdf", dpi=300, bbox_inches='tight')
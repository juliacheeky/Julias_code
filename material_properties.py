import xraylib as xrl

E_in_keV = 50

bone = "H0.39234C0.15008N0.03487O0.31620Na0.00051Mg0.00096P0.03867S0.00109Ca0.06529"
marrow = "H0.61743C0.20427N0.01438O0.16261P0.00019S0.00037Cl0.00033K0.00030Fe0.00011"
air = "N0.78084O0.20946Ar0.00934C0.00036Ne0.000018He0.000005Kr0.000001"


rho_bone_in_g_cm3 = 1.92
rho_marrow_in_g_cm3 = 1.03
rho_air_in_g_cm3 = 0.001225


mu_bone_in_1_m = xrl.CS_Total_CP(bone, E_in_keV) * rho_bone_in_g_cm3 * 100
delta_bone = 1 - xrl.Refractive_Index_Re(bone,E_in_keV,rho_bone_in_g_cm3)


mu_marrow_in_1_m = xrl.CS_Total_CP(marrow, E_in_keV) * rho_marrow_in_g_cm3 * 100
delta_marrow = 1 - xrl.Refractive_Index_Re(marrow,E_in_keV,rho_marrow_in_g_cm3)

mu_air_in_1_m = xrl.CS_Total_CP(air, E_in_keV) * rho_air_in_g_cm3 * 100
delta_air = 1 - xrl.Refractive_Index_Re(air,E_in_keV,rho_air_in_g_cm3)

print()
print(f"Bone: Delta: {delta_bone:.3e}, Mu 1/m: {mu_bone_in_1_m:.3e} 1/m at {E_in_keV} keV")
print(f"Bone mu/rho: {mu_bone_in_1_m*1e-2/rho_bone_in_g_cm3:.3e} cm2/g at {E_in_keV} keV")
print(f"Marrow: Delta: {delta_marrow:.3e}, Mu 1/m: {mu_marrow_in_1_m:.3e} 1/m at {E_in_keV} keV")
print(f"Air: Delta: {delta_air:.3e}, Mu 1/m: {mu_air_in_1_m:.3e} 1/m at {E_in_keV} keV")
print(f"Air mu/rho: {mu_air_in_1_m*1e-2/rho_air_in_g_cm3:.3e} cm2/g at {E_in_keV} keV")
print()
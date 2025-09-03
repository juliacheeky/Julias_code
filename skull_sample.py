import numpy as np
import scipy.fft
import xraylib as xrl
from typing import Tuple
import cv2  
from skull_parameter import * 
import matplotlib.pyplot as plt


class Sample_Skull:                             
    
    def __init__(self, 
                 mat_bone: str, 
                 mat_air: str
                 ) -> None:

        self.mat_bone = mat_bone 
        self.mat_air = mat_air

        image = cv2.imread("04-02__rec_Sag1236.bmp", cv2.IMREAD_GRAYSCALE)
        pixel_size_skull = 6e-6
        ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = (binary > 0).astype(np.uint8) 
        binary_cropped = binary[900:2300,600:2600]
        scale_factor = pixel_size_skull / sim_pix_size_in_m

        self.positive = cv2.resize(
            binary_cropped,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_NEAREST
        )
        t_samp_in_pix = self.positive.shape[1]
        self.t_samp_in_mm = t_samp_in_pix * sim_pix_size_in_m * 1e3
        print(f"Sample thickness in mm: {self.t_samp_in_mm:.3f}")
        self.num_slc = int(t_samp_in_pix /t_slc_in_pix) 

        self.rho_bone_in_g_cm3 = rho_bone_in_g_cm3
        self.rho_air_in_g_cm3 = rho_air_in_g_cm3

        self.mu_bone_in_1_m = xrl.CS_Total_CP(self.mat_bone, E_in_keV_skull) * self.rho_bone_in_g_cm3 * 100
        self.delta_bone = 1 - xrl.Refractive_Index_Re(self.mat_bone,E_in_keV_skull,self.rho_bone_in_g_cm3)
        
        self.mu_air_in_1_m = xrl.CS_Total_CP(self.mat_air, E_in_keV_skull) * self.rho_air_in_g_cm3 * 100
        self.delta_air = 1 - xrl.Refractive_Index_Re(self.mat_air,E_in_keV_skull,self.rho_air_in_g_cm3)

        print(f"Bone: Delta: {self.delta_bone:.3e}, Mu 1/m: {self.mu_bone_in_1_m:.3e} 1/m at {E_in_keV_skull} keV")
        print(f"Air: Delta: {self.delta_air:.3e}, Mu 1/m: {self.mu_air_in_1_m:.3e} 1/m at {E_in_keV_skull} keV")
    
    def create_slice2d(self)-> Tuple[np.ndarray, 
                                           np.ndarray]:

        negative = np.ones(self.positive.shape) - self.positive
        return self.positive, negative
    
    
    def create_projected_1d_slices(self) -> Tuple[np.ndarray,np.ndarray]:

 
        slice_profiles_sph = []
        slice_profiles_bkg = []
        bone, pores = self.create_slice2d()
        
        print(f"Number of slices: {self.num_slc}") 
        print(t_slc_in_pix) 
        for i in range(self.num_slc):
            start = i * t_slc_in_pix
            end = start + t_slc_in_pix
            slice_chunk_sph = bone[:,start:end] 
            slice_chunk_bkg = pores[:, start:end]          # select slice
            profile_sph = np.sum(slice_chunk_sph, axis=1)           # sum over rows
            profile_bkg = np.sum(slice_chunk_bkg, axis=1)
            slice_profiles_sph.append(profile_sph)
            slice_profiles_bkg.append(profile_bkg)
        
        return np.array(slice_profiles_sph), np.array(slice_profiles_bkg)

    def samp_with_refract_property(self, 
                           samp_sph_in_m: np.ndarray,
                           samp_bkg_in_m: np.ndarray) -> np.ndarray:
        """
        Applies the refractive properties of the sample to the binary array.

        This method modifies the input wave field based on the phase shifts 
        and attenuation resulting from the sample properties.

        Args:
            samp_sph_in_m (np.ndarray): _description_
            samp_bkg_in_m (np.ndarray): _description_

        Returns:
            np.ndarray: The modified wave field after interaction with the 
                        sample.
        """
        return np.exp(-1j * k_in_1_m * (self.delta_bone * \
                                samp_sph_in_m + self.delta_air * \
                                samp_bkg_in_m)) * \
               np.exp(-((self.mu_bone_in_1_m / 2) * samp_sph_in_m + \
                      (self.mu_air_in_1_m / 2) * samp_bkg_in_m)) 


"""samp2d = Sample_Skull(mat_bone = mat_bone, mat_air = mat_air) 

positive, negative = samp2d.create_slice2d()

plt.imshow(positive, cmap='gray', aspect='equal')

# Set tick positions (in pixels)
tick_spacing = 400  # adjust as needed
x_ticks = np.arange(0, positive.shape[1], tick_spacing)
y_ticks = np.arange(0, positive.shape[0], tick_spacing)

# Set tick labels in mm
plt.xticks(x_ticks, [f"{x*0.001:.1f}" for x in x_ticks])
plt.yticks(y_ticks, [f"{y*0.001:.1f}" for y in y_ticks])
plt.xlabel("mm")
plt.ylabel("mm")

plt.grid(True, color='red', linestyle='-', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.tight_layout()
plt.savefig("loeschen.pdf", dpi=300)"""

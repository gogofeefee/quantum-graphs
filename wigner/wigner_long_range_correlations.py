#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 15:54:46 2025

@author: hugo
"""

import numpy as np
import scipy.special as sp
import scipy.integrate as spi
import scipy.interpolate as interp
import os

###############################################################################
################## LONG RANGE CORRELATION FUNCTIONS ###########################
###############################################################################

# File name for caching results
L_min, L_max, L_points = 0, 100, 10000  # Define range and number of points
npz_path = "/home/hugo/Documents/Research/Simul/Graphs/wigner/"
npz_filename = f"wigner_functions_L{L_min}-{L_max}_N{L_points}.npz"

def two_level_cluster_func(E, wigner="GOE"):
    """
    Computes the two-level cluster functions for GOE, GUE, and GSE.
    """
    wigner = wigner.lower()
    E = np.asarray(E)  # Ensure E is an array to handle both scalar and array inputs
    
    if wigner == "goe":
        Y2 = np.ones_like(E)
        nonzero_mask = E != 0  
        if np.any(nonzero_mask):
            E_nz = E[nonzero_mask]
            si = sp.sici(np.pi * E_nz)[0]
            sinc = np.sinc(E_nz)
            Y2[nonzero_mask] = sinc**2 + (np.pi / 2 * np.sign(E_nz) - si) * (
                np.cos(np.pi * E_nz) / (np.pi * E_nz) - np.sin(np.pi * E_nz) / (np.pi * E_nz)**2
            )
        return Y2 if Y2.shape else Y2.item()
    
    elif wigner == "gue":
        return np.sinc(E)**2
    
    elif wigner == "gse":
        Y2 = np.ones_like(E)
        nonzero_mask = E != 0  
        if np.any(nonzero_mask):
            E_nz = E[nonzero_mask]
            si = sp.sici(2 * np.pi * E_nz)[0]
            sinc = np.sinc(2 * E_nz)
            Y2[nonzero_mask] = sinc**2 - si * (
                np.cos(2 * np.pi * E_nz) / (2 * np.pi * E_nz) - 
                np.sin(2 * np.pi * E_nz) / (2 * np.pi * E_nz)**2
            )
        return Y2 if Y2.shape else Y2.item()
    
    else:
        raise ValueError("Invalid wigner type. Choose from 'GOE', 'GUE', or 'GSE'.")

def _compute_sigma_func(L, wigner="GOE", show_progress=False):
    """
    Computes the number variance Σ²(L) based on the exact formula when possible.

    Parameters:
        L (float or array-like): The length scale.
        wigner (str): The ensemble type, one of "poissonian", "GOE", "GUE", or "GSE".
                      The input is case-insensitive.
        show_progress (bool): If True, prints progress in the shell.

    Returns:
        float or array-like: The computed number variance Σ²(L).
    """
    wigner = wigner.lower()
    L = np.asarray(L)

    if wigner == "poissonian":
        return L  

    elif wigner in ["goe", "gue", "gse"]:
        def integral_func(L_val):
            try:
                integral, error = spi.quad(
                    lambda E: (L_val - E) * two_level_cluster_func(E, wigner=wigner), 
                    0, L_val, 
                    limit=100,  # Increase subdivisions
                    epsabs=1e-8, epsrel=1e-6  # Adaptive error tolerance
                )
                return L_val - 2 * integral
            except Exception as e:
                print(f"Integration failed for L={L_val}: {e}")
                return np.nan  # Return NaN if integration fails

        results = []
        for i, L_val in enumerate(L):
            results.append(integral_func(L_val))
            if show_progress:
                print(f"Computing Σ²(L, {wigner}): {(i + 1) / len(L) * 100:.1f}% completed", end="\r", flush=True)
        print()
        return np.array(results)

def _compute_delta_func(L, wigner="GOE", show_progress=False):
    """
    Computes the spectral rigidity Δ₃(L) based on the exact formula when possible.

    Parameters:
        L (float or array-like): The length scale.
        wigner (str): The ensemble type, one of "poissonian", "GOE", "GUE", or "GSE".
                      The input is case-insensitive.
        show_progress (bool): If True, prints progress in the shell.

    Returns:
        float or array-like: The computed spectral rigidity Δ₃(L).
    """
    wigner = wigner.lower()
    L = np.asarray(L)
    result = np.zeros_like(L, dtype=float)

    if np.any(L == 0):
        result[L == 0] = 0  

    if wigner == "poissonian":
        return L / 15  

    elif wigner in ["goe", "gue", "gse"]:
        def integral_func(L_val):
            try:
                integral, error = spi.quad(
                    lambda E: (L_val - E)**3 * (2*L_val**2 - 9*L_val*E - 3*E**2) * two_level_cluster_func(E, wigner=wigner), 
                    0, L_val, 
                    limit=100,  # Increase subdivisions
                    epsabs=1e-8, epsrel=1e-6  # Adaptive error tolerance
                )
                return L_val / 15 - 1 / (15 * L_val**4) * integral
            except Exception as e:
                print(f"Integration failed for L={L_val}: {e}")
                return np.nan  # Return NaN if integration fails

        mask = L > 0  
        results = []
        for i, L_val in enumerate(L[mask]):
            results.append(integral_func(L_val))
            if show_progress:
                print(f"Computing Δ₃(L, {wigner}): {(i + 1) / len(L) * 100:.1f}% completed", end="\r", flush=True)
        print()
        result[mask] = results
        return result

###############################################################################
#################### Load or Compute Wigner Functions #########################
###############################################################################

if os.path.exists(npz_path + npz_filename):
    print(f"Loading precomputed data from {npz_filename}...")
    data = np.load(npz_path + npz_filename)
    L = data['L']
    delta_GOE, sigma_GOE = data['delta_GOE'], data['sigma_GOE']
    delta_GUE, sigma_GUE = data['delta_GUE'], data['sigma_GUE']
    delta_GSE, sigma_GSE = data['delta_GSE'], data['sigma_GSE']
else:
    print("No precomputed data found. Computing Wigner functions...")
    L = np.linspace(L_min, L_max, L_points)

    delta_GOE = _compute_delta_func(L, wigner='GOE', show_progress=True)
    sigma_GOE = _compute_sigma_func(L, wigner='GOE', show_progress=True)

    delta_GUE = _compute_delta_func(L, wigner='GUE', show_progress=True)
    sigma_GUE = _compute_sigma_func(L, wigner='GUE', show_progress=True)

    delta_GSE = _compute_delta_func(L, wigner='GSE', show_progress=True)
    sigma_GSE = _compute_sigma_func(L, wigner='GSE', show_progress=True)

    # Save computed values to an .npz file
    np.savez(npz_filename, L=L, delta_GOE=delta_GOE, sigma_GOE=sigma_GOE, 
             delta_GUE=delta_GUE, sigma_GUE=sigma_GUE, 
             delta_GSE=delta_GSE, sigma_GSE=sigma_GSE)
    print(f"Precomputed data saved as {npz_filename}")

###############################################################################
#################### Interpolated Functions ####################################
###############################################################################

interp_sigma_GOE = interp.interp1d(L, sigma_GOE, kind='cubic', bounds_error=False, fill_value=np.nan)
interp_sigma_GUE = interp.interp1d(L, sigma_GUE, kind='cubic', bounds_error=False, fill_value=np.nan)
interp_sigma_GSE = interp.interp1d(L, sigma_GSE, kind='cubic', bounds_error=False, fill_value=np.nan)

interp_delta_GOE = interp.interp1d(L, delta_GOE, kind='cubic', bounds_error=False, fill_value=np.nan)
interp_delta_GUE = interp.interp1d(L, delta_GUE, kind='cubic', bounds_error=False, fill_value=np.nan)
interp_delta_GSE = interp.interp1d(L, delta_GSE, kind='cubic', bounds_error=False, fill_value=np.nan)

def sigma2_func(L, wigner="GOE"):
    wigner = wigner.lower()
    return {"goe": interp_sigma_GOE, "gue": interp_sigma_GUE, "gse": interp_sigma_GSE}.get(wigner, lambda x: x)(L)

def delta3_func(L, wigner="GOE"):
    wigner = wigner.lower()
    return {"goe": interp_delta_GOE, "gue": interp_delta_GUE, "gse": interp_delta_GSE}.get(wigner, lambda x: x)(L)

###############################################################################
#################### Usage Example ############################################
###############################################################################

if __name__ == "__main__":
    
    def test_long_range_correlations(L_max=20, N_L=5000, save_fig=False, file_format='svg', save_dir='./'):
        """
        Test function to plot long-range spectral correlations using the sigma and delta functions 
        for GOE, GUE, and GSE ensembles.
    
        Parameters:
        - L_max: int, maximum interval length to consider (default: 20).
        - N_L: int, number of points to sample in L range (default: 5000).
        - save_fig: bool, if True, saves the figure.
        - file_format: str, format for saving (e.g., 'svg', 'png', 'pdf').
        - save_dir: str, directory to save the figure.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import os
    
        # Define the range for L
        L_values = np.linspace(0, L_max, N_L)
    
        # Compute sigma and delta values for the three ensembles
        sigma2_GOE = sigma2_func(L_values, wigner="GOE")
        sigma2_GUE = sigma2_func(L_values, wigner="GUE")
        sigma2_GSE = sigma2_func(L_values, wigner="GSE")
    
        delta3_GOE = delta3_func(L_values, wigner="GOE")
        delta3_GUE = delta3_func(L_values, wigner="GUE")
        delta3_GSE = delta3_func(L_values, wigner="GSE")
    
        # Create the figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0})
    
        # Plot Number Variance Σ²(L)
        ax_sigma, ax_delta = axes
        ax_sigma.plot(L_values, sigma2_GOE, label="GOE", linestyle='-', color='blue', linewidth=2)
        ax_sigma.plot(L_values, sigma2_GUE, label="GUE", linestyle='--', color='red', linewidth=2)
        ax_sigma.plot(L_values, sigma2_GSE, label="GSE", linestyle='-.', color='green', linewidth=2)
        ax_sigma.set_ylabel(r'$\Sigma^2(L)$', fontsize=22)
        ax_sigma.set_xlim(0.0, L_max)
        ax_sigma.set_ylim(0.0, max(max(sigma2_GOE), max(sigma2_GUE), max(sigma2_GSE)) * 1.0)
        ax_sigma.legend(fontsize=18, loc='upper left')
    
        # Configure ticks
        ax_sigma.minorticks_on()
        ax_sigma.tick_params(axis='both', which='major', direction='in', length=6, width=2, labelsize=22, top=True, right=True)
        ax_sigma.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5, top=True, right=True)
    
        # Plot Spectral Rigidity Δ₃(L)
        ax_delta.plot(L_values, delta3_GOE, label="GOE", linestyle='-', color='blue', linewidth=2)
        ax_delta.plot(L_values, delta3_GUE, label="GUE", linestyle='--', color='red', linewidth=2)
        ax_delta.plot(L_values, delta3_GSE, label="GSE", linestyle='-.', color='green', linewidth=2)
        ax_delta.set_ylabel(r'$\Delta_3(L)$', fontsize=22)
        ax_delta.set_xlabel('Interval length $L$', fontsize=22)
        ax_delta.set_xlim(0.0, L_max)
        ax_delta.set_ylim(0.0, max(max(delta3_GOE), max(delta3_GUE), max(delta3_GSE)) * 1.0)
        #ax_delta.legend(fontsize=18, loc='upper left')
    
        # Configure ticks
        ax_delta.minorticks_on()
        ax_delta.tick_params(axis='both', which='major', direction='in', length=6, width=2, labelsize=22, top=True, right=True)
        ax_delta.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5, top=True, right=True)
    
        plt.tight_layout()
    
        # ✅ Save figure if requested
        if save_fig:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"Test_LongRange_Lmax{L_max}_NL{N_L}.{file_format}"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, format=file_format, dpi=300, bbox_inches='tight')
            print(f"Test plot saved as {filepath}")
    
        plt.show()
        
    # Save the test plot as an SVG in the "./plots" directory
    test_long_range_correlations(save_fig=True, file_format="svg", save_dir="./")

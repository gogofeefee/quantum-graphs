#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 15:54:46 2025

@author: hugo

This codes aims at defining the delta3 functions from the list of 
values computed by Barbara with RMT
"""

import numpy as np
import scipy.interpolate as interp
import os

###############################################################################
###################### SIGMA CORRELATION FUNCTION #############################
###############################################################################

# Working directory
working_dir = '/home/hugo/Documents/Research/Simul/Graphs/wigner/'

# Load the wigner delta3 functions
data_GOE = np.loadtxt(working_dir + 'delta3goe.txt')
data_GUE = np.loadtxt(working_dir + 'delta3gue.txt')
data_GSE = np.loadtxt(working_dir + 'delta3_RP_model_N65536_gamma_0.9_GSE.txt')

# Split into two separate arrays
L_GOE, delta3_GOE = data_GOE[:, 0], data_GOE[:, 1]
L_GUE, delta3_GUE = data_GUE[:, 0], data_GUE[:, 1]
L_GSE, delta3_GSE = data_GSE[:, 0], data_GSE[:, 1]

# Adding the (0, 0) value to each arrays
L_GOE = np.concatenate(([0.0], L_GOE))
delta3_GOE = np.concatenate(([0.0], delta3_GOE))
L_GUE = np.concatenate(([0.0], L_GUE))
delta3_GUE = np.concatenate(([0.0], delta3_GUE))
L_GSE = np.concatenate(([0.0], L_GSE))
delta3_GSE = np.concatenate(([0.0], delta3_GSE))

###############################################################################
####################### Interpolated Functions ################################
###############################################################################

interp_delta3_GOE = interp.interp1d(L_GOE, delta3_GOE, kind='cubic', bounds_error=False, fill_value=np.nan)
interp_delta3_GUE = interp.interp1d(L_GUE, delta3_GUE, kind='cubic', bounds_error=False, fill_value=np.nan)
interp_delta3_GSE = interp.interp1d(L_GSE, delta3_GSE, kind='cubic', bounds_error=False, fill_value=np.nan)

def delta3_RMT(L, wigner="GOE"):
    wigner = wigner.lower()
    return {"goe": interp_delta3_GOE, "gue": interp_delta3_GUE, "gse": interp_delta3_GSE}.get(wigner, lambda x: x)(L)

###############################################################################
#################### Usage Example ############################################
###############################################################################

if __name__ == "__main__":
    
    def test_long_range_correlations(L_max=20, N_L=5000, save_fig=False, file_format='svg', save_dir=working_dir):
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
        import matplotlib.pyplot as plt
    
        # Define the range for L
        L_values = np.linspace(0, L_max, N_L)
    
        # Compute sigma and delta values for the three ensembles
        delta3_GOE = delta3_RMT(L_values, wigner="GOE")
        delta3_GUE = delta3_RMT(L_values, wigner="GUE")
        delta3_GSE = delta3_RMT(L_values, wigner="GSE")
    
        # Create the figure with two subplots
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Plot Number Variance Σ²(L)
        ax.plot(L_values, delta3_GOE, label="GOE", linestyle='-', color='blue', linewidth=2)
        ax.plot(L_values, delta3_GUE, label="GUE", linestyle='--', color='red', linewidth=2)
        ax.plot(L_values, delta3_GSE, label="GSE", linestyle='-.', color='green', linewidth=2)
        ax.set_ylabel(r'$\Sigma^2(L)$', fontsize=22)
        ax.set_xlim(0.0, L_max)
        ax.set_ylim(0.0, max(max(delta3_GOE), max(delta3_GUE), max(delta3_GSE)) * 1.0)
        ax.legend(fontsize=18, loc='upper left')
    
        # Configure ticks
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', direction='in', length=6, width=2, labelsize=22, top=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.5, top=True, right=True)
    
        plt.tight_layout()
    
        # Save figure if requested
        if save_fig:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"Test_delta3_RMT_Lmax{L_max}_NL{N_L}.{file_format}"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, format=file_format, dpi=300, bbox_inches='tight')
            print(f"Test plot saved as {filepath}")
    
        plt.show()
        
    # Save the test plot as an SVG in the "./plots" directory
    test_long_range_correlations(save_fig=True, file_format="svg", save_dir=working_dir)

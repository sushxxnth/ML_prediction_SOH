"""
SOC-Aware EIS Visualization

Generates Nyquist plots showing impedance patterns across:
- Different SOC levels (0%, 50%, 100%)
- Different temperatures (-40°C to 50°C)

This visualization supports Research Contribution #1: SOC-aware degradation modeling
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.eis_impedance_loader import EISImpedanceLoader


def plot_nyquist_by_soc(loader: EISImpedanceLoader, output_dir: str = "reports/figures"):
    """
    Create Nyquist plots grouped by SOC level.
    
    Shows how impedance changes with SOC - key for Contribution #1.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Color maps
    soc_colors = {0.0: '#E74C3C', 50.0: '#F39C12', 100.0: '#27AE60'}
    temp_markers = {-40.0: 'o', -5.0: 's', 25.0: '^', 50.0: 'D'}
    
    # Plot by temperature, colored by SOC
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    temps = sorted(loader.TEMPERATURE_DIRS.values())
    
    for ax, temp in zip(axes, temps):
        ax.set_title(f'Temperature: {temp}°C', fontsize=14, fontweight='bold')
        
        for cell_key, cell in loader.cells.items():
            if cell.temperature != temp:
                continue
            
            for spectrum in cell.spectra[:3]:  # Limit to first 3 per cell
                soc = spectrum.soc
                color = soc_colors.get(soc, 'gray')
                
                # Plot Nyquist (Z_real vs -Z_imag)
                ax.plot(
                    spectrum.z_real * 1000,  # Convert to mOhm
                    -spectrum.z_imag * 1000,
                    color=color,
                    alpha=0.6,
                    linewidth=1.5
                )
        
        ax.set_xlabel("Z' (mΩ)", fontsize=12)
        ax.set_ylabel("-Z'' (mΩ)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=soc_colors[0.0], label='SOC 0%', linewidth=3),
        plt.Line2D([0], [0], color=soc_colors[50.0], label='SOC 50%', linewidth=3),
        plt.Line2D([0], [0], color=soc_colors[100.0], label='SOC 100%', linewidth=3),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=12)
    
    plt.suptitle('SOC-Dependent Impedance Spectra (Nyquist Plots)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'nyquist_by_soc_and_temp.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_impedance_features_vs_soc(loader: EISImpedanceLoader, output_dir: str = "reports/figures"):
    """
    Plot extracted impedance features vs SOC level.
    
    Shows R0, Rct trends with SOC - quantifies SOC dependency.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect features
    features_df = loader.get_all_features()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    feature_names = ['R0', 'Rct_estimate', 'z_real_mean', 'warburg_slope']
    feature_labels = ['R₀ (Ohmic Resistance)', 'Rct (Charge Transfer)', 'Mean Z_real', 'Warburg Slope']
    
    temp_colors = {-40.0: '#3498DB', -5.0: '#9B59B6', 25.0: '#E67E22', 50.0: '#E74C3C'}
    
    for ax, feat, label in zip(axes.flatten(), feature_names, feature_labels):
        for temp in sorted(features_df['temperature'].unique()):
            temp_data = features_df[features_df['temperature'] == temp]
            
            # Group by SOC
            soc_means = temp_data.groupby('soc')[feat].mean()
            soc_stds = temp_data.groupby('soc')[feat].std()
            
            ax.errorbar(
                soc_means.index,
                soc_means.values,
                yerr=soc_stds.values,
                marker='o',
                markersize=8,
                capsize=5,
                label=f'{temp}°C',
                color=temp_colors.get(temp, 'gray'),
                linewidth=2
            )
        
        ax.set_xlabel('SOC (%)', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks([0, 50, 100])
    
    plt.suptitle('EIS Features vs SOC Level (by Temperature)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'eis_features_vs_soc.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_temperature_effect(loader: EISImpedanceLoader, output_dir: str = "reports/figures"):
    """
    Plot impedance changes across temperature range.
    
    Supports Contribution #4: Extreme temperature generalization.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    temp_colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, 4))
    temps = sorted(loader.TEMPERATURE_DIRS.values())
    
    for temp, color in zip(temps, temp_colors):
        # Get all spectra at this temperature (SOC 50% only for clarity)
        spectra = [s for s in loader._spectra_list if s.temperature == temp and s.soc == 50.0]
        
        if spectra:
            # Average spectrum
            z_real_avg = np.mean([s.z_real for s in spectra], axis=0)
            z_imag_avg = np.mean([s.z_imag for s in spectra], axis=0)
            
            ax.plot(
                z_real_avg * 1000,
                -z_imag_avg * 1000,
                color=color,
                linewidth=3,
                label=f'{temp}°C'
            )
    
    ax.set_xlabel("Z' (mΩ)", fontsize=14)
    ax.set_ylabel("-Z'' (mΩ)", fontsize=14)
    ax.set_title('Temperature Effect on Impedance (SOC 50%)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    output_path = os.path.join(output_dir, 'temperature_effect_nyquist.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def main():
    """Generate all visualizations."""
    print("Loading EIS data...")
    loader = EISImpedanceLoader('.')
    loader.load()
    
    print(f"\nGenerating visualizations for {len(loader._spectra_list)} spectra...\n")
    
    paths = []
    paths.append(plot_nyquist_by_soc(loader))
    paths.append(plot_impedance_features_vs_soc(loader))
    paths.append(plot_temperature_effect(loader))
    
    print("\n=== Visualization Complete ===")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()

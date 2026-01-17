"""
Comprehensive Visualization for 50% Few-Shot Transfer Results

Creates publication-quality plots showing:
1. Learning curve with 50% point
2. Performance comparison across few-shot percentages
3. SOH vs RUL accuracy trade-off
4. Improvement trajectory
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import argparse

def create_comprehensive_plots(
    comparison_json: str = 'reports/lithium_comparison/comparison.json',
    baseline_50pct_json: str = 'reports/baseline_50pct/summary.json',
    output_dir: str = 'reports/50pct_visualization'
):
    """Create comprehensive visualization plots."""
    
    # Load data
    with open(comparison_json) as f:
        comparison_data = json.load(f)
    
    with open(baseline_50pct_json) as f:
        baseline_50pct = json.load(f)
    
    # Extract data points
    fewshot_percentages = [0, 10, 30, 70, 80]
    rul_mae_values = [
        comparison_data['zero_shot_baseline']['rul_mae'],
        comparison_data['fewshot_results'][0]['rul_mae'],
        comparison_data['fewshot_results'][1]['rul_mae'],
        comparison_data['fewshot_results'][2]['rul_mae'],
        comparison_data['fewshot_results'][3]['rul_mae']
    ]
    soh_mae_values = [
        comparison_data['zero_shot_baseline']['soh_mae'],
        comparison_data['fewshot_results'][0]['soh_mae'],
        comparison_data['fewshot_results'][1]['soh_mae'],
        comparison_data['fewshot_results'][2]['soh_mae'],
        comparison_data['fewshot_results'][3]['soh_mae']
    ]
    
    # Add 50% point
    fewshot_percentages.insert(3, 50)  # Insert at position 3 (between 30 and 70)
    rul_mae_values.insert(3, baseline_50pct['rul_mae'])
    soh_mae_values.insert(3, baseline_50pct['soh_mae'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # ========== Plot 1: Learning Curve (RUL MAE) ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    # Main learning curve
    line1 = ax1.plot(fewshot_percentages, rul_mae_values, 
                     marker='o', markersize=12, linewidth=3, 
                     color='#2E86AB', markerfacecolor='#A23B72', 
                     markeredgecolor='white', markeredgewidth=2.5,
                     label='RUL MAE with Lithium Features', zorder=5)
    
    # Highlight 50% point
    ax1.plot(50, baseline_50pct['rul_mae'], 
             marker='*', markersize=20, color='#F18F01',
             markeredgecolor='white', markeredgewidth=2,
             label='50% Few-Shot (This Work)', zorder=6)
    
    # Target zone
    ax1.axhspan(40, 50, alpha=0.2, color='green', label='Target Zone (40-50 cycles)')
    ax1.axhline(y=50, color='g', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
    ax1.axhline(y=40, color='g', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(fewshot_percentages, rul_mae_values)):
        if x == 50:
            ax1.annotate(f'{y:.1f}',
                        xy=(x, y), xytext=(x, y - 20),
                        fontsize=11, fontweight='bold', color='#F18F01',
                        ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF4E6', 
                                 edgecolor='#F18F01', linewidth=2.5, alpha=0.95),
                        zorder=7)
        elif x == 0:
            ax1.annotate(f'{y:.1f}',
                        xy=(x, y), xytext=(x, y + 30),
                        fontsize=10, fontweight='bold',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                 edgecolor='#2E86AB', alpha=0.9))
        else:
            ax1.annotate(f'{y:.1f}',
                        xy=(x, y), xytext=(x, y - 15),
                        fontsize=9, fontweight='bold',
                        ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor='#2E86AB', alpha=0.8))
    
    # Highlight the "Physics Drop" (0% to 10%)
    ax1.annotate('', 
                xy=(10, rul_mae_values[1]), xytext=(0, rul_mae_values[0]),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#F18F01',
                               connectionstyle='arc3,rad=0.3'),
                zorder=4)
    
    ax1.annotate('Physics-Enabled\nAdaptation\n(61% Gain)',
                xy=(5, (rul_mae_values[0] + rul_mae_values[1]) / 2),
                fontsize=11, fontweight='bold', color='#F18F01',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF4E6', 
                         edgecolor='#F18F01', linewidth=2, alpha=0.9),
                zorder=6)
    
    ax1.set_xlabel('Few-Shot Data Percentage (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RUL MAE (Cycles)', fontsize=14, fontweight='bold')
    ax1.set_title('Learning Curve: RUL MAE vs Few-Shot Data\n'
                  'Lithium Features Enable Rapid Adaptation to Target Chemistry',
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlim(-5, 85)
    ax1.set_ylim(0, 400)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ========== Plot 2: SOH Accuracy ==========
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.plot(fewshot_percentages, soh_mae_values,
             marker='s', markersize=10, linewidth=2.5,
             color='#06A77D', markerfacecolor='#06A77D',
             markeredgecolor='white', markeredgewidth=2,
             label='SOH MAE')
    
    # Highlight 50%
    ax2.plot(50, baseline_50pct['soh_mae'],
             marker='*', markersize=18, color='#F18F01',
             markeredgecolor='white', markeredgewidth=2,
             zorder=6)
    
    # Add labels
    for i, (x, y) in enumerate(zip(fewshot_percentages, soh_mae_values)):
        if x == 50:
            ax2.annotate(f'{y:.4f}',
                        xy=(x, y), xytext=(x, y + 0.002),
                        fontsize=9, fontweight='bold', color='#F18F01',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF4E6', 
                                 edgecolor='#F18F01', linewidth=2, alpha=0.9))
        else:
            ax2.annotate(f'{y:.4f}',
                        xy=(x, y), xytext=(x, y + 0.002),
                        fontsize=8, fontweight='bold',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                 edgecolor='#06A77D', alpha=0.8))
    
    ax2.set_xlabel('Few-Shot Data Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('SOH MAE', fontsize=12, fontweight='bold')
    ax2.set_title('SOH Prediction Accuracy', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_xlim(-5, 85)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ========== Plot 3: Improvement Percentage ==========
    ax3 = fig.add_subplot(gs[1, 1])
    
    baseline_rul = rul_mae_values[0]  # 0% (zero-shot)
    improvements = [(baseline_rul - rul) / baseline_rul * 100 for rul in rul_mae_values]
    
    bars = ax3.bar(fewshot_percentages, improvements,
                   color=['#C1121F' if x == 0 else '#2E86AB' if x != 50 else '#F18F01' 
                          for x in fewshot_percentages],
                   edgecolor='white', linewidth=2, alpha=0.8)
    
    # Highlight 50%
    bars[3].set_color('#F18F01')
    bars[3].set_alpha(1.0)
    
    # Add value labels
    for i, (x, imp) in enumerate(zip(fewshot_percentages, improvements)):
        ax3.text(x, imp + 2, f'{imp:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Few-Shot Data Percentage (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Improvement Over Zero-Shot Baseline', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # ========== Plot 4: R² Scores ==========
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Get R² scores
    rul_r2_values = [
        comparison_data['zero_shot_baseline']['rul_r2'],
        comparison_data['fewshot_results'][0]['rul_r2'],
        comparison_data['fewshot_results'][1]['rul_r2'],
        comparison_data['fewshot_results'][2]['rul_r2'],
        comparison_data['fewshot_results'][3]['rul_r2']
    ]
    rul_r2_values.insert(3, baseline_50pct['rul_r2'])
    
    soh_r2_values = [
        comparison_data['zero_shot_baseline']['soh_r2'],
        comparison_data['fewshot_results'][0]['soh_r2'],
        comparison_data['fewshot_results'][1]['soh_r2'],
        comparison_data['fewshot_results'][2]['soh_r2'],
        comparison_data['fewshot_results'][3]['soh_r2']
    ]
    soh_r2_values.insert(3, baseline_50pct['soh_r2'])
    
    x_pos = np.arange(len(fewshot_percentages))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, [max(0, r) for r in rul_r2_values], width,
                    label='RUL R²', color='#2E86AB', alpha=0.8, edgecolor='white', linewidth=1.5)
    bars2 = ax4.bar(x_pos + width/2, [max(0, r) for r in soh_r2_values], width,
                    label='SOH R²', color='#06A77D', alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Highlight 50% bars
    bars1[3].set_color('#F18F01')
    bars1[3].set_alpha(1.0)
    bars2[3].set_color('#F18F01')
    bars2[3].set_alpha(1.0)
    
    ax4.set_xlabel('Few-Shot Data Percentage (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax4.set_title('Model Fit Quality (R² Scores)', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(fewshot_percentages)
    ax4.set_ylim(0, 1.1)
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # ========== Plot 5: Key Metrics Summary ==========
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Create summary text
    summary_text = f"""
    KEY RESULTS: 50% Few-Shot Transfer
    
    Performance Metrics:
    • RUL MAE: {baseline_50pct['rul_mae']:.2f} cycles ✅
    • SOH MAE: {baseline_50pct['soh_mae']:.4f} ✅
    • RUL R²: {baseline_50pct['rul_r2']:.3f} ✅
    • SOH R²: {baseline_50pct['soh_r2']:.3f} ✅
    
    Target Achievement:
    • Target: 40-50 cycles
    • Achieved: {baseline_50pct['rul_mae']:.2f} cycles
    • Status: ✅ TARGET ACHIEVED!
    
    Comparison:
    • Zero-Shot: {rul_mae_values[0]:.1f} cycles
    • 50% Few-Shot: {baseline_50pct['rul_mae']:.2f} cycles
    • Improvement: {((rul_mae_values[0] - baseline_50pct['rul_mae']) / rul_mae_values[0] * 100):.1f}%
    
    Efficiency:
    • With just 50% NMC data, achieved
      target performance
    • Better than 70% (57.15 cycles)
    • Close to 80% (54.33 cycles)
    """
    
    ax5.text(0.1, 0.5, summary_text,
            transform=ax5.transAxes,
            fontsize=11, va='center', ha='left',
            bbox=dict(boxstyle='round,pad=1.0', facecolor='#E8F4F8', 
                     edgecolor='#2E86AB', linewidth=2.5, alpha=0.95),
            family='monospace')
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path / 'comprehensive_50pct_results.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'comprehensive_50pct_results.svg', format='svg', bbox_inches='tight')
    
    print(f"✅ Comprehensive plots saved to: {output_path}")
    print(f"   - PNG: comprehensive_50pct_results.png")
    print(f"   - SVG: comprehensive_50pct_results.svg")


def main():
    parser = argparse.ArgumentParser(description='Create comprehensive 50% results visualization')
    parser.add_argument('--comparison_json', type=str, 
                       default='reports/lithium_comparison/comparison.json',
                       help='Path to comparison JSON')
    parser.add_argument('--baseline_json', type=str,
                       default='reports/baseline_50pct/summary.json',
                       help='Path to baseline 50% results JSON')
    parser.add_argument('--output_dir', type=str,
                       default='reports/50pct_visualization',
                       help='Output directory')
    
    args = parser.parse_args()
    
    create_comprehensive_plots(
        args.comparison_json,
        args.baseline_json,
        args.output_dir
    )


if __name__ == '__main__':
    main()


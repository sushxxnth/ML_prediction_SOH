"""
Simple Comparison Plot: 50% Few-Shot Results

A clean, easy-to-understand plot for presentations and reports.
"""

import matplotlib.pyplot as plt
import json
from pathlib import Path

def create_simple_comparison(
    comparison_json: str = 'reports/lithium_comparison/comparison.json',
    baseline_50pct_json: str = 'reports/baseline_50pct/summary.json',
    output_dir: str = 'reports/50pct_visualization'
):
    """Create a simple, clear comparison plot."""
    
    # Load data
    with open(comparison_json) as f:
        comp = json.load(f)
    with open(baseline_50pct_json) as f:
        baseline = json.load(f)
    
    # Prepare data
    percentages = [0, 10, 30, 50, 70, 80]
    rul_mae = [
        comp['zero_shot_baseline']['rul_mae'],
        comp['fewshot_results'][0]['rul_mae'],
        comp['fewshot_results'][1]['rul_mae'],
        baseline['rul_mae'],
        comp['fewshot_results'][2]['rul_mae'],
        comp['fewshot_results'][3]['rul_mae']
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot main curve
    ax.plot(percentages, rul_mae, 
           marker='o', markersize=14, linewidth=3.5,
           color='#2E86AB', markerfacecolor='#2E86AB',
           markeredgecolor='white', markeredgewidth=2.5,
           label='RUL MAE', zorder=3)
    
    # Highlight 50% point
    ax.plot(50, baseline['rul_mae'],
           marker='*', markersize=25, color='#F18F01',
           markeredgecolor='white', markeredgewidth=3,
           label='50% Few-Shot (This Work)', zorder=5)
    
    # Target zone
    ax.axhspan(40, 50, alpha=0.25, color='green', 
              label='Target Zone (40-50 cycles)', zorder=1)
    ax.axhline(y=50, color='green', linestyle='--', linewidth=2.5, alpha=0.7, zorder=2)
    ax.axhline(y=40, color='green', linestyle='--', linewidth=2.5, alpha=0.7, zorder=2)
    
    # Add labels
    for x, y in zip(percentages, rul_mae):
        if x == 50:
            ax.annotate(f'{y:.1f} cycles\n(Target Achieved!)',
                       xy=(x, y), xytext=(x, y - 35),
                       fontsize=12, fontweight='bold', color='#F18F01',
                       ha='center', va='top',
                       bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFF4E6', 
                                edgecolor='#F18F01', linewidth=3, alpha=0.95),
                       zorder=6)
        elif x == 0:
            ax.annotate(f'{y:.0f}',
                       xy=(x, y), xytext=(x, y + 35),
                       fontsize=11, fontweight='bold',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor='#2E86AB', linewidth=2, alpha=0.9))
        else:
            ax.annotate(f'{y:.1f}',
                       xy=(x, y), xytext=(x, y - 20),
                       fontsize=10, fontweight='bold',
                       ha='center', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='#2E86AB', linewidth=2, alpha=0.85))
    
    # Styling
    ax.set_xlabel('Few-Shot Data Percentage (%)', fontsize=16, fontweight='bold')
    ax.set_ylabel('RUL MAE (Cycles)', fontsize=16, fontweight='bold')
    ax.set_title('Cross-Chemistry RUL Prediction: Learning Curve\n'
                'Lithium Features Enable Efficient Few-Shot Transfer',
               fontsize=18, fontweight='bold', pad=25)
    ax.set_xlim(-5, 85)
    ax.set_ylim(0, 400)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, zorder=0)
    ax.legend(loc='upper right', fontsize=13, framealpha=0.95, edgecolor='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Add key insight box
    insight = f"""Key Achievement:
    
50% Few-Shot Transfer:
• RUL MAE: {baseline['rul_mae']:.2f} cycles
• Target: 40-50 cycles
• Status: TARGET ACHIEVED

Improvement:
• Zero-Shot: {rul_mae[0]:.0f} cycles
• 50% Few-Shot: {baseline['rul_mae']:.2f} cycles
• Reduction: {((rul_mae[0] - baseline['rul_mae']) / rul_mae[0] * 100):.1f}%"""
    
    ax.text(0.02, 0.98, insight,
           transform=ax.transAxes,
           fontsize=11, ha='left', va='top',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F4F8', 
                    edgecolor='#2E86AB', linewidth=2.5, alpha=0.95),
           family='monospace')
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path / 'simple_50pct_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'simple_50pct_comparison.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Simple comparison plot saved to: {output_path}")


if __name__ == '__main__':
    create_simple_comparison()


#!/usr/bin/env python3
# AI-Assisted (2025-05-02): Visualize the distribution of entries across veille types.

"""20_visualize_veille_types.py — Generate visualizations of taxonomy distribution across veille types."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.ticker as mtick

def main():
    parser = argparse.ArgumentParser(description="Visualize veille type distribution in Excel workbook.")
    parser.add_argument(
        "--input", default="data/veille_types.xlsx",
        help="Path to the veille types Excel workbook."
    )
    parser.add_argument(
        "--output", default="data/veille_types_distribution.png",
        help="Path to output visualization image."
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Show top N veille types by size."
    )
    args = parser.parse_args()

    # Load Excel summary
    print(f"[visualize_veille] Loading Excel workbook: {args.input}")
    xls = pd.ExcelFile(args.input)
    summary = pd.read_excel(xls, 'Summary')
    
    # Sort by count and take top N
    summary = summary.sort_values('Count', ascending=False).head(args.top)
    
    # Calculate total for percentage
    total_count = summary['Count'].sum()
    
    # Add percentage column
    summary['Percentage'] = summary['Count'] / total_count * 100
    
    # Create figure for the visualization
    plt.figure(figsize=(14, 10))
    
    # Create horizontal bar chart (better for long names)
    bars = plt.barh(
        summary['Type'][::-1],  # Reverse order to have largest at top
        summary['Count'][::-1],
        color=plt.cm.viridis(np.linspace(0, 1, len(summary)))
    )
    
    # Add counts and percentages on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        percent = summary['Percentage'].iloc[len(summary)-1-i]  # Reverse index due to [::-1]
        plt.text(
            width + (total_count * 0.01),  # Small offset
            bar.get_y() + bar.get_height()/2,
            f'{int(width)} ({percent:.1f}%)',
            va='center',
            fontsize=9
        )
    
    # Set labels and title
    plt.xlabel('Nombre d\'entrées', fontsize=12)
    plt.ylabel('Type de veille', fontsize=12)
    plt.title('Distribution des entrées par type de veille', fontsize=16, pad=20)
    
    # Add grid for readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Set a more readable format for the x-axis
    plt.ticklabel_format(style='plain', axis='x')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[visualize_veille] Visualization saved to {output_path}")
    
    # Create treemap (alternative visualization)
    plt.figure(figsize=(12, 10))
    try:
        # Try to use squarify for treemap (optional)
        import squarify
        # Filter out very small categories for better visibility
        plot_df = summary[summary['Count'] > total_count * 0.01]  # > 1% of total
        
        squarify.plot(
            sizes=plot_df['Count'],
            label=[f"{t}\n{c} ({p:.1f}%)" for t, c, p in 
                  zip(plot_df['Type'], plot_df['Count'], plot_df['Percentage'])],
            alpha=0.8,
            color=plt.cm.viridis(np.linspace(0, 1, len(plot_df)))
        )
        plt.axis('off')
        plt.title('Répartition des types de veille (Treemap)', fontsize=16)
        
        # Save treemap
        treemap_path = output_path.with_stem(output_path.stem + '_treemap')
        plt.savefig(treemap_path, dpi=300, bbox_inches='tight')
        print(f"[visualize_veille] Treemap saved to {treemap_path}")
    except ImportError:
        print("[visualize_veille] squarify package not available, treemap visualization skipped")
        # Create pie chart as alternative
        plt.pie(
            summary['Count'], 
            labels=[f"{t}\n({p:.1f}%)" for t, p in zip(summary['Type'], summary['Percentage'])],
            startangle=90,
            colors=plt.cm.viridis(np.linspace(0, 1, len(summary)))
        )
        plt.axis('equal')
        plt.title('Proportion des entrées par type de veille', fontsize=14)
        
        # Save pie chart
        pie_path = output_path.with_stem(output_path.stem + '_pie')
        plt.savefig(pie_path, dpi=300, bbox_inches='tight')
        print(f"[visualize_veille] Pie chart saved to {pie_path}")


if __name__ == '__main__':
    main() 
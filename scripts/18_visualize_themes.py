#!/usr/bin/env python3
# AI-Assisted (2025-04-18): Visualize the distribution of entries across thematic categories.

"""18_visualize_themes.py — Generate visualizations of taxonomy distribution across themes."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize thematic distribution in Excel workbook.")
    parser.add_argument(
        "--input", default="data/thematic_taxonomy_custom.xlsx",
        help="Path to the thematic Excel workbook."
    )
    parser.add_argument(
        "--output", default="data/theme_distribution.png",
        help="Path to output visualization image."
    )
    parser.add_argument(
        "--top", type=int, default=15,
        help="Show top N themes by size."
    )
    args = parser.parse_args()

    # Load Excel summary
    print(f"[visualize_themes] Loading Excel workbook: {args.input}")
    xls = pd.ExcelFile(args.input)
    summary = pd.read_excel(xls, 'Summary')
    
    # Sort by count and take top N
    summary = summary.sort_values('Count', ascending=False).head(args.top)
    
    # Create figure for the visualization
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    bars = plt.bar(
        summary['Theme'], 
        summary['Count'],
        color=plt.cm.viridis(np.linspace(0, 1, len(summary)))
    )
    
    # Add counts above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 5,
            f'{int(height)}',
            ha='center', 
            va='bottom',
            rotation=0,
            fontsize=9
        )
    
    # Set labels and title
    plt.xlabel('Thématiques', fontsize=12)
    plt.ylabel('Nombre d\'entrées', fontsize=12)
    plt.title('Distribution des entrées par thématique', fontsize=14, pad=20)
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[visualize_themes] Visualization saved to {output_path}")
    
    # Create pie chart (alternative visualization)
    plt.figure(figsize=(10, 10))
    plt.pie(
        summary['Count'], 
        labels=summary['Theme'],
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.viridis(np.linspace(0, 1, len(summary)))
    )
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Proportion des entrées par thématique', fontsize=14)
    
    # Save pie chart
    pie_path = output_path.with_stem(output_path.stem + '_pie')
    plt.savefig(pie_path, dpi=300, bbox_inches='tight')
    print(f"[visualize_themes] Pie chart saved to {pie_path}")


if __name__ == '__main__':
    main() 
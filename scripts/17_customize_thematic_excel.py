#!/usr/bin/env python3
# AI-Assisted (2025-04-18): Customize thematic Excel workbook.

"""17_customize_thematic_excel.py — Customize the automatically generated thematic Excel with better names and manual reassignments."""
import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Customize thematic Excel workbook.")
    parser.add_argument(
        "--input", default="data/thematic_taxonomy.xlsx",
        help="Path to the input thematic Excel workbook."
    )
    parser.add_argument(
        "--theme-mapping", default="data/theme_mapping.csv",
        help="Path to CSV with old_theme,new_theme mapping for renaming."
    )
    parser.add_argument(
        "--reassignments", default="data/reassignments.csv",
        help="Path to CSV with label_id,old_theme,new_theme for manual reassignments."
    )
    parser.add_argument(
        "--output", default="data/thematic_taxonomy_custom.xlsx",
        help="Path to output customized Excel workbook."
    )
    args = parser.parse_args()

    # Check if mapping file exists, create template if not
    mapping_path = Path(args.theme_mapping)
    if not mapping_path.exists():
        print(f"[customize_thematic] Creating template theme mapping file: {mapping_path}")
        
        # Load Excel to get current theme names
        xls = pd.ExcelFile(args.input)
        summary = pd.read_excel(xls, 'Summary')
        themes = summary['Theme'].tolist()
        
        # Create mapping template (old_name -> suggested_new_name)
        mapping_df = pd.DataFrame({
            'old_theme': themes,
            'new_theme': themes  # Start with same names, user can edit
        })
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_df.to_csv(mapping_path, index=False)
        print(f"[customize_thematic] Please edit {mapping_path} to define your preferred theme names.")
        print(f"[customize_thematic] Then run this script again.")
        return

    # Check if reassignment file exists, create template if not
    reassign_path = Path(args.reassignments)
    if not reassign_path.exists():
        print(f"[customize_thematic] Creating template reassignments file: {reassign_path}")
        
        # Create empty reassignment template
        reassign_df = pd.DataFrame(columns=['label_id', 'label', 'old_theme', 'new_theme'])
        reassign_path.parent.mkdir(parents=True, exist_ok=True)
        reassign_df.to_csv(reassign_path, index=False)
        print(f"[customize_thematic] If needed, edit {reassign_path} to manually reassign specific entries.")
        print(f"[customize_thematic] Format: label_id,label,old_theme,new_theme")
        print(f"[customize_thematic] Then run this script again.")
        
        # If both files need to be created, exit now
        if not mapping_path.exists():
            return
    
    # Load mapping and reassignment files if they exist
    if mapping_path.exists():
        theme_mapping_df = pd.read_csv(mapping_path)
        theme_mapping = dict(zip(theme_mapping_df['old_theme'], theme_mapping_df['new_theme']))
    else:
        theme_mapping = {}

    if reassign_path.exists() and reassign_path.stat().st_size > 5:  # Not empty
        reassign_df = pd.read_csv(reassign_path)
        has_reassignments = len(reassign_df) > 0
    else:
        has_reassignments = False

    # Load the Excel file
    xls = pd.ExcelFile(args.input)
    sheet_names = xls.sheet_names
    
    # Create a new Excel file with renamed sheets and reassigned entries
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # First, load the summary
        summary_df = pd.read_excel(xls, 'Summary')
        
        # Update theme names in summary
        if theme_mapping:
            summary_df['Theme'] = summary_df['Theme'].map(lambda x: theme_mapping.get(x, x))
        
        # Write updated summary
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Process each original sheet (except Summary)
        for sheet_name in sheet_names:
            if sheet_name == 'Summary':
                continue
            
            # Load this sheet's data
            df = pd.read_excel(xls, sheet_name)
            old_theme = df['theme'].iloc[0] if not df.empty else sheet_name
            
            # Rename theme in the data
            if old_theme in theme_mapping:
                new_theme = theme_mapping[old_theme]
                df['theme'] = new_theme
            else:
                new_theme = old_theme
            
            # Handle reassignments if any
            reassigned_ids = []
            if has_reassignments:
                # Find entries to reassign from this theme
                to_reassign = reassign_df[reassign_df['old_theme'] == old_theme]
                if len(to_reassign) > 0:
                    reassigned_ids = to_reassign['label_id'].tolist()
                    # Remove reassigned entries from current theme
                    df = df[~df['label_id'].isin(reassigned_ids)]
            
            # Make Excel-compatible sheet name
            new_sheet_name = "".join(c for c in new_theme[:31] if c.isalnum() or c in " _").strip()
            if not new_sheet_name:
                new_sheet_name = f"Theme_{sheet_names.index(sheet_name)}"
            
            # Write the updated sheet
            df.to_excel(writer, sheet_name=new_sheet_name, index=False)
        
        # Handle items that need to be reassigned to different themes
        if has_reassignments:
            # Group by new theme
            for new_theme, group in reassign_df.groupby('new_theme'):
                # Make Excel-compatible sheet name
                new_sheet_name = "".join(c for c in new_theme[:31] if c.isalnum() or c in " _").strip()
                if not new_sheet_name:
                    continue
                
                # Check if this sheet already exists (we need to append)
                if new_sheet_name in writer.sheets:
                    existing_df = pd.read_excel(output_path, sheet_name=new_sheet_name)
                    
                    # Create entries to add
                    to_add = []
                    for _, row in group.iterrows():
                        to_add.append({
                            'label_id': row['label_id'],
                            'label': row['label'],
                            'theme': new_theme
                        })
                    
                    # Combine and sort
                    combined_df = pd.concat([existing_df, pd.DataFrame(to_add)])
                    combined_df = combined_df.sort_values('label')
                    
                    # Write back
                    combined_df.to_excel(writer, sheet_name=new_sheet_name, index=False)
                else:
                    # This is a completely new theme, create it from scratch
                    entries = []
                    for _, row in group.iterrows():
                        entries.append({
                            'label_id': row['label_id'],
                            'label': row['label'],
                            'theme': new_theme
                        })
                    
                    new_df = pd.DataFrame(entries)
                    new_df = new_df.sort_values('label')
                    new_df.to_excel(writer, sheet_name=new_sheet_name, index=False)
    
    print(f"[customize_thematic] Successfully customized thematic Excel at {output_path}")
    if theme_mapping:
        print(f"[customize_thematic] Applied {len(theme_mapping)} theme name changes")
    if has_reassignments:
        print(f"[customize_thematic] Processed {len(reassign_df)} manual reassignments")


if __name__ == '__main__':
    main() 
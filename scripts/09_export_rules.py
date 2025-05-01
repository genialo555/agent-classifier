#!/usr/bin/env python3
# AI-Assisted (2025-04-18): Export a summary of Cursor taxonomy rules to CSV.

"""09_export_rules.py — Parse .cursor/rules/sprint1.xml and output a CSV summarizing each rule."""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd


def parse_rule(elem: ET.Element) -> dict:
    """Extract fields from a <cursorRule> element."""
    rule_id = elem.attrib.get('id', '').strip()
    priority = elem.attrib.get('priority', '').strip()
    desc_elem = elem.find('description')
    description = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ''

    # Build condition string
    cond_elem = elem.find('condition')
    conditions = []
    if cond_elem is not None:
        for child in cond_elem:
            tag = child.tag
            if child.attrib:
                attrs = ', '.join(f"{k}={v}" for k, v in child.attrib.items())
                conditions.append(f"{tag}({attrs})")
            else:
                conditions.append(tag)
    condition = ' & '.join(conditions)

    # Build action string
    action_elem = elem.find('action')
    action = ''
    if action_elem is not None:
        act_type = action_elem.attrib.get('type', '').strip()
        params = {k: v for k, v in action_elem.attrib.items() if k != 'type'}
        if params:
            attrs = ', '.join(f"{k}={v}" for k, v in params.items())
            action = f"{act_type}({attrs})"
        else:
            action = act_type

    return {
        'id': rule_id,
        'priority': priority,
        'description': description,
        'condition': condition,
        'action': action,
    }


def main():
    parser = argparse.ArgumentParser(description="Export Cursor rules summary to CSV.")
    parser.add_argument(
        "--xml", default=".cursor/rules/sprint1.xml",
        help="Path to the Cursor rules XML file."
    )
    parser.add_argument(
        "--output", default="data/rules_summary.csv",
        help="Path to write the rules summary CSV."
    )
    args = parser.parse_args()

    # Parse XML
    xml_path = Path(args.xml)
    if not xml_path.is_file():
        raise FileNotFoundError(f"Cursor rules XML not found: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract rules
    rules = [parse_rule(elem) for elem in root.findall('cursorRule')]
    if not rules:
        print(f"No rules found in {xml_path}")
        return

    # Create DataFrame and sort by numeric priority
    df = pd.DataFrame(rules)
    df['priority'] = pd.to_numeric(df['priority'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values(by='priority')

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export CSV
    df.to_csv(output_path, index=False)
    print(f"[export_rules] wrote rules summary to {output_path}")


if __name__ == '__main__':
    main() 
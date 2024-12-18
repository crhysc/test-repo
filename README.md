import json
import os
import argparse
import glob
import sys
import csv
from pymatgen.core import Structure

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate POSCAR files and an id_prop.csv file from JSON files containing 2D structures."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the directory containing the input JSON files."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="POSCAR_files",
        help="Directory to save the generated POSCAR files and the id_prop.csv file. Defaults to 'POSCAR_files'."
    )
    parser.add_argument(
        "-f",
        "--filename_prefix",
        type=str,
        default="POSCAR_",
        help="Prefix for the POSCAR filenames. Defaults to 'POSCAR_'."
    )
    return parser.parse_args()

def load_json(file_path):
    """
    Loads JSON data from a file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded JSON data from '{file_path}'.")
        return data
    except Exception as e:
        print(f"Error loading JSON file '{file_path}': {e}")
        return None

def create_output_directory(directory):
    """
    Creates the output directory if it does not exist.
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created output directory '{directory}'.")
        except Exception as e:
            print(f"Error creating directory '{directory}': {e}")
            sys.exit(1)
    else:
        print(f"Output directory '{directory}' already exists.")

def generate_poscar(structure, filename):
    """
    Generates a POSCAR file from a pymatgen Structure object.
    """
    try:
        structure.to(fmt="poscar", filename=filename)
        print(f"POSCAR file written to '{filename}'.")
    except Exception as e:
        print(f"Error writing POSCAR file '{filename}': {e}")

def sanitize_filename(name):
    """
    Sanitizes the filename by removing or replacing invalid characters.
    """
    return "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)

def main():
    args = parse_arguments()
    input_dir = args.input
    output_dir = args.output
    filename_prefix = args.filename_prefix

    # Create output directory
    create_output_directory(output_dir)

    # Prepare to write id_prop.csv
    id_prop_path = os.path.join(output_dir, "id_prop.csv")

    try:
        csvfile = open(id_prop_path, 'w', newline='')
    except Exception as e:
        print(f"Error opening id_prop.csv for writing: {e}")
        sys.exit(1)

    writer = csv.writer(csvfile)
    skipped = 0

    # Find and sort all JSON files in the input directory
    pattern = os.path.join(input_dir, "*.json")
    json_files = sorted(glob.glob(pattern))

    if not json_files:
        print(f"No JSON files found in directory '{input_dir}'.")
        csvfile.close()
        sys.exit(1)

    for file_index, file_path in enumerate(json_files):
        print(f"\nProcessing file {file_index}: '{file_path}'")
        data = load_json(file_path)
        if data is None:
            continue

        base_filename = os.path.basename(file_path)
        sanitized_base_filename = sanitize_filename(os.path.splitext(base_filename)[0])

        # Iterate over the top-level keys in the JSON data
        for top_key in data:
            print(f"Processing top-level key: '{top_key}'")
            entries = data[top_key]
            if not entries:
                print(f"No entries found under key '{top_key}'. Skipping.")
                continue

            sanitized_top_key = sanitize_filename(top_key) or "key_empty"

            for entry_index, entry in enumerate(entries):
                steps = entry.get('steps', [])
                if not steps:
                    print(f"No 'steps' found in entry {entry_index} under key '{top_key}'. Skipping.")
                    continue

                # Select only the last step
                last_step = steps[-1]
                structure_dict = last_step.get('structure')
                energy = last_step.get('energy', None)

                if not structure_dict or energy is None:
                    print(f"Missing 'structure' or 'energy' in the last step of entry {entry_index} under key '{top_key}'. Skipping.")
                    skipped += 1
                    continue

                # Reconstruct the Structure object from the last step
                try:
                    structure = Structure.from_dict(structure_dict)
                except Exception as e:
                    print(f"Error reconstructing Structure for the last step of entry {entry_index} under key '{top_key}': {e}")
                    skipped += 1
                    continue

                # Create a single POSCAR file for this entry (from its last step)
                poscar_filename = f"{filename_prefix}{sanitized_base_filename}_{sanitized_top_key}_entry{entry_index}_laststep.vasp"
                poscar_path = os.path.join(output_dir, poscar_filename)

                generate_poscar(structure, poscar_path)

                # Write a single line to id_prop.csv for this entry
                try:
                    writer.writerow([poscar_filename, energy])
                except Exception as e:
                    print(f"Error writing to id_prop.csv for {poscar_filename}: {e}")
                    skipped += 1

    csvfile.close()
    print("\nAll POSCAR files have been generated and id_prop.csv has been created.")
    print(f"Number of entries skipped: {skipped}")

if __name__ == "__main__":
    main()

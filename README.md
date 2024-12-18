import json
import os
import argparse
import glob
import sys
import csv
from pymatgen.core import Structure

def parse_arguments():
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
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded JSON data from '{file_path}'.")
        return data
    except Exception as e:
        print(f"Error loading JSON file '{file_path}': {e}")
        return None

def create_output_directory(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created output directory '{directory}'.")
        except Exception as e:
            print(f"Error creating directory '{directory}': {e}")
            sys.exit(1)
 

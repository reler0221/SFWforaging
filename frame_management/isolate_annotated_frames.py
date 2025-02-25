import os
import shutil
import pandas as pd

# Define paths
main_directory = "/Volumes/T5 EVO/Foraging HD/Raw_frames_to_annotate"  # Replace with your actual main directory
destination_directory = "/Volumes/T5 EVO/Foraging HD/only_annotated_frames"  # Replace with your desired destination
filenames_csv = pd.read_csv("/Users/hyewonjun/UZH/Thesis/annotated_filenames.csv")
filenames_to_find = filenames_csv["filename"].tolist()# List of filenames to search for

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Walk through the directory and find matching files
for root, _, files in os.walk(main_directory):
    for file in files:
        if file in filenames_to_find:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_directory, file)
            shutil.copy2(source_path, destination_path)  # Preserves metadata
            print(f"Copied: {source_path} -> {destination_path}")

print("File copying completed.")
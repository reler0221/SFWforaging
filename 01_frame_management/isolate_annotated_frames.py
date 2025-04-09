import os
import shutil
import pandas as pd

# Define paths
main_directory_frames = "/Volumes/T5 EVO/Foraging HD/Raw_frames_to_annotate"
main_directory_annotations ="/Volumes/T5 EVO/Foraging HD/Annotation"
destination_directory = "/Volumes/T5 EVO/Foraging HD/only_annotated_frames"

filenames_csv = pd.read_csv("/Users/hyewonjun/UZH/Thesis/annotated_filenames.csv")
filenames_to_find = filenames_csv["filename"].tolist()# List of filenames to search for

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

#%% Walk through the directory and find matching files
for root, _, files in os.walk(main_directory_frames):
    for file in files:
        if file in filenames_to_find:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_directory, file)
            shutil.copy2(source_path, destination_path)  # Preserves metadata
            print(f"Copied:  {source_path} -> {destination_path}")

print("File copying completed.")

#%% Merge all annotation files and isolate the rows with annotation into a separate file
frames_csv = [os.path.join(main_directory_annotations,f) for f in os.listdir(main_directory_annotations) if f.endswith(".csv") and not f.startswith(".")]
all_annotations_df = pd.concat((pd.read_csv(file) for file in frames_csv))
nonempty_annotations_df = all_annotations_df[all_annotations_df["region_count"]>0]
nonempty_annotations_df.to_csv(destination_directory + "/annotations.csv", index=False)
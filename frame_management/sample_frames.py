import os
import random
import shutil

#%%
import_dir = "/Volumes/T5 EVO/Foraging HD/Environment_all_frames/Batch12"
export_dir = "/Volumes/T5 EVO/Foraging HD/Isolated_frames/background_frames"
n = 100 # number of frames to sample from directory

#%% Sample
filenames = [f for f in os.listdir(import_dir) if f.endswith(".jpg") and not f.startswith(".")]
sampled_filenames = random.sample(filenames, n)
print(len(sampled_filenames))
#%% Copy sampled frames to export directory
for file in sampled_filenames:
    source_path = os.path.join(import_dir, file)
    destination_path = os.path.join(export_dir, file)
    shutil.copy2(source_path, destination_path)
    print(f"File {file} copied in {destination_path}")

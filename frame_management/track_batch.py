import os
import pandas as pd

#%%
import_dir = "/Volumes/T5 EVO/Foraging HD/Isolated_frames/background_frames"
search_dir = "/Volumes/T5 EVO/Foraging HD/Raw_frames_to_annotate"
#%%
filenames = [f for f in os.listdir(import_dir) if f.endswith(".jpg") and not f.startswith(".")]
print(len(filenames))


#%%
batch_list = []
for root, dirs, files in os.walk(search_dir):
    for file in files:
        if file in filenames: batch_list.append(root.split("/")[-1])

#%%
batch_series = pd.Series(batch_list)
print(set(batch_series))
table_batch = batch_series.value_counts()
print(table_batch)

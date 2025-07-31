import pandas as pd
import numpy as np

#%% import csv
f = pd.read_csv("/Volumes/T5 EVO/Foraging HD/Manual_analyses/Pecking_manual_annotations/Pecking_manual_annotations.csv")
f = f[1:] # first row was an example row - should delete it later in the raw file

#%% convert time to seconds
f["Video_length"] = f["Video_length"].apply(lambda x: int(x.split(":")[0])*60 + int(x.split(":")[1]))
#%%
pecking_per_video_df = f.groupby("Video_name")["Pecking_events"].sum()
print(pecking_per_video_df)

video_length_df = f.groupby("Video_name")["Video_length"].first()
print(video_length_df)


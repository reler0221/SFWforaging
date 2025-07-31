import os
import pandas as pd
import random
import shutil

#%%
random.seed(0)
annotation_file = "/Volumes/T5 EVO/Foraging HD/Annotation/extra1.csv"
frames_dir = "/Volumes/T5 EVO/Foraging HD/Raw_frames_to_annotate/Extra1"
birds_outdir = "/Volumes/T5 EVO/Foraging HD/Isolated_frames/annotated_frames_extra"
background_outdir = "/Volumes/T5 EVO/Foraging HD/Isolated_frames/background_frames_extra"
#%%
df = pd.read_csv(annotation_file)
df_birds = df.loc[df["region_count"] > 0]
df_background = df.loc[df["region_count"] == 0]
#%%
bird_filenames = df_birds["#filename"].tolist()
background_filenames = df_background["#filename"].tolist()
frames_list = [filename for filename in os.listdir(frames_dir) if filename.endswith(".jpg") and not filename.startswith(".")]
print(len(frames_list))
#%%
for frame in frames_list:
    if frame in background_filenames:
        shutil.copyfile(frames_dir + "/" + frame, background_outdir + "/" + frame)
    elif frame in bird_filenames:
        shutil.copyfile(frames_dir + "/" + frame, birds_outdir + "/" + frame)



#%%
df_birds.to_csv("/Volumes/T5 EVO/Foraging HD/Isolated_frames/extra_birds.csv", index=False)

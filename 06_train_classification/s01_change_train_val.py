import os
import random
import pandas as pd

#%%
df = pd.read_csv("/Volumes/T5 EVO/Foraging HD/Classification_model_input/01_Pecking/pecking.csv")
df["video_id"] = df["filename"].str.split("_").apply(lambda x: '_'.join(x[:3]))
#label = df["type"].tolist()

#%% check exceptional videos
y_per_videos = df[df["type"] == "y"].groupby("video_id").size().reset_index()
y_list = y_per_videos.video_id.tolist()

print(y_list)
print(len(y_list))
print(sum(df["type_num"])/len(y_list))
#%% exclude videos that could bias
exclusions = ["20240805_1_00035"]
y_list = [el for el in y_list if el not in exclusions]
print(len(y_list))
#%%
goal_range = [len(y_list)*0.35, len(y_list)*0.4]
val_count = 0
while val_count < goal_range[0] or val_count > goal_range[1]:
    val_videos = random.sample(y_list, 60)
    val_count= y_per_videos[y_per_videos["video_id"].isin(val_videos)][0].sum()
    print(val_count)

#%%
df.loc[df["video_id"].isin(val_videos), "val_or_train"] = "val"
df.loc[~df["video_id"].isin(val_videos), "val_or_train"] = "train"

#%%
print(df.groupby(
    ["type_num", "val_or_train"]
).size().unstack(fill_value=0))

#%%
df.to_csv("/Volumes/T5 EVO/Foraging HD/Classification_model_input/01_Pecking/pecking4.csv")








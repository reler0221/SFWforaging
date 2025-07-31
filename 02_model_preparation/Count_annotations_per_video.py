# Count annotated frames per videos
# Assume there is already a merged filed of all annotations.

import os
import pandas as pd


#%% 1. import annotation csv
f = pd.read_csv("/Volumes/T5 EVO/Foraging HD/Isolated_frames/revised_3425.csv")
filenames = f[f["region_shape_attributes"].str.len()>10]["#filename"].str.replace(".jpg","")

print("Number of birds: ", len(filenames))
print("Number of frames: ", len(filenames.unique()))
print("Nuber of videos: ", len(filenames.str.slice(stop=-5).unique()))

#%% 2-1. count all birds in each videos - counts multiple birds in frame
birds_per_video_dict = {}
frames_per_video_dict = {}
birds_per_frame_dict = {}
for i,filename in filenames.items():
    videoname = filename[:-5] # 20240805_1_00035_0007 -> 20240805_1_00035
    if (videoname in frames_per_video_dict.keys()) and (filename in birds_per_frame_dict.keys()):
        birds_per_video_dict[videoname] +=1
        birds_per_frame_dict[filename] += 1

    elif (videoname in frames_per_video_dict.keys()) and (filename not in birds_per_frame_dict.keys()):
        frames_per_video_dict[videoname] += 1
        birds_per_video_dict[videoname] += 1
        birds_per_frame_dict[filename] = 1

    else:
        frames_per_video_dict[videoname] = 1
        birds_per_video_dict[videoname] = 1
        birds_per_frame_dict[filename] = 1


#%% 3. Make dataframe
sorted_keys = sorted(frames_per_video_dict.keys())
video_framecount_df = pd.DataFrame({
    "Videoname": sorted_keys,
    "Birds_per_video": [birds_per_video_dict[videoname] for videoname in sorted_keys],
    "Frames_per_video": [frames_per_video_dict[videoname] for videoname in sorted_keys]
})

#%% 4. Check -- Looks good(equal to the sums calculated in the first section)
print(sum(video_framecount_df["Frames_per_video"]))
print(sum(video_framecount_df["Birds_per_video"]))
print(len(video_framecount_df))

#%% 5. Export
video_framecount_df.to_csv("Frames_per_video.csv", index=False)




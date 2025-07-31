## Preprocessing
# Smooth data: Detection window size = 5 frames , Pecking window size = 3? none?
# Fill Tracking ID NaN if it is surrounded by same ID back and forth
# Each Tracking is considered as 1 occurrence of foraging bout
# Each cluster of pecking (even if there is only 1 smoothed frame) is considered as 1 occurrence of pecking
# Get start and end frame of each Foraging bout -> Add as foraging bout length
# Add the number of pecking occurrences per each foraging bout

#%%
import numpy as np
import pandas as pd
import os
from collections import Counter

#%%
input_dir = "/Users/hyewonjun/UZH/Thesis/Data/Raw18060102"
output_dir = "08_preprocessing"
output_csv = "18060102_pad3.csv"

input_files = os.listdir(input_dir)
files = os.listdir(input_dir)
filepaths = [os.path.join(input_dir,filename) for filename in files]

#%%

data = pd.DataFrame()# pd.read_csv("08_preprocessing/Test_allmodels_raw_01.csv")

for filepath in filepaths:
    f = pd.read_csv(filepath, encoding="latin1")
    data = pd.concat([data,f], ignore_index=True)

data["GROUP"] = data["FILENAME"].str.split("_").str[1]
colnames = data.columns

print(colnames)


#%%
videos = data["FILENAME"].unique().tolist()
videos.sort()

print(len(videos))
print(videos)

#%%
fps = 25
frame_interval = 1
detection_padding = 2 # window size = 5 (0.2 sec)
pecking_padding = 1 # window size = 3 (0.12 sec)

summarized_results = {
    "TRACK_ID":[],
    "FILENAME":[],
    "BOUT_START": [],
    "BOUT_END": [],
    "PECKING": [],
    "PECKING_FRAMES": [],
    # "SHADE": [],
    "MOULT": [],
    "VIDEO_DURATION": []
}

#%%
for video in videos:
    observed_times = []
    subset_video: pd.DataFrame = data[data["FILENAME"] == video].copy()

    # Smooth detection
    detection_raw: pd.Series = subset_video["X1"].fillna(0).astype(bool)
    detection_smoothed = detection_raw.rolling(window=detection_padding * 2 + 1, center=True, min_periods=1).mean().round()
    subset_video["DETECT"] = detection_raw
    subset_video["DETECT_SMOOTHED"] = detection_smoothed

    # Fill in track id of the initial detection (the first frame detected doesn't have an id detected)
    mask = (
            subset_video['BOX_ID'].isna() &
            subset_video['BOX_ID'].shift(-1).notna() &
            subset_video['X1'].notna()
    )
    subset_video.loc[mask, 'BOX_ID'] = subset_video['BOX_ID'].shift(-1)

    # Fill BOX_ID nans if it is between same id e.g. 123 123 nan 123
    only_with_bird = subset_video[subset_video["DETECT_SMOOTHED"]==1].copy()
    idx_indices = only_with_bird[only_with_bird['BOX_ID'].isna()].index
    for i in idx_indices:
        last_nonna_from_prev10: int = only_with_bird['BOX_ID'].loc[:i-1].last_valid_index()
        first_nonna_from_next10: int = only_with_bird['BOX_ID'].loc[i+1:].first_valid_index()
        if last_nonna_from_prev10 is not None and first_nonna_from_next10 is not None:
            if only_with_bird["BOX_ID"].loc[first_nonna_from_next10] == only_with_bird["BOX_ID"].loc[last_nonna_from_prev10]:
                only_with_bird.loc[i, "BOX_ID"] = only_with_bird["BOX_ID"].loc[first_nonna_from_next10]

    # Group by ID
    ids = [x for x in only_with_bird["BOX_ID"].unique() if not np.isnan(x)]
    sequence_starts_per_id = []
    sequence_ends_per_id = []
    pecking_frequencies_per_id = []
    pecking_frames_per_id = []
    moult_per_id = []
    # shade_frames_per_id = []


    for track_id in ids:
        id_data:pd.DataFrame = only_with_bird[only_with_bird["BOX_ID"] == track_id].copy().reset_index(drop=True)
        start_frame, end_frame = id_data["FRAME_NUM"].iloc[0], id_data["FRAME_NUM"].iloc[-1]
        sequence_starts_per_id.append(start_frame)
        sequence_ends_per_id.append(end_frame)
        #Smooth pecking and
        pecking_smoothed = id_data["PECKING_LABEL"].rolling(window=pecking_padding * 2 + 1, center=True, min_periods=1).mean().round()
        pecking_starts = []
        pecking_ends = []

        flag = 0
        for i, label in enumerate(pecking_smoothed):
            if label == 0:
                if flag == 1:
                    pecking_ends.append(id_data["FRAME_NUM"].iloc[i-1])
                    flag = 0
                else: continue
            else:
                if flag == 0:
                    pecking_starts.append(id_data["FRAME_NUM"].iloc[i])
                    flag = 1
                else: continue
        if flag == 1: # get the last frame num if the series ends with 1
            pecking_ends.append(id_data["FRAME_NUM"].iloc[-1])

        pecking_frequencies_per_id.append(len(pecking_starts))
        pecking_frames_per_id.append(sum(pecking_smoothed))
        moult_per_id.append(id_data["BLUE_LABEL"].mean())
        # shade_frames_per_id.append(id_data["SHADE_LABEL"].sum)


    video_duration = subset_video["FRAME_NUM"].iloc[-1]*frame_interval/25 # in seconds


    summarized_results["TRACK_ID"] += ids
    summarized_results["FILENAME"] += [video]*len(ids)
    summarized_results["VIDEO_DURATION"] += [video_duration]*len(ids)
    summarized_results["BOUT_START"] += sequence_starts_per_id
    summarized_results["BOUT_END"] += sequence_ends_per_id
    summarized_results["PECKING"] += pecking_frequencies_per_id
    summarized_results["PECKING_FRAMES"] += pecking_frames_per_id
    summarized_results["MOULT"] += moult_per_id
    # summarized_results["SHADE"] += shade_frames_per_id

    # print(video, summarized_results,"Done")




#%%
results = pd.DataFrame(summarized_results)
results["BOUT_LENGTH"] = results["BOUT_END"] - results["BOUT_START"]+frame_interval
results["DATE"] = results["FILENAME"].str.split("_").apply(lambda x: x[0])
results["GROUP"] = results["FILENAME"].str.split("_").apply(lambda x: x[1])
print(results)

#%% Add metadata information: time of day, exclusions
#%% Load data
data = results.copy()
meta = pd.read_csv("Metadata/SFW_metainfo.csv")
meta_training = pd.read_csv("Metadata/training_notes_expanded.csv") # only for summer

#%% Format meta_training
meta_training["TIME"] = meta_training["TIME"].apply(lambda x: x[:5]) #remove "AEDT

#%% filter with dates
start_date = 20240800
end_date = 20250800

data = data[data["DATE"].astype('int').between(start_date,end_date)].reset_index(drop=True)
meta = meta[meta["Date"].between(start_date,end_date)].reset_index(drop=True)

#%% Get date / group / time from meta - no filename
date_group = meta["Group_id"]
time_h = meta["Time_stop"].apply(lambda x: int(x[:2]))
time_m = meta["Time_stop"].apply(lambda x: int(x[-2:])/60)
time = time_h + time_m

main_date_group_time = pd.concat([date_group, time], axis=1)



#%% Get filename / time from training meta
time_h = meta_training["TIME"].apply(lambda x: int(x[:2]))
time_m = meta_training["TIME"].apply(lambda x: int(x[-2:])/60)
time = time_h + time_m
training_filename_time = pd.concat([meta_training["FILE_NAME"], time], axis=1)

#%%
time_arr = np.empty(data.shape[0])
#%%
for i, row in enumerate(data.itertuples()):
    if row.GROUP == "training":
        filename = row.FILENAME[:-4]
        time_matches = training_filename_time.loc[training_filename_time["FILE_NAME"] == filename, "TIME"]
        if time_matches.empty:
            time = None
            print(filename, "not in sheet")
        else:
            time = time_matches.values[0]
    else:
        date_group = "_".join(row.FILENAME[:-4].split("_")[:2])
        time_matches = main_date_group_time.loc[main_date_group_time["Group_id"] == date_group, "Time_stop"]
        if time_matches.empty:
            time = None
            print(date_group, "not in sheet")
        else:
            time = time_matches.values[0]

    time_arr[i] = time
print(time_arr)
#%%
data["TIME"] = time_arr

#%%
data = data[data["TIME"].notna()].reset_index(drop=True)
print(data.head())
#%% Assign groups
#%% Load data
df = data.copy() # pd.read_csv("08_preprocessing/Test_pecking_pad3_time.csv")
meta_group = pd.read_csv("Metadata/SFW_group_compo.csv")
training_meta = pd.read_csv("Metadata/training_notes_expanded.csv")

#%% Inspect metadata
print(training_meta["GROUP_NAME"].unique())
print(training_meta[training_meta["GROUP_NAME"]=="cmra/jgyr"]) # 6
print(training_meta[training_meta["GROUP_NAME"].isnull()]) # 18


#%%
start_date = 20240800
end_date = 20250800
meta_group["Date"] = meta_group["Group_id"].apply(lambda x: int(x.split("_")[0]))
breeding_meta_group = meta_group[meta_group["Date"].between(start_date,end_date).reset_index(drop=True)].copy()

birds_identified = breeding_meta_group["Ind"].unique()
birds_identified = pd.Series(birds_identified).str.lower().tolist()



#%% Breeding groups *only for 2024 breeding
breeding_groups = {
    "cmra": ["cmra","pnaba","jbbwb","hona","hranr","hgrq"],
    "gbb": ["gbb", "conr", "soo", "snro", "hnwo", "sgrm"],
    "jbrb": ["jbrb","jmrw","iwq","swa"],
    "jgr": ["jgyr", "pwrbo", "zowogn", "sbnr", "hnrwg", "honrw", "hma","hyyr", "hoon", "hbny"],
    "pbgrb": ["pbgrb", "inyb", "jbn", "sgn", "smag", "hoab", "hwbo", "hragr"],
    "srbaba": ["srbaba", "cgrb"],
    "syw": ["syw", "smow", "hgrwn", "hgrb"],
    "unb": ["jmna", "hoa","hyab", "hbby"] # didn't put unb in values because it will count all unb as this unb when this transforms to ind_to_group.
}
birds_known = np.concatenate([inds for inds in breeding_groups.values()]).tolist()
ind_to_group = {bird_id: group for group, members in breeding_groups.items() for bird_id in members}

#%%
print(df["FILENAME"].apply(len).unique()) # 20, 27

#%%
df["GROUP_ID"] = [None]*len(df)

for i, filename in enumerate(df["FILENAME"].apply(lambda x: x[:-4])):

    if df["GROUP"][i] == "training":
        if filename in training_meta["FILE_NAME"].values:
            group = training_meta.loc[training_meta["FILE_NAME"] == filename, "GROUP_NAME"].values[0]
            df.loc[i, "GROUP_ID"] = group
        else: print(filename, "not in metadata")

    else:
        file_label = "_".join(filename.split("_")[:2])
        if file_label in meta_group["Group_id"].values:
            members = meta_group.loc[meta_group["Group_id"] == file_label,"Ind"]
            group_ids = [ind_to_group[member] for member in members.str.lower() if member in ind_to_group]
            if group_ids:
                group_counts = Counter(group_ids)
                max_freq = max(group_counts.values())
                top_groups = [group for group, count in group_counts.items() if count == max_freq]

                if len(top_groups) == 1:
                    group = top_groups[0]
                    df.loc[i, "GROUP_ID"] = group
                else:
                    print(filename, "too mixed")

            else: print(filename, "no known inds")

        else: print(filename, "not in metadata")

#%%
print(df.loc[df["GROUP_ID"].isnull(), :])

#%%

df.to_csv("08_preprocessing/18060102_pad3.csv")






import numpy as np
import pandas as pd
import os



#%%
data = pd.read_csv("08_preprocessing/Test_pecking_pad3.csv")
meta = pd.read_csv("Metadata/SFW_metainfo.csv")
meta_training = pd.read_csv("Metadata/training_notes_expanded.csv") # only for summer

#%% Format meta_training
meta_training["TIME"] = meta_training["TIME"].apply(lambda x: x[:5]) #remove "AEDT

#%% filter with dates
start_date = 20241200
end_date = 20241230

data = data[data["DATE"].between(start_date,end_date)].reset_index(drop=True)
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
#%%
data.to_csv("08_preprocessing/Test_pecking_pad3_time.csv")









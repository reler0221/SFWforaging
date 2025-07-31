import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
#%%
df = pd.read_csv("08_preprocessing/Test_pecking_pad3_time.csv")
meta_group = pd.read_csv("Metadata/SFW_group_compo.csv")
training_meta = pd.read_csv("Metadata/training_notes_expanded.csv")

#%% Inspect metadata
print(training_meta["GROUP_NAME"].unique())
print(training_meta[training_meta["GROUP_NAME"]=="cmra/jgyr"]) # 6
print(training_meta[training_meta["GROUP_NAME"].isnull()]) # 18


#%%
start_date = 20241200
end_date = 20250200
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
#%% export
df.to_csv("09_analyses/Test_pecking_pad3_time_withgroup.csv", index=False)










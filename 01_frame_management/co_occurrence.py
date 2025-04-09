# -*- coding: utf-8 -*-

"""
Revised 2025/03/06
"""


#%% Setups
# Import libraries
import os
import pandas as pd
import re
import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from annotation_functions import *

#%% Set environment
#os.chdir("/Users/hyewonjun/UZH/Thesis/")

#%% Parameters
folder_with_annotations = "/Volumes/T5 EVO/Foraging HD/only_annotated_frames" #/Users/hyewonjun/UZH/Thesis/annotations
filetype = ".csv"
categories = sorted(["pecking", "shade","blue"])
colnames = ["filename", "batch", "error", "region_count"] + categories
response_type_yesno = ["y", "n", "u"] #u: undeterminable


#%% Initialize summary variables
total_frames = 0
total_annotated_frames = 0
region_attributes_metainfo = pd.DataFrame(columns=colnames)
errors = []


#%% Run Code
# Import files
csv_paths = sorted([os.path.join(folder_with_annotations, f) for f in os.listdir(folder_with_annotations) if f.endswith(filetype) and not f.startswith(".")])
print(len(csv_paths))

# Fill region attributes information for all batches
for filename in csv_paths:
    file = pd.read_csv(filename)
    region_attributes_clean_df = make_clean_df(filename, file, colnames)
    region_attributes_clean_df = fill_region_att_and_errorcheck(file, categories, region_attributes_clean_df, response_type_yesno)
    region_attributes_metainfo = pd.concat([region_attributes_metainfo, region_attributes_clean_df], ignore_index=True)

#%% Isolate pecking, shade, blue columns
isolated_df = region_attributes_metainfo[categories].copy()

#Only take the annotations that has all values(comment out to include 0.0s)
isolated_df = isolated_df[isolated_df.ne(0.0).all(axis=1)]

#%%Check all response types
response_types = isolated_df[categories].stack().value_counts()
#%% Co_occurrence pairwise

co_pecking_shade = pd.crosstab(isolated_df["pecking"], isolated_df["shade"])
co_pecking_blue = pd.crosstab(isolated_df["pecking"], isolated_df["blue"])
co_shade_blue = pd.crosstab(isolated_df["shade"], isolated_df["blue"])
#%% Pairwise heatmap
sns.heatmap(co_pecking_shade, annot=True, fmt="d", cmap="Blues")
plt.show()
sns.heatmap(co_pecking_blue, annot=True, fmt="d",cmap="Blues")
plt.show()
sns.heatmap(co_shade_blue, annot=True, fmt="d",cmap="Blues")
plt.show()



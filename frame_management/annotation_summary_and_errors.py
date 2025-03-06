
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

#%% Summary statistics and errors
# Total frames and annotated frames
total_frames = len(region_attributes_metainfo)
total_annotated_frames = len(region_attributes_metainfo[region_attributes_metainfo["region_count"] > 0])
print("Total frames:", total_frames)
print("Total annotated frames:", total_annotated_frames)
print("\n")

# Frequency table for each variable
for variable in categories:
    table = region_attributes_metainfo[variable].value_counts()
    print(f"{table}\n")

# Errors
errors = region_attributes_metainfo[region_attributes_metainfo["error"] != 0]
print("ERRORS:\n", errors)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:40:28 2024

@author: hyewonjun
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised 2025/02/19
"""

#%%READMEEE
# Libraries: os, pandas, re, numpy

# Reads the annotation files exported from via and (1)counts total frame (2)counts annotated frames (3)counts annotated frames for each defined variables(categories) (4)gives errornous annotations

# A directory(folder) with the annoatated files should be inside the working directory. If not, either set the working directory to make it that way, or paste an absolute path of the folder for the variable "folder_with_annotations" in #%%Parameters

# The code reads all csv files in the folder where you have put your annotated files. The code would not work if there are other csv types files in the same folder.

# Do not change colnames.

# Every variable with "substrate" will later change to another non-binary variable. Ignore for now.

#%% Setups
# Import libraries
import os
import pandas as pd
import re
import numpy as np
import csv

#%% Set environment
os.chdir("/Users/hyewonjun/UZH/Thesis/")

#%% Parameters
folder_with_annotations = "annotations"
filetype = ".csv"
categories = sorted(["pecking", "shade", "substrate", "covered"])
colnames = ["filename", "batch", "error", "region_count"] + categories
response_type_yesno = ["y", "n", "u"] #u: undeterminable
response_type_substrate = ["b", "v", "l", "g", "p", "d", "r", "s", "t","m" "n"]

#%% Initialize summary variables
total_frames = 0
total_annotated_frames = 0
region_attributes_metainfo = pd.DataFrame(columns=colnames)
errors = []

#%% Functions
# Create a clean DataFrame to save region attributes on separate columns
def make_clean_df(filename, file, colnames):
    region_attributes_clean_df = pd.DataFrame(np.zeros((len(file), len(colnames))), columns=colnames)

    region_attributes_clean_df["filename"] = file["#filename"].copy()
    region_attributes_clean_df["batch"] = re.search(r"(\d+)(?=\.)", filename).group()
    region_attributes_clean_df["region_count"] = file["region_count"].copy()
    return region_attributes_clean_df

# Fill in region attributes and perform error checking
def fill_region_att_and_errorcheck(file, categories, region_attributes_clean_df, response_type_substrate, response_type_yesno):
    for i, attributes in enumerate(file["region_attributes"]):
        attributes_dict = eval(attributes)  # Ensure attributes are parsed correctly
        for key, value in attributes_dict.items():
            if key == "substrate":
                region_attributes_clean_df.loc[i, key] = value
                # Error check
                if value not in response_type_substrate:
                    region_attributes_clean_df.loc[i, "error"] = key
            elif key in categories:
                region_attributes_clean_df.loc[i, key] = value
                # Error check
                if value not in response_type_yesno:
                    region_attributes_clean_df.loc[i, "error"] = key
    return region_attributes_clean_df

def create_list_of_annotated_frames_by_class(region_attributes_metainfo, class_str):
    filenames_annotated = region_attributes_metainfo[region_attributes_metainfo[class_str] != 0.0]
    filenames_annotated = set(filenames_annotated["filename"])
    return filenames_annotated

#%% Run Code
# Import files
csv_paths = sorted([os.path.join(folder_with_annotations, f) for f in os.listdir(folder_with_annotations) if f.endswith(filetype)])


# Fill region attributes information for all batches
for filename in csv_paths:
    file = pd.read_csv(filename)
    region_attributes_clean_df = make_clean_df(filename, file, colnames)
    region_attributes_clean_df = fill_region_att_and_errorcheck(file, categories, region_attributes_clean_df, response_type_substrate, response_type_yesno)
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

#%% Get list of annotated frames
# Full list of annotated frames
filenames_annotated = region_attributes_metainfo[region_attributes_metainfo["region_count"] > 0]
filenames_annotated = set(filenames_annotated["filename"])
# Lists of annotated frames per class
annotated_byclass = {}
for class_str in categories:
    annotated_byclass[class_str] = create_list_of_annotated_frames_by_class(region_attributes_metainfo, class_str)
    print(class_str, len(annotated_byclass[class_str]))

print(annotated_byclass["pecking"]-annotated_byclass["shade"])
print(annotated_byclass["shade"]-annotated_byclass["pecking"])

#%% Save annotated filenames as csv.

list_filenames_annotated = sorted((list(filenames_annotated)))

df = pd.DataFrame(list_filenames_annotated, columns=["filename"])
df.to_csv("~annotated_filenames.csv", index=False)
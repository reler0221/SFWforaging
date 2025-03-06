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


#%% Setups
# Import libraries
import os
import pandas as pd
import re
import numpy as np
import csv

#%% Set environment
#os.chdir("/Users/hyewonjun/UZH/Thesis/")

#%% Parameters
folder_with_annotations = "/Volumes/T5 EVO/Foraging HD/only_annotated_frames" #/Users/hyewonjun/UZH/Thesis/annotations
filetype = ".csv"
categories = sorted(["pecking", "shade", "substrate", "covered","blue"])
colnames = ["filename", "batch", "error", "region_count"] + categories
response_type_yesno = ["y", "n", "u"] #u: undeterminable


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
def fill_region_att_and_errorcheck(file, categories, region_attributes_clean_df, response_type_yesno):
    for i, attributes in enumerate(file["region_attributes"]):
        attributes_dict = eval(attributes)  # Ensure attributes are parsed correctly
        for key, value in attributes_dict.items():
            if key in categories:
                region_attributes_clean_df.loc[i, key] = value
                # Error check
                if value not in response_type_yesno:
                    region_attributes_clean_df.loc[i, "error"] = key
    return region_attributes_clean_df

def create_list_of_annotated_frames_by_class(region_attributes_metainfo, class_str):
    filenames_annotated = region_attributes_metainfo[region_attributes_metainfo[class_str] != 0.0]
    filenames_annotated = set(filenames_annotated["filename"])
    return filenames_annotated

def frames_bycol(region_attributes_metainfo, col_str):
    col_annotated_frames = region_attributes_metainfo[region_attributes_metainfo[col_str] != 0]
    col_y_frame_list = list(col_annotated_frames[col_annotated_frames[col_str] == "y"]["filename"])
    col_n_frame_list = list(col_annotated_frames[col_annotated_frames[col_str] == "n"]["filename"])
    col_u_frame_list = list(col_annotated_frames[col_annotated_frames[col_str] == "u"]["filename"])
    col_all_frame_set = list(col_annotated_frames["filename"])
    return({"y_frame_list": col_y_frame_list,
            "n_frame_list": col_n_frame_list,
            "u_frame_list": col_u_frame_list,
            "all_frame_set": col_all_frame_set})

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
folder_with_annotations = "/Volumes/T5 EVO/Foraging HD/Isolated_frames/annotated_frames" #/Users/hyewonjun/UZH/Thesis/annotations
filetype = ".csv"
categories = sorted(["pecking", "shade", "blue"])
colnames = ["filename", "batch", "error", "region_count"] + categories
response_type_yesno = ["y", "n", "u"] #u: undeterminable


#%% Initialize summary variables
total_frames = 0
total_annotated_frames = 0
region_attributes_metainfo = pd.DataFrame(columns=colnames)
errors = []



#%% Get dataframe
# Import files
csv_paths = sorted([os.path.join(folder_with_annotations, f) for f in os.listdir(folder_with_annotations) if f.endswith(filetype) and not f.startswith(".")])
print(len(csv_paths))

# Fill region attributes information for all batches
for filename in csv_paths:
    file = pd.read_csv(filename)
    region_attributes_clean_df = make_clean_df(filename, file, colnames)
    region_attributes_clean_df = fill_region_att_and_errorcheck(file, categories, region_attributes_clean_df, response_type_yesno)
    region_attributes_metainfo = pd.concat([region_attributes_metainfo, region_attributes_clean_df], ignore_index=True)


#%% pecking
pecking_frames_lists = frames_bycol(region_attributes_clean_df, "pecking")
pecking_y_frames = pecking_frames_lists["y_frame_list"]
pecking_n_frames = pecking_frames_lists["n_frame_list"]
pecking_u_frames = pecking_frames_lists["u_frame_list"]
pecking_all_frames = pecking_frames_lists["all_frame_set"]

pecking_y_frame_set = set(pecking_y_frames)
pecking_n_frame_set = set(pecking_n_frames)
pecking_u_frame_set = set(pecking_u_frames)
pecking_all_frame_set = set(pecking_all_frames)


#%% shade
shade_frames_lists = frames_bycol(region_attributes_clean_df, "shade")
shade_y_frames = shade_frames_lists["y_frame_list"]
shade_n_frames = shade_frames_lists["n_frame_list"]
shade_u_frames = shade_frames_lists["u_frame_list"]
shade_all_frames = shade_frames_lists["all_frame_set"]

shade_y_frame_set = set(shade_y_frames)
shade_n_frame_set = set(shade_n_frames)
shade_u_frame_set = set(shade_u_frames)
shade_all_frame_set = set(shade_all_frames)



#%% blue
blue_frames_lists = frames_bycol(region_attributes_clean_df, "blue")
blue_y_frames = blue_frames_lists["y_frame_list"]
blue_n_frames = blue_frames_lists["n_frame_list"]
blue_u_frames = blue_frames_lists["u_frame_list"]
blue_all_frames = blue_frames_lists["all_frame_set"]

blue_y_frame_set = set(blue_y_frames)
blue_n_frame_set = set(blue_n_frames)
blue_u_frame_set = set(blue_u_frames)
blue_all_frame_set = set(blue_all_frames)
#%% Get list of annotated frames
# Full list of annotated frames
filenames_annotated = region_attributes_metainfo[region_attributes_metainfo["region_count"] > 0]
filenames_annotated = set(filenames_annotated["filename"])
# Lists of annotated frames per class
annotated_byclass = {}
for class_str in categories:
    annotated_byclass[class_str] = create_list_of_annotated_frames_by_class(region_attributes_metainfo, class_str)
    print(class_str, len(annotated_byclass[class_str]))


# #%% Save annotated filenames as csv.
#
# list_filenames_annotated = sorted((list(filenames_annotated)))
#
# df = pd.DataFrame(list_filenames_annotated, columns=["filename"])
# df.to_csv("annotated_filenames.csv", index=False)
#%% Print frame counts
print(f"Pecking: \nTotal Frames: {len(pecking_all_frame_set)} \nY Frames: {len(pecking_y_frame_set)} \nN Frames: {len(pecking_n_frame_set)} \nU Frames: {len(pecking_u_frame_set)} \nTotal Regions: {len(pecking_all_frames)} \nY Regions: {len(pecking_y_frames)} \nN Regions: {len(pecking_n_frames)} \nU Regions: {len(pecking_u_frames)}\n")

print(f"Shade: \nTotal Frames: {len(shade_all_frame_set)} \nY Frames: {len(shade_y_frame_set)} \nN Frames: {len(shade_n_frame_set)} \nU Frames: {len(shade_u_frame_set)} \nTotal Regions: {len(shade_all_frames)} \nY Regions: {len(shade_y_frames)} \nN Regions: {len(shade_n_frames)} \nU Regions: {len(shade_u_frames)}\n")

print(f"Blue: \nTotal Frames: {len(blue_all_frame_set)} \nY Frames: {len(blue_y_frame_set)} \nN Frames: {len(blue_n_frame_set)} \nU Frames: {len(blue_u_frame_set)} \nTotal Regions: {len(blue_all_frames)} \nY Regions: {len(blue_y_frames)} \nN Regions: {len(blue_n_frames)} \nU Regions: {len(blue_u_frames)}\n")

print("Regions without region attributes:", len(region_attributes_metainfo[
    (region_attributes_metainfo["pecking"] == 0.0) &
    (region_attributes_metainfo["shade"] == 0.0) &
    (region_attributes_metainfo["blue"] == 0.0)
    ])
      )

print("Frames without region attributes:", len(set(region_attributes_metainfo[
    (region_attributes_metainfo["pecking"] == 0.0) &
    (region_attributes_metainfo["shade"] == 0.0) &
    (region_attributes_metainfo["blue"] == 0.0)
    ]["filename"]))
      )
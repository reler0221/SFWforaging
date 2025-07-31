import pandas as pd
from dask.dataframe import to_csv
from imutils import paths
import cv2
import json
import os
import shutil
import matplotlib.pyplot as plt
from imutils import paths
import random as rand
import numpy as np
from patsy.user_util import balanced

#%% Set directories and load files
# Load input_csv for desired variable
export_dir = "/Volumes/T5 EVO/Foraging HD/Classification_model_input/01_Pecking"
pecking_df = pd.read_csv("/Volumes/T5 EVO/Foraging HD/Classification_model_input/01_Pecking/pecking_moreval_combined.csv")
shade_df = pd.read_csv("/Volumes/T5 EVO/Foraging HD/Classification_model_input/03_Shade/shade_moreval.csv")

print(shade_df)

#%% function
def get_maximum_num_in_training(df: pd.DataFrame) -> tuple:
    counts = df.groupby(["type_num", "val_or_train"]).size().unstack(fill_value=0)

    maximum_type, maximum_num = counts["train"].idxmax(), counts["train"].max()

    return int(maximum_type), int(maximum_num)

def get_minimum_num_in_validation(df: pd.DataFrame) -> tuple:
    counts = df.groupby(["type_num", "val_or_train"]).size().unstack(fill_value=0)
    minimum_type, minimum_num = counts["val"].idxmin(), counts["val"].min()

    return int(minimum_type), int(minimum_num)


def sample_frames(frames_indices: list, sample_size: int) -> list:
    indices = rand.sample(frames_indices, sample_size)
    return indices


#%% set parameters
df = shade_df
rand.seed(0)

#%% check frequecies
print(df.groupby(
    ["type_num", "val_or_train"]
).size().unstack(fill_value=0))
#%% duplicate the frames in the training set to equalize the amount of frames in all classes
types = list(df["type_num"].unique())

training_df = df[df["val_or_train"]=="train"]

training_maximum_type, training_maximum_num = get_maximum_num_in_training(df)
skipped_type_list_training = [type_num for type_num in types if type_num != training_maximum_type]


for class_type in skipped_type_list_training:
    frames_subset = training_df[training_df["type_num"] == class_type]
    num_of_frames_to_copy = training_maximum_num - len(frames_subset)
    print(num_of_frames_to_copy)
    frames_subset_indices = list(frames_subset.index)
    indices_of_frames_to_duplicate = sample_frames(frames_subset_indices, num_of_frames_to_copy)
    frames_to_duplicate = frames_subset.loc[indices_of_frames_to_duplicate].copy()
    training_df = pd.concat([training_df, frames_to_duplicate], ignore_index=True)

print(training_df.groupby(
    ["type_num", "val_or_train"]
).size().unstack(fill_value=0))

#%% remove the frames in the validation set to equalize the amount of frames in all classes
validation_df = df[df["val_or_train"]=="val"]

validation_minimum_type, validation_minimum_num = get_minimum_num_in_validation(df)
skipped_type_list_validation = [type_num for type_num in types if type_num != validation_minimum_type]

for class_type in skipped_type_list_validation:
    frames_subset = validation_df[validation_df["type_num"] == class_type]
    num_of_frames_to_remove = len(frames_subset) - validation_minimum_num
    print(num_of_frames_to_remove)

    frames_subset_indices = list(frames_subset.index)
    indices_of_frames_to_remove = sample_frames(frames_subset_indices, num_of_frames_to_remove)
    validation_df = validation_df.drop(indices_of_frames_to_remove)

print(validation_df.groupby(
    ["type_num", "val_or_train"]
).size().unstack(fill_value=0))

#%% concat the new training and validation df to make a new balanced dataset
balanced_df = pd.concat([training_df, validation_df], ignore_index=True)
print(balanced_df.groupby(
    ["type_num", "val_or_train"]
).size().unstack(fill_value=0))

#%% Export the balanced dataframe as csv
balanced_df.to_csv(os.path.join(export_dir, "balanced_shade_moreval.csv"))









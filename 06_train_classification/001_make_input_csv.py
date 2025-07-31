import pandas as pd
from dask.dataframe import to_csv
from imutils import paths
import cv2
import json
import os

import shutil
import matplotlib.pyplot as plt
from imutils import paths
import random
import numpy as np

#%% Set directories and load files
# Load all_annotations
df_annotations = pd.read_csv("/Volumes/T5 EVO/Foraging HD/Isolated_frames/revised_3425.csv")
print(len(df_annotations))


# Load list of validation videos
validation_videos = pd.read_csv("/Volumes/T5 EVO/Foraging HD/Isolated_frames/Validation_videos_list_5.csv", header=0)
validation_videos = [x for x in validation_videos["Val_videos"]]

# Set images directory
import_images_dir="/Volumes/T5 EVO/Foraging HD/Isolated_frames/cropped_frames"
#%%
# Set export directory
export_dir="/Volumes/T5 EVO/Foraging HD/Classification_model_input/03_Shade"




#%% 01. Format and subset dataframe

#%% select only annotated images i.e. the ones with something written the region shape

df_annotations=df_annotations[df_annotations['region_attributes'].str.len()> 35]
print(len(df_annotations))


#%% add classes as separate columns
annotations_list = df_annotations["region_attributes"].tolist()
# Initialize empty lists
pecking, shade, blue = [], [], []
# Process each JSON string
for item in annotations_list:
    parsed = json.loads(item)  # Convert string to dictionary


    # Append values, using default 'None' if key is missing
    pecking.append(parsed.get("pecking", None))
    shade.append(parsed.get("shade", None))
    blue.append(parsed.get("blue", None))

df_annotations["pecking"]=pecking
df_annotations["shade"]=shade
df_annotations["blue"]=blue

#%% mark validation or training
filenames = df_annotations["#filename"].str.slice(stop = -4).tolist()
val_or_train = [None]*len(df_annotations)
for i, img_name in enumerate(filenames):
    img_source_video = img_name[:-5]
    if img_source_video in validation_videos:
        val_or_train[i] = "val"
    else:
        val_or_train[i] = "train"

print(val_or_train.count("val"))
print(val_or_train.count("train"))

df_annotations["val_or_train"]=val_or_train

#%% 02. Check the file formats of the cropped images
#%% list all cropped frames in image directory
images_list = os.listdir(import_images_dir)
print(len(images_list) == len(df_annotations))


#%% remove extra images from the image list
# Images without "cropped" are extra frames. Exclude them
images_list = [image for image in images_list if "cropped" in os.path.basename(image)]
# images_list_split = list(map(lambda x: x.split("cropped", 1)[0], images_list))
# images_list.sort()

#%% check if all filenames have the same formats -- OK
filename_len = set(map(lambda x: len(x), filenames))
#filename_split_len = set(map(lambda x: len(x), images_list_split))
print(filename_len)

#%% OK
print([file for file in images_list if len(file)==41]) # with "training"
print([file for file in images_list if len(file)==34]) # groupname 1 digit
print([file for file in images_list if len(file)==35]) # groupname 2 digits

#%% put into a df
def make_dataframe_for_classification(type_name: str, main_df: pd.DataFrame, img_dir: str, img_list: list):
    possible_types = sorted(main_df[type_name].unique().tolist())
    images_type_num = [possible_types.index(x) for x in main_df[type_name]]

    images_df = pd.DataFrame(
        {"filename": img_list,
         "images_paths": [os.path.join(img, img_dir) for img in img_list],
         "type": main_df[type_name],
         "type_num": images_type_num,
         "val_or_train": main_df["val_or_train"]
         }
    )

    return(images_df)

#%%
# pecking <- video list 4
pecking_df = make_dataframe_for_classification("pecking", df_annotations, import_images_dir, images_list)


#%%
# # blue <- video list 3
# blue_df = make_dataframe_for_classification("blue", df_annotations, import_images_dir, images_list)

#%%
# # shade <- video list 5
shade_df = make_dataframe_for_classification("shade", df_annotations, import_images_dir, images_list)
# replace "u"(unable to determine due to cloud coverage) as "y"
shade_df = shade_df.replace(2, 1)
shade_df = shade_df.replace("u", "y")

#%%
# pecking_df.to_csv(
#     os.path.join(export_dir, f"pecking_moreval.csv"),
#     index=False
# )


shade_df.to_csv(
    os.path.join(export_dir, f"shade_moreval.csv"),
    index=False
)
#%% check balance
counts = shade_df.groupby(
    ["type_num", "val_or_train"]
).size().unstack(fill_value=0)
print(counts)

#%%










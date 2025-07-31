import pandas as pd
from imutils import paths
import cv2
import json
import os
import shutil
import matplotlib.pyplot as plt
from imutils import paths
import random
import numpy as np

#%% Load import files
# Load all_annotations
df_annotations = pd.read_csv("/Volumes/T5 EVO/")
print(len(df_annotations))


# Load list of validation videos
validation_videos = pd.read_csv("/Volumes/T5 EVO/Foraging HD/Isolated_frames/Validation_videos_list.csv", header=0)
validation_videos = [x for x in validation_videos["Val_videos"]]

#%% select only annotated images i.e. the ones with something written the region shape

df_annotations=df_annotations[df_annotations['region_shape_attributes'].str.len()> 20]

print(len(df_annotations))

#%% check that there are no wierd values on the annotations
print("Number of characters")
print(list(set(df_annotations['region_shape_attributes'].str.len().tolist())))
# [52, 53, 54, 55, 56, 57]
# change the lengths below if the results above are different
print(df_annotations[df_annotations['region_shape_attributes'].str.len()==52]["region_shape_attributes"].iloc[0])
print(df_annotations[df_annotations['region_shape_attributes'].str.len()==53]["region_shape_attributes"].iloc[0])
print(df_annotations[df_annotations['region_shape_attributes'].str.len()==54]["region_shape_attributes"].iloc[0])
print(df_annotations[df_annotations['region_shape_attributes'].str.len()==55]["region_shape_attributes"].iloc[0])
print(df_annotations[df_annotations['region_shape_attributes'].str.len()==56]["region_shape_attributes"].iloc[0])
print(df_annotations[df_annotations['region_shape_attributes'].str.len()==57]["region_shape_attributes"].iloc[0])
#seems all ok

#%% separate the annotation df into training and validation
slice_videoname = df_annotations['#filename'].str[:-9]
annotations_validation = df_annotations[slice_videoname.isin(validation_videos)]

#print(annotations_validation)

print("validation images: ", len(annotations_validation)) # looks good

annotations_training = df_annotations[~df_annotations['#filename'].isin(annotations_validation["#filename"])]
#print(annotations_training)
print("training images: ", len(annotations_training)) # looks good



#%% directories to export
import_images_dir="/Volumes/T5 EVO/Foraging HD/Isolated_frames/annotated_frames"
background_images_dir="/Volumes/T5 EVO/Foraging HD/Isolated_frames/background_frames"
export_dir="/Volumes/T5 EVO/Foraging HD/YOLO_model_input/06_Birds_background_ver3"
# define subdirectory if it doesn't exist
os.makedirs(os.path.join(export_dir + "/labels/val"), exist_ok=True)
os.makedirs(os.path.join(export_dir + "/images/val"), exist_ok=True)
os.makedirs(os.path.join(export_dir + "/labels/train"), exist_ok=True)
os.makedirs(os.path.join(export_dir + "/images/train"), exist_ok=True)
#%% list all images uniquely
images_validation= list(set(annotations_validation["#filename"]))
images_training = list(set(annotations_training["#filename"]))

images_background = [f for f in os.listdir(background_images_dir) if f.endswith(".jpg") and not f.startswith(".")]
#%% define a function that converts and distributes the annotations/images into a yolo format
def save_as_yolo(val_or_train: str, images: list, annotations: pd.DataFrame, images_dir: str, export_dir: str):
    if val_or_train != "val" and val_or_train != "train":
        print("Check parameter: val_or_train")
        return False

    # loop through all rows in dataframe
    for i in range(0, len(images)):
        # load image to get the dimensions for the normalization
        image_path = images_dir + "/" + images[i]
        image = cv2.imread(image_path)
        img_height, img_width, _ = image.shape
        if not annotations.empty:
            print("Annotation image: ", image_path )# get bounding box information if annotations df is not empty
            annotations_i = annotations[annotations["#filename"] == images[i]]
            for row in range(0, len(annotations_i)):
                # get the bounding boxes
                bbox_data = json.loads(annotations_i['region_shape_attributes'].iloc[row])

                if bbox_data["name"] != "rect": # sanity check
                    print("check annotation: ", annotations_i["#filename"][0] )
                    continue

                class_id = 0  # class_label(annotations_i.iloc[row], len(class_categories))
                x, y, width, height = bbox_data["x"], bbox_data["y"], bbox_data["width"], bbox_data["height"]

                x_center = (x + width / 2) / img_width
                y_center = (y + height / 2) / img_height
                width_norm = width / img_width
                height_norm = height / img_height

                # format label as yolo bbox label
                label_content: str = f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n"
                label_filename = os.path.splitext(annotations_i['#filename'].iloc[row])[0] + ".txt"
                label_path = os.path.join(export_dir + f"/labels/{val_or_train}/", label_filename)
                with open(label_path, "a") as f:
                    f.write(label_content)

                # copy the respective image
                shutil.copyfile(image_path, label_path.replace("labels", "images").replace("txt", "jpg"))

        else: # for background images (no annotations)
            print("Background image: ", image_path)
            label_path = os.path.join(export_dir + f"/labels/{val_or_train}/", images[i].replace("jpg","txt"))
            open(label_path, "a").close()

            # copy the respective image
            shutil.copyfile(image_path, label_path.replace("labels", "images").replace("txt", "jpg"))

#%% prepare annotated frames:
#%% save validation images/labels
save_as_yolo("val", images_validation, annotations_validation, import_images_dir, export_dir)
print("Done")
#%% save training images/labels
save_as_yolo("train", images_training, annotations_training, import_images_dir, export_dir)
print("Done")
#%% prepare background frames:
empty_df = pd.DataFrame()
random.seed(0)
sample_size = int(len(images_background)*0.1)
images_background_val = random.sample(images_background, sample_size)
images_background_training = list(set(images_background)-set(images_background_val))
#%% background validation
save_as_yolo("val", images_background_val, empty_df, background_images_dir, export_dir)
#%% background training
save_as_yolo("train", images_background_training, empty_df, background_images_dir, export_dir)








import os
import cv2
import pandas as pd


#%% Load import files
# Load all_annotations
df_annotations = pd.read_csv("/Volumes/T5 EVO/Foraging HD/Isolated_frames/revised_3425.csv")
print(len(df_annotations))


# Load list of validation videos
validation_videos = pd.read_csv("/Volumes/T5 EVO/Foraging HD/Isolated_frames/Validation_videos_list.csv", header=0)
validation_videos = [x for x in validation_videos["Val_videos"]]


#%% select only annotated images with all variables labeled

df_annotations=df_annotations[df_annotations['region_attributes'].str.len()> 36]

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

#%%
# yolo needs images and labels on different folders and labels on different files for each
# first select 10% pictures for validation
import json

print("images:")
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
#%%
df_annotations["pecking"]=pecking
df_annotations["shade"]=shade
df_annotations["blue"]=blue
print(df_annotations)

#%%
#%% separate the annotation df into training and validation
slice_videoname = df_annotations['#filename'].str[:-9]
annotations_validation = df_annotations[slice_videoname.isin(validation_videos)]

print("validation images: ", len(annotations_validation)) # looks good

annotations_training = df_annotations[~df_annotations['#filename'].isin(annotations_validation['#filename'])]
#print(annotations_training)
print("training images: ", len(annotations_training)) # looks good

#%% check the ratio of each variables in training and validation sets.
print(annotations_training["pecking"].value_counts())
print(annotations_validation["pecking"].value_counts())

print(annotations_training["blue"].value_counts())
print(annotations_validation["blue"].value_counts())

print(annotations_training["shade"].value_counts())
print(annotations_validation["shade"].value_counts()) # This should be redistributed

#%%directories of the images and to where to export
images_dir="/Volumes/T5 EVO/Foraging HD/Isolated_frames/annotated_frames"
export_dir="/Volumes/T5 EVO/Foraging HD/Isolated_frames/cropped_frames"

#%%
def crop_image(image_name: str, image_dir: str, export_dir: str):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    annotations_i = df_annotations[df_annotations["#filename"]==image_name]

    for row in range(len(annotations_i)):
        row = row
        bbox_data = json.loads(annotations_i['region_shape_attributes'].iloc[row])
        x = int(bbox_data["x"])
        y = int(bbox_data["y"])
        width = int(bbox_data["width"])
        height = int(bbox_data["height"])


        crop = image[y:y+height, x:x+width]
        crop_name = os.path.splitext(image_name)[0]+f"cropped_{row}.jpg"
        crop_path = os.path.join(export_dir, crop_name)
        cv2.imwrite(crop_path, crop)
#%% crop images
for filename in df_annotations['#filename']:
    crop_image(filename, images_dir, export_dir)

#%% check
print(len(os.listdir(export_dir)))

#%% save annotated df
df_annotations.to_csv("/Volumes/T5 EVO/Foraging HD/Isolated_frames/full_annotations.csv", index=False)

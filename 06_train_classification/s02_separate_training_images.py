import pandas as pd
import os
import shutil

#%%
outdir = "/Volumes/T5 EVO/Foraging HD/Isolated_frames/shade"
#%%
image_df = pd.read_csv("06_train_classification/shade.csv")
print(image_df.dtypes)

image_df["full_path"] = image_df.apply(lambda x: os.path.join(x["images_paths"], x["filename"]),axis=1)
path_label = image_df[["full_path", "type"]]

#%%
unique_labels = path_label["type"].unique()
for label in unique_labels:
    os.makedirs(os.path.join(outdir, label), exist_ok=True)


#%%

for row in path_label.itertuples():
    label = row.type
    src_path = row.full_path
    filename = os.path.basename(src_path)
    dst_path = os.path.join(outdir, label, filename)
    shutil.copy(src_path, dst_path)




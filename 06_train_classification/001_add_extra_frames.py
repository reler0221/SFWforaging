import pandas as pd
import os

#%%
y_dir = "/Volumes/T5 EVO/Foraging HD/Isolated_frames/cropped_frames_extra/pecking"
n_dir = "/Volumes/T5 EVO/Foraging HD/Isolated_frames/cropped_frames_extra/not_pecking"
all_crops_dir = "/Volumes/T5 EVO/Foraging HD/Isolated_frames/cropped_frames"

main_df = pd.read_csv("/Volumes/T5 EVO/Foraging HD/Classification_model_input/01_Pecking/pecking.csv")
print("Existing dataframe loaded:")
print(main_df.shape)
print(main_df.dtypes)
print(main_df.head())
print(main_df["type"].unique())
print(main_df["type_num"].unique())

#%%
y_filenames = os.listdir(y_dir)
n_filenames = os.listdir(n_dir)

filenames = y_filenames + n_filenames
filepaths = [all_crops_dir]*len(filenames)#[os.path.join(all_crops_dir, f) for f in filenames]
types = ["y"]*len(os.listdir(y_dir)) + ["n"]*len(os.listdir(n_dir))
type_nums = [1]*len(os.listdir(y_dir)) + [0]*len(os.listdir(n_dir))
val_or_train = ["train"]*len(filenames)

print(len(filenames))
print(len(types))

extra_df = pd.DataFrame(
    {"filename": filenames,
     "images_path": filepaths,
     "type": types,
     "type_num": type_nums,
     "val_or_train": val_or_train
     }
)

#%% make sure all images in annotation is added in the combined directory

not_in_dir = [f for f in filepaths if os.path.exists(f) == False]
print(not_in_dir)

#%%
extra_df.to_csv("/Volumes/T5 EVO/Foraging HD/Classification_model_input/01_Pecking/extra.csv", index=False)









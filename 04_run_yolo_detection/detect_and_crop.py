#from ultralytics import yolo
import os
import cv2

#Load model

# model = YOLO("/Volumes/T5 EVO/Foraging HD/YOLO_model_output/Runs/train05/weights/best.pt")

#%%Set directories
input_images_dir = "/Volumes/T5 EVO/Foraging HD/Test_set/Test_frames"
export_dir = "/Volumes/T5 EVO/Foraging HD/Test_set/Test_frames"

#Get images path
images = [f for f in os.listdir(input_images_dir) if f.endswith(".jpg") and not f.startswith(".")]
paths = list(map(lambda x: os.path.join(input_images_dir, x), images))

#Choose fewer images
paths = paths[0]

#%% Run predictions
for image_path in paths:
    image = cv2.imread(image_path)
    results = model(image)
    print(results)








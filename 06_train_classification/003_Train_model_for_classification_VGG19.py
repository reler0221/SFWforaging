#import modules
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
#os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
#ImageFile.LOAD_TRUNCATED_IMAGES = True

#%%
tf.config.list_physical_devices('GPU')
#%% set directories and load files
images_dir="/Volumes/T5 EVO/Foraging HD/Isolated_frames/cropped_frames"
export_dir = "/Volumes/T5 EVO/Foraging HD/VGG_model_output"
miscalculated_images_export_dir = "/Volumes/T5 EVO/Foraging HD/VGG_model_output/01_Pecking_misclassified"
pecking_df = pd.read_csv("/Volumes/T5 EVO/Foraging HD/Classification_model_input/01_Pecking/balanced_pecking.csv")

pecking_export_path = os.path.join(export_dir, "pecking_classification.h5")
#%% check data
print(pecking_df)
print(pecking_df.groupby(
    ["type_num", "val_or_train"]
).size().unstack(fill_value=0))

pecking_types = list(pecking_df["type"].unique())
#%% set parameters
# which dataframe to use - should be generated from 002_balance_data.py
df = pecking_df
export_path = pecking_export_path
types = list(map(lambda x: "pecking_"+x, pecking_types))

# image generator parameters
gen_rescale = 1./255
gen_x_col = "full_path"
gen_y_col = "type_num"
gen_target_size = (224, 224)
gen_batch = 4
# gen_rotation_range =
# gen_zoom_range =
gen_class_mode="sparse"

# model parameters
output_categories_num = 1 # 1 if binary. len(xx_types) for more
activation_function = "sigmoid" # signoid if binary, softmax for more

# model export settings
cp_monitor='val_loss'
cp_mode='min'
earlystop_epochs=5

# model compile settings
compile_loss = "binary_crossentropy" # binary crossentrophy or sparse_categorical_crossentropy

# model fit settings
epochs = 50

#%% add "full_path" column in the dataframe
df["full_path"] = df["filename"].apply(lambda x: os.path.join(images_dir, x))
print(df)
#%% prepare images for training
training_df = df[df["val_or_train"] == "train"]
train_data = ImageDataGenerator(rescale = gen_rescale)
train_generator_data=train_data.flow_from_dataframe(training_df,
                                x_col= gen_x_col,
                                y_col= gen_y_col,
                                target_size = gen_target_size,
                                #rotation_range=40,
                                #zoom_range=0.2,
                                batch_size= gen_batch,
                                shuffle=True,
                                class_mode=gen_class_mode)

#%% prepare images for validation
validation_df = df[df["val_or_train"] == "val"]
validation_data = ImageDataGenerator(rescale = gen_rescale)
validation_generator_data= validation_data.flow_from_dataframe(validation_df,
                                x_col= gen_x_col,
                                y_col= gen_y_col,
                                target_size = gen_target_size,
                                #rotation_range=40,
                                #zoom_range=0.2,
                                batch_size= gen_batch,
                                shuffle=False,
                                class_mode=gen_class_mode)


#%%
## Import vgg pretrained model on imagenet dataset
vgg19 = VGG19(input_shape=(224,224,3), weights='imagenet', include_top=False)
## Take the output of the last convolutional layer
x = vgg19.layers[-1].output
## Flatten into 1D (it is 3D originally)
x = Flatten()(x)
## x = Dropout(0.5)(x) # when few images (<1000)
## Add a Dence layer (number of neurons, activation function)
x = Dense(256, activation='relu')(x)
#x = Dropout(0.5)(x)
## Final layer
predictors = Dense(output_categories_num, activation = activation_function)(x)
model = Model(inputs=vgg19.input, outputs=predictors)

#%%
model.summary()

#%% save model - only if there was an improvement in the model comparing
checkpoint = ModelCheckpoint(export_path, monitor=cp_monitor, verbose=1, save_best_only=True, mode=cp_mode)
# stop training if there is no improvement in model for n consecutives epochs.
early_stopping_monitor = EarlyStopping(patience=earlystop_epochs)
callbacks_list = [checkpoint, early_stopping_monitor]

#%% compile the model
model.compile(loss= compile_loss,
              optimizer=Adam(learning_rate=1e-5), #learning rate usually -5 works the best for pre trained models. but -6 also sometimes
              metrics=['accuracy']) # Accuracy = (Number of correct predictions) ÷ (Total number of predictions)

#%%
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

history = model.fit(train_generator_data,
          steps_per_epoch=len(train_generator_data),
          epochs= epochs,
          validation_data=validation_generator_data,
          validation_steps=len(validation_generator_data),
          callbacks=callbacks_list)

#%% plot model results
#training vs validation loss plot
training_loss = history.history['loss'] #extract loss values at each epoch
validation_loss = history.history['val_loss']#extract accuracy values at each epoch

actual_epochs = range(1, len(training_loss) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_loss, label='Training Loss', color="green")
plt.plot(epochs, validation_loss, label='Validation Loss', color="blue")
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#%% training vs validation accuracy plot
training_acc = history.history['acc']
validation_acc = history.history['val_acc']

epochs = range(1, len(training_acc) + 1)
plt.figure(figsize=(10, 6))  # You can adjust the width (10) and height (6) as needed
plt.plot(epochs, training_acc, label='Training accuracy', color="green")
plt.plot(epochs, validation_acc, label='Validation accuracy', color="blue")
plt.ylim(0,1)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%% confusion matrix
#get the predicted labels and the true labels of the validation dataset
predicted_label=[]
true_label=[]
for step in range(len(validation_generator_data)):
    x, y = next(validation_generator_data) #generate image data and true label from the validation dataset
    results=model.predict(x, verbose=0) #predict the labels
    for i in range(0,len(results)):
      predicted_label.append(np.argmax(results[i])) #store predicted label
      true_label.append(int(y[i]))#store true labels

# Compute the confusion matrix
confusion = confusion_matrix(true_label, predicted_label)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels= types,
            yticklabels= types
            )
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#%% find misclassified images
results = model.predict(validation_generator_data, verbose=1)
predicted_classes = np.argmax(results, axis=1)
true_classes = validation_generator_data.classes
misclassified_indices = np.where(predicted_classes != true_classes)[0]
true_classes = np.array(validation_generator_data.classes)
misclassified_indices = np.array(misclassified_indices)

filenames = validation_generator_data.filenames
misclassified_images = [filenames[i] for i in misclassified_indices]
misclassified_df = pd.DataFrame({
    'image': misclassified_images,
    'true_label': true_classes[misclassified_indices],
    'predicted_label': predicted_classes[misclassified_indices]
})


print(misclassified_df.head(10000).to_string(index=False))

for img_path in misclassified_images:
    src_path = os.path.join(images_dir, img_path)  # corrected source path
    dst_path = os.path.join(miscalculated_images_export_dir, os.path.basename(img_path))  # corrected destination path
    shutil.copy(src_path, dst_path)

print(f"Images copied to : {miscalculated_images_export_dir}")
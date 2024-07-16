import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

# Parameters
image_height, image_width = 80, 80
num_classes = 3
batch_size = 32
current_path = "Kaggle/"

train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')

# Loading images and labels based on the csv file and the folder
def load_data(df, folder):
    images = []
    labels = []
    for i, row in df.iterrows():
        img_path = os.path.join(folder, row[0])+".png"
        image = load_img(img_path, target_size=(image_height, image_width)) #resizing just in case
        image = img_to_array(image) / 255.0 #normalizing
        images.append(image)
        labels.append(row[1])
    return np.array(images), np.array(labels)

X_train, y_train = load_data(train_df, 'train')
X_val, y_val = load_data(val_df, 'validation')

# Convert labels to categorical (one-hot encoded vectors)
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

#To train with the whole data
X_train = np.concatenate((X_train, X_val))
y_train = np.concatenate((y_train, y_val))


#Augumentation
datagen = ImageDataGenerator(
    rotation_range=20,  #randomly rotate images up to 20 degrees (counter)clockwise
    width_shift_range=0.2,  #because not all of the content of images is centered, we do random shifts up to 0.2 * width
    height_shift_range=0.2, #equivalent to width_shift_range on the vertical axis
    shear_range=0.2,  #add distorsion along axis
    zoom_range=0.2,
    horizontal_flip=True, #reverse the image as in a mirror
    fill_mode='nearest'  #it will fill missing pixels caused by augumentation with the nearest pixels
)


# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='leaky_relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='leaky_relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='leaky_relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='leaky_relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#the loss function is used with one-hot encoded labels and compares the distribution of 
# predicted labels with the distribution of actual labels


# Print the model summary
model.summary()

# Timing callback to measure training time per epoch
class TimingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        print(f'Epoch {epoch + 1} duration: {epoch_duration:.2f} seconds')

timing_callback = TimingCallback()

# Train the model with timing
start_time = time.time()
#Stops training after 7 epochs without improvement. 
# Restores model weights from the epoch with the best value of the val_loss.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#Reduces the learning rate by a factor of 0.2 after 5 epochs without improvement.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
#Fits the model using datagen.flow to apply data augmentation to training data. It trains the model for 50 epochs.
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr, timing_callback])
end_time = time.time()


# Calculate the total training time
total_time = end_time - start_time
print(f'Total training time: {total_time / 60:.2f} minutes')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_val, y_val)
print(f'Test accuracy: {test_accuracy}')

# Make predictions
predictions = model.predict(X_val)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_val, axis=1)

# Print classification report
print(classification_report(true_classes, predicted_classes))



#Generate submission file based on sample
test_df = pd.read_csv("sample_submission.csv")
X_test, _ = load_data(test_df, 'test')
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
test_df['label'] = predicted_classes
#save as
test_df.to_csv('first_try_submission.csv', index=False)

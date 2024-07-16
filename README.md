# Convolutional-Neural-Network-to-classify-images
A Convolutional Neural Network which is trained using supervised learning on images and classifies them in three classes. It uses Keras, Pandas and other libraries.

Images were converted to numpy.array and normalized with values between [0,1].
Applied to_categorical to labels, which transforms them into one-hot encoded vectors.

## Data augumentation:
It helps the model generalize by making small changes to images like rotations, zoom, flips, shifts.

```python

datagen = ImageDataGenerator(
rotation_range=20,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest'
)

```

- rotation_range = randomly rotate images up to 20 degrees (counter)clockwise
- width_shift_range = because not all of the content of images is centered, we do random shifts up to
- 0.2 * width
- height_shift_range = equivalent to width_shift_range on the vertical axis
- shear_range = add distorsion along axis
- zoom_range = apply zoom to image
- horizontal_flip = reverse the image as in a mirror
- fill_mode = it will fill missing pixels caused by augumentation with the nearest pixels

## Training
- early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
  - Stops training after 10 epochs without improvement. Restores model weights from the epoch with the best value of the val_loss.
- reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
  - Reduces the learning rate by a factor of 0.2 after 5 epochs without improvement.
- history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr, timing_callback])
  - Fits the model using datagen.flow to apply data augmentation to training data. It trains the second CNN model for 50 epochs.

```python

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

```

The model uses Convolutional layers to learn step by step diverse edges, textures and, further on,
more complex patterns while MaxPooling in between Convolutional Layers, which helps prevent
overfitting and reduce computational cost.

Then the Flatten layers converts the features into a 1D vector, preparing the data for the Dense
layer. The Dense layer learns high-level representations and combinations of the extracted features.
The Dropout layer “drops” 50% of the input units during training to prevent overfitting.
The model uses an increasing number of filters (32, 64, 128) in successive layers, balancing
complexity and computational efficiency.

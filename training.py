import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Set your data directories
train_data_dir = 'train' #training sets
validation_data_dir = 'validation' #validation sets
batch_size = 32
epochs = 20

# Preprocess Your Data
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255) 

train_generator = train_datagen.flow_from_directory(     
    train_data_dir,
    target_size=(48, 48), #img size
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(48, 48),#img size
    batch_size=batch_size,
    class_mode='categorical'
)

# Build a Convolutional Neural Network (CNN) 
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))  

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator  # Pass the validation generator here
) 

#  Save the Model
model.save('test.h5')






























































""" Type: Convolutional layer (Conv2D)
Number of Filters (Kernels): 32
Filter (Kernel) Size: (3, 3)
Activation Function: ReLU (Rectified Linear Unit)
Input Shape: (48, 48, 3)
Explanation: This layer applies 32 convolutional filters"""
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model

# Set the path to your dataset
train_dataset_path = 'D:/capstone_project/skin_cancer/test_Copy'
test_dataset_path = 'D:/capstone_project/skin_cancer/test_Copy'

# Constants
img_height, img_width = 224, 224
batch_size = 32

# Create data generator with data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create data generator for testing (no data augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load VGG16 model with pre-trained weights (excluding top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model with VGG16 and additional layers
model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(7, activation='softmax'))  # Adjust based on the number of classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important for keeping track of filenames
)

# Train the model
model.fit(train_generator, epochs=50, validation_data=test_generator)

# Evaluate the model on the test set
# evaluation = model.evaluate(test_generator)
# print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")
    
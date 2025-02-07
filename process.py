import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Define dataset path
dataset_path = 'C:\\Users\\ASLAM\\Desktop\\ISL\\dataset\\original_images'

# Create ImageDataGenerator for data augmentation and rescaling
datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values
    rotation_range=30,       # Randomly rotate images by 30 degrees
    width_shift_range=0.2,   # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,         # Apply shearing transformation
    zoom_range=0.2,          # Randomly zoom images
    horizontal_flip=True,    # Randomly flip images horizontally
    fill_mode='nearest',     # Fill missing pixels after transformations
    validation_split=0.2     # Reserve 20% of the data for validation
)

# Create generators for training and validation data
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(250, 250),  # Resize images to 250x250
    batch_size=32,           # Set the batch size
    class_mode='categorical', # Use categorical labels
    subset='training'        # Use training subset
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(250, 250),  # Resize images to 250x250
    batch_size=32,           # Set the batch size
    class_mode='categorical', # Use categorical labels
    subset='validation'      # Use validation subset
)

# Get number of classes
num_classes = len(train_generator.class_indices)

# Print the number of samples in each set
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Number of classes: {num_classes}")

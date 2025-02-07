import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

# Define dataset path
dataset_path = 'C:\\Users\\ASLAM\\Desktop\\ISL\\dataset\\original_images'

# Create ImageDataGenerator for data augmentation and rescaling
datagen = ImageDataGenerator(
    rescale=1./255,          
    rotation_range=30,      
    width_shift_range=0.2,   
    height_shift_range=0.2,  
    shear_range=0.2,         
    zoom_range=0.2,          
    horizontal_flip=True,    
    fill_mode='nearest',     
    validation_split=0.2     
)

# Create generators for training and validation data
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(250, 250),  
    batch_size=32,           
    class_mode='categorical', 
    subset='training'        
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(250, 250),  
    batch_size=32,           
    class_mode='categorical', 
    subset='validation'      
)

# Get number of classes
num_classes = len(train_generator.class_indices)

# Print the number of samples in each set
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Number of classes: {num_classes}")

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Define dataset path
dataset_path = 'C:\\Users\\ASLAM\\Desktop\\ISL\\dataset\\original_images'

# Create ImageDataGenerator for data augmentation and rescaling
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2  # Reserve 20% of the data for validation
)

# Create generators for training and validation data
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(250, 250),  # Resize images to 250x250
    batch_size=32,  # Set the batch size
    class_mode='categorical',  # Use categorical labels
    subset='training'  # Use training subset
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(250, 250),  # Resize images to 250x250
    batch_size=32,  # Set the batch size
    class_mode='categorical',  # Use categorical labels
    subset='validation'  # Use validation subset
)

# Get the number of classes (categories)
num_classes = len(train_generator.class_indices)

# Load the MobileNetV2 model without the top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(250, 250, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10  # Adjust the number of epochs based on your needs
)

# Save the model
model.save('C:\\Users\\ASLAM\\Desktop\\ISL\\isl_mobilenetv2.h5')

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy}")
print(f"Validation Loss: {loss}")

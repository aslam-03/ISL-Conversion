from tensorflow.keras.models import load_model # type: ignore

# Load your trained model
model = load_model('C:\\Users\\ASLAM\\Desktop\\ISL\\trained_model.h5')
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Path to your manually split test set
test_dataset_path = 'C:\\Users\\ASLAM\\Desktop\\ISL\\dataset\\test_images'

# Create an ImageDataGenerator for rescaling test images
test_datagen = ImageDataGenerator(rescale=1./255)

# Create the test data generator
test_generator = test_datagen.flow_from_directory(
    test_dataset_path,
    target_size=(250, 250),  # Ensure images are resized to match training input size
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important: Do not shuffle the test data
)
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}")

import data_preprocessing
import model as model_script
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Parameters
IMAGE_SIZE = (128, 128)  # Resize images to 128x128 pixels
BATCH_SIZE = 32
DATA_DIR = 'D:/vscode/intern/Hunarpe/divesh/dataset'  # Path to your dataset directory

# Create data generators
train_generator, validation_generator = data_preprocessing.create_generators(
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    data_dir=DATA_DIR
)

# Build the model
model = model_script.build_model(input_shape=(128, 128, 3))  # Assuming RGB images

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=10
)

# Evaluate the model on test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'D:/vscode/intern/Hunarpe/divesh/test_dataset',  # Path to your test dataset directory
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print(f'Test Accuracy: {test_acc:.2f}')

# Make predictions and print classification report
test_generator.reset()
preds = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE, verbose=1)
preds = (preds > 0.5).astype(int).reshape(-1)

y_true = test_generator.classes
print(classification_report(y_true, preds, target_names=['Cat', 'Dog']))

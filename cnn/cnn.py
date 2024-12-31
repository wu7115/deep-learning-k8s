import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

dataset_path = os.getenv('DATASET_PATH', '/app/dataset')

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)
training_set = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'training_set'),
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary'
)

test_datagen = ImageDataGenerator(rescale = 1./255)
testing_set = test_datagen.flow_from_directory(
    os.path.join(dataset_path, 'test_set'),
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary'
)

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = cnn.fit(
    x=training_set,
    validation_data=testing_set,
    epochs=10
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='CNN Training Accuracy')
plt.plot(history.history['val_accuracy'], label='CNN Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('CNN Training vs Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='CNN Training Loss')
plt.plot(history.history['val_loss'], label='CNN Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('CNN Training vs Validation Loss')
plt.legend()

plt.tight_layout()

plot_path = '/app/plots'
os.makedirs(plot_path, exist_ok=True)
plt.savefig(f'{plot_path}/cnn_training_plot.png')  # Save plot inside the container
plt.close()
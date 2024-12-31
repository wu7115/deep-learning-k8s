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

lstm = tf.keras.models.Sequential()
lstm.add(tf.keras.layers.Reshape((64, 64 * 3), input_shape=[64, 64, 3]))
lstm.add(tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True))
lstm.add(tf.keras.layers.Dropout(0.2))
lstm.add(tf.keras.layers.LSTM(units=64, activation='tanh'))
lstm.add(tf.keras.layers.Dropout(0.2))
lstm.add(tf.keras.layers.Dense(units=128, activation='relu'))
lstm.add(tf.keras.layers.Dropout(0.2))
lstm.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = lstm.fit(
    x=training_set,
    validation_data=testing_set,
    epochs=10
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='LSTM Training Accuracy')
plt.plot(history.history['val_accuracy'], label='LSTM Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('LSTM Training vs Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='LSTM Training Loss')
plt.plot(history.history['val_loss'], label='LSTM Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('LSTM Training vs Validation Loss')
plt.legend()

plt.tight_layout()

plot_path = '/app/plots'
os.makedirs(plot_path, exist_ok=True)
plt.savefig(f'{plot_path}/lstm_training_plot.png')
plt.close()
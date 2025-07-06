# Gak pakek fine tuning

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# Path dataset baru
base_dir = "D:/DataMining_Sem6/new_dataset"
train_dir = os.path.join(base_dir, "training")
val_dir = os.path.join(base_dir, "validation")

# Data Augmentation & Generator (lebih kuat)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    brightness_range=[0.6, 1.4],
    channel_shift_range=30.0,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),  # Ukuran optimal untuk MobileNetV2
    batch_size=16,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical'
)

# Simpan label encoder (kelas)
np.save('classes_new4.npy', list(train_generator.class_indices.keys()))

# Transfer Learning dengan MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False  # Freeze base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=7, verbose=1)

print("[INFO] Training model (head only)...")
history = model.fit(
    train_generator,
    epochs=60,
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr]
)

# FINE-TUNING: Unfreeze lebih banyak layer MobileNetV2
base_model.trainable = True
for layer in base_model.layers[:-80]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("[INFO] Fine-tuning model...")
history_finetune = model.fit(
    train_generator,
    epochs=80,
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr]
)

loss, accuracy = model.evaluate(val_generator)
print(f"[RESULT] Test Accuracy: {accuracy * 100:.2f}%")

model.save('saved_model_new4.keras')
print("[INFO] Model saved as 'saved_model_new.keras'")

# Plot
acc = history.history['accuracy'] + history_finetune.history['accuracy']
val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
losses = history.history['loss'] + history_finetune.history['loss']
val_losses = history.history['val_loss'] + history_finetune.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, losses, 'r', label='Training loss')
plt.plot(epochs, val_losses, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

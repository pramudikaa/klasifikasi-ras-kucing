# pakek fine tunning cok

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# Path dataset baru
base_dir = "D:/DataMining_Sem6/new_dataset"
train_dir = os.path.join(base_dir, "training")
val_dir = os.path.join(base_dir, "validation")

# Data Augmentation & Generator
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Simpan label encoder (kelas)
np.save('classes_new1.npy', list(train_generator.class_indices.keys()))

# Transfer Learning dengan MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    optimizer=RMSprop(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=1)

print("[INFO] Training model with transfer learning (head only)...")
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[reduce_lr]
)

# FINE-TUNING: Unfreeze beberapa layer terakhir MobileNetV2
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=RMSprop(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("[INFO] Fine-tuning model...")
history_finetune = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[reduce_lr]
)

loss, accuracy = model.evaluate(val_generator)
print(f"[RESULT] Test Accuracy: {accuracy * 100:.2f}%")

model.save('saved_model_new.keras')
print("[INFO] Model saved as 'saved_model.keras'")

print(model.summary())

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Running model script on port {port}...")
    # Kamu bisa panggil fungsi training di sini kalau mau modular
    # Tapi kalau sudah otomatis jalan dari atas, cukup print saja

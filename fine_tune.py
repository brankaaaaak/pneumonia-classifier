from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from sklearn.utils.class_weight import compute_class_weight
import os, numpy as np, matplotlib.pyplot as plt

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
DATA_DIR = r"C:\Users\User\Desktop\archive\chest_xray_split"
MODEL_PATH = "best_pneumonia_resnet.keras"

def load_data(data_dir, image_size, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1]
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(os.path.join(data_dir, 'train'),
                                                  target_size=image_size,
                                                  batch_size=batch_size,
                                                  class_mode='binary')
    val_gen = test_datagen.flow_from_directory(os.path.join(data_dir, 'val'),
                                               target_size=image_size,
                                               batch_size=batch_size,
                                               class_mode='binary')
    return train_gen, val_gen

def fine_tune_resnet(model_path):
    model = load_model(model_path)

    base_model = None
    for layer in model.layers:
        if isinstance(layer, models.Model):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Base model (ResNet) nije pronađen!")

    for layer in base_model.layers[:-100]:
        layer.trainable = False
    for layer in base_model.layers[-100:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=5e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

train_gen, val_gen = load_data(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
model = fine_tune_resnet(MODEL_PATH)

class_weights = dict(enumerate(
    compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
))

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint('best_pneumonia_resnet_finetuned.keras', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
]

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
                    callbacks=callbacks, class_weight=class_weights)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(), plt.title('Fine-Tuned ResNet50 Accuracy')
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(), plt.title('Fine-Tuned ResNet50 Loss')
plt.show()

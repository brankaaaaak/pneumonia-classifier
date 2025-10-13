import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = r"C:\Users\User\Desktop\archive\chest_xray_split"  # path to dataset

def load_data_t(data_dir, image_size, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    val_gen = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    return train_gen, val_gen, test_gen

def build_resnet_transfer(image_size):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*image_size, 3)
    )
    base_model.trainable = False  

    inputs = layers.Input(shape=(*image_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model_t(model, train_gen, val_gen, epochs):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))
    print("Class Weights:", class_weights)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=16, restore_best_weights=True),
        ModelCheckpoint('best_pneumonia_resnet.keras', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    return history

if __name__ =="__main__":
    train_gen, val_gen, test_gen = load_data_t(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
    model = build_resnet_transfer(IMAGE_SIZE)
    model.summary()

    history = train_model_t(model, train_gen, val_gen, EPOCHS)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs (ResNet50)')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs (ResNet50)')
    plt.legend()

    plt.tight_layout()
    plt.show()
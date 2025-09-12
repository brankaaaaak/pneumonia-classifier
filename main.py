import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = r'C:\Users\User\Desktop\archive\chest_xray' #path to the data

def load_data(data_dir, image_size, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
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

if __name__ == "__main__":
    print("Hello world")
    train_gen, val_gen, test_gen = load_data(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
    print("Train samples:", train_gen.samples)
    print("Validation samples:", val_gen.samples)
    print("Test samples:", test_gen.samples)
    print("Classes:", train_gen.class_indices)

    #visualization
    x_batch, y_batch = next(train_gen)
    print("Batch shape:", x_batch.shape)
    print("Labels shape:", y_batch.shape)

    plt.figure(figsize=(10, 6))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(x_batch[i])
        plt.title("PNEUMONIA" if y_batch[i] == 1 else "NORMAL")
        plt.axis("off")
    plt.show()
import os
import shutil
import random

"""
    additional partitioning which, after division according to the 70/15/15 principle,
    will equalize the training set
    because initially there is a big difference between the NORMAL and PNEUMONIA sets

"""

DATA_DIR = r"C:\Users\User\Desktop\archive\chest_xray"
OUTPUT_DIR = r"C:\Users\User\Desktop\archive\chest_xray_split_2"

CLASSES = ["NORMAL", "PNEUMONIA"]

for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

all_images = {cls: [] for cls in CLASSES}
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, split, cls)
        if not os.path.exists(folder):
            continue
        for img in os.listdir(folder):
            all_images[cls].append(os.path.join(folder, img))

for cls, imgs in all_images.items():
    random.shuffle(imgs)
    total = len(imgs)
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)

    train_imgs = imgs[:train_end]
    val_imgs = imgs[train_end:val_end]
    test_imgs = imgs[val_end:]

    for img in train_imgs:
        shutil.copy(img, os.path.join(OUTPUT_DIR, "train", cls))
    for img in val_imgs:
        shutil.copy(img, os.path.join(OUTPUT_DIR, "val", cls))
    for img in test_imgs:
        shutil.copy(img, os.path.join(OUTPUT_DIR, "test", cls))

print("\nBalancing the number of images in the TRAIN set...")
normal_dir = os.path.join(OUTPUT_DIR, "train", "NORMAL")
pneumonia_dir = os.path.join(OUTPUT_DIR, "train", "PNEUMONIA")

normal_count = len(os.listdir(normal_dir))
pneumonia_count = len(os.listdir(pneumonia_dir))

if normal_count < pneumonia_count:
    normal_files = os.listdir(normal_dir)
    while len(os.listdir(normal_dir)) < pneumonia_count:
        file = random.choice(normal_files)
        new_name = f"aug_{len(os.listdir(normal_dir))}_{file}"
        shutil.copy(os.path.join(normal_dir, file), os.path.join(normal_dir, new_name))

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}:")
    for cls in CLASSES:
        count = len(os.listdir(os.path.join(OUTPUT_DIR, split, cls)))
        print(f"  {cls}: {count} images")

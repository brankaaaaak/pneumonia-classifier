import os
import shutil
import random

# folders
DATA_DIR = r"C:\Users\User\Desktop\archive\chest_xray"
OUTPUT_DIR = r"C:\Users\User\Desktop\archive\chest_xray_split"

# classes
CLASSES = ["NORMAL", "PNEUMONIA"]


for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

all_images = {cls: [] for cls in CLASSES}
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, split, cls)
        for img in os.listdir(folder):
            all_images[cls].append(os.path.join(folder, img))

# partitioning 70/15/15
for cls, imgs in all_images.items():
    random.shuffle(imgs)
    total = len(imgs)
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)

    train_imgs = imgs[:train_end]
    val_imgs = imgs[train_end:val_end]
    test_imgs = imgs[val_end:]

    # copy images
    for img in train_imgs:
        shutil.copy(img, os.path.join(OUTPUT_DIR, "train", cls))
    for img in val_imgs:
        shutil.copy(img, os.path.join(OUTPUT_DIR, "val", cls))
    for img in test_imgs:
        shutil.copy(img, os.path.join(OUTPUT_DIR, "test", cls))

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}:")
    for cls in CLASSES:
        count = len(os.listdir(os.path.join(OUTPUT_DIR, split, cls)))
        print(f"  {cls}: {count} images")
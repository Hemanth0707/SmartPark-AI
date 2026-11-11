import os
import shutil
import random

SOURCE = "CNRPark-Patches-150x150"
OUTPUT = "dataset"

random.seed(42)

classes = ["busy", "free"]
splits = ["train", "val"]

for split in splits:
    for class_name in classes:
        os.makedirs(
            os.path.join(OUTPUT, split, class_name),
            exist_ok=True
        )

for camera in ["A", "B"]:
    for class_name in classes:

        source_folder = os.path.join(
            SOURCE, camera, class_name
        )

        images = [
            file for file in os.listdir(source_folder)
            if file.lower().endswith(".jpg")
        ]

        random.shuffle(images)

        split_index = int(len(images) * 0.8)

        train_images = images[:split_index]
        val_images = images[split_index:]

        for file in train_images:
            shutil.copy(
                os.path.join(source_folder, file),
                os.path.join(OUTPUT, "train", class_name, file)
            )

        for file in val_images:
            shutil.copy(
                os.path.join(source_folder, file),
                os.path.join(OUTPUT, "val", class_name, file)
            )

print("Dataset preparation complete.")
import os
import shutil
import random

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\projects\Vehicle_AI_Insurance"

SOURCE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "data1a"
)

YOLO_DIR = os.path.join(
    BASE_DIR,
    "yolo_dataset"
)

TRAIN_IMAGE_DIR = os.path.join(
    YOLO_DIR,
    "images",
    "train"
)

VAL_IMAGE_DIR = os.path.join(
    YOLO_DIR,
    "images",
    "val"
)

TRAIN_LABEL_DIR = os.path.join(
    YOLO_DIR,
    "labels",
    "train"
)

VAL_LABEL_DIR = os.path.join(
    YOLO_DIR,
    "labels",
    "val"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for folder in [
    TRAIN_IMAGE_DIR,
    VAL_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    VAL_LABEL_DIR
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):

    extensions = (
        ".jpg",
        ".jpeg",
        ".JPG",
        ".JPEG",
        ".png",
        ".PNG"
    )

    images = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.endswith(extensions):

                images.append(
                    os.path.join(root, file)
                )

    return images


# ============================================================
# LOAD DATA
# ============================================================

damage_train = get_images(
    os.path.join(
        SOURCE_DIR,
        "training",
        "00-damage"
    )
)

whole_train = get_images(
    os.path.join(
        SOURCE_DIR,
        "training",
        "01-whole"
    )
)

damage_val = get_images(
    os.path.join(
        SOURCE_DIR,
        "validation",
        "00-damage"
    )
)

whole_val = get_images(
    os.path.join(
        SOURCE_DIR,
        "validation",
        "01-whole"
    )
)


print()
print("======================================")
print("SOURCE DATASET")
print("======================================")

print("Damage training :", len(damage_train))
print("Whole training   :", len(whole_train))
print("Damage validation:", len(damage_val))
print("Whole validation :", len(whole_val))


# ============================================================
# NOTE
# ============================================================
#
# Your current dataset is a CLASSIFICATION dataset.
#
# It contains:
#
# 00-damage
# 01-whole
#
# It does NOT contain bounding-box annotations.
#
# Therefore, we cannot create real YOLO damage bounding boxes
# from these images automatically.
#
# For now we create YOLO-compatible placeholder labels.
#
# IMPORTANT:
# This is only for testing the YOLO pipeline.
# It is NOT a properly annotated damage-detection dataset.
#
# ============================================================


def copy_images_and_labels(
    image_list,
    image_destination,
    label_destination,
    class_id
):

    count = 0

    for image_path in image_list:

        original_name = os.path.basename(
            image_path
        )

        name_without_extension = os.path.splitext(
            original_name
        )[0]

        # Prevent duplicate names
        new_name = f"{class_id}_{name_without_extension}.jpg"

        destination_image = os.path.join(
            image_destination,
            new_name
        )

        destination_label = os.path.join(
            label_destination,
            f"{class_id}_{name_without_extension}.txt"
        )

        # Copy image
        shutil.copy2(
            image_path,
            destination_image
        )

        # ----------------------------------------------------
        # Placeholder YOLO bounding box
        # ----------------------------------------------------
        #
        # class_id
        # x_center
        # y_center
        # width
        # height
        #
        # Values are normalized 0-1.
        #
        # Full-image bounding box:
        #
        # center = 0.5, 0.5
        # width  = 1.0
        # height = 1.0
        #
        # ----------------------------------------------------

        with open(
            destination_label,
            "w"
        ) as f:

            f.write(
                f"{class_id} 0.5 0.5 1.0 1.0\n"
            )

        count += 1

    return count


# ============================================================
# COPY TRAINING DATA
# ============================================================

print()
print("======================================")
print("PREPARING TRAINING DATA")
print("======================================")


damage_count = copy_images_and_labels(
    damage_train,
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    0
)

whole_count = copy_images_and_labels(
    whole_train,
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    1
)


# ============================================================
# COPY VALIDATION DATA
# ============================================================

print()
print("======================================")
print("PREPARING VALIDATION DATA")
print("======================================")


damage_val_count = copy_images_and_labels(
    damage_val,
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR,
    0
)

whole_val_count = copy_images_and_labels(
    whole_val,
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR,
    1
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("======================================")
print("YOLO DATASET CREATED")
print("======================================")

print()
print("Training images :", damage_count + whole_count)
print("Validation images:", damage_val_count + whole_val_count)

print()
print("YOLO dataset:")
print(YOLO_DIR)

print()
print("Training images:")
print(TRAIN_IMAGE_DIR)

print()
print("Training labels:")
print(TRAIN_LABEL_DIR)

print()
print("Validation images:")
print(VAL_IMAGE_DIR)

print()
print("Validation labels:")
print(VAL_LABEL_DIR)

print()
print("======================================")
print("IMPORTANT")
print("======================================")

print(
    "The current data1a dataset has no real "
    "bounding-box annotations."
)

print(
    "The generated labels are full-image "
    "placeholder boxes."
)

print(
    "For real damage localization, use a "
    "properly annotated vehicle damage dataset."
)

print()
print("DONE!")
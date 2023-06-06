"""SCript construido para ejecutar partición de datos"""
import os
import random
import shutil

import numpy as np
import tensorflow as tf

tf.random.set_seed(123)
np.random.seed(123)

main_dir = "data/raw/"
processed_dir = "data/processed/"

# Directorio con las imágenes de fake
fake_dir = os.path.join(main_dir, "fake")

# Directorio con las imágenes de real
real_dir = os.path.join(main_dir, "real")

# The directory to save training images
train_dir = os.path.join(processed_dir, "train")

# The directory to save validation images
val_dir = os.path.join(processed_dir, "validation")

# The directory to save validation images
test_dir = os.path.join(processed_dir, "test")

# The proportion of images to use for the validation set (0.2 means 20%)
val_ratio = [0.7, 0.2, 0.1]

# Create subdirectories for each category in the training and validation directories
categories = ["fake", "real"]
for category in categories:
    os.makedirs(os.path.join(train_dir, category))
    os.makedirs(os.path.join(val_dir, category))
    os.makedirs(os.path.join(test_dir, category))

# Split the images in each category into training and validation sets
for category in categories:
    category_dir = os.path.join(main_dir, category)
    all_files = os.listdir(category_dir)
    random.shuffle(all_files)
    val_size = int(len(all_files) * val_ratio[1])
    test_size = int(len(all_files) * val_ratio[2])
    train_size = len(all_files) - val_size - test_size
    train_files = all_files[:train_size]
    val_files = all_files[train_size : train_size + val_size]
    test_files = all_files[train_size + val_size :]
    for filename in train_files:
        src_path = os.path.join(category_dir, filename)
        dst_path = os.path.join(train_dir, category, filename)
        shutil.copy(src_path, dst_path)
    for filename in val_files:
        src_path = os.path.join(category_dir, filename)
        dst_path = os.path.join(val_dir, category, filename)
        shutil.copy(src_path, dst_path)
    for filename in test_files:
        src_path = os.path.join(category_dir, filename)
        dst_path = os.path.join(test_dir, category, filename)
        shutil.copy(src_path, dst_path)

print("Image split complete.")

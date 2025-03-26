import os
import shutil
import random

def split_dataset(folder_A, folder_B, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    images_A = sorted([f for f in os.listdir(folder_A) if f.endswith(('.png', '.jpg', '.jpeg'))])
    images_B = sorted([f for f in os.listdir(folder_B) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    assert len(images_A) == len(images_B), "Both folders must have the same number of images"
    
    paired_images = list(zip(images_A, images_B))
    random.shuffle(paired_images)
    
    total_images = len(paired_images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    
    train_set = paired_images[:train_count]
    val_set = paired_images[train_count:train_count + val_count]
    test_set = paired_images[train_count + val_count:]
    
    for split_name, split_set in zip(['train', 'val', 'test'], [train_set, val_set, test_set]):
        split_A = os.path.join(output_folder, split_name, 'A')
        split_B = os.path.join(output_folder, split_name, 'B')
        os.makedirs(split_A, exist_ok=True)
        os.makedirs(split_B, exist_ok=True)
        
        for img_A, img_B in split_set:
            shutil.copy(os.path.join(folder_A, img_A), os.path.join(split_A, img_A))
            shutil.copy(os.path.join(folder_B, img_B), os.path.join(split_B, img_B))
    
    print("Dataset split completed!")

split_dataset('../BBDM-RRC/data/source_images', '../BBDM-RRC/data/target_images', '../BBDM-RRC/data')

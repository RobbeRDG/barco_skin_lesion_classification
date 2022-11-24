import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

from util import config


class SegmentationDataset(Dataset):
    def __init__(self, feature_images_base_path, label_images_base_path, transformations_both = None, transformations_features=None):
        self.feature_images_base_path = feature_images_base_path
        self.label_images_base_path = label_images_base_path
        self.transformations_both = transformations_both
        self.transformations_features = transformations_features
        
        # Extract the image id's
        feature_image_names = os.listdir(self.feature_images_base_path)
        self.image_ids = list(map(lambda x: x.split('_')[1][:-4], feature_image_names))

    def __getitem__(self, idx):
        # Get the image id
        image_id = self.image_ids[idx]
        
        # Set the image paths
        feature_image_path = os.path.join(self.feature_images_base_path, f'ISIC_{str(image_id).zfill(7)}.jpg')
        label_image_path = os.path.join(self.label_images_base_path, f'ISIC_{str(image_id).zfill(7)}_segmentation.png')

        # Get the images
        feature_image = Image.open(feature_image_path)
        label_image = Image.open(label_image_path)

        # Apply the image transformations
        feature_image = self.transformations_both(feature_image)
        label_image = self.transformations_both(label_image)

        # Apply the additional transformations to the features
        feature_image = self.transformations_features(feature_image)

        return feature_image, label_image

    def __len__(self):
        return len(os.listdir(self.feature_images_base_path))


if __name__ == '__main__':
    # Test dataset
    test_dataset = SegmentationDataset(
        config.SEGMENTATION_DATA_PATH_TRAIN_FEATURES,
        config.SEGMENTATION_DATA_PATH_TRAIN_LABELS
    )

    # Print the length
    print(f'length: {test_dataset.__len__()}')

    feature, label = test_dataset.__getitem__(6)
    
    plt.subplot(1,2,1)
    plt.imshow(feature,aspect="auto")

    plt.subplot(1,2,2)
    plt.imshow(label,aspect="auto")

    plt.savefig("test.png")


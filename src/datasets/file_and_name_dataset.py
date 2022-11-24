import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

from util import config


class FileAndNameDataset(Dataset):
    def __init__(self, feature_images_base_path, transformations = None,):
        self.feature_images_base_path = feature_images_base_path
        self.transformations = transformations
        
        # Extract the image names
        self.image_ids = os.listdir(self.feature_images_base_path)

    def __getitem__(self, idx):
        # Get the image id
        image_id = self.image_ids[idx]
        
        # Set the image path
        feature_image_path = os.path.join(self.feature_images_base_path, image_id)

        # Get the image
        feature_image = Image.open(feature_image_path)

        # Apply the image transformations
        #feature_image = self.transformations(feature_image)

        return feature_image, image_id

    def __len__(self):
        return len(os.listdir(self.feature_images_base_path))


if __name__ == '__main__':
    # Test dataset
    test_dataset = FileAndNameDataset(
        config.CLASSIFICATION_DATA_PATH_TRAIN_FEATURES,
    )

    # Print the length
    print(f'length: {test_dataset.__len__()}')

    image, name = test_dataset.__getitem__(6)
    
    plt.subplot(1,2,1)
    plt.imshow(image,aspect="auto")

    plt.savefig(name)


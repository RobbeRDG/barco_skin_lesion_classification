import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from util import config
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):
    def __init__(self, feature_images_base_path, label_images_base_path):
        self.feature_images_base_path = feature_images_base_path
        self.label_images_base_path = label_images_base_path

    def __getitem__(self, idx):
        # Set the image paths
        feature_image_path = os.path.join(self.feature_images_base_path, f'ISIC_{str(idx).zfill(7)}.jpg')
        label_image_path = os.path.join(self.label_images_base_path, f'ISIC_{str(idx).zfill(7)}_segmentation.png')

        # Get the images
        feature_image = np.array(Image.open(feature_image_path))
        label_image = np.array(Image.open(label_image_path))

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

    feature, label = test_dataset.__getitem__(4)
    
    plt.subplot(1,2,1)
    plt.imshow(feature,aspect="auto")

    plt.subplot(1,2,2)
    plt.imshow(label,aspect="auto")

    plt.savefig("test.png")


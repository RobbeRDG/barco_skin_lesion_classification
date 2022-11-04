from os.path import join

import torch
from torchvision import transforms

# Data paths
SEGMENTATION_DATA_PATH_TRAIN_FEATURES = "data/segmentation/train_features"
SEGMENTATION_DATA_PATH_TRAIN_LABELS = "data/segmentation/train_labels"
SEGMENTATION_DATA_PATH_TEST_FEATURES = "data/segmentation/test_features"
SEGMENTATION_DATA_PATH_TEST_LABELS = "data/segmentation/test_labels"
METADATA_PATH = "data/metadata"

# Checkpoint paths
SEGMENTATION_MODEL_CHECKPOINT_PATH = 'checkpoints/'

# Data
TRAIN_KEY = "train"
VAL_KEY = "val"
KEYS = [TRAIN_KEY, VAL_KEY]
DATA_FOLDER_NAMES = {TRAIN_KEY: "legacy", VAL_KEY: "target"}

# Model params for segmentation model
SEGMENTATION_EPOCHS = 100
SEGMENTATION_BATCH_SIZE = 16
SEGMENTATION_NUM_WORKERS = 2
SEGMENTATION_LR = 0.0001

# Set torch to use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentations segmentation
SEGMENTATION_TRAIN_TRANSFORMATIONS_BOTH = transforms.Compose([
    transforms.Resize((187, 250)),
    transforms.ToTensor()
])
SEGMENTATION_TEST_TRANSFORMATIONS_BOTH = transforms.Compose([
    transforms.Resize((187, 250)),
    transforms.ToTensor()
])

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    TRAIN_KEY: transforms.Compose([
        transforms.RandomResizedCrop((63, 255)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    VAL_KEY: transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
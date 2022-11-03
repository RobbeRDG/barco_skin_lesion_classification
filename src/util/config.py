from os.path import join

import torch
from torchvision import transforms

# Paths
BASE_PATH = '/workspaces/barco_skin_lesion_classification'
CODE_PATH = join(BASE_PATH,"src/")
SEGMENTATION_DATA_PATH_TRAIN_FEATURES = join(BASE_PATH,"data/segmentation/train_features")
SEGMENTATION_DATA_PATH_TRAIN_LABELS = join(BASE_PATH,"data/segmentation/train_labels")
SEGMENTATION_DATA_PATH_TEST_FEATURES = join(BASE_PATH,"data/segmentation/test_features")
SEGMENTATION_DATA_PATH_TEST_LABELS = join(BASE_PATH,"data/segmentation/test_labels")
METADATA_PATH = join(BASE_PATH,"data/metadata")

# Data
TRAIN_KEY = "train"
VAL_KEY = "val"
KEYS = [TRAIN_KEY, VAL_KEY]
DATA_FOLDER_NAMES = {TRAIN_KEY: "legacy", VAL_KEY: "target"}

# Model params for segmentation model
SEGMENTATION_EPOCHS = 10
SEGMENTATION_BATCH_SIZE = 4
SEGMENTATION_NUM_WORKERS = 2
SEGMENTATION_LR = 0.01

# Set torch to use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentations segmentation
SEGMENTATION_TRAIN_TRANSFORMATIONS_BOTH = transforms.Compose([
    transforms.RandomResizedCrop((190, 255)),
    transforms.ToTensor()
])

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    TRAIN_KEY: transforms.Compose([
        transforms.RandomResizedCrop((190, 255)),
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
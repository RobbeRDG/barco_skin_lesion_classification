from os.path import join
import sys

import torch
from torchvision import transforms

# Set torch to use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
SEGMENTATION_DATA_PATH_TRAIN_FEATURES = "data/segmentation/train_features"
SEGMENTATION_DATA_PATH_TRAIN_LABELS = "data/segmentation/train_labels"
SEGMENTATION_DATA_PATH_TEST_FEATURES = "data/segmentation/test_features"
SEGMENTATION_DATA_PATH_TEST_LABELS = "data/segmentation/test_labels"
METADATA_PATH = "data/metadata"

# Checkpoint paths
SEGMENTATION_MODEL_CHECKPOINT_PATH = 'checkpoints/'

# Model params for segmentation model
SEGMENTATION_EPOCHS = 30
SEGMENTATION_BATCH_SIZE = 8
SEGMENTATION_NUM_WORKERS = 2
SEGMENTATION_LR = 0.0001
SEGMENTATION_IMAGE_HEIGHT = 225
SEGMENTATION_IMAGE_WIDTH = 300
SEGMENTATION_START_FROM_ARTIFACT = True
SEGMENTATION_START_ARTIFACT = "dermapool/segmentation/final_model:v1"
SEGMENTATION_START_ARTIFACT_MODEL = "chechpoint_11_05_2022_14_19_24.pth"

# Data augmentations segmentation
SEGMENTATION_TRAIN_TRANSFORMATIONS_BOTH = transforms.Compose([
    transforms.Resize((SEGMENTATION_IMAGE_HEIGHT, SEGMENTATION_IMAGE_WIDTH)),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
SEGMENTATION_TEST_TRANSFORMATIONS_BOTH = transforms.Compose([
    transforms.Resize((SEGMENTATION_IMAGE_HEIGHT, SEGMENTATION_IMAGE_WIDTH)),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
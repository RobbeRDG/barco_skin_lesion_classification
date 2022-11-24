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
CLASSIFICATION_DATA_PATH_TRAIN_FEATURES = "data/classification/train"
CLASSIFICATION_DATA_PATH_TRAIN_SEGMENTED_FEATURES = "data/classification/train_segmented"
CLASSIFICATION_DATA_PATH_TRAIN_UNSEGMENTED_FEATURES = "data/classification/train_unsegmented"
CLASSIFICATION_DATA_PATH_TEST_FEATURES = "data/classification/test"
METADATA_TRAIN_PATH = "data/classification/metadata/metadata_test.csv"
METADATA_TEST_PATH = "data/classification/metadata/metadata_test.csv"

# Checkpoint paths
SEGMENTATION_MODEL_CHECKPOINT_PATH = 'checkpoints/segmentation'
CLASSIFICATION_MODEL_CHECKPOINT_PATH = 'checkpoints/classification'

# Model params for segmentation model
SEGMENTATION_EPOCHS = 50
SEGMENTATION_BATCH_SIZE = 16
SEGMENTATION_NUM_WORKERS = 2
SEGMENTATION_LR = 0.0001
SEGMENTATION_IMAGE_HEIGHT = 225
SEGMENTATION_IMAGE_WIDTH = 300
SEGMENTATION_START_FROM_ARTIFACT = False
SEGMENTATION_START_ARTIFACT = "dermapool/segmentation/final_model:v1"
SEGMENTATION_START_ARTIFACT_MODEL = "chechpoint_11_05_2022_14_19_24.pth"

# Model params for classification model
CLASSIFICATION_EPOCHS = 10
CLASSIFICATION_BATCH_SIZE = 16
CLASSIFICATION_LR = 0.0001
CLASSIFICATION_IMAGE_HEIGHT = 225
CLASSIFICATION_IMAGE_WIDTH = 300

# Data augmentations segmentation
SEGMENTATION_TRAIN_TRANSFORMATIONS_BOTH = transforms.Compose([
    transforms.Resize((SEGMENTATION_IMAGE_HEIGHT, SEGMENTATION_IMAGE_WIDTH)),
    transforms.ToTensor(),
])
SEGMENTATION_TRAIN_TRANSFORMATIONS_FEATURES = transforms.Compose([
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
SEGMENTATION_TEST_TRANSFORMATIONS_BOTH = transforms.Compose([
    transforms.Resize((SEGMENTATION_IMAGE_HEIGHT, SEGMENTATION_IMAGE_WIDTH)),
    transforms.ToTensor(),
])
SEGMENTATION_TEST_TRANSFORMATIONS_FEATURES = transforms.Compose([
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
SEGMENTATION_RUN_TRANSFORMATIONS = transforms.Compose([
    transforms.Resize((SEGMENTATION_IMAGE_HEIGHT, SEGMENTATION_IMAGE_WIDTH)),
    transforms.ToTensor()
])


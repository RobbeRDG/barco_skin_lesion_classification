from os.path import join
import sys

import torch
from torchvision import transforms

# Set torch to use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
#segmentation
SEGMENTATION_DATA_PATH_TRAIN_FEATURES = "data/segmentation/train_features"
SEGMENTATION_DATA_PATH_TRAIN_LABELS = "data/segmentation/train_labels"
SEGMENTATION_DATA_PATH_TEST_FEATURES = "data/segmentation/test_features"
SEGMENTATION_DATA_PATH_TEST_LABELS = "data/segmentation/test_labels"

CLASSIFICATION_DATA_PATH_TRAIN_FEATURES = "data/classification/train"
CLASSIFICATION_DATA_PATH_TRAIN_SEGMENTED_FEATURES = "data/classification/train_segmented"
CLASSIFICATION_DATA_PATH_TRAIN_UNSEGMENTED_FEATURES = "data/classification/train_unsegmented"
CLASSIFICATION_DATA_PATH_TEST_FEATURES = "data/classification/test"
CLASSIFICATION_DATA_PATH_VAL_FEATURES = "data/classification/target"
CLASSIFICATED_DATA_PATH_TRAIN_CLASSIFIED = "data/classification/results" 
METADATA_TRAIN_PATH = "data/classification/metadata/metadata_train.csv"
METADATA_TEST_PATH = "data/classification/metadata/metadata_test.csv"
METADATA_VAL_PATH = "data/classification/metadata/metadata_target.csv"

# Checkpoint paths
SEGMENTATION_MODEL_CHECKPOINT_PATH = 'checkpoints/segmentation'
UNSEGMENTED_CLASSIFICATION_MODEL_CHECKPOINT_PATH = 'checkpoints/classification/unsegmented'
SEGMENTED_CLASSIFICATION_MODEL_CHECKPOINT_PATH = 'checkpoints/classification/segmented'

# Model params for segmentation model
SEGMENTATION_EPOCHS = 50
SEGMENTATION_BATCH_SIZE = 16
SEGMENTATION_NUM_WORKERS = 2
SEGMENTATION_LR = 0.0001
SEGMENTATION_IMAGE_HEIGHT = 225
SEGMENTATION_IMAGE_WIDTH = 300
SEGMENTATION_ARTIFACT = "dermapool/segmentation/final_model:v1"
SEGMENTATION_ARTIFACT_CHECKPOINT = "chechpoint_11_05_2022_14_19_24.pth"

# Model params for classification model
CLASSIFICATION_EPOCHS = 5
CLASSIFICATION_BATCH_SIZE = 16
CLASSIFICATION_LR = 0.0001
CLASSIFICATION_IMAGE_HEIGHT = 225
CLASSIFICATION_IMAGE_WIDTH = 300
UNSEGMENTED_CLASSIFICATION_ARTIFACT = "dermapool/classification/final_classification_model_unsegmented:v1"
UNSEGMENTED_CLASSIFICATION_ARTIFACT_CHECKPOINT = "chechpoint_11_28_2022_19_28_15.pth"
SEGMENTED_CLASSIFICATION_ARTIFACT = "dermapool/classification/final_classification_model_segmented:v1"
SEGMENTED_CLASSIFICATION_ARTIFACT_CHECKPOINT = "chechpoint_11_28_2022_19_05_59.pth"

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

# Data augmentations classification
CLASSIFICATION_TRAIN_TRANSFORMATIONS = transforms.Compose([
    transforms.Resize((CLASSIFICATION_IMAGE_HEIGHT, CLASSIFICATION_IMAGE_WIDTH)),
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
CLASSIFICATION_VAL_TRANSFORMATIONS = transforms.Compose([
    transforms.Resize((CLASSIFICATION_IMAGE_HEIGHT, CLASSIFICATION_IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
CLASSIFICATION_TEST_TRANSFORMATIONS = transforms.Compose([
    transforms.Resize((CLASSIFICATION_IMAGE_HEIGHT, CLASSIFICATION_IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
from os.path import join
from torchvision import transforms
import torch

# Paths
BASE_PATH = '/workspaces/barco_skin_lesion_classification'
CODE_PATH = join(BASE_PATH,"src/")
DATASET_PATH = join(BASE_PATH,"data/")
METADATA_PATH = join(BASE_PATH,"data/metadata")

# Data
TRAIN_KEY = "train"
VAL_KEY = "val"
KEYS = [TRAIN_KEY, VAL_KEY]
DATA_FOLDER_NAMES = {TRAIN_KEY: "legacy", VAL_KEY: "target"}

# Model params
BATCH_SIZE = 4
NUM_WORKERS = 2

# Set torch to use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    TRAIN_KEY: transforms.Compose([
        transforms.RandomResizedCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    VAL_KEY: transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
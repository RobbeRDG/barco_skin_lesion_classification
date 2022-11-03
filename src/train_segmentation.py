import matplotlib as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

import wandb
from datasets.segmentationdataset import SegmentationDataset
from models.unet_model import UNet
from trainers.segmentation_model_trainer import train_segmentation_model
from util import config

# Get the data
train_segmentation_dataset = SegmentationDataset(
    config.SEGMENTATION_DATA_PATH_TRAIN_FEATURES,
    config.SEGMENTATION_DATA_PATH_TRAIN_LABELS,
    config.SEGMENTATION_TRAIN_TRANSFORMATIONS_BOTH
    )

test_segmentation_dataset = SegmentationDataset(
    config.SEGMENTATION_DATA_PATH_TEST_FEATURES,
    config.SEGMENTATION_DATA_PATH_TEST_LABELS
    )

# Place the datasets in dataloaders
train_segmentation_dataloader = DataLoader(train_segmentation_dataset, batch_size=config.SEGMENTATION_BATCH_SIZE)
test_segmentation_dataloader = DataLoader(test_segmentation_dataset, batch_size=1)

# Get the model
model = UNet(n_channels=3, n_classes=1, bilinear=False)
model.to(config.DEVICE)

# Set the optimizer
optimizer = optim.Adam(model.parameters(), lr=config.SEGMENTATION_LR)

# Set the loss fn
criteria = nn.CrossEntropyLoss()

# Set the gradient scaler
#grad_scaler = torch.cuda.amp.grad_scaler.GradScaler()


# Setup weights and biasses
#wandb.login()

# Start wandb
#wandb.init(
#    settings=wandb.Settings(start_method="fork"),
#    project="test-project", 
#    name=f"experiment_1", 
#    config={
#        "learning_rate": config.SEGMENTATION_LR,
#        "batch_size": config.SEGMENTATION_BATCH_SIZE,
#        "epochs": config.SEGMENTATION_EPOCHS,
#    }
#)

for epoch in range(config.SEGMENTATION_EPOCHS):
    # Set the model in training mode
    model.train()

    # Train the model
    train_loss_this_epoch = train_segmentation_model(
        model,
        optimizer,
        criteria,
        #grad_scaler,
        train_segmentation_dataloader
    )
    
    # Set the model in evaluation mode
    model.eval()

    # Validate the model
    val_loss_this_epoch, sample_image_array = validate_segmentation_model(
        model,
        criteria,
        test_segmentation_dataloader,
        test_segmentation_dataset
    )

    # convert the image array into an image
    sample_image = wandb.Image(sample_image_array, caption="Top: Output, Bottom: Input")

    # Log the train loss this epoch
    wandb.log({
        'train_loss': train_loss_this_epoch/len(train_segmentation_dataloader.dataset),
        'val_loss': val_loss_this_epoch/len(test_segmentation_dataloader.dataset),
        'sample_image': sample_image
    })


# Mark the run as finished
#wandb.finish()
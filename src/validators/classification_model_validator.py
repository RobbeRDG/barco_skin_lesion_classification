import torch
from tqdm import tqdm

from util import config
from util.dice_score import dice_loss


def validate_classification_model(
        model,
        loss_fn,
        dataloader,
    ):
    total_loss = 0.0

    # No gradients needed during validation
    with torch.no_grad():
        # First calculate the validation loss
        loop = tqdm(dataloader, leave=True)
        for idx, (input, label) in enumerate(loop):
            # Send the input and label to device
            input, label = input.to(config.DEVICE), label.to(config.DEVICE)

            # Runs the forward pass.
            output = model(input)
            loss = loss_fn(output, label)

            # Add the loss to the total
            total_loss += loss

    return total_loss

    



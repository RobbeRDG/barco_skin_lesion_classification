import torch
from tqdm import tqdm

from util import config
from util.dice_score import dice_loss


def validate_segmentation_model(
        model,
        loss_fn,
        dataloader,
        dataset
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
            loss = loss_fn(output, label) + dice_loss(torch.sigmoid(output).float(), label.float())

            # Add the loss to the total
            total_loss += loss

        # Second, take a sample of the segmentation performance
        input, label
        for sample_input, sample_label in dataloader:
          input = sample_input
          label = sample_label

        input, label = input.to(config.DEVICE), label.to(config.DEVICE)

        output = model(input)
        output = torch.sigmoid(output)

        # Rescale the pixel values
        input = torch.mul(input[0][0], 255)
        output = torch.mul(output[0][0], 255)
        label = torch.mul(label[0][0], 255)

        # Make one big image with the tree image arrays
        image_array = torch.cat([input, output, label])

    # Return both the image and the total loss
    return total_loss, image_array

    



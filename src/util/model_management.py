import torch
from os.path import join
import wandb
import datetime
from util import config

def save_model(model, checkpoint_path, save_as_artifact):
    # Set the best model
    best_model_state = model.state_dict()

    # Save the best model
    torch.save(best_model_state, checkpoint_path)

    # Also save the model as an artifact
    if save_as_artifact:
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

    return best_model_state
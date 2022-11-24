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

    # Also save the final model as an artifact
    if save_as_artifact:
        artifact = wandb.Artifact('final_model', type='model')
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

    return best_model_state

def get_artifact_model_weights():
    # Download the artifact
    artifact = wandb.use_artifact(config.SEGMENTATION_START_ARTIFACT, type='model')

    model_weights = artifact.get_path(config.SEGMENTATION_START_ARTIFACT_MODEL)

    return model_weights.download()


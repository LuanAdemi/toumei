import torch.nn as nn
import pickle


def convert_unit_string(x: str):
    identifiers = x.split(":")
    indices = map(int, identifiers[1:])
    return tuple([identifiers[0]] + list(indices))


def freeze_model(model: nn.Module):
    """
    Freezes the parameters of a pytorch model

    :param model: the model
    """
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_model(model: nn.Module):
    """
    Un-Freezes the parameters of a pytorch model

    :param model: the model
    """
    for p in model.parameters():
        p.requires_grad_(True)


def save(objective, filename):
    if objective.model is not None:
        objective.detach()
    pickle.dump(objective, open(filename, "wb"))


def load(filename):
    return pickle.load(open(filename, "rb"))

import torch.nn as nn


def convertUnitString(x: str):
    identifiers = x.split(":")
    indices = map(int, identifiers[1:])
    return tuple([identifiers[0]] + list(indices))


def freezeModel(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


def unfreezeModel(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(True)

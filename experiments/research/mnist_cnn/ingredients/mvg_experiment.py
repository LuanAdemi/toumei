import os

import torch
from sacred import Experiment
from sacred.observers import MongoObserver

from train import train_ingredient, train_model
from evaluate import evaluate_ingredient, evaluate_model

import uuid

ex = Experiment('mnist_cnn', ingredients=[train_ingredient, evaluate_ingredient])

ex.observers.append(MongoObserver(
    url="mongodb://luan:QnUctUPz2SamYadp@159.69.81.38/sacred?authMechanism=SCRAM-SHA-256",
    db_name="sacred"))

@ex.config
def cfg():
    save_path = f"results/mvg/{uuid.uuid4()}/"
    device = torch.device('cuda')
    mvg = True

@ex.automain
def run(_run, save_path, device, mvg):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ntks, delta_ntk_norms, model = train_model(device=device, save_path=save_path, mvg=mvg)
    evaluate_model(save_path=save_path, model=model)






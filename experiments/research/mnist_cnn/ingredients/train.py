import numpy as np
import torch
import torch.optim as opt
import torch.nn as nn
from tqdm import trange
from mnist_dataset import MNISTDataset
from torch.utils.data import DataLoader

from toumei.models import SimpleCNN
from utils import ntk
from sacred import Ingredient
from dataset import data_ingredient

train_ingredient = Ingredient('train', ingredients=[data_ingredient])


@train_ingredient.config
def cfg():
    batch_size = 32
    lr = 3e-3
    epochs = 3000
    switching_time = 25


@train_ingredient.capture
def train_model(_run, batch_size, device, lr, epochs, switching_time, mvg, save_path):
    dataset = MNISTDataset(device=device)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    model = SimpleCNN(2, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = opt.SGD(params=model.parameters(), lr=lr)

    a, b = 1, 1

    current_task = 0

    ntks = []
    delta_ntk_norms = []

    with trange(epochs) as t:
        for e in t:

            if mvg:
                if e % switching_time == 0:
                    if current_task == 0:
                        a, b = -1, 1
                        current_task = 1
                    else:
                        a, b = 1, 1
                        current_task = 0

            loss_train = []

            for inputs, outputs in dataloader:
                out = model(inputs)

                gt = a * outputs[:, 0] + b * outputs[:, 1]

                loss = criterion(out.view(-1), gt.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_train.append(loss.item())

            if e % 4 == 0:
                f, tk = ntk(model, torch.stack(dataset.inputs[:64]))

                delta_ntk_norm = torch.tensor([0])

                if len(ntks) >= 1:
                    delta_ntk_norm = torch.norm(ntks[len(ntks) - 1] - tk)

                ntks.append(tk)
                delta_ntk_norms.append(delta_ntk_norm)
                _run.log_scalar("training.dntk", delta_ntk_norm.item(), e)

            t.set_description(f"EPOCH #{e}")
            t.set_postfix(loss=np.mean(loss_train), coeffs=[a, b], d_ntk_norm=delta_ntk_norm.item())
            _run.log_scalar("training.loss", np.mean(loss_train), e)

    torch.save(model.state_dict(), save_path + 'model.pth')
    _run.add_artifact(save_path + 'model.pth', 'model_weights.pth')

    return ntks, delta_ntk_norms, model


import math
import random

import matplotlib.pyplot as plt
import numpy as np
from experiments.research.binary_addition.two_numbers_datasets import OneHotEncodingDataset
from toumei.models import SimpleMLP
import torch
from torch.utils.data import DataLoader
from toumei.misc import MLPGraph
from toumei.cnns.objectives.utils import set_seed


set_seed(42)

N_NETWORKS = 30
STEP_SIZE = 8

MVG_EPOCHS = 10000
MAX_EPOCHS = 10000
NETWORK_SAMPLES = 1
LR = 0.001
LOSS = torch.nn.MSELoss()

METHOD = "MVG"
SWITCHING_TIME = 15

DEVICE = torch.device("cuda")

NETWORKS = np.arange(start=STEP_SIZE, stop=N_NETWORKS*STEP_SIZE, step=STEP_SIZE)
GLOBAL_Q_VALUES = []

for i, n in enumerate(NETWORKS):
    Q_VALUES = []

    dataset = OneHotEncodingDataset(elements=n, length=4000)
    DATASET_SIZE = len(dataset)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    for s in range(NETWORK_SAMPLES):

        dimensions = [2*n, n*4, n*2, n, 1]

        model = SimpleMLP(*dimensions).to(DEVICE)
        opt = torch.optim.Adam(lr=LR, params=model.parameters())

        current_task = 0
        a = 1
        b = 1

        global_losses = []
        loss_train = []
        e = 0
        while True:
            e += 1
            r = random.random()

            if e < MVG_EPOCHS:

                if e % SWITCHING_TIME == 0:
                    if METHOD == "MVG":
                        if current_task == 0:
                            a = 1
                            b = 2
                            current_task = 1
                        else:
                            a = 1
                            b = 1
                            current_task = 0
                    elif METHOD == "RMVG":
                        a = random.random() * 10
                        b = random.random() * 10
                    else:
                        pass
            else:
                a = 1
                b = 1

            loss_train = []

            for _ in range(1):
                for h, (element, label) in enumerate(dataLoader):
                    (x1, x2) = label
                    result = x1 * a + x2 * b
                    predicted_result = model(element)
                    opt.zero_grad()
                    loss = LOSS(predicted_result.view(-1), result.view(-1))
                    loss.backward()
                    opt.step()
                    global_losses.append(loss.item())
                    loss_train.append(loss.item())

            if e % 100 == 0:
                print(f"({i}/{len(NETWORKS)}) DIM={dimensions}, "
                        f"S={s}, EPOCH={e}, LOSS={np.mean(loss_train)}, "
                        f"SWITCH={SWITCHING_TIME}, A={a}, B={b}, DATASET_SIZE={DATASET_SIZE}")

            if e >= MAX_EPOCHS:
                break

        graph = MLPGraph(model)
        Q, clusters = graph.get_model_modularity(method="louvain")

        print(f"Q={Q}")
        Q_VALUES.append(Q)

        x1 = np.linspace(1, len(global_losses), num=len(global_losses))
        plt.plot(x1, global_losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(f"mvg_full_batch/loss_{sum(dimensions)}_{s}_{METHOD}.png")
        plt.clf()

        torch.save(model.state_dict(), f"mvg_full_batch/{sum(dimensions)}_{s}_{METHOD}.pth")
    GLOBAL_Q_VALUES.append(Q_VALUES)

GLOBAL_Q_VALUES = np.array(GLOBAL_Q_VALUES)
np.save("mvg_full_batch/q_values.npy", GLOBAL_Q_VALUES)

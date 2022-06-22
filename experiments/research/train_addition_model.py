import random

import numpy as np
import torch
from toumei.models import SimpleMLP
from toumei.misc import MLPGraph


def number_to_tensor(n: int):
    out = torch.zeros(10, dtype=torch.float)
    out[n] = 1.
    return out


device = torch.device("cuda")

model = SimpleMLP(20, 4, 1).to(device)
loss_fc = torch.nn.MSELoss()
opt = torch.optim.Adam(lr=0.001, params=model.parameters())

epochs = 100
batches = 128
batch_size = 16

for e in range(epochs):
    model.train()
    loss_train = []
    for i in range(batches):
        elements = []
        labels = []
        for j in range(batch_size):
            a = random.randint(0, 9)
            b = random.randint(0, 9)
            elements.append(torch.cat((number_to_tensor(a), number_to_tensor(b))))
            labels.append(torch.FloatTensor([a + b]))

        elements = torch.stack(elements).to(device)
        labels = torch.stack(labels).to(device)

        predicted_result = model(elements)
        opt.zero_grad()
        loss = loss_fc(predicted_result, labels)
        loss.backward()
        opt.step()
        loss_train.append(loss.item())
        print('TRAIN: EPOCH %d: BATCH %d: LOSS: %.4f' %
              (e, i, np.mean(loss_train)))

graph = MLPGraph(model)
print(graph.get_model_modularity())

torch.save(model.state_dict(), "models/addition_model.pth")
from toumei.cnns.featurevis.objectives.utils import freeze_model, unfreeze_model
from toumei.parents.objective import Objective
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch

import tqdm


def hash_tensor(x):
    """
    A custom hash function for torch tensors for making the same tensors have the same hash value.
    :param x: the tensor
    :return: the hash value
    """
    hash_list = []
    for v in x:
        hash_list.append(v.item())

    return " ".join(map(str, hash_list))


class DatasetFinder(Objective):
    """
    Finds a sample from the dataset that maximizes the given objective
    """
    def __init__(self, dataset: Dataset, obj_func, sample_size=32):
        super(DatasetFinder, self).__init__()

        self.labels = None
        self.losses = None
        self.images = None
        self.sample_size = sample_size
        self.dataset = dataset
        self.iterator = iter(DataLoader(dataset, batch_size=sample_size))

        self.objective = obj_func

    def __str__(self):
        return f"DatasetFinder(dataset={self.dataset}, objective={self.objective}, sample_size={self.sample_size})"

    def attach(self, model: nn.Module):
        """
        Attach the modules to the model.
        :param model: the model
        """
        if self.model is not None:
            self.detach()

        self.model = model

        # freeze the model
        freeze_model(model)

        # call attach on the root
        self.root.attach(model)

    def detach(self):
        """
        Detach the modules from the current model
        :return: nothing
        """

        # check if attached
        if self.model is None:
            print("Cannot detach the current objective, since it was not detached in the first place.")
            return

        # unfreeze the model
        unfreeze_model(self.model)

        # call detach on the root
        self.root.detach()

        # reset the current model
        self.model = None

    @property
    def root(self):
        """
        Returns the root node of the objective function tree
        :return: the root node
        """
        return self.objective

    def forward(self) -> torch.Tensor:
        """
        The forward function for the objective.
        This calls the root, which induces recursive calls for each child of the module
        :return: a tensor for optimization
        """
        return self.root()

    def optimize(self, verbose=True):
        """
        Optimize the current objective using a selection method for samples from the dataset
        """
        self.model.eval()
        self.images = []
        self.labels = []
        self.losses = []

        batch = next(self.iterator)
        with tqdm.trange(len(batch[0]), disable=not verbose) as t:
            t.set_description(self.__str__())
            for i in t:
                img = batch[0][i]
                label = batch[1][i]

                self.images.append(img)
                self.labels.append(label)

                _ = self.model(img)
                self.losses.append(self.forward())

        self.losses = torch.tensor(self.losses)
        self.labels = torch.stack(self.labels)

    def get_topk_inputs(self, k):
        v, idxs = torch.topk(self.losses, k, largest=False)

        return [torch.permute(self.images[i], (1, 2, 0)).cpu() for i in idxs]

    def get_top_label(self):
        values, idxs = torch.topk(self.losses, len(self.losses), largest=False)

        classes = torch.unique(self.labels, dim=0)
        label_to_idx = {hash_tensor(c): i for i, c in enumerate(classes)}
        cumulative_loss = torch.zeros(size=(len(classes), ), dtype=torch.float)

        for value, idx in zip(values, idxs):
            label = hash_tensor(self.labels[idx])
            cumulative_loss[label_to_idx[label]] += value

        return classes[torch.argmin(cumulative_loss, dim=0)]

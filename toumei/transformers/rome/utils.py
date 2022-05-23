from collections import OrderedDict

import torch.nn as nn
import torch
from typing import List


class TracingHook(object):
    """
    A simple network hook for wrapping the forward function of modules.
    It can store and edit the hidden state.
    """
    def __init__(self, module: nn.Module, out_f=None):
        self.module = module
        self.output = None

        def create_hook(m, inputs, output):
            if out_f is not None:
                output = out_f(output)

            self.output = output

            return output

        self.hook = self.module.register_forward_hook(create_hook)

    @property
    def hidden_state(self):
        """
        Returns the hidden_state
        """
        return self.output

    def remove(self):
        """
        Removes the hook
        """
        self.hook.remove()


class TracingHookDict(OrderedDict):
    """
    A OrderedDict storing TracingHooks for the specified layers
    """
    def __init__(self, model: nn.Module, layers: List[nn.Module], out_f=None):
        super().__init__()

        self.model = model
        self.layers = layers

        for layer in layers:
            self[layer] = TracingHook(layer, out_f)

    def remove(self):
        """
        Remove the TracingHookDict and all it's TracingHooks
        """
        for layer, hook in reversed(self.items()):
            hook.remove()


def generate_inputs(prompt, tokenizer, batch_size=10):
    """
    Creates a batch of the same tokens using the given prompt
    :param prompt: the input prompt to tokenize
    :param tokenizer: the tokenizer
    :param batch_size: the batch size
    :returns: the created batch
    """
    inputs = []
    for s in range(batch_size):
        inputs.append(tokenizer(prompt, return_tensors='pt').input_ids)

    return torch.stack(inputs, dim=0)


def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        return m
    else:
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output

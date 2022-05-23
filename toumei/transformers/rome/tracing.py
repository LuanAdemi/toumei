from collections import defaultdict

import torch
import torch.nn as nn

from transformers import PreTrainedModel, PreTrainedTokenizer

from utils import *


def untuple(x):
    return x[0] if isinstance(x, tuple) else x


def _patching_rule(x: torch.Tensor, module,
                   patching_states: dict, token_range: tuple, device,
                   noise_coeff: float = 0.1):
    """
    Defines the patching rules for the different modules (layers) used in the tracing process
    :param x: the previous hidden state of the knowledge flow, where h[0] is uncorrupted h[1] is corrupted
    :param module: the current module
    :returns: the patched hidden state
    """

    # define the patching rule for the word embedding in the wte layer
    if isinstance(module, nn.Embedding):
        start, end = token_range
        # corrupt the tokens in the given range by adding random gaussian noise
        x[1:, start:end] += noise_coeff * torch.randn(x.shape[0] - 1, end - start, x.shape[2]).to(device)
        return x

    if module not in patching_states:
        return x

    h = untuple(x)
    # restore the uncorrupted hidden state from the first forward pass
    for token in patching_states[module]:
        h[1:, token] = h[0, token]
    return x


class CausalTracer(object):
    """
    Performs causal tracing on hugging face transformers models (should work on other transformer
    implementations when the syntax matches)

    Based on https://github.com/kmeng01/rome
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # set the default device to the models device
        self.device = model.device

        self.layers = nested_children(self.model)

    def to(self, device):
        """
        Sends the CausalTracer to the corresponding device
        :param device: the new device
        """
        self.device = device

        # update the model device
        self.model.to(device)

    def trace(self, prompt: str, subject):
        """

        :param subject:
        :param prompt:
        :return:
        """

        # create a batch for the forward pass
        inputs = generate_inputs([prompt] * (10 + 1), self.tokenizer, self.device)

        # collect the original output of the model
        with torch.no_grad():
            answer_token, base_score = [d[0] for d in predict_from_input(self.model, inputs)]

        # decode the answer with the tokenizer
        [answer] = decode_tokens(self.tokenizer, [answer_token])

        token_range = find_token_range(self.tokenizer, inputs["input_ids"][0], subject)

        low_score = self._forward_pass(inputs, [], answer_token, token_range).item()

        table = []

        for t in range(inputs["input_ids"].shape[1]):
            row = []
            for layer in self.layers["transformer"]["h"]:
                # add the current mlp to the patching layers
                patching_layer = [(t, self.layers["transformer"]["h"][layer])]
                # perform a causal tracing step
                result = self._forward_pass(inputs, patching_layer, answer_token, token_range)
                row.append(result)
            table.append(torch.stack(row))

        table = torch.stack(table).detach().cpu()

        result = dict(
            scores=table,
            low_score=low_score,
            high_score=base_score,
            input_ids=inputs["input_ids"][0],
            input_tokens=decode_tokens(self.tokenizer, inputs["input_ids"][0]),
            subject_range=token_range,
            answer=answer,
            window=False,
            kind="",
        )

        plot_trace_heatmap(result)

    def _forward_pass(self, inputs, patching_states: list, answer_token: int, token_range: tuple):
        """
        Performs a single causal trace
        :param inputs:
        :param patching_states:
        :return:
        """

        patch_spec = defaultdict(list)
        for t, l in patching_states:
            patch_spec[l].append(t)

        with torch.no_grad():
            # hook the model to later extract the hidden_states
            hook_dict = TracingHookDict(model=self.model,
                                        layers={**dict([("wte", self.layers["transformer"]["wte"])]),
                                                **dict(patching_states)},
                                        out_f=_patching_rule, out_f_params=(patch_spec, token_range, self.device))

            # forward pass
            out = self.model(**inputs)

            # remove the hook
            hook_dict.remove()

        # collect the model output probs for the original answer token
        probs = torch.softmax(out.logits[1:, -1, :], dim=1).mean(dim=0)[answer_token]

        return probs

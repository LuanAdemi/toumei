import torch
import torch.nn as nn

from transformers import PreTrainedModel, PreTrainedTokenizer

from utils import TracingHookDict, generate_inputs, nested_children


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

    def __call__(self, prompt: str):
        """"""

    def to(self, device):
        """
        Sends the CausalTracer to the corresponding device
        :param device: the new device
        """
        self.device = device

        # update the model device
        self.model.to(device)

    def _forward_pass(self, prompt: str):
        """
        Performs two forward passes through the transformers model with the second one being the corrupted pass
        :param prompt: the input prompt
        :param do_samples: sample the transformers output
        :param temp: temperature
        :param max_length: max token length for the output
        :returns: the generated text
        """

        # create a batch for the forward pass
        inputs = generate_inputs(prompt, self.tokenizer)

        # collect the original output of the model
        with torch.no_grad():
            answer_token, base_score = [o for o in torch.argmax(self.model(inputs)).item()]

        # decode the answer with the tokenizer
        answer = self.tokenizer.decode(answer_token)

        token_range = ()

        low_score = self._trace(inputs[0], {}, answer_token, token_range).item()

        table = []

        for t in range(inputs.shape[1]):
            row = []
            for layer in self.layers["transformer"]["h"]:
                result = self._trace(inputs, [(layer, )], answer_token, token_range, )

    def _patching_rule(self, h: torch.Tensor, module: nn.Module,
                       patching_states: dict, token_range: tuple,
                       noise_coeff: float = 0.1):
        """
        Defines the patching rules for the different modules (layers) used in the tracing process
        :param h: the previous hidden state of the knowledge flow, where h[0] is uncorrupted h[1] is corrupted
        :param module: the current module
        :returns: the patched hidden state
        """
        # define the patching rule for the word embedding in the wte layer
        if isinstance(module, nn.Embedding) and "wte" in module.__str__():
            start, end = token_range
            # corrupt the tokens in the given range by adding random gaussian noise
            h[1:, start:end] += noise_coeff * torch.randn(h.shape[0] - 1, end - start, h.shape[2]).to(self.device)
        elif module in patching_states:
            # restore the uncorrupted hidden state from the first forward pass
            for token in patching_states[module]:
                h[1:, token] = h[0, token]
        return h

    def _trace(self, inputs, patching_states: dict, answer_token: int, tracing_modules=None):
        """
        Performs a single causal trace
        :param inputs:
        :param patching_states:
        :param tracing_modules:
        :return:
        """
        tracing_modules = [] if tracing_modules is None else tracing_modules

        # hook the model to later extract the hidden_states
        hook_dict = TracingHookDict(model=self.model,
                                    layers=list(patching_states.keys()) + tracing_modules,
                                    out_f=self._patching_rule)

        # forward pass
        out = self.model(**inputs)

        # remove the hook
        hook_dict.remove()

        # collect the model output probs for the original answer token
        probs = torch.softmax(out.logits[1:, -1, :], dim=1).mean(dim=0)[answer_token]

        if tracing_modules is not None:
            traced = torch.stack([hook_dict[module].hidden_state.detach().cpu() for module in tracing_modules], dim=2)
            return probs, traced

        return probs

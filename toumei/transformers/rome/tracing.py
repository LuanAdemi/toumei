import torch
import torch.nn as nn

from transformers import PreTrainedModel, PreTrainedTokenizer


class CausalTracer(object):
    """
    Performs causal tracing on hugging face transformers models (should work on other implementations when the syntax
    matches)

    Based on https://github.com/kmeng01/rome
    """
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # set the default device to the models device
        self.device = model.device

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

    def _forward_pass(self, prompt: str, do_samples: bool=False, temp: float=0.9, max_length: int=100):
        """
        Performs two forward passes through the transformers model with the second one being the corrupted pass
        :param prompt: the input prompt
        :param do_samples: sample the transformers output
        :param temp: temperature
        :param max_length: max token length for the output
        :returns: the generated text
        """
        input_ids = self.tokenizer(prompt, return_tensor='pt').input_ids
        gen_tokens = self.model.generate(input_ids, do_samples=do_samples, temp=temp, max_length=max_length)

        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]

        return gen_text

    def _patching_rule(self, h: torch.Tensor, module: nn.Module,
                       patching_states: dict, token_range: tuple,
                       noise_coeff: float=0.1):
        """
        Defines the patching rules for the different modules (layers) used in the tracing process
        :param h: the previous hidden state of the knowledge flow h[0] is uncorrupted h[1] is corrupted
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

    def _trace(self, inputs, patching_states, tracing_modules=None):
        """
        Performs a single causal trace
        :param inputs:
        :param patching_states:
        :param tracing_modules:
        :return:
        """


from collections import defaultdict
from toumei.cnns.objectives.utils import freeze_model

from tqdm import trange

from transformers import PreTrainedModel, PreTrainedTokenizer

from toumei.transformers.rome.utils import *


class CausalTracer(object):
    """
    Performs causal tracing on hugging face transformers models
    (should work on other transformer implementations when the syntax matches)

    Based on https://github.com/kmeng01/rome
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initiates a new CausalTracer object using the given model and tokenizer

        :param model: the transformer model we would like to trace
        :param tokenizer: the tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

        # prep the model for tracing
        self.model.eval()
        freeze_model(self.model)

        # set the default device to the models device
        self.device = model.device

        # get the mlp layers
        self.layers = nested_children(self.model)

    def to(self, device):
        """
        Sends the CausalTracer to the corresponding device

        :param device: the new device
        """
        self.device = device
        self.model.to(device)

    def trace(self, prompt: str, subject, samples=10, verbose=False):
        """
        Performs causal tracing using the specified prompt and subject

        :param subject: The subject of the prompt
        :param prompt: The prompt
        :param samples: The amount of samples for the tracing process
        :param verbose: Print some info about the tracing process
        :returns: The tracing result
        """

        # create a batch for the forward pass
        inputs = generate_inputs([prompt] * (samples + 1), self.tokenizer, self.device)

        # collect the original output of the model
        with torch.no_grad():
            answer_token, base_score = [d[0] for d in predict_from_input(self.model, inputs)]

        # decode the answer with the tokenizer
        [answer] = decode_tokens(self.tokenizer, [answer_token])

        # the token range of the subject
        token_range = find_token_range(self.tokenizer, inputs["input_ids"][0], subject)

        # calculate the lowest score possible by corrupting and patching no layer at all
        low_score = self._forward_pass(inputs, [], answer_token, token_range).item()

        # build the tracing score table
        table = []

        num_tokens = inputs["input_ids"].shape[1]

        if verbose:
            print(f"Prompt: {prompt},")
            print(f"Subject: {subject},")
            print(f"Answer: {answer}")
            print(f"Performing causal tracing for {num_tokens} tokens...")

        for t in trange(num_tokens, disable=not verbose):
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
            best_score=base_score,
            worst_score=low_score,
            input_ids=inputs["input_ids"][0],
            input_tokens=decode_tokens(self.tokenizer, inputs["input_ids"][0]),
            subject_range=token_range,
            answer=answer
        )

        if verbose:
            plot_trace_heatmap_sns(result)

        return result

    def _forward_pass(self, inputs, patching_states: list, answer_token: int, token_range: tuple,
                      noise_coeff: float = 0.1):
        """
        Performs a single causal trace

        :param inputs: the input tokens
        :param patching_states: the state we are going to patch
        :param answer_token: the token of the answer
        :param token_range: the token range of the subject
        :returns: the probability of the answer token
        """

        patch_spec = defaultdict(list)
        for t, l in patching_states:
            patch_spec[l].append(t)

        def _patching_rule(x: torch.Tensor, module):
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
                x[1:, start:end] += noise_coeff * torch.randn(x.shape[0] - 1, end - start, x.shape[2]).to(self.device)
                return x

            if module not in patch_spec:
                return x

            h = x[0] if isinstance(x, tuple) else x
            # restore the uncorrupted hidden state from the first forward pass
            for token in patch_spec[module]:
                h[1:, token] = h[0, token]
            return x

        with torch.no_grad():

            # build the patching layers dict
            layers = {**dict([("wte", self.layers["transformer"]["wte"])]), **dict(patching_states)}

            # hook the model to later extract the hidden_states
            hook_dict = TracingHookDict(model=self.model, layers=layers, out_f=_patching_rule)

            # forward pass
            out = self.model(**inputs)

            # remove the hook
            hook_dict.remove()

        # collect the model output probs for the original answer token
        probs = torch.softmax(out.logits[1:, -1, :], dim=1).mean(dim=0)[answer_token]

        return probs

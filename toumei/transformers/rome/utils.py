from collections import OrderedDict

import torch.nn as nn
import torch

import transformers

import matplotlib.pyplot as plt
import seaborn as sns


class TracingHook(object):
    """
    A simple network hook for wrapping the forward function of modules.
    It can store and edit the hidden state.
    """
    def __init__(self, name, module: nn.Module, out_f=None):
        self.name = name
        self.module = module
        self.output = None

        def create_hook(m, inputs, output):
            if out_f is not None:
                output = out_f(output, self.module)

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
    def __init__(self, model: nn.Module, layers: dict, out_f=None):
        super().__init__()

        self.model = model
        self.layers = layers

        for key, value in layers.items():
            self[key] = TracingHook(key, value, out_f)

    def remove(self):
        """
        Remove the TracingHookDict and all it's TracingHooks
        """
        for layer, hook in reversed(self.items()):
            hook.remove()


def nested_children(m: torch.nn.Module):
    """
    Returns the submodules of a model

    :param m: the model
    :returns: the submodules
    """
    children = dict(m.named_children())
    output = {}

    if "mlp" in children:
        return m
    if children == {}:
        return m
    else:
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output


def plot_trace_heatmap_sns(result):
    """
    Plots the tracing heat map using the given result

    :param result: the tracing result dict
    """
    differences = result["scores"]
    answer = result["answer"]

    labels = list(result["input_tokens"])

    ax = sns.heatmap(differences, yticklabels=labels, cmap="Blues",
                     cbar_kws={'label': f"p({str(answer).strip()})"},
                     vmin=result["worst_score"], vmax=result["best_score"])

    ax.set_title("Causal tracing result")
    ax.set_xlabel("Block")
    ax.set_ylabel("Token")

    ax.figure.subplots_adjust(left=0.23)

    plt.show()


"""
The following util functions are from the official ROME implementation
"""


def generate_inputs(prompts, tokenizer: transformers.PreTrainedTokenizer, device):
    """
    Creates a batch of input tokens using the prompts by padding them to maxlen

    :param prompts: the input prompts to tokenize
    :param tokenizer: the tokenizer
    :param device: the device for the generated tensors
    :returns: the created batch
    """

    # get all tokens
    tokens = [tokenizer.encode(p) for p in prompts]
    max_len = max(len(t) for t in tokens)

    # get the padding token id
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0

    # pad the input to maxlen
    input_ids = [[pad_id] * (max_len - len(t)) + t for t in tokens]
    position_ids = [[0] * (max_len - len(t)) + list(range(len(t))) for t in tokens]
    attention_mask = [[0] * (max_len - len(t)) + [1] * len(t) for t in tokens]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    """
    Decodes the given tokens to their corresponding string using the specified tokenizer

    :param tokenizer: the tokenizer for decoding
    :param token_array: a list of the tokens to decode
    :returns: a list with the decoded tokens
    """
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    """
    Find the token range (for the subject) by searching for a substring in the token_array

    :param tokenizer: the tokenizer
    :param token_array: a list of tokens
    :param substring: the substring for calculating the token range
    :returns: the token range
    """
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return tok_start, tok_end


def predict_from_input(model, inp):
    """
    Predict the next token given the input and the model

    :param model: the model for inference
    :param inp: the input
    :returns: the predicted token and it's probability
    """
    # forward-pass
    out = model(**inp)["logits"]

    # pass the logits through a softmax function to get probs
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p

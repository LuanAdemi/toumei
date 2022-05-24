from collections import OrderedDict

import torch.nn as nn
import torch

import transformers

import matplotlib.pyplot as plt


class TracingHook(object):
    """
    A simple network hook for wrapping the forward function of modules.
    It can store and edit the hidden state.
    """
    def __init__(self, name, module: nn.Module, out_f=None, out_f_params=None):
        self.name = name
        self.module = module
        self.output = None
        self.out_f_params = out_f_params

        def create_hook(m, inputs, output):
            if out_f is not None:
                output = out_f(output, self.module, *self.out_f_params)

            self.output = recursive_copy(output, clone=False, detach=False, retain_grad=False)

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
    def __init__(self, model: nn.Module, layers: dict, out_f=None, out_f_params=None):
        super().__init__()

        self.model = model
        self.layers = layers

        for key, value in layers.items():
            self[key] = TracingHook(key, value, out_f, out_f_params)

    def remove(self):
        """
        Remove the TracingHookDict and all it's TracingHooks
        """
        for layer, hook in reversed(self.items()):
            hook.remove()


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
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
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


def predict_token(mt, prompts, return_p=False):
    inp = generate_inputs(mt.tokenizer, prompts)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def nested_children(m: torch.nn.Module):
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


def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None):
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel("single restored layer within GPT-2-XL")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)

        plt.show()


def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."

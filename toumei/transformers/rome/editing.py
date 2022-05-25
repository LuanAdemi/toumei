from transformers import PreTrainedModel, PreTrainedTokenizer

from utils import *


class RankOneEditor(object):
    """
    Performs rank one model editing for an MLP block in the specified model.
    This forces the model to adapt a new knowledge-triple.
    """
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initiates a new RankOneEditor object used for performing rank one model editing on the given model

        :param model: the model we would like to edit
        :param tokenizer: a tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer

        self.device = self.model.device

        self.layers = list(nested_children(self.model))

    def to(self, device):
        """
        Sends the RankOneEditor to the specified device

        :param device: the new device
        """
        self.device = device
        self.model.to(device)

    def edit(self, block: int, knowledge_triple: tuple, inplace=True):
        """
        Edits a model to adapt the given knowledge-triple.

        :param block: The block the knowledge is located (perform causal tracing first)
        :param knowledge_triple: The new knowledge-triple consisting out of an object, a relation and a subject
        :param inplace: Perform the editing inplace (default) and not on a copy of the model (heavy memory usage)
        """

        module = self.layers[block]
        
        o, r, s = knowledge_triple


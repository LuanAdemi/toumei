import torch
from torch.utils.hooks import RemovableHandle
import torch.nn as nn

from toumei.objectives.module import Module


class Atom(Module):
    """
    The building block for creating objectives.
    An atom creates a forward_hook in the specified model and stores the activation as a hook endpoint in the object.
    """
    def __init__(self, unit: str, layer: str):
        super(Atom, self).__init__()
        # hook stuff
        self.forward_hook = None
        self.hook_endpoint = None

        # identifier stuff
        self.unit = unit
        self.layer = layer

        # model
        self.attached_model = None

    def __str__(self) -> str:
        return f"{self.name}(unit={self.key}, model={self.model.__class__.__name__})"

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        The call method for the atom.
        It executes the overwritten forward function and returns its output
        :param args: the arguments
        :param kwargs: the keyword arguments
        :return: the output of forward()
        """
        if self.activation is None:
            raise Exception("Could not fetch activation map. This usually means no forward pass has happened prior to "
                            "this call.")
        return self.forward(args, kwargs)

    def attach(self, model: nn.Module):
        """
        Attach to the specified model
        :param model: the model to attach to
        :return: nothing
        """
        for p in model.parameters():
            if p.requires_grad:
                raise Exception("The model has not been frozen. Please make sure you call objective.freezeModel() "
                                "before you attach an atom to it.")

        self.attached_model = model

        # define the hook generation
        def create_hook(name):
            def hook(m, i, o):
                # copy the output of the given layer
                self.hook_endpoint = o.squeeze(0)

            return hook

        # add the forward hook
        self.forward_hook = model.get_submodule(self.module).register_forward_hook(create_hook(self.key))

    def detach(self):
        """
        Detach from the model
        :return: nothing
        """
        if self.hook is not None:
            self.hook.remove()
            self.hook_endpoint = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        The function defining the function of an atom.
        This has to be overwritten by every child
        :param args: the arguments
        :param kwargs: the keyword arguments
        :return: a tensor
        """
        return torch.zeros(1)

    @property
    def key(self) -> str:
        """
        The key of the atom
        :return: the key
        """
        return self.unit

    @property
    def module(self) -> str:
        """
        The model module the atom is attached to
        :return: the module
        """
        return self.layer

    @property
    def hook(self) -> RemovableHandle:
        """
        Returns a RemovableHandle object for the hook of the atom
        :return: RemovableHandle object
        """
        return self.forward_hook

    @property
    def activation(self) -> torch.Tensor:
        """
        The tensor returned by the forward hook
        :return: activation tensor
        """
        return self.hook_endpoint

    @property
    def model(self) -> nn.Module:
        """
        The attached model
        :return: the model
        """
        return self.attached_model

    @property
    def name(self) -> str:
        return "Atom"

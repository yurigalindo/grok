"""Utilities for working with saturated transformer models."""

from typing import Union, Iterable
import torch
from torch.nn import Module, Parameter


def _identity(x):
    return x


def get_saturation(module, *args, **kwargs, extract_fn=_identity, infinity: float = 1e9):
    """Get the saturation scores of a model over a dataset that can be specified through `*args, **kwargs`.
    
    Note that `module` should return a [batch_size, feature_dim] vector, or `extract_fn` should format its output this way.
    """
    with saturate(module, infinity=infinity):
        hard_outputs = model(*args, **kwargs)
        hard_vector = extract_fn(hard_outputs)
    soft_outputs = model(*args, **kwargs)
    soft_vector = extract_fn(soft_outputs)
    
    inner_prods = torch.einsum("bi, bi -> b", hard_vector, soft_vector)
    soft_norm = soft_vector.norm(p=2, dim=1)
    hard_norm = hard_vector.norm(p=2, dim=1)
    # Add 1e-9 for numerical stability.
    return inner_prods / (soft_norm * hard_norm + 1e-9)


class saturate:

    """Context manager in which a model's parameters and forward pass will be saturated.
    We can pass in either a torch module or an iterable of torch parameters to saturated. Gradients will be disabled.
    
    Example usage:
    
    ```python
    with saturate(model):
        outputs = model(inputs)
    ```
    """

    def __init__(
        self,
        model_or_params: Union[Module, Iterable[Parameter]],
        infinity: float = 1000,
        no_grad: bool = True,
    ):
        if isinstance(model_or_params, Module):
            self.params = model_or_params.parameters()
        else:
            self.params = model_or_params

        self.params = list(self.params)
        self.infinity = infinity
        self.old_param_data = []
        self.no_grad = torch.no_grad() if no_grad else None

    def __enter__(self):
        if self.no_grad:
            self.no_grad.__enter__()
        for param in self.params:
            self.old_param_data.append(param.data)
            param.data = param.data.mul(self.infinity)

    def __exit__(self, type, value, traceback):
        for param, data in zip(self.params, self.old_param_data):
            param.data = data
        if self.no_grad:
            self.no_grad.__exit__(type, value, traceback)


class masked_saturate:
    def __init__(
        self,
        params: Iterable[Parameter],
        masks: Iterable[Parameter],
        infinity: float = 1000,
    ):
        self.params = list(params)
        self.masks = masks
        self.infinity = infinity
        self.old_param_data = []
        self.no_grad = torch.no_grad()

    def __enter__(self):
        self.no_grad.__enter__()
        for param, mask in zip(self.params, self.masks):
            self.old_param_data.append(param.data)
            param.data = torch.where(mask, param.data.mul(self.infinity), param.data)

    def __exit__(self, type, value, traceback):
        for param, data in zip(self.params, self.old_param_data):
            param.data = data
        self.no_grad.__exit__(type, value, traceback)

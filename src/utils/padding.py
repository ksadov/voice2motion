import torch
import numpy as np
from typing import Union, Tuple


def pad_to_multiple(
    tensor: Union[torch.Tensor, np.ndarray],
    multiple: int,
    dim: int = 1,
    min_length: int = 1,
) -> Tuple[Union[torch.Tensor, np.ndarray], int]:
    """
    Pads a tensor along a specified dimension so its length is divisible by multiple.
    Ensures the padded length is at least min_length.

    Args:
        tensor: Input tensor to pad
        multiple: The multiple to pad to
        dim: Dimension along which to pad
        min_length: Minimum length after padding

    Returns:
        Tuple of (padded_tensor, padding_amount)
    """
    size = max(tensor.shape[dim], 1)  # Ensure size is at least 1

    # First ensure we meet minimum length
    min_padding = max(0, min_length - size)

    # Then ensure we're divisible by multiple
    remaining = (size + min_padding) % multiple
    if remaining > 0:
        multiple_padding = multiple - remaining
    else:
        multiple_padding = 0

    total_padding = min_padding + multiple_padding

    if total_padding == 0:
        return tensor, 0

    if isinstance(tensor, np.ndarray):
        pad_width = [(0, 0)] * tensor.ndim
        pad_width[dim] = (0, total_padding)
        padded = np.pad(tensor, pad_width, mode="constant", constant_values=0)
    else:  # torch.Tensor
        pad_dims = [0] * (2 * tensor.ndim)
        pad_dims[-(2 * dim + 1)] = total_padding
        padded = torch.nn.functional.pad(tensor, pad_dims)

    return padded, total_padding


def unpad_tensor(
    tensor: Union[torch.Tensor, np.ndarray],
    padding_amount: int,
    dim: int = 1,
    min_length: int = 1,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Removes padding from a tensor along a specified dimension.
    Ensures output tensor maintains minimum length.

    Args:
        tensor: Padded input tensor
        padding_amount: Amount of padding to remove
        dim: Dimension from which to remove padding
        min_length: Minimum length to maintain

    Returns:
        Tensor with padding removed
    """
    if padding_amount == 0:
        return tensor

    if dim < 0:
        dim = tensor.ndim + dim

    # Calculate how much padding we can actually remove while maintaining min_length
    current_length = tensor.shape[dim]
    max_removable = max(0, current_length - min_length)
    padding_to_remove = min(padding_amount, max_removable)

    if padding_to_remove == 0:
        return tensor

    slicing = [slice(None)] * tensor.ndim
    slicing[dim] = slice(0, -padding_to_remove if padding_to_remove > 0 else None)

    return tensor[tuple(slicing)]

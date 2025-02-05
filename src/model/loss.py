import torch

from src.utils.constants import BLENDSHAPE_NAMES, HEAD_ANGLE_NAMES


def masked_difference_loss(
    preds: torch.Tensor,
    tgts: torch.Tensor,
    bad_frame_masks: torch.Tensor,
    tgt_lens: torch.Tensor,
    device: torch.device,
    calculate_shapekey_dict: bool = False,
    metric: str = "l2",
) -> torch.Tensor:
    """
    Calculate difference loss for padded sequences by masking out padded values.
    Args:
        preds (torch.Tensor): Predicted values tensor of shape (batch_size, seq_len, n_feats)
        tgts (torch.Tensor): Target values tensor of shape (batch_size, seq_len, n_feats)
        bad_frame_masks (torch.Tensor): Boolean tensor of shape (batch_size, seq_len), where True indicates a bad frame
        tgt_lens (torch.Tensor): Tensor of shape (batch_size,) containing the lengths of the target sequences
        device (torch.device): Device on which the tensors reside
        calculate_shapekey_dict (bool): Whether to calculate the shapekey dictionary
        metric (str): Metric to use for loss calculation
    Returns:
        torch.Tensor: Scalar tensor containing the masked MSE loss
    """
    batch_size, seq_len = tgts.shape[:2]
    seq_range = torch.arange(seq_len, device=tgts.device)
    seq_length_mask = seq_range[None, :] < tgt_lens[:, None]

    valid_mask = seq_length_mask & ~bad_frame_masks

    valid_mask = valid_mask.unsqueeze(-1).expand_as(tgts)

    if metric == "l2":
        diff = (preds - tgts) ** 2
    elif metric == "l1":
        diff = torch.abs(preds - tgts)
    elif metric == "l1_smooth":
        diff = torch.nn.SmoothL1Loss(reduction="none", beta=0.1)(preds, tgts)
    else:
        raise ValueError(f"Invalid metric: {metric}")

    if calculate_shapekey_dict:
        shapekey_dict = {}
        for i, shapekey in enumerate(BLENDSHAPE_NAMES + HEAD_ANGLE_NAMES):
            shapekey_dict[shapekey] = (
                (diff[:, :, i].sum() / valid_mask[:, :, 0].sum()).detach().cpu().item()
            )
    else:
        shapekey_dict = None

    masked_squared_diff = diff * valid_mask.float()

    total_loss = masked_squared_diff.sum()
    n_valid_elements = valid_mask.sum()

    return total_loss / n_valid_elements.clamp(min=1.0), shapekey_dict

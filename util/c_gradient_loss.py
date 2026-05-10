import torch
import torch.nn.functional as F


def c_gradient_loss_weighted(c_translated, c_target, edge_weight=10.0):
    """C-component gradient loss with edge-aware weighting."""
    dx_trans = c_translated[..., 1:] - c_translated[..., :-1]
    dy_trans = c_translated[..., 1:, :] - c_translated[..., :-1, :]
    dx_target = c_target[..., 1:] - c_target[..., :-1]
    dy_target = c_target[..., 1:, :] - c_target[..., :-1, :]
    edge_x = 1.0 + edge_weight * torch.abs(dx_target).mean(dim=1, keepdim=True).detach()
    edge_y = 1.0 + edge_weight * torch.abs(dy_target).mean(dim=1, keepdim=True).detach()
    loss_dx = (edge_x * torch.abs(dx_trans - dx_target)).mean()
    loss_dy = (edge_y * torch.abs(dy_trans - dy_target)).mean()
    return loss_dx + loss_dy

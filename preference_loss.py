import torch
import torch.nn.functional as F


def preference_loss(r_chosen, r_rejected):
    """Bradley-Terry pairwise preference loss."""
    return -F.logsigmoid(r_chosen - r_rejected).mean()

import torch

from mini_lm import MiniLM
from reward_model import RewardModel
from tokenizer import Tokenizer


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_policy(path):
    checkpoint = load_checkpoint(path)
    tokenizer = Tokenizer(checkpoint["vocab"])
    policy = MiniLM(tokenizer.vocab_size, **checkpoint["config"])
    policy.load_state_dict(checkpoint["state_dict"])
    return policy, tokenizer, checkpoint


def load_reward(path, tokenizer):
    checkpoint = load_checkpoint(path)
    if tokenizer.vocab != checkpoint["vocab"]:
        raise ValueError("SFT and reward checkpoints must use the same vocabulary")
    reward_model = RewardModel(tokenizer.vocab_size, **checkpoint["config"])
    reward_model.load_state_dict(checkpoint["state_dict"])
    reward_model.eval()
    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)
    return reward_model

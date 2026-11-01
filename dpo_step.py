import torch
import torch.nn.functional as F

from tokenizer import pad_sequences


def response_sequence(tokenizer, prompt, response):
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    response_ids = tokenizer.encode(response, add_eos=True)
    return prompt_ids + response_ids, len(prompt_ids)


def response_logprobs(policy, sequences, prompt_lengths, pad_id):
    """Sum log-probabilities for response tokens only."""
    device = next(policy.parameters()).device
    input_ids, mask = pad_sequences(sequences, pad_id)
    input_ids = input_ids.to(device)
    mask = mask.to(device)
    logits, _ = policy(input_ids)
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    targets = input_ids[:, 1:]
    token_logprobs = log_probs.gather(
        2, targets.unsqueeze(-1)
    ).squeeze(-1)
    positions = torch.arange(token_logprobs.size(1), device=device)
    response_mask = positions.unsqueeze(0) >= (
        torch.tensor(prompt_lengths, device=device).unsqueeze(1) - 1
    )
    return (token_logprobs * (mask[:, 1:] & response_mask)).sum(dim=1)


def dpo_loss(
    policy,
    reference,
    chosen_sequences,
    rejected_sequences,
    chosen_prompt_lengths,
    rejected_prompt_lengths,
    pad_id,
    beta=0.1,
):
    policy_chosen = response_logprobs(
        policy, chosen_sequences, chosen_prompt_lengths, pad_id
    )
    policy_rejected = response_logprobs(
        policy, rejected_sequences, rejected_prompt_lengths, pad_id
    )
    with torch.no_grad():
        reference_chosen = response_logprobs(
            reference, chosen_sequences, chosen_prompt_lengths, pad_id
        )
        reference_rejected = response_logprobs(
            reference, rejected_sequences, rejected_prompt_lengths, pad_id
        )

    logits = beta * (
        (policy_chosen - reference_chosen)
        - (policy_rejected - reference_rejected)
    )
    loss = -F.logsigmoid(logits).mean()
    accuracy = (logits > 0).float().mean()
    return loss, logits.detach(), accuracy.detach()

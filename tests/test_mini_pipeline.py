import unittest

import torch

from dpo_step import dpo_loss, response_sequence
from grpo_step import collect_grpo_groups, group_advantages, grpo_update
from mini_lm import MiniLM
from ppo_step import collect_rollouts, compute_gae, make_reference, ppo_update
from reward_model import RewardModel
from tokenizer import EOS_TOKEN, PAD_TOKEN, Tokenizer, pad_sequences


class MiniPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer().build_vocab(
            ["Explain phishing", "phishing is a cyber attack"]
        )

    def test_tokenizer_round_trip_and_padding(self):
        ids = self.tokenizer.encode("Explain phishing", add_bos=True)
        self.assertEqual(self.tokenizer.decode(ids), "explain phishing")
        padded, mask = pad_sequences([[1, 2], [3]], self.tokenizer.pad_id)
        self.assertEqual(tuple(padded.shape), (2, 2))
        self.assertEqual(mask.tolist(), [[True, True], [True, False]])
        self.assertIn(PAD_TOKEN, self.tokenizer.vocab)
        self.assertIn(EOS_TOKEN, self.tokenizer.vocab)

    def test_dpo_preference_margin_is_finite(self):
        torch.manual_seed(5)
        policy = MiniLM(self.tokenizer.vocab_size, embed=8, hidden=16)
        reference = make_reference(policy)
        chosen_sequence, chosen_length = response_sequence(
            self.tokenizer, "Explain phishing", "phishing is cyber attack"
        )
        rejected_sequence, rejected_length = response_sequence(
            self.tokenizer, "Explain phishing", "phishing is fish"
        )
        loss, logits, accuracy = dpo_loss(
            policy,
            reference,
            [chosen_sequence],
            [rejected_sequence],
            [chosen_length],
            [rejected_length],
            self.tokenizer.pad_id,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(logits.abs().item(), 0.0, places=5)
        self.assertGreaterEqual(accuracy.item(), 0.0)

    def test_gae_matches_hand_calculation(self):
        advantages, returns = compute_gae([1.0, 1.0], [0.5, 0.5], gamma=1.0, lam=1.0)
        self.assertAlmostEqual(advantages[0], 1.5)
        self.assertAlmostEqual(advantages[1], 0.5)
        self.assertAlmostEqual(returns[0], 2.0)
        self.assertAlmostEqual(returns[1], 1.0)

    def test_ppo_updates_policy_but_not_reference(self):
        torch.manual_seed(3)
        policy = MiniLM(self.tokenizer.vocab_size, embed=8, hidden=16)
        reward_model = RewardModel(self.tokenizer.vocab_size, embed=8, hidden=16)
        reference = make_reference(policy)
        reference_before = [parameter.detach().clone() for parameter in reference.parameters()]
        policy_before = [parameter.detach().clone() for parameter in policy.parameters()]
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)

        rollouts = collect_rollouts(
            policy,
            reference,
            reward_model,
            self.tokenizer,
            ["Explain phishing"],
            max_new_tokens=4,
            beta=0.02,
        )
        self.assertEqual(len(rollouts[0].response_ids), len(rollouts[0].old_logprobs))
        stats = ppo_update(policy, rollouts, optimizer, self.tokenizer, epochs=2)

        reference_after = list(reference.parameters())
        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(reference_before, reference_after)
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(policy_before, policy.parameters())
            )
        )
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in stats.values()))

    def test_grpo_group_advantages_and_update(self):
        torch.manual_seed(7)
        policy = MiniLM(self.tokenizer.vocab_size, embed=8, hidden=16)
        reward_model = RewardModel(self.tokenizer.vocab_size, embed=8, hidden=16)
        reference = make_reference(policy)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)
        groups = collect_grpo_groups(
            policy,
            reference,
            reward_model,
            self.tokenizer,
            ["Explain phishing"],
            group_size=2,
            max_new_tokens=3,
        )
        advantages = group_advantages(groups)
        self.assertAlmostEqual(advantages[0].mean().item(), 0.0, places=5)
        stats = grpo_update(policy, groups, optimizer, self.tokenizer, epochs=1)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in stats.values()))


if __name__ == "__main__":
    unittest.main()

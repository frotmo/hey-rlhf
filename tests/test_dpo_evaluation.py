import unittest

from evaluate_dpo_preferences import summarize_preference_records


class DPOPreferenceSummaryTests(unittest.TestCase):
    def test_summarize_preference_records_reports_accuracy_and_margin(self):
        base = {
            "chosen_implicit_reward": 0.3,
            "rejected_implicit_reward": 0.1,
            "policy_chosen_mean_logprob": -0.2,
            "policy_rejected_mean_logprob": -0.4,
            "chosen_token_length": 10,
            "rejected_token_length": 12,
            "truncated": False,
        }
        records = [
            {
                **base,
                "margin": 0.2,
                "correct": True,
                "dpo_loss": 0.5,
            },
            {
                **base,
                "margin": -0.1,
                "correct": False,
                "dpo_loss": 0.8,
                "truncated": True,
            },
        ]

        summary = summarize_preference_records(
            records,
            {"beta": 0.1},
        )

        self.assertEqual(summary["pair_count"], 2)
        self.assertEqual(summary["pairwise_accuracy"], 0.5)
        self.assertAlmostEqual(summary["margin"]["mean"], 0.05)
        self.assertEqual(summary["zero_margin_count"], 0)
        self.assertEqual(summary["non_truncated"]["pair_count"], 1)
        self.assertEqual(
            summary["non_truncated"]["pairwise_accuracy"],
            1.0,
        )
        self.assertEqual(summary["truncated_count"], 1)
        self.assertEqual(summary["truncated_rate"], 0.5)
        self.assertEqual(summary["metadata"]["beta"], 0.1)

    def test_summary_handles_zero_margin_ties(self):
        audit = {
            "chosen_implicit_reward": 0.0,
            "rejected_implicit_reward": 0.0,
            "policy_chosen_mean_logprob": -0.2,
            "policy_rejected_mean_logprob": -0.2,
            "chosen_token_length": 10,
            "rejected_token_length": 10,
            "truncated": False,
            "margin": 0.0,
            "correct": False,
            "dpo_loss": 0.693147,
        }

        summary = summarize_preference_records([audit])

        self.assertEqual(summary["pairwise_accuracy"], 0.0)
        self.assertEqual(
            summary["pairwise_accuracy_with_ties_half"],
            0.5,
        )
        self.assertIsNone(summary["pairwise_accuracy_non_tied"])
        self.assertEqual(summary["tie_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

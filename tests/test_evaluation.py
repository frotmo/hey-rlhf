import unittest

from evaluate_hf_policy import summarize_records


class EvaluationSummaryTests(unittest.TestCase):
    def test_summarize_records_reports_paired_comparison(self):
        records = [
            {
                "policy_reward": 2.0,
                "reference_reward": 1.0,
                "reward_delta": 1.0,
                "sampled_kl": 0.2,
                "policy_token_length": 4,
                "reference_token_length": 3,
            },
            {
                "policy_reward": 0.5,
                "reference_reward": 0.5,
                "reward_delta": 0.0,
                "sampled_kl": -0.1,
                "policy_token_length": 6,
                "reference_token_length": 8,
            },
        ]

        summary = summarize_records(records, {"prompts_file": "eval.jsonl"})

        self.assertEqual(summary["prompt_count"], 2)
        self.assertEqual(summary["policy"]["reward"]["mean"], 1.25)
        self.assertEqual(summary["comparison"]["policy_win_rate"], 0.5)
        self.assertEqual(summary["comparison"]["tie_rate"], 0.5)
        self.assertEqual(summary["comparison"]["policy_loss_rate"], 0.0)
        self.assertAlmostEqual(
            summary["comparison"]["sampled_kl"]["mean"],
            0.05,
        )
        self.assertAlmostEqual(
            summary["comparison"]["sampled_kl_abs_mean"],
            0.15,
        )
        self.assertEqual(
            summary["metadata"]["prompts_file"],
            "eval.jsonl",
        )


if __name__ == "__main__":
    unittest.main()

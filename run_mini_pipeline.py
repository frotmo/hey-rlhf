import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=("ppo", "grpo", "dpo", "all"),
        default="ppo",
    )
    args = parser.parse_args()
    runners = {
        "ppo": ["train_sft.py", "train_reward_model.py", "train_ppo.py"],
        "grpo": ["train_sft.py", "train_reward_model.py", "train_grpo.py"],
        "dpo": ["train_sft.py", "train_dpo.py"],
        "all": [
            "train_sft.py",
            "train_reward_model.py",
            "train_ppo.py",
            "train_grpo.py",
            "train_dpo.py",
        ],
    }
    for script in runners[args.algorithm]:
        subprocess.run([sys.executable, script], check=True)


if __name__ == "__main__":
    main()

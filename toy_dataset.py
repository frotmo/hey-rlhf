"""Small cybersecurity examples for a CPU-friendly teaching pipeline."""

data = [
    (
        "Explain phishing",
        "Phishing is a cyber attack that tricks users into revealing information",
    ),
    (
        "What is malware",
        "Malware is malicious software designed to harm systems",
    ),
    (
        "What is encryption",
        "Encryption converts data into a secure format",
    ),
    (
        "Why update software",
        "Updates fix security holes and reduce risk",
    ),
    (
        "What is a strong password",
        "A strong password is long unique and hard to guess",
    ),
    (
        "What is a firewall",
        "A firewall filters network traffic",
    ),
]


preferences = [
    {
        "prompt": "Explain phishing",
        "chosen": "Phishing is a cyber attack that tricks users",
        "rejected": "Phishing is a type of fish",
    },
    {
        "prompt": "What is malware",
        "chosen": "Malware is malicious software that harms systems",
        "rejected": "Malware is a computer mouse",
    },
    {
        "prompt": "What is encryption",
        "chosen": "Encryption converts data into a secure format",
        "rejected": "Encryption is a game console",
    },
    {
        "prompt": "Why update software",
        "chosen": "Updates fix security holes and reduce risk",
        "rejected": "Updates make every program slower",
    },
    {
        "prompt": "What is a strong password",
        "chosen": "A strong password is long unique and hard to guess",
        "rejected": "A strong password is easy to remember",
    },
    {
        "prompt": "What is a firewall",
        "chosen": "A firewall filters network traffic",
        "rejected": "A firewall is a physical wall",
    },
    {
        "prompt": "What is social engineering",
        "chosen": "Social engineering manipulates people into giving access",
        "rejected": "Social engineering builds bridges",
    },
    {
        "prompt": "How to reduce risk",
        "chosen": "Use updates strong passwords and backups",
        "rejected": "Use one password for every account",
    },
]


prompts = [prompt for prompt, _ in data] + [item["prompt"] for item in preferences]
prompts = list(dict.fromkeys(prompts))

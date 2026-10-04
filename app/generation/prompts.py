from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def load_prompt_template(version: str = "qa_v1") -> str:
    """Load prompt template file by version name."""
    prompt_file = PROMPTS_DIR / f"{version}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template version '{version}' not found at {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()

"""LLM-assisted data dictionary drafting (capstone Step 9: Use of Generative AI).

Profiles every raw Olist table (dtypes, missingness, ranges, cardinality, sample
values) and asks Claude to draft a complete data dictionary in markdown. The
prompt and the raw model output are both saved under docs/llm/ so the final,
human-verified docs/data_dictionary.md can be diffed against the draft.

Requires ANTHROPIC_API_KEY in .env (never committed). Idempotent: skips the API
call if the raw draft already exists; pass --force to re-generate.
"""

import sys
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from src.data import RAW_TABLES, load_table

REPO_ROOT = Path(__file__).resolve().parents[1]
LLM_DOCS_DIR = REPO_ROOT / "docs" / "llm"
PROMPT_PATH = LLM_DOCS_DIR / "data_dictionary_prompt.md"
RAW_PATH = LLM_DOCS_DIR / "data_dictionary_raw.md"

MODEL = "claude-opus-4-8"

# Free-text fields whose sample values shouldn't be quoted into prompts/docs
SKIP_EXAMPLES = {"review_comment_title", "review_comment_message"}


def profile_table(name: str, df: pd.DataFrame) -> str:
    lines = [f"### {RAW_TABLES[name]}  ({len(df)} rows)", ""]
    lines.append("| column | dtype | missing % | n_unique | range / examples |")
    lines.append("|---|---|---|---|---|")
    for col in df.columns:
        s = df[col]
        missing = f"{100 * s.isna().mean():.1f}"
        nunique = s.nunique()
        if col in SKIP_EXAMPLES:
            detail = "(free text - examples omitted)"
        elif pd.api.types.is_numeric_dtype(s):
            detail = f"min {s.min():g}, max {s.max():g}, median {s.median():g}"
        else:
            examples = [str(v)[:40] for v in s.dropna().unique()[:4]]
            detail = ", ".join(examples)
        lines.append(f"| {col} | {s.dtype} | {missing} | {nunique} | {detail} |")
    lines.append("")
    return "\n".join(lines)


def build_prompt() -> str:
    profiles = []
    for name in RAW_TABLES:
        try:
            profiles.append(profile_table(name, load_table(name)))
        except FileNotFoundError:
            profiles.append(f"### {RAW_TABLES[name]}\n\n(file not present locally)\n")

    return f"""You are documenting the Olist Brazilian eCommerce dataset
(https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) for a machine
learning project's data dictionary.

Below are schema profiles for each raw CSV, computed directly from the data.
Write a complete data dictionary in markdown: one section per file, with a table
containing exactly these columns: Column | Type | Missing % | Description | Units / allowed values.

Rules:
- Copy Type and Missing % verbatim from the profiles - do not restate or round them.
- Descriptions must be concise (one sentence) and specific to Olist's marketplace
  context (e.g. customer_id is per-order; customer_unique_id identifies the person).
- Monetary columns are in Brazilian reais (BRL); dimensions in cm; weight in grams.
- For any column where you are not confident about the meaning, append "(verify)"
  to the description rather than guessing confidently.
- Start with a short intro paragraph explaining the dataset's relational structure,
  and end with a note that timestamps are local Brazilian time.
- Output only the markdown document, no preamble.

{chr(10).join(profiles)}"""


def draft_dictionary(force: bool = False) -> None:
    LLM_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt()
    PROMPT_PATH.write_text(prompt, encoding="utf-8")
    print(f"wrote {PROMPT_PATH}")

    if RAW_PATH.exists() and not force:
        print(f"{RAW_PATH} already exists - skipping API call (use --force to re-draft)")
        return

    load_dotenv(REPO_ROOT / ".env")
    client = Anthropic()
    with client.messages.stream(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = stream.get_final_message()

    text = next(b.text for b in message.content if b.type == "text")
    RAW_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {RAW_PATH}")
    print(
        f"usage: {message.usage.input_tokens} in / {message.usage.output_tokens} out "
        f"({MODEL})"
    )


if __name__ == "__main__":
    draft_dictionary(force="--force" in sys.argv)

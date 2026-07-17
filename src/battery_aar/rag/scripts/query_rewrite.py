"""Query rewriting: turn the FeatureScientist task prompt into retrieval queries.

The FeatureScientist prompt (workflows/role_prompts.py) is a long agent
instruction full of schema and bookkeeping text -- a poor search string for
both BM25 and embedding retrieval. This module calls the workflow LLM
(configured via .env, OPEN_BATTERY_AGENTS_MODEL et al.; distinct from the
fixed embedding model) to rewrite that prompt into a handful of short,
domain-focused queries suitable for the RAG knowledge base. The queries feed
``hybrid_search.retrieve`` in the prompt-augmentation stage.

Usage:
    python -m battery_aar.rag.scripts.query_rewrite [--n 4]
        [--dataset-profile FILE.json] [--feature-probe FILE.json]
        [--show-original]

Example:
    python -m battery_aar.rag.scripts.query_rewrite --n 4 --show-original
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from battery_aar.agents.llm_client import load_llm_client_config
from battery_aar.workflows.role_prompts import feature_scientist_prompt

DEFAULT_N_QUERIES = 4
REWRITE_TEMPERATURE = 0.3

REWRITER_SYSTEM_PROMPT = (
    "You are a query rewriter for a battery-science retrieval system. Given an "
    "agent task prompt, you extract the underlying scientific information needs "
    "and express them as short search queries. You return only a JSON array of "
    "strings, nothing else."
)


def rewriter_user_prompt(original_prompt: str, n_queries: int) -> str:
    return f"""Rewrite the agent task prompt below into {n_queries} retrieval queries for a
knowledge base of electrochemistry textbooks and battery aging papers.

Rules:
- Each query is one short phrase (roughly 5-15 words) about battery science
  concepts the agent needs to ground its decisions in.
- Cover distinct aspects of the task (degradation physics, measurable
  early-cycle signals, feature/predictor constructions, protocol effects);
  do not produce near-duplicates.
- Use domain vocabulary likely to appear in textbooks and papers.
- Ignore output-format instructions, JSON schemas, agent bookkeeping, and
  dataset column names.

Return a JSON array of exactly {n_queries} strings.

Agent task prompt:
---
{original_prompt}
---"""


def _chat(system: str, user: str, temperature: float) -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    from openai import OpenAI

    config = load_llm_client_config()
    if not config.api_key:
        raise RuntimeError(
            "no API key configured; set OPEN_BATTERY_AGENTS_API_KEY or "
            "STANFORD_AI_API_KEY (see .env.example)"
        )
    kwargs = {"api_key": config.api_key, "base_url": config.base_url}
    if config.default_headers:
        kwargs["default_headers"] = config.default_headers
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def _extract_string_array(text: str) -> list[str]:
    """Parse a JSON array of strings from an LLM reply, tolerating prose/fences."""
    candidates = [text.strip()]
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and all(isinstance(q, str) and q.strip() for q in parsed):
            deduped = list(dict.fromkeys(q.strip() for q in parsed))
            if deduped:
                return deduped
    raise ValueError(f"LLM reply is not a JSON array of strings:\n{text}")


def rewrite_queries(
    original_prompt: str,
    n_queries: int = DEFAULT_N_QUERIES,
) -> list[str]:
    """Rewrite an agent task prompt into retrieval queries via the .env LLM."""
    if n_queries < 1:
        raise ValueError("n_queries must be >= 1")
    reply = _chat(
        REWRITER_SYSTEM_PROMPT,
        rewriter_user_prompt(original_prompt, n_queries),
        REWRITE_TEMPERATURE,
    )
    return _extract_string_array(reply)


def feature_scientist_queries(
    dataset_profile: dict | None = None,
    feature_probe: dict | None = None,
    n_queries: int = DEFAULT_N_QUERIES,
) -> tuple[str, list[str]]:
    """Build the FeatureScientist prompt and rewrite it into retrieval queries.

    Returns (original_prompt, queries).
    """
    original = feature_scientist_prompt(dataset_profile or {}, feature_probe or {})
    return original, rewrite_queries(original, n_queries=n_queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=DEFAULT_N_QUERIES)
    parser.add_argument("--dataset-profile", type=Path, default=None,
                        help="JSON file with the dataset profile fed to the prompt")
    parser.add_argument("--feature-probe", type=Path, default=None,
                        help="JSON file with the feature probe fed to the prompt")
    parser.add_argument("--show-original", action="store_true",
                        help="also print the original FeatureScientist prompt")
    args = parser.parse_args()

    profile = json.loads(args.dataset_profile.read_text()) if args.dataset_profile else {}
    probe = json.loads(args.feature_probe.read_text()) if args.feature_probe else {}
    original, queries = feature_scientist_queries(profile, probe, n_queries=args.n)

    if args.show_original:
        print("--- original FeatureScientist prompt ---")
        print(original)
        print("--- rewritten retrieval queries ---")
    for i, query in enumerate(queries, 1):
        print(f"[{i}] {query}")


if __name__ == "__main__":
    main()

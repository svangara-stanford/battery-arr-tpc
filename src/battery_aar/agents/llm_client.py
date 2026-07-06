from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentResponse:
    code: str
    prompt: str
    response_text: str


@dataclass
class LLMClientConfig:
    api_key: str | None
    base_url: str | None
    model: str
    alias: str | None
    alias_header: str | None
    extra_headers: dict[str, str]

    @property
    def default_headers(self) -> dict[str, str]:
        headers = dict(self.extra_headers)
        if self.alias and self.alias_header:
            headers[self.alias_header] = self.alias
        return headers

    def safe_summary(self) -> dict[str, Any]:
        return {
            "api_key_configured": bool(self.api_key),
            "base_url_configured": bool(self.base_url),
            "model": self.model,
            "alias_configured": bool(self.alias),
            "alias_header_configured": bool(self.alias_header),
            "extra_headers_configured": bool(self.extra_headers),
            "default_header_names": sorted(self.default_headers),
        }


def load_llm_client_config(model: str | None = None) -> LLMClientConfig:
    api_key = (
        os.getenv("OPEN_BATTERY_AGENTS_API_KEY")
        or os.getenv("STANFORD_AI_API_KEY")
        or os.getenv("STANFORD_AI_PLAYGROUND_API_KEY")
    )
    base_url = (
        os.getenv("OPEN_BATTERY_AGENTS_BASE_URL")
        or os.getenv("STANFORD_AI_BASE_URL")
        or os.getenv("STANFORD_AI_PLAYGROUND_BASE_URL")
    )
    alias = os.getenv("OPEN_BATTERY_AGENTS_ALIAS") or os.getenv("STANFORD_AI_ALIAS")
    alias_header = os.getenv("OPEN_BATTERY_AGENTS_ALIAS_HEADER")
    extra_headers = _load_extra_headers()
    return LLMClientConfig(
        api_key=api_key,
        base_url=base_url,
        model=model or os.getenv("OPEN_BATTERY_AGENTS_MODEL") or "gpt-4o-mini",
        alias=alias,
        alias_header=alias_header,
        extra_headers=extra_headers,
    )


def _load_extra_headers() -> dict[str, str]:
    raw = os.getenv("OPEN_BATTERY_AGENTS_EXTRA_HEADERS_JSON")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("OPEN_BATTERY_AGENTS_EXTRA_HEADERS_JSON must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("OPEN_BATTERY_AGENTS_EXTRA_HEADERS_JSON must decode to a JSON object")
    headers: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            raise ValueError("OPEN_BATTERY_AGENTS_EXTRA_HEADERS_JSON header names must be strings")
        if value is None:
            continue
        headers[key] = str(value)
    return headers


class OfflineHeuristicAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def propose(self, prompt: str, iteration: int) -> AgentResponse:
        code = _ridge_candidate_code() if iteration % 2 == 0 else _forest_candidate_code()
        return AgentResponse(code=code, prompt=prompt, response_text="offline heuristic candidate")


class OpenAICompatibleAgent:
    def __init__(self, agent_id: str, model: str | None = None):
        self.agent_id = agent_id
        self.config = load_llm_client_config(model=model)
        self.api_key = self.config.api_key
        self.base_url = self.config.base_url
        self.model = self.config.model
        if not self.config.api_key:
            raise RuntimeError("No Open Battery Agents API key found")

    def propose(self, prompt: str, iteration: int) -> AgentResponse:
        from openai import OpenAI

        default_headers = self.config.default_headers
        client_kwargs: dict[str, Any] = {"api_key": self.config.api_key, "base_url": self.config.base_url}
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return only Python code for one candidate file."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        text = response.choices[0].message.content or ""
        code = _strip_code_fences(text)
        return AgentResponse(code=code, prompt=prompt, response_text=text)


def make_agent(agent_id: str, offline: bool, model: str | None = None):
    if offline:
        return OfflineHeuristicAgent(agent_id)
    config = load_llm_client_config(model=model)
    if not config.api_key:
        return OfflineHeuristicAgent(agent_id)
    return OpenAICompatibleAgent(agent_id, model=model)


def llm_startup_summary(model: str | None = None) -> dict[str, Any]:
    return load_llm_client_config(model=model).safe_summary()


_CODE_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)(?:\n```|\Z)", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    # Models often preface code with prose ("Here is the repaired code:").
    # Extract the first fenced block wherever it appears; fall back to the
    # raw text when the reply contains no fence at all.
    match = _CODE_FENCE_RE.search(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _ridge_candidate_code() -> str:
    return r'''
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def _features(meta, cycles, max_cycle):
    rows = []
    for rid, grp in cycles[cycles["cycle_index"] <= max_cycle].groupby("row_id"):
        g = grp.sort_values("cycle_index")
        q = g["discharge_capacity"].to_numpy(float)
        c = g["cycle_index"].to_numpy(float)
        if len(q) < 3:
            feats = [np.nan] * 5
        else:
            slope, intercept = np.polyfit(c, q, 1)
            feats = [q[0], q[min(1, len(q)-1)], q[-1], slope, np.nanmax(q) - q[min(1, len(q)-1)]]
        rows.append([rid] + feats)
    return pd.DataFrame(rows, columns=["row_id", "q0", "q2", "qN", "slope", "max_delta"]).fillna(0.0)

def fit(train_metadata, train_cycle_summary, train_labels, config):
    max_cycle = int(config.get("max_cycle", 100))
    X = _features(train_metadata, train_cycle_summary, max_cycle)
    y = train_labels.set_index("row_id").loc[X["row_id"], "y"].to_numpy(float)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X.drop(columns=["row_id"]), y)
    return {"model": model, "max_cycle": max_cycle}

def predict(model, test_metadata, test_cycle_summary, config):
    X = _features(test_metadata, test_cycle_summary, int(model["max_cycle"]))
    y_pred = model["model"].predict(X.drop(columns=["row_id"]))
    return pd.DataFrame({"row_id": X["row_id"], "y_pred": y_pred})
'''


def _forest_candidate_code() -> str:
    return r'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def _features(meta, cycles, max_cycle):
    rows = []
    for rid, grp in cycles[cycles["cycle_index"] <= max_cycle].groupby("row_id"):
        q = grp.sort_values("cycle_index")["discharge_capacity"].to_numpy(float)
        if len(q) == 0:
            feats = [0.0] * 7
        else:
            diffs = np.diff(q) if len(q) > 1 else np.array([0.0])
            feats = [q[0], q[min(1, len(q)-1)], q[-1], np.mean(q), np.std(q), np.min(diffs), np.max(diffs)]
        rows.append([rid] + feats)
    return pd.DataFrame(rows, columns=["row_id", "q0", "q2", "qN", "q_mean", "q_std", "min_dq", "max_dq"]).fillna(0.0)

def fit(train_metadata, train_cycle_summary, train_labels, config):
    X = _features(train_metadata, train_cycle_summary, int(config.get("max_cycle", 100)))
    y = train_labels.set_index("row_id").loc[X["row_id"], "y"].to_numpy(float)
    model = RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=17)
    model.fit(X.drop(columns=["row_id"]), y)
    return {"model": model, "max_cycle": int(config.get("max_cycle", 100))}

def predict(model, test_metadata, test_cycle_summary, config):
    X = _features(test_metadata, test_cycle_summary, model["max_cycle"])
    return pd.DataFrame({"row_id": X["row_id"], "y_pred": model["model"].predict(X.drop(columns=["row_id"]))})
'''

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AgentResponse:
    code: str
    prompt: str
    response_text: str


class OfflineHeuristicAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def propose(self, prompt: str, iteration: int) -> AgentResponse:
        code = _ridge_candidate_code() if iteration % 2 == 0 else _forest_candidate_code()
        return AgentResponse(code=code, prompt=prompt, response_text="offline heuristic candidate")


class OpenAICompatibleAgent:
    def __init__(self, agent_id: str, model: str | None = None):
        self.agent_id = agent_id
        self.api_key = os.getenv("OPEN_BATTERY_AGENTS_API_KEY") or os.getenv("STANFORD_AI_PLAYGROUND_API_KEY")
        self.base_url = os.getenv("OPEN_BATTERY_AGENTS_BASE_URL") or os.getenv("STANFORD_AI_PLAYGROUND_BASE_URL")
        self.model = model or os.getenv("OPEN_BATTERY_AGENTS_MODEL") or "gpt-4o-mini"
        if not self.api_key:
            raise RuntimeError("No Open Battery Agents API key found")

    def propose(self, prompt: str, iteration: int) -> AgentResponse:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
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
    api_key = os.getenv("OPEN_BATTERY_AGENTS_API_KEY") or os.getenv("STANFORD_AI_PLAYGROUND_API_KEY")
    if not api_key:
        return OfflineHeuristicAgent(agent_id)
    return OpenAICompatibleAgent(agent_id, model=model)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
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

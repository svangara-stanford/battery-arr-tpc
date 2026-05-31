from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem

PAPER_TOP_PROTOCOLS = (
    (4.8, 5.2, 5.2, 4.160),
    (5.2, 5.2, 4.8, 4.160),
    (4.4, 5.6, 5.2, 4.252),
)


@dataclass
class BayesGapConfig:
    gamma: float = 1.0
    likelihood_std: float = 164.0
    init_beta: float = 5.0
    epsilon: float = 0.5
    standardization_mean: float = 947.0
    standardization_std: float = 164.0
    bsize: int = 48
    seed: int = 0


class BayesGap:
    def __init__(self, policies: pd.DataFrame, config: BayesGapConfig | None = None):
        self.config = config or BayesGapConfig()
        policies = policies.loc[:, ["C1", "C2", "C3", "C4"]].copy()
        rng = np.random.default_rng(self.config.seed)
        order = rng.permutation(len(policies))
        self.policies = policies.iloc[order].reset_index(drop=True)
        self.param_space = self.policies[["C1", "C2", "C3"]].to_numpy(float)
        self.num_arms = self.param_space.shape[0]
        self.X = self._design_matrix()
        self.num_dims = self.X.shape[1]
        self.eta = self.config.standardization_std

    @classmethod
    def from_csv(cls, policies_csv: str | Path, config: BayesGapConfig | None = None) -> "BayesGap":
        return cls(pd.read_csv(policies_csv), config)

    def _design_matrix(self) -> np.ndarray:
        mapper = Nystroem(gamma=self.config.gamma, n_components=self.num_arms, random_state=1)
        return mapper.fit_transform(self.param_space)

    def posterior_theta(self, X_t: np.ndarray | None = None, Y_t: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        prior_mean = np.zeros(self.num_dims)
        if X_t is None or Y_t is None or len(X_t) == 0:
            return prior_mean, self.eta * self.eta * np.identity(self.num_dims)
        sigma = self.config.likelihood_std
        posterior_covar = np.linalg.inv(X_t.T @ X_t / (sigma * sigma) + np.identity(self.num_dims) / (self.eta * self.eta))
        posterior_mean = np.linalg.multi_dot((posterior_covar, X_t.T, Y_t)) / (sigma * sigma)
        return np.squeeze(posterior_mean), posterior_covar

    def get_posterior_bounds(self, beta: float, X_t: np.ndarray | None = None, Y_t: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        mean, covar = self.posterior_theta(X_t, Y_t)
        marginal_mean = self.X @ mean
        marginal_var = np.sum((self.X @ covar) * self.X, axis=1)
        upper = np.around(marginal_mean + beta * np.sqrt(marginal_var), 4)
        lower = np.around(marginal_mean - beta * np.sqrt(marginal_var), 4)
        return upper, lower

    def _arm_indices_for_predictions(self, early_pred: pd.DataFrame) -> list[int]:
        indices: list[int] = []
        rounded_space = np.round(self.param_space, 3)
        for _, row in early_pred.iterrows():
            target = np.round(row[["C1", "C2", "C3"]].to_numpy(float), 3)
            matches = np.where(np.all(np.isclose(rounded_space, target, atol=1e-3), axis=1))[0]
            if matches.size == 0:
                raise ValueError(f"prediction policy not found in policy space: {target}")
            indices.append(int(matches[0]))
        return indices

    def run_round(
        self,
        round_idx: int,
        out_dir: str | Path,
        previous_state_path: str | Path | None = None,
        previous_predictions_csv: str | Path | None = None,
    ) -> dict[str, object]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        beta = self.config.init_beta
        proposal_arms: list[int] = []
        proposal_gaps: list[float] = []
        X_t: list[np.ndarray] = []
        Y_t: list[np.ndarray] = []
        best_arm_params: np.ndarray | None = None

        if round_idx == 0:
            upper, lower = self.get_posterior_bounds(beta)
        else:
            if previous_state_path is None or previous_predictions_csv is None:
                raise ValueError("round_idx > 0 requires previous state and previous predictions")
            with Path(previous_state_path).open("rb") as handle:
                proposal_arms, proposal_gaps, X_t, Y_t, beta = pickle.load(handle)
            beta = np.around(beta * self.config.epsilon, 4)
            early_pred = read_early_predictions(previous_predictions_csv)
            arm_idx = self._arm_indices_for_predictions(early_pred)
            rewards = early_pred["Prediction"].to_numpy(float).reshape(-1, 1)
            rewards = rewards - self.config.standardization_mean
            X_t.append(self.X[arm_idx])
            Y_t.append(rewards)
            upper, lower = self.get_posterior_bounds(beta, np.vstack(X_t), np.vstack(Y_t))
            previous_proposal = proposal_arms[round_idx - 1]
            proposal_gaps.append(float(np.max(np.delete(upper, previous_proposal)) - lower[previous_proposal]))
            best_arm = proposal_arms[int(np.argmin(np.asarray(proposal_gaps)))]
            best_arm_params = self.param_space[best_arm]

        batch_arms: list[int] = []
        candidate_arms = list(range(self.num_arms))
        for batch_elem in range(min(self.config.bsize, self.num_arms)):
            j_big, _ = _find_J_t(candidate_arms, upper, lower, self.num_arms)
            j_small = _find_j_t(candidate_arms, j_big, upper, self.num_arms)
            a_t = j_big if (upper[j_big] - lower[j_big]) >= (upper[j_small] - lower[j_small]) else j_small
            if batch_elem == 0:
                proposal_arms.append(j_big)
            batch_arms.append(a_t)
            candidate_arms.remove(a_t)

        state_path = out / f"round_{round_idx}_state.pkl"
        with state_path.open("wb") as handle:
            pickle.dump([proposal_arms, proposal_gaps, X_t, Y_t, beta], handle)

        selected = self.policies.iloc[batch_arms].reset_index(drop=True)
        next_batch_path = out / f"round_{round_idx}_next_batch.csv"
        selected.to_csv(next_batch_path, index=False)
        bounds = self.policies.copy()
        bounds["upper_bound"] = upper + self.config.standardization_mean
        bounds["lower_bound"] = lower + self.config.standardization_mean
        bounds["mean_bound"] = (bounds["upper_bound"] + bounds["lower_bound"]) / 2
        bounds["posterior_half_width"] = (bounds["upper_bound"] - bounds["lower_bound"]) / 2
        bounds_path = out / f"round_{round_idx}_bounds.csv"
        bounds.to_csv(bounds_path, index=False)
        bounds_pickle_path = out / f"round_{round_idx}_bounds.pkl"
        with bounds_pickle_path.open("wb") as handle:
            pickle.dump([self.param_space, bounds["upper_bound"].to_numpy(), bounds["lower_bound"].to_numpy(), bounds["mean_bound"].to_numpy()], handle)

        best_summary = None
        if best_arm_params is not None:
            best_summary = {"C1": float(best_arm_params[0]), "C2": float(best_arm_params[1]), "C3": float(best_arm_params[2])}
        return {
            "round_idx": round_idx,
            "state_path": state_path,
            "next_batch": selected,
            "next_batch_path": next_batch_path,
            "bounds": bounds,
            "bounds_path": bounds_path,
            "bounds_pickle_path": bounds_pickle_path,
            "best_arm": best_summary,
        }


def _find_J_t(candidate_arms: list[int], upper: np.ndarray, lower: np.ndarray, num_arms: int) -> tuple[int, float]:
    b_values: list[float] = []
    for k in range(num_arms):
        if k in candidate_arms:
            b_values.append(float(np.max(np.delete(upper, k)) - lower[k]))
        else:
            b_values.append(float("inf"))
    arr = np.asarray(b_values)
    return int(np.argmin(arr)), float(np.min(arr))


def _find_j_t(candidate_arms: list[int], preselected_arm: int, upper: np.ndarray, num_arms: int) -> int:
    values = np.full(num_arms, -np.inf)
    for k in candidate_arms:
        if k != preselected_arm:
            values[k] = upper[k]
    return int(np.argmax(values))


def read_early_predictions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    if set(["C1", "C2", "C3", "C4", "Prediction"]).issubset(df.columns):
        return df.loc[:, ["C1", "C2", "C3", "C4", "Prediction"]].dropna()
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 5:
        raise ValueError(f"early prediction CSV must have at least five columns: {path}")
    df = df.iloc[:, :5]
    df.columns = ["C1", "C2", "C3", "C4", "Prediction"]
    return df.dropna()


def run_closed_loop(
    policies_csv: str | Path,
    out_dir: str | Path,
    prediction_csvs: list[str | Path] | None = None,
    config: BayesGapConfig | None = None,
) -> list[dict[str, object]]:
    agent = BayesGap.from_csv(policies_csv, config)
    out = Path(out_dir)
    results = [agent.run_round(0, out)]
    results[0]["consumed_prediction_file"] = None
    results[0]["input_rows"] = 0
    results[0]["round_description"] = "generate/select initial batch if needed"
    previous_state = results[0]["state_path"]
    for idx, pred_csv in enumerate(prediction_csvs or [], start=1):
        early_pred = read_early_predictions(pred_csv)
        round_result = agent.run_round(idx, out, previous_state_path=previous_state, previous_predictions_csv=pred_csv)
        round_result["consumed_prediction_file"] = str(pred_csv)
        round_result["input_rows"] = int(len(early_pred))
        round_result["round_description"] = f"consume predictions from {Path(pred_csv).name}"
        results.append(round_result)
        previous_state = results[-1]["state_path"]
    if results:
        final_ranking = write_final_posterior_ranking(results[-1]["bounds"], out / "final_posterior_ranking.csv")
        paper_check = write_paper_top_protocol_check(final_ranking, out / "final_paper_top_protocol_check.csv")
        results[-1]["final_posterior_ranking"] = final_ranking
        results[-1]["final_posterior_ranking_path"] = out / "final_posterior_ranking.csv"
        results[-1]["paper_top_protocol_check"] = paper_check
        results[-1]["paper_top_protocol_check_path"] = out / "final_paper_top_protocol_check.csv"
    return results


def write_final_posterior_ranking(bounds: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    ranking = bounds.copy()
    sort_col = "mean_bound" if "mean_bound" in ranking.columns else "posterior_mean"
    ranking = ranking.sort_values(sort_col, ascending=False).reset_index(drop=True)
    ranking.insert(0, "final_posterior_rank", np.arange(1, len(ranking) + 1))
    ranking = ranking.rename(
        columns={
            "mean_bound": "final_posterior_mean",
            "upper_bound": "final_posterior_upper",
            "lower_bound": "final_posterior_lower",
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(path, index=False)
    return ranking


def write_paper_top_protocol_check(ranking: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top3 = ranking.head(3)[["C1", "C2", "C3", "C4"]].round(3).to_numpy(float).tolist()
    paper_top = [list(np.round(proto, 3)) for proto in PAPER_TOP_PROTOCOLS]
    final_top_three_exact_match = top3 == paper_top
    for c1, c2, c3, c4 in PAPER_TOP_PROTOCOLS:
        match = ranking[
            np.isclose(ranking["C1"], c1, atol=1e-3)
            & np.isclose(ranking["C2"], c2, atol=1e-3)
            & np.isclose(ranking["C3"], c3, atol=1e-3)
            & np.isclose(ranking["C4"], c4, atol=1e-3)
        ]
        if match.empty:
            rows.append(
                {
                    "protocol": f"{c1:.1f}C-{c2:.1f}C-{c3:.1f}C-{c4:.3f}C",
                    "C1": c1,
                    "C2": c2,
                    "C3": c3,
                    "C4": c4,
                    "exists_in_policy_space": False,
                    "final_posterior_rank": np.nan,
                    "final_posterior_mean": np.nan,
                    "final_posterior_uncertainty": np.nan,
                    "final_top_three_exact_match": final_top_three_exact_match,
                }
            )
            continue
        row = match.iloc[0]
        rows.append(
            {
                "protocol": f"{c1:.1f}C-{c2:.1f}C-{c3:.1f}C-{c4:.3f}C",
                "C1": c1,
                "C2": c2,
                "C3": c3,
                "C4": c4,
                "exists_in_policy_space": True,
                "final_posterior_rank": int(row["final_posterior_rank"]),
                "final_posterior_mean": float(row["final_posterior_mean"]),
                "final_posterior_uncertainty": float(row.get("posterior_half_width", np.nan)),
                "final_top_three_exact_match": final_top_three_exact_match,
            }
        )
    out = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out

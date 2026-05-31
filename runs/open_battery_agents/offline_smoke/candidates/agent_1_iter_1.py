
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

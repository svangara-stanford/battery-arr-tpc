
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

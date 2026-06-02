import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from battery_aar.tools.client import NativeToolClient
from battery_aar.tools.server import create_app
from battery_aar.workflows.candidate_compiler import candidate_spec_from_plans, compile_candidate_spec_to_python
from battery_aar.workflows.schemas import FeaturePlan, ModelPlan


def _write_toy_processed(base: Path):
    base.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "cell_id": ["c0", "c1", "c2", "c3"],
            "batch_id": ["b0", "b0", "b1", "b1"],
            "protocol_readable": ["p0", "p1", "p0", "p1"],
            "cycle_life": [700.0, 720.0, 760.0, 780.0],
            "label_source": ["toy"] * 4,
        }
    )
    cycles = []
    for row_id in metadata["row_id"]:
        for cycle in range(1, 12):
            cycles.append(
                {
                    "row_id": row_id,
                    "cell_id": f"c{row_id}",
                    "cycle_index": cycle,
                    "discharge_capacity": 1.1 - 0.001 * cycle - 0.0002 * row_id,
                    "charge_capacity": None,
                }
            )
    metadata.to_csv(base / "cell_metadata.csv", index=False)
    pd.DataFrame(cycles).to_csv(base / "cycle_summary.csv", index=False)
    return metadata, pd.DataFrame(cycles)


def test_fastapi_health_and_tools(tmp_path):
    client = TestClient(create_app())

    health = client.get("/health")
    tools = client.get("/tools")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert tools.status_code == 200
    payload = tools.json()
    assert payload["success"] is True
    assert {tool["name"] for tool in payload["tools"]} == {
        "profile_dataset",
        "build_battery_features",
        "list_feature_programs",
        "review_candidate",
        "evaluate_candidate",
        "compare_runs",
    }
    programs = client.get("/features/programs", params={"run_id": "programs_http", "run_dir": str(tmp_path / "programs_http")})
    assert programs.status_code == 200
    assert "attia_severson_like" in programs.json()["recipes"]
    assert "cross_cycle_curve_delta" in {operator["name"] for operator in programs.json()["operators"]}


def test_native_tool_client_lists_feature_programs_and_logs_trace(tmp_path):
    run_dir = tmp_path / "feature_program_tools"
    response = NativeToolClient().list_feature_programs(run_id="feature_program_tools", run_dir=str(run_dir))

    assert response.success is True
    assert "broad_physics" in response.recipes
    records = [json.loads(line) for line in (run_dir / "artifacts" / "tool_calls.jsonl").read_text().splitlines() if line.strip()]
    assert records[-1]["tool_name"] == "list_feature_programs"
    assert records[-1]["success"] is True


def test_native_tool_client_profile_dataset_writes_trace(tmp_path):
    processed = tmp_path / "processed"
    _write_toy_processed(processed)
    run_dir = tmp_path / "run"
    response = NativeToolClient().profile_dataset(
        run_id="native_profile",
        run_dir=str(run_dir),
        processed_dir=str(processed),
        data_source="toy",
    )

    assert response.success is True
    assert response.profile["metadata_row_count"] == 4
    assert response.output_artifact_ids
    tool_calls = (run_dir / "artifacts" / "tool_calls.jsonl").read_text().splitlines()
    assert len(tool_calls) == 1
    record = json.loads(tool_calls[0])
    assert record["tool_name"] == "profile_dataset"
    assert record["tool_call_id"] == response.tool_call_id
    assert record["success"] is True


def test_fastapi_dataset_profile_works_on_toy_data_and_logs(tmp_path):
    processed = tmp_path / "processed"
    _write_toy_processed(processed)
    run_dir = tmp_path / "run_http"
    client = TestClient(create_app())

    response = client.post(
        "/dataset/profile",
        json={
            "run_id": "http_profile",
            "run_dir": str(run_dir),
            "processed_dir": str(processed),
            "data_source": "toy",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["profile"]["metadata_row_count"] == 4
    assert (run_dir / "artifacts" / "dataset_profile.json").exists()
    assert (run_dir / "artifacts" / "tool_calls.jsonl").exists()


def test_failed_tool_call_is_logged_with_error_type_and_message(tmp_path):
    run_dir = tmp_path / "run_fail"
    client = TestClient(create_app())

    response = client.post(
        "/dataset/profile",
        json={
            "run_id": "failed_profile",
            "run_dir": str(run_dir),
            "processed_dir": str(tmp_path / "missing"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert payload["error_type"]
    assert payload["error_message"]
    records = (run_dir / "artifacts" / "tool_calls.jsonl").read_text().splitlines()
    assert records
    record = json.loads(records[-1])
    assert record["success"] is False
    assert record["error_type"] == payload["error_type"]
    assert record["error_message"] == payload["error_message"]


def test_review_candidate_preflight_catches_wrong_toolbox_keyword(tmp_path):
    candidate = tmp_path / "bad_candidate.py"
    candidate.write_text(
        """
import pandas as pd
from battery_aar.features.battery_lifetime_features import build_all_battery_features

def fit(train_metadata, train_cycle_summary, train_labels, config):
    build_all_battery_features(train_metadata, train_cycle_summary, max_cycle=100, include_protocol_currents=True)
    return {}

def predict(model, test_metadata, test_cycle_summary, config):
    return pd.DataFrame({"row_id": test_metadata["row_id"], "y_pred": [1.0] * len(test_metadata)})
"""
    )
    response = NativeToolClient().review_candidate(
        run_id="review_bad_kw",
        run_dir=str(tmp_path / "run_review_bad_kw"),
        candidate_path=str(candidate),
    )

    assert response.verdict == "needs_repair"
    assert response.failure_reason
    assert "unexpected keyword argument" in response.failure_reason
    assert any("preflight_failure" in issue for issue in response.issues)


def test_review_candidate_allows_identifier_drop_list_usage(tmp_path):
    candidate = tmp_path / "drop_identifier_candidate.py"
    candidate.write_text(
        """
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from battery_aar.features.battery_lifetime_features import build_all_battery_features

IDENTIFIER_COLUMNS = {"row_id", "cell_id", "batch_id", "source_path", "protocol_readable", "anonymized_cell_id"}

def _features(metadata, cycle_summary, config, cols=None):
    X = build_all_battery_features(metadata, cycle_summary, max_cycle=100, include_protocol=True)
    X = X.reset_index().rename(columns={X.index.name or "index": "row_id"})
    ids = metadata[["row_id"]].drop_duplicates()
    X = ids.merge(X, on="row_id", how="left")
    X = X.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in X.columns], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if cols is None:
        X = X.dropna(axis=1, how="all")
        cols = list(X.columns)
    else:
        for c in cols:
            if c not in X:
                X[c] = np.nan
        X = X[cols]
    return ids, X, cols

def fit(train_metadata, train_cycle_summary, train_labels, config):
    ids, X, cols = _features(train_metadata, train_cycle_summary, config)
    y = train_labels.set_index("row_id").loc[ids["row_id"], "y"].to_numpy(float)
    model = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
    model.fit(X, y)
    return {"model": model, "cols": cols}

def predict(model, test_metadata, test_cycle_summary, config):
    ids, X, _ = _features(test_metadata, test_cycle_summary, config, model["cols"])
    return pd.DataFrame({"row_id": ids["row_id"], "y_pred": model["model"].predict(X)})
"""
    )
    response = NativeToolClient().review_candidate(
        run_id="review_drop_list",
        run_dir=str(tmp_path / "run_review_drop_list"),
        candidate_path=str(candidate),
    )

    assert response.verdict == "pass"
    assert not any("batch_id" in issue for issue in response.issues)
    assert response.failure_reason is None


def test_review_candidate_flags_identifier_feature_usage(tmp_path):
    candidate = tmp_path / "leaky_identifier_candidate.py"
    candidate.write_text(
        """
import pandas as pd

def fit(train_metadata, train_cycle_summary, train_labels, config):
    features = train_metadata[["batch_id"]].copy()
    return {"mean": float(train_labels["y"].mean()), "features": list(features.columns)}

def predict(model, test_metadata, test_cycle_summary, config):
    features = test_metadata[["batch_id"]].copy()
    return pd.DataFrame({"row_id": test_metadata["row_id"], "y_pred": [model["mean"]] * len(features)})
"""
    )
    response = NativeToolClient().review_candidate(
        run_id="review_leaky_identifier",
        run_dir=str(tmp_path / "run_review_leaky_identifier"),
        candidate_path=str(candidate),
    )

    assert response.verdict in {"needs_attention", "needs_repair"}
    assert any("batch_id may be used as a model feature" in issue for issue in response.issues)


def test_review_compiled_candidate_identifier_drop_list_is_not_warning(tmp_path):
    feature_plan = FeaturePlan(
        run_id="review_compiled",
        human_readable_summary="features",
        agent_id="feature_scientist",
        feature_families=["capacity_summary", "protocol"],
        include_protocol_features=True,
        max_cycle=100,
    )
    model_plan = ModelPlan(
        run_id="review_compiled",
        human_readable_summary="model",
        agent_id="model_architect",
        model_family="Ridge",
        estimator_name="Ridge",
        preprocessing_steps=["SimpleImputer", "StandardScaler"],
        hyperparameters={"alpha": 1.0},
    )
    candidate_path = tmp_path / "compiled_candidate.py"
    spec = candidate_spec_from_plans(
        run_id="review_compiled",
        candidate_id="compiled_candidate",
        agent_id="code_generator",
        iteration=0,
        candidate_path=candidate_path,
        feature_plan=feature_plan,
        model_plan=model_plan,
    )
    compile_candidate_spec_to_python(spec, candidate_path)

    response = NativeToolClient().review_candidate(
        run_id="review_compiled",
        run_dir=str(tmp_path / "run_review_compiled"),
        candidate_path=str(candidate_path),
    )

    assert response.verdict == "pass"
    assert not any("batch_id" in issue or "protocol_readable" in issue for issue in response.issues)


def test_evaluate_candidate_writes_predictions_and_diagnostics(tmp_path):
    processed = tmp_path / "processed_eval"
    _write_toy_processed(processed)
    feature_plan = FeaturePlan(
        run_id="eval_predictions",
        human_readable_summary="features",
        agent_id="feature_scientist",
        feature_families=["capacity_summary"],
        include_protocol_features=False,
        max_cycle=100,
    )
    model_plan = ModelPlan(
        run_id="eval_predictions",
        human_readable_summary="model",
        agent_id="model_architect",
        model_family="Ridge",
        estimator_name="Ridge",
        preprocessing_steps=["SimpleImputer", "StandardScaler"],
        hyperparameters={"alpha": 1.0},
    )
    candidate_path = tmp_path / "candidate_eval.py"
    spec = candidate_spec_from_plans(
        run_id="eval_predictions",
        candidate_id="candidate_eval",
        agent_id="code_generator",
        iteration=0,
        candidate_path=candidate_path,
        feature_plan=feature_plan,
        model_plan=model_plan,
    )
    compile_candidate_spec_to_python(spec, candidate_path)

    response = NativeToolClient().evaluate_candidate(
        run_id="eval_predictions",
        run_dir=str(tmp_path / "run_eval_predictions"),
        candidate_path=str(candidate_path),
        metadata_path=str(processed / "cell_metadata.csv"),
        cycle_summary_path=str(processed / "cycle_summary.csv"),
        split_mode="random",
        max_cycle=100,
        allow_protocol_features=False,
    )

    assert response.success is True
    assert response.prediction_path
    pred_path = Path(response.prediction_path)
    assert pred_path.exists()
    pred = pd.read_csv(pred_path)
    assert {"row_id", "y_true", "y_pred", "residual", "split_mode"}.issubset(pred.columns)
    assert pred["split_mode"].eq("random").all()
    assert response.metrics["n_predictions"] == len(pred)
    assert "y_true_mean" in response.metrics
    assert "y_pred_mean" in response.metrics
    assert "n_negative_predictions" in response.metrics
    report_payload = json.loads((tmp_path / "run_eval_predictions" / "artifacts" / "tool_outputs" / "evaluation_report.json").read_text())
    assert report_payload["prediction_path"] == response.prediction_path
    assert report_payload["extra_metrics"]["n_predictions"] == len(pred)
    assert "y_pred_mean" in report_payload["extra_metrics"]

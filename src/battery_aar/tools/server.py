from __future__ import annotations

from fastapi import FastAPI

from .implementations import build_battery_features, compare_runs, evaluate_candidate, list_tools, profile_dataset, review_candidate
from .schemas import (
    BuildFeaturesRequest,
    BuildFeaturesResponse,
    CandidateEvaluateRequest,
    CandidateEvaluateResponse,
    CandidateReviewRequest,
    CandidateReviewResponse,
    DatasetProfileRequest,
    DatasetProfileResponse,
    RunCompareRequest,
    RunCompareResponse,
    ToolListResponse,
)


def create_app() -> FastAPI:
    app = FastAPI(title="Open Battery Agents Tool Server", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/tools", response_model=ToolListResponse)
    def tools():
        return list_tools(run_id="server", tool_call_id="tools")

    @app.post("/dataset/profile", response_model=DatasetProfileResponse)
    def dataset_profile(request: DatasetProfileRequest):
        return profile_dataset(request)

    @app.post("/features/build", response_model=BuildFeaturesResponse)
    def features_build(request: BuildFeaturesRequest):
        return build_battery_features(request)

    @app.post("/candidate/review", response_model=CandidateReviewResponse)
    def candidate_review(request: CandidateReviewRequest):
        return review_candidate(request)

    @app.post("/candidate/evaluate", response_model=CandidateEvaluateResponse)
    def candidate_evaluate(request: CandidateEvaluateRequest):
        return evaluate_candidate(request)

    @app.post("/runs/compare", response_model=RunCompareResponse)
    def runs_compare(request: RunCompareRequest):
        return compare_runs(request)

    return app


app = create_app()

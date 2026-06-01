from __future__ import annotations

import json
from typing import Any
from urllib import request as urlrequest

from .implementations import build_battery_features, compare_runs, evaluate_candidate, profile_dataset, review_candidate
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
)


class NativeToolClient:
    """In-process tool client with the same method surface as HTTPToolClient."""

    def profile_dataset(self, **kwargs: Any) -> DatasetProfileResponse:
        return profile_dataset(DatasetProfileRequest(**kwargs))

    def build_battery_features(self, **kwargs: Any) -> BuildFeaturesResponse:
        return build_battery_features(BuildFeaturesRequest(**kwargs))

    def review_candidate(self, **kwargs: Any) -> CandidateReviewResponse:
        return review_candidate(CandidateReviewRequest(**kwargs))

    def evaluate_candidate(self, **kwargs: Any) -> CandidateEvaluateResponse:
        return evaluate_candidate(CandidateEvaluateRequest(**kwargs))

    def compare_runs(self, **kwargs: Any) -> RunCompareResponse:
        return compare_runs(RunCompareRequest(**kwargs))


class HTTPToolClient:
    """HTTP client for the FastAPI Open Battery Agents tool server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _post(self, endpoint: str, request_model: Any, response_model: type[Any]):
        payload = request_model.model_dump(mode="json")
        req = urlrequest.Request(
            self.base_url + endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req) as response:  # nosec B310 - caller controls configured tool server URL
            data = json.loads(response.read().decode("utf-8"))
        return response_model.model_validate(data)

    def profile_dataset(self, **kwargs: Any) -> DatasetProfileResponse:
        return self._post("/dataset/profile", DatasetProfileRequest(**kwargs), DatasetProfileResponse)

    def build_battery_features(self, **kwargs: Any) -> BuildFeaturesResponse:
        return self._post("/features/build", BuildFeaturesRequest(**kwargs), BuildFeaturesResponse)

    def review_candidate(self, **kwargs: Any) -> CandidateReviewResponse:
        return self._post("/candidate/review", CandidateReviewRequest(**kwargs), CandidateReviewResponse)

    def evaluate_candidate(self, **kwargs: Any) -> CandidateEvaluateResponse:
        return self._post("/candidate/evaluate", CandidateEvaluateRequest(**kwargs), CandidateEvaluateResponse)

    def compare_runs(self, **kwargs: Any) -> RunCompareResponse:
        return self._post("/runs/compare", RunCompareRequest(**kwargs), RunCompareResponse)

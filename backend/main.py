"""Server-side evaluator API for the two-laptop PathSnap demo."""
from __future__ import annotations

import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from joblib import load

from src.data.loader import discover_trip_dirs, load_trip
from src.features.candidate_generator import generate_candidates
from src.features.feature_extractor import (
    FEATURE_COLUMNS,
    align_route_labels,
    extract_features,
)
from src.routing import decode_trip, load_road_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

_configured_data_root = Path(os.getenv("PATHSNAP_DATA_ROOT", CONFIG["data_root"]))
DATA_ROOT = (
    _configured_data_root
    if _configured_data_root.is_absolute()
    else PROJECT_ROOT / _configured_data_root
).resolve()
EVALUATOR_PARTITION = CONFIG.get("evaluator_partition", "evaluator_testing")
EVALUATOR_ROOT = DATA_ROOT / EVALUATOR_PARTITION
MODEL_PATH = Path(os.getenv(
    "PATHSNAP_MODEL_PATH", str(PROJECT_ROOT / "artifacts" / "tuned_model.joblib"),
)).resolve()

logging.basicConfig(
    level=os.getenv("PATHSNAP_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pathsnap.api")
app = FastAPI(
    title="PathSnap Evaluator API",
    description="Server-side route prediction and aggregate evaluator scoring.",
    version="1.0.0",
)


class TripsResponse(BaseModel):
    trip_ids: list[str]


class PredictRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class PredictResponse(BaseModel):
    trip_id: str
    model: str
    total_observations: int
    matched_observations: int
    top1_accuracy: float | None
    route_arc_ids: list[int]
    elapsed_seconds: float


@lru_cache(maxsize=1)
def _load_final_model(model_path: str, modified_ns: int):
    """Load the tuned model once per file version."""
    del modified_ns  # Included in the cache key to invalidate after a new tune.
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Tuned model not found at {path}; run the optimize stage first"
        )
    return load(path)


def _get_model():
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Tuned model is missing at {MODEL_PATH}; run --stage optimize first.",
        )
    try:
        return _load_final_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    except Exception as exc:
        logger.exception("Unable to load the tuned model")
        raise HTTPException(status_code=503, detail="Tuned model could not be loaded") from exc


def _evaluator_trip_dir(trip_id: str) -> Path:
    # The allow-list lookup prevents path traversal and confines requests to the
    # evaluator partition configured on the server.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trip_id):
        raise HTTPException(status_code=422, detail="Invalid trip_id")
    if not EVALUATOR_ROOT.is_dir():
        raise HTTPException(
            status_code=503,
            detail=f"Evaluator data directory is not available: {EVALUATOR_ROOT}",
        )
    trip_dir = EVALUATOR_ROOT / trip_id
    if not trip_dir.is_dir() or trip_dir.resolve().parent != EVALUATOR_ROOT.resolve():
        raise HTTPException(status_code=404, detail="Evaluator trip not found")
    return trip_dir


@app.get("/trips", response_model=TripsResponse)
def list_evaluator_trips() -> TripsResponse:
    """Return evaluator trip IDs only; no trip files or labels are exposed."""
    if not EVALUATOR_ROOT.is_dir():
        raise HTTPException(
            status_code=503,
            detail=f"Evaluator data directory is not available: {EVALUATOR_ROOT}",
        )
    trip_ids = [path.name for path in discover_trip_dirs(EVALUATOR_ROOT)]
    logger.info("GET /trips returned %d evaluator trip IDs", len(trip_ids))
    return TripsResponse(trip_ids=trip_ids)


@app.post("/predict", response_model=PredictResponse)
def predict_evaluator_trip(request: PredictRequest) -> PredictResponse:
    """Predict a continuous route and return aggregate accuracy only.

    Reference routes are read inside the server solely to calculate aggregate
    top-1 accuracy. Neither route labels nor per-observation correctness are
    included in the response.
    """
    started = time.perf_counter()
    trip_dir = _evaluator_trip_dir(request.trip_id)
    logger.info("POST /predict started trip_id=%s", request.trip_id)

    try:
        trip = load_trip(trip_dir)
        candidates = generate_candidates(
            trip["track"], trip["nodes"], trip["arcs"],
            CONFIG["candidate"]["radius_m"],
            CONFIG["candidate"]["max_candidates"],
            trip["trip_id"],
        )
        if candidates.empty:
            raise HTTPException(
                status_code=422,
                detail="No road candidates were generated for this trip.",
            )

        features = extract_features(
            trip["track"], trip["nodes"], trip["arcs"], candidates,
        )
        graph = load_road_graph(trip["nodes"], trip["arcs"])
        decoded = decode_trip(
            _get_model(), trip["track"], features, graph, FEATURE_COLUMNS,
        )

        # Evaluator truth remains in this process. Only the aggregate score leaves it.
        reference = align_route_labels(
            trip["track"], trip["nodes"], trip["arcs"], trip["route"],
            CONFIG["label_max_distance_m"],
        )
        scored = decoded.observations.merge(
            reference[["observation_id", "label_arc_id", "label_status"]],
            on="observation_id", how="left",
        )
        matched = scored[scored.label_status.eq("matched")]
        accuracy = (
            float((matched.candidate_arc_id == matched.label_arc_id).mean())
            if len(matched) else None
        )
        elapsed = time.perf_counter() - started
        logger.info(
            "POST /predict finished trip_id=%s top1_accuracy=%s matched=%d/%d "
            "route_arcs=%d elapsed=%.3fs",
            request.trip_id,
            f"{accuracy:.4f}" if accuracy is not None else "unavailable",
            len(matched), len(trip["track"]), len(decoded.route), elapsed,
        )
        return PredictResponse(
            trip_id=request.trip_id,
            model=MODEL_PATH.name,
            total_observations=len(trip["track"]),
            matched_observations=len(matched),
            top1_accuracy=accuracy,
            route_arc_ids=decoded.route.arc_id.astype(int).tolist(),
            elapsed_seconds=round(elapsed, 3),
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.exception("Trip files missing for trip_id=%s", request.trip_id)
        raise HTTPException(status_code=422, detail="Trip files are incomplete") from exc
    except ValueError as exc:
        logger.warning("Trip decoding failed trip_id=%s: %s", request.trip_id, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed trip_id=%s", request.trip_id)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

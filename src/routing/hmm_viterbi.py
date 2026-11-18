"""Log-space HMM decoding of candidate road arcs into continuous routes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from src.data.geo_utils import haversine_m
from src.routing.graph import RoadGraph


@dataclass
class RouteDecodeResult:
    """Decoded observation choices and the continuous network path."""

    observations: pd.DataFrame
    route: pd.DataFrame
    log_probability: float


def score_candidates(model: Any, candidates: pd.DataFrame,
                     feature_columns: list[str], score_column: str = "score") -> pd.DataFrame:
    """Attach positive-class probabilities from a fitted binary classifier."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("The candidate model must implement predict_proba")
    probabilities = np.asarray(model.predict_proba(candidates[feature_columns]))
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        raise ValueError("The fitted model has no positive class labelled 1")
    result = candidates.copy()
    result[score_column] = probabilities[:, classes.index(1)]
    return result


def _fraction(row: pd.Series) -> float:
    for column in ("projection_fraction",):
        if column in row and pd.notna(row[column]):
            return float(np.clip(row[column], 0.0, 1.0))
    if "distance_to_start_m" in row and "segment_length_m" in row:
        length = float(row["segment_length_m"])
        if length > 0 and pd.notna(row["distance_to_start_m"]):
            return float(np.clip(row["distance_to_start_m"] / length, 0.0, 1.0))
    return 0.5


def _gps_distance(track: pd.DataFrame, previous_obs: int, current_obs: int) -> float:
    if not {"longitude", "latitude"}.issubset(track.columns):
        return 0.0
    if previous_obs not in track.index or current_obs not in track.index:
        return 0.0
    a, b = track.loc[previous_obs], track.loc[current_obs]
    return float(haversine_m(a.longitude, a.latitude, b.longitude, b.latitude))


def _elapsed_seconds(track: pd.DataFrame, previous_obs: int, current_obs: int) -> float:
    """Return a positive GPS time gap when timestamps are available."""
    if "timestamp" not in track.columns:
        return 0.0
    if previous_obs not in track.index or current_obs not in track.index:
        return 0.0
    delta = float(track.loc[current_obs, "timestamp"] - track.loc[previous_obs, "timestamp"])
    return max(delta, 0.0) if np.isfinite(delta) else 0.0


def decode_route(track: pd.DataFrame, candidates: pd.DataFrame, road_graph: RoadGraph,
                 score_column: str = "score", observation_column: str = "observation_id",
                 arc_column: str = "candidate_arc_id", transition_sigma_m: float = 35.0,
                 relative_sigma: float = 0.35, gap_uncertainty_mps: float = 5.0,
                 probability_floor: float = 1e-12
                 ) -> RouteDecodeResult:
    """Decode candidate probabilities with a route-connectivity HMM.

    ``track`` must have longitude/latitude columns and be indexed by observation
    ID (or contain them as a column). ``candidates`` has one or more candidate
    arcs per observation and a model probability in ``score_column``. Optional
    ``projection_fraction`` (or distance-to-start/segment-length features) makes
    the network-distance transition model more precise.

    Transition states are connected only when a directed path exists in the road
    graph. The returned route includes intermediate road arcs, so consecutive
    decoded observations form a continuous directed path. A clear error is raised
    when no continuous route can span the observations.
    """
    required = {observation_column, arc_column, score_column}
    if missing := required.difference(candidates.columns):
        raise ValueError(f"candidates is missing columns: {sorted(missing)}")
    if candidates.empty:
        raise ValueError("No candidate rows were supplied")
    if transition_sigma_m <= 0 or relative_sigma < 0:
        raise ValueError("Transition uncertainty must be positive")

    points = track.copy()
    if "observation_id" in points.columns:
        points = points.set_index("observation_id", drop=False)
    grouped: list[pd.DataFrame] = []
    for obs_id, group in candidates.groupby(observation_column, sort=True):
        # There can be duplicate candidate arcs; retain the strongest emission.
        group = group.sort_values(score_column, ascending=False).drop_duplicates(arc_column).copy()
        group[arc_column] = group[arc_column].astype(int)
        scores = pd.to_numeric(group[score_column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError(f"Invalid model probabilities at observation {obs_id}")
        scores = np.maximum(scores, probability_floor)
        group["_emission_logp"] = np.log(scores / scores.sum())
        group["_fraction"] = group.apply(_fraction, axis=1)
        grouped.append(group.reset_index(drop=True))

    if not grouped:
        raise ValueError("No observations have candidate arcs")

    backpointers: list[np.ndarray] = []
    delta = grouped[0]["_emission_logp"].to_numpy(dtype=float)

    for t in range(1, len(grouped)):
        previous, current = grouped[t-1], grouped[t]
        previous_id = int(previous[observation_column].iloc[0])
        current_id = int(current[observation_column].iloc[0])
        observed_distance = _gps_distance(points, previous_id, current_id)
        elapsed = _elapsed_seconds(points, previous_id, current_id)
        sigma = max(
            float(transition_sigma_m),
            float(relative_sigma) * observed_distance,
            float(gap_uncertainty_mps) * elapsed,
        )
        raw = np.full((len(previous), len(current)), -np.inf, dtype=float)

        for i, state_a in previous.iterrows():
            for j, state_b in current.iterrows():
                transition = road_graph.transition(
                    int(state_a[arc_column]), int(state_b[arc_column]),
                    float(state_a["_fraction"]), float(state_b["_fraction"]),
                )
                if transition is None:
                    continue
                network_distance, _ = transition
                # Road travel should be at least the straight-line displacement;
                # GPS noise is handled by a wider sigma on longer observation gaps.
                residual = (network_distance - observed_distance) / sigma
                raw[i, j] = -0.5 * residual * residual

        row_norm = logsumexp(raw, axis=1, keepdims=True)
        valid_rows = np.isfinite(row_norm[:, 0])
        transitions = np.full_like(raw, -np.inf)
        transitions[valid_rows] = raw[valid_rows] - row_norm[valid_rows]

        next_delta = np.full(len(current), -np.inf, dtype=float)
        predecessor = np.full(len(current), -1, dtype=int)
        for j in range(len(current)):
            scores = delta + transitions[:, j]
            best_i = int(np.argmax(scores))
            if np.isfinite(scores[best_i]):
                predecessor[j] = best_i
                next_delta[j] = scores[best_i] + float(current["_emission_logp"].iloc[j])
        if not np.isfinite(next_delta).any():
            raise ValueError(
                f"No directed road-network route connects candidate observations "
                f"{previous_id} and {current_id}"
            )
        backpointers.append(predecessor)
        delta = next_delta

    state_index = int(np.argmax(delta))
    best_log_probability = float(delta[state_index])
    chosen: list[int] = [state_index]
    for predecessor in reversed(backpointers):
        state_index = int(predecessor[state_index])
        if state_index < 0:
            raise RuntimeError("Viterbi backtrace encountered an unreachable state")
        chosen.append(state_index)
    chosen.reverse()

    decoded_rows = []
    for group, index in zip(grouped, chosen):
        row = group.iloc[index]
        decoded_rows.append({
            "observation_id": int(row[observation_column]),
            "candidate_arc_id": int(row[arc_column]),
            "score": float(row[score_column]),
        })
    observations = pd.DataFrame(decoded_rows)

    route_arc_ids: list[int] = []
    first = grouped[0].iloc[chosen[0]]
    route_arc_ids.append(int(first[arc_column]))
    for t in range(1, len(grouped)):
        a = grouped[t-1].iloc[chosen[t-1]]
        b = grouped[t].iloc[chosen[t]]
        transition = road_graph.transition(
            int(a[arc_column]), int(b[arc_column]),
            float(a["_fraction"]), float(b["_fraction"]),
        )
        if transition is None:
            raise RuntimeError("Decoded state pair lost its road-network connection")
        path = transition[1]
        route_arc_ids.extend(path[1:])

    route_rows = []
    for route_index, arc_id in enumerate(route_arc_ids):
        arc = road_graph.arcs[arc_id]
        route_rows.append({
            "route_index": route_index,
            "arc_id": arc_id,
            "from_node_id": arc.from_node_id,
            "to_node_id": arc.to_node_id,
            "length_m": arc.length_m,
            "bearing_deg": arc.bearing_deg,
        })
    return RouteDecodeResult(
        observations=observations,
        route=pd.DataFrame(route_rows),
        log_probability=best_log_probability,
    )


def decode_trip(model: Any, track: pd.DataFrame, candidates: pd.DataFrame,
                road_graph: RoadGraph, feature_columns: list[str], **decode_options: Any
                ) -> RouteDecodeResult:
    """Score candidates with a fitted model, then decode their continuous route."""
    scored = score_candidates(model, candidates, feature_columns)
    return decode_route(track, scored, road_graph, **decode_options)

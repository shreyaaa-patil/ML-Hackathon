"""Run PathSnap data, model-selection, tuning, and route-decoding stages."""
import argparse
import re
from pathlib import Path

import pandas as pd
import yaml
from joblib import dump, load

from src.data.loader import discover_trip_dirs, load_trip
from src.eda.eda import run_eda
from src.features.candidate_generator import generate_candidates, candidate_diagnostics
from src.features.feature_extractor import (
    align_route_labels, extract_features, FEATURE_COLUMNS,
)
from src.models.bayesian_opt import tune_selected
from src.models.evaluate import evaluate_model, save_comparison
from src.models.train import train_five_models
from src.routing import decode_trip, load_road_graph


def build_features(root, partition, cfg):
    rows, quality = [], []
    for trip_dir in discover_trip_dirs(Path(root) / partition):
        trip = load_trip(trip_dir)
        labels = align_route_labels(
            trip["track"], trip["nodes"], trip["arcs"], trip["route"],
            cfg["label_max_distance_m"],
        )
        candidates = generate_candidates(
            trip["track"], trip["nodes"], trip["arcs"],
            cfg["candidate"]["radius_m"], cfg["candidate"]["max_candidates"],
            trip["trip_id"],
        )
        if candidates.empty:
            continue
        candidates = candidates.merge(
            labels[["observation_id", "label_arc_id", "label_status"]],
            on="observation_id", how="left",
        )
        candidates = candidates[candidates.label_status.eq("matched")].copy()
        candidates["label"] = (
            candidates.candidate_arc_id == candidates.label_arc_id
        ).astype(int)
        rows.append(extract_features(
            trip["track"], trip["nodes"], trip["arcs"], candidates,
        ))
        quality.append({
            "trip_id": trip["trip_id"],
            **candidate_diagnostics(
                candidates, labels[labels.label_status.eq("matched")],
            ),
            "label_matched": int((labels.label_status == "matched").sum()),
            "label_unmatched": int((labels.label_status != "matched").sum()),
        })
    if not rows:
        raise ValueError(f"No labeled candidate features found for partition {partition!r}")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(quality)


def select_and_evaluate(cfg):
    processed = Path("data/processed")
    train = pd.read_pickle(processed / f"{cfg['training_partition']}_features.pkl")
    cv = pd.read_csv("artifacts/training_cv_results.csv")
    test = pd.read_pickle(processed / f"{cfg['student_partition']}_features.pkl")
    metrics = []
    for model_name in cv.model:
        model = load(Path("artifacts") / f"{model_name}.joblib")
        row, _ = evaluate_model(model, test, FEATURE_COLUMNS)
        metrics.append({"model": model_name, **row})
    comparison = cv.merge(
        save_comparison(metrics, "outputs/model_comparison"), on="model",
        suffixes=("_cv", "_student"),
    )
    comparison.to_csv("outputs/model_comparison/model_comparison.csv", index=False)
    best = comparison.sort_values(
        ["student_top1_accuracy", "cv_macro_f1_mean"],
        ascending=False,
    ).iloc[0]
    Path("outputs/model_comparison/model_selection_report.md").write_text(
        "# Model selection\n\n"
        f"Selected **{best.model}** using student Top-1 accuracy "
        f"({best.student_top1_accuracy:.4f}) and group-CV Macro F1 "
        f"({best.cv_macro_f1_mean:.4f}). Evaluator data is never read.\n"
    )
    return train, test, comparison, str(best.model)


def run_optimization(train, test, comparison, selected_model, cfg):
    study, tuned_model = tune_selected(
        train, FEATURE_COLUMNS, selected_model, "outputs/optimization",
        cfg["optuna_trials"], cfg["cv_splits"], cfg["random_seed"],
        cfg["max_training_candidate_rows"],
    )
    tuned_metrics, _ = evaluate_model(tuned_model, test, FEATURE_COLUMNS)
    untuned = comparison[comparison.model.eq(selected_model)].iloc[0].to_dict()
    rows = [
        {"version": "untuned", **untuned},
        {
            "version": "tuned", "model": selected_model,
            "cv_macro_f1_mean": study.best_value, **tuned_metrics,
        },
    ]
    Path("outputs/optimization").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        "outputs/optimization/tuned_vs_untuned.csv", index=False,
    )
    dump(tuned_model, "artifacts/tuned_model.joblib")
    return tuned_model


def _selected_model_name():
    report = Path("outputs/model_comparison/model_selection_report.md")
    if report.is_file():
        match = re.search(r"Selected \*\*(.+?)\*\*", report.read_text())
        if match:
            return match.group(1)
    comparison = Path("outputs/model_comparison/model_comparison.csv")
    if comparison.is_file():
        frame = pd.read_csv(comparison)
        return str(frame.sort_values(
            ["student_top1_accuracy", "cv_macro_f1_mean"],
            ascending=False,
        ).iloc[0].model)
    raise FileNotFoundError(
        "Run the model training/evaluation stages first, or provide "
        "artifacts/tuned_model.joblib before decoding routes."
    )


def decode_partition(root, cfg):
    tuned_path = Path("artifacts/tuned_model.joblib")
    if tuned_path.is_file():
        model = load(tuned_path)
        model_name = "tuned_model"
    else:
        model_name = _selected_model_name()
        model_path = Path("artifacts") / f"{model_name}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Selected model artifact not found: {model_path}")
        model = load(model_path)

    partition = cfg["student_partition"]
    output_dir = Path("outputs/routes") / partition
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for trip_dir in discover_trip_dirs(Path(root) / partition):
        trip = load_trip(trip_dir)
        candidates = generate_candidates(
            trip["track"], trip["nodes"], trip["arcs"],
            cfg["candidate"]["radius_m"], cfg["candidate"]["max_candidates"],
            trip["trip_id"],
        )
        if candidates.empty:
            summaries.append({
                "trip_id": trip["trip_id"], "status": "no_candidates",
                "decoded_observations": 0, "route_arcs": 0,
                "top1_accuracy": float("nan"),
            })
            continue
        features = extract_features(
            trip["track"], trip["nodes"], trip["arcs"], candidates,
        )
        graph = load_road_graph(trip["nodes"], trip["arcs"])
        try:
            decoded = decode_trip(
                model, trip["track"], features, graph, FEATURE_COLUMNS,
            )
        except ValueError as exc:
            summaries.append({
                "trip_id": trip["trip_id"], "status": f"decode_failed: {exc}",
                "decoded_observations": 0, "route_arcs": 0,
                "top1_accuracy": float("nan"),
            })
            continue

        trip_id = trip["trip_id"]
        decoded.observations.to_csv(
            output_dir / f"{trip_id}_observations.csv", index=False,
        )
        decoded.route.to_csv(output_dir / f"{trip_id}_route.csv", index=False)
        labels = align_route_labels(
            trip["track"], trip["nodes"], trip["arcs"], trip["route"],
            cfg["label_max_distance_m"],
        )
        scored = decoded.observations.merge(
            labels[["observation_id", "label_arc_id", "label_status"]],
            on="observation_id", how="left",
        )
        matched = scored[scored.label_status.eq("matched")]
        accuracy = float(
            (matched.candidate_arc_id == matched.label_arc_id).mean()
        ) if len(matched) else float("nan")
        route_rows = list(decoded.route.itertuples())
        continuous = all(
            int(a.to_node_id) == int(b.from_node_id)
            for a, b in zip(route_rows, route_rows[1:])
        )
        summaries.append({
            "trip_id": trip_id, "status": "ok" if continuous else "invalid_route",
            "model": model_name,
            "decoded_observations": len(decoded.observations),
            "route_arcs": len(decoded.route),
            "top1_accuracy": accuracy,
            "log_probability": decoded.log_probability,
        })

    pd.DataFrame(summaries).to_csv(output_dir / "decode_summary.csv", index=False)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True,
        choices=[
            "inventory", "eda", "preprocess", "features", "train",
            "evaluate", "select", "optimize", "decode", "final",
        ],
    )
    parser.add_argument("--data-root")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path("config/config.yaml").read_text())
    root = args.data_root or cfg["data_root"]
    processed = Path("data/processed")
    processed.mkdir(parents=True, exist_ok=True)

    if args.stage in {"inventory", "eda"}:
        run_eda(root, "outputs/eda", (
            cfg["training_partition"], cfg["student_partition"],
        ))

    if args.stage in {"features", "preprocess", "final"}:
        Path("outputs/preprocessing").mkdir(parents=True, exist_ok=True)
        for partition in (cfg["training_partition"], cfg["student_partition"]):
            frame, quality = build_features(root, partition, cfg)
            frame.to_pickle(processed / f"{partition}_features.pkl")
            quality.to_csv(
                Path("outputs/preprocessing") / f"{partition}_label_quality.csv",
                index=False,
            )

    model_stages = {"train", "evaluate", "select", "optimize", "final"}
    if args.stage in model_stages:
        train = pd.read_pickle(processed / f"{cfg['training_partition']}_features.pkl")
        if args.stage in {"train", "final"}:
            train_five_models(
                train, FEATURE_COLUMNS, "artifacts", cfg["random_seed"],
                cfg["cv_splits"], cfg["max_training_candidate_rows"],
            )
        if args.stage in {"evaluate", "select", "optimize", "final"}:
            train, test, comparison, selected_model = select_and_evaluate(cfg)
            if args.stage in {"optimize", "final"}:
                run_optimization(train, test, comparison, selected_model, cfg)

    if args.stage in {"decode", "final"}:
        decode_partition(root, cfg)


if __name__ == "__main__":
    main()

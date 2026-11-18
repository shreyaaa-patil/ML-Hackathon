"""Run PathSnap data, model-selection, tuning, and route-decoding stages."""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from joblib import dump, load
from sklearn.metrics import f1_score

from src.data.loader import discover_trip_dirs, load_trip
from src.eda.eda import run_eda
from src.features.candidate_generator import generate_candidates, candidate_diagnostics
from src.features.feature_extractor import (
    align_route_labels, extract_features, FEATURE_COLUMNS,
)
from src.models.bayesian_opt import tune_selected, search_space_description
from src.models.evaluate import evaluate_model, save_comparison
from src.models.train import train_five_models
from src.routing import decode_trip, load_road_graph
from src.routing.hmm_viterbi import score_candidates
from src.visualization.folium_maps import create_trip_map
from src.visualization.model_plots import (
    plot_cross_validation, plot_student_comparison,
    plot_tuned_vs_untuned, plot_route_method_comparison,
)


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
        matched_labels = labels[labels.label_status.eq("matched")]
        if candidates.empty:
            quality.append({
                "trip_id": trip["trip_id"],
                **candidate_diagnostics(candidates, matched_labels),
                "label_matched": int(len(matched_labels)),
                "label_unmatched": int(len(labels) - len(matched_labels)),
            })
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
            **candidate_diagnostics(candidates, matched_labels),
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
    quality_path = Path("outputs/preprocessing") / f"{cfg['student_partition']}_label_quality.csv"
    quality = pd.read_csv(quality_path) if quality_path.is_file() else pd.DataFrame()
    candidate_observation_count = int(test.groupby("trip_id").observation_id.nunique().sum())
    student_label_counts = {
        "student_matched_observations": int(quality.label_matched.sum()) if "label_matched" in quality else candidate_observation_count,
        "student_total_observations": int((quality.label_matched + quality.label_unmatched).sum()) if {"label_matched", "label_unmatched"}.issubset(quality.columns) else candidate_observation_count,
    }
    student_label_counts["student_label_coverage"] = (
        student_label_counts["student_matched_observations"] /
        max(student_label_counts["student_total_observations"], 1)
    )
    metrics = []
    for model_name in cv.model:
        model = load(Path("artifacts") / f"{model_name}.joblib")
        row, _ = evaluate_model(model, test, FEATURE_COLUMNS)
        metrics.append({"model": model_name, **student_label_counts, **row})
    comparison = cv.merge(
        save_comparison(metrics, "outputs/model_comparison"), on="model",
        suffixes=("_cv", "_student"),
    )
    comparison.to_csv("outputs/model_comparison/model_comparison.csv", index=False)
    plot_cross_validation(cv, "outputs/model_comparison/plots")
    plot_student_comparison(comparison, "outputs/model_comparison/plots")
    best = comparison.sort_values(
        ["student_top1_accuracy", "cv_macro_f1_mean"],
        ascending=False,
    ).iloc[0]
    Path("outputs/model_comparison/model_selection_report.md").write_text(
        "# Model selection\n\n"
        f"Selected **{best.model}** using student Top-1 accuracy "
        f"({best.student_top1_accuracy:.4f}) and group-CV Macro F1 "
        f"({best.cv_macro_f1_mean:.4f}). Student label coverage was "
        f"{student_label_counts['student_label_coverage']:.1%} "
        f"({student_label_counts['student_matched_observations']}/"
        f"{student_label_counts['student_total_observations']} observations). "
        "Evaluator data is never read.\n"
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
    metric_columns = (
        "cv_macro_f1_mean", "cv_macro_f1_std", "student_label_coverage",
        "student_top1_accuracy",
        "candidate_macro_f1", "student_top1_macro_f1", "log_loss",
        "inference_seconds_per_row",
    )
    untuned_row = {"version": "untuned", "model": selected_model}
    for key in metric_columns:
        source_key = f"{key}_cv" if key.startswith("cv_") else f"{key}_student"
        untuned_row[key] = untuned.get(source_key, untuned.get(key))
    tuned_row = {
        "version": "tuned", "model": selected_model,
        "cv_macro_f1_mean": study.best_value,
        "cv_macro_f1_std": float(pd.Series(study.best_trial.user_attrs["fold_macro_f1_scores"]).std(ddof=0)),
        "student_label_coverage": untuned.get("student_label_coverage"),
        **tuned_metrics,
    }
    rows = [untuned_row, tuned_row]
    Path("outputs/optimization").mkdir(parents=True, exist_ok=True)
    comparison_frame = pd.DataFrame(rows)
    comparison_frame.to_csv(
        "outputs/optimization/tuned_vs_untuned.csv", index=False,
    )
    plot_tuned_vs_untuned(comparison_frame, "outputs/optimization/plots")
    params = study.best_params
    space = search_space_description(selected_model)
    changed = {
        "cv_macro_f1_mean": (untuned_row["cv_macro_f1_mean"], tuned_row["cv_macro_f1_mean"]),
        "student_top1_accuracy": (untuned_row["student_top1_accuracy"], tuned_row["student_top1_accuracy"]),
        "student_label_coverage": (untuned_row["student_label_coverage"], tuned_row["student_label_coverage"]),
        "candidate_macro_f1": (untuned_row["candidate_macro_f1"], tuned_row["candidate_macro_f1"]),
        "student_top1_macro_f1": (untuned_row["student_top1_macro_f1"], tuned_row["student_top1_macro_f1"]),
        "log_loss": (untuned_row["log_loss"], tuned_row["log_loss"]),
        "inference_seconds_per_row": (untuned_row["inference_seconds_per_row"], tuned_row["inference_seconds_per_row"]),
    }
    verdict = "outperformed" if tuned_row["student_top1_accuracy"] > untuned_row["student_top1_accuracy"] else "did not improve Top-1 accuracy"
    lines = [
        "# Bayesian hyperparameter optimization",
        "",
        f"Chosen model: **{selected_model}**. Optuna used the TPE sampler and maximized mean GroupKFold validation Macro F1 over {len(study.trials)} trials.",
        "",
        "## Search space",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in space.items())
    lines += ["", "## Best hyperparameters", "", "```json", json.dumps(params, indent=2), "```", "", "## Untuned versus tuned", "", f"On the student-testing partition, the tuned model {verdict}.", "", "| Metric | Untuned | Tuned |", "|---|---:|---:|"]
    for key, values in changed.items():
        display = {
            "candidate_macro_f1": "Student candidate Macro F1",
            "cv_macro_f1_mean": "Group CV Macro F1",
            "student_label_coverage": "Student label coverage",
            "log_loss": "Student candidate Log Loss",
            "inference_seconds_per_row": "Inference seconds per candidate",
        }.get(key, key)
        lines.append(f"| {display} | {values[0]:.6f} | {values[1]:.6f} |")
    lines += ["", "CV Macro F1 is the Optuna objective and uses the same training trips and grouped folds for both rows. Student-test metrics use the same held-out partition. Inference times are measured on that partition and can vary with machine load.", ""]
    Path("outputs/optimization/optimization_report.md").write_text("\n".join(lines), encoding="utf-8")
    dump(tuned_model, "artifacts/tuned_model.joblib")
    return tuned_model


def _route_metrics(predictions, labels, total_observations):
    scored = predictions.merge(
        labels[["observation_id", "label_arc_id", "label_status"]],
        on="observation_id", how="left",
    )
    matched = scored[scored.label_status.eq("matched")]
    if matched.empty:
        return {"top1_accuracy": float("nan"), "macro_f1": float("nan"),
                "matched_observations": 0,
                "label_coverage": 0.0}
    return {
        "top1_accuracy": float((matched.candidate_arc_id == matched.label_arc_id).mean()),
        "macro_f1": float(f1_score(matched.label_arc_id, matched.candidate_arc_id,
                                   average="macro", zero_division=0)),
        "matched_observations": int(len(matched)),
        "label_coverage": float(len(matched) / max(total_observations, 1)),
    }


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
    map_written = False
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
                "classifier_only_top1_accuracy": float("nan"),
                "classifier_only_macro_f1": float("nan"),
                "hmm_top1_accuracy": float("nan"), "hmm_macro_f1": float("nan"),
                "matched_observations": 0, "label_coverage": 0.0,
            })
            continue
        features = extract_features(
            trip["track"], trip["nodes"], trip["arcs"], candidates,
        )
        graph = load_road_graph(trip["nodes"], trip["arcs"])
        labels = align_route_labels(
            trip["track"], trip["nodes"], trip["arcs"], trip["route"],
            cfg["label_max_distance_m"],
        )
        classifier_scored = score_candidates(model, features, FEATURE_COLUMNS)
        classifier_only = (
            classifier_scored.sort_values(
                ["observation_id", "score"], ascending=[True, False],
            ).drop_duplicates("observation_id")
            [["observation_id", "candidate_arc_id", "score"]]
        )
        classifier_metrics = _route_metrics(
            classifier_only, labels, len(trip["track"]),
        )
        classifier_only.to_csv(
            output_dir / f"{trip['trip_id']}_classifier_only.csv", index=False,
        )
        try:
            decoded = decode_trip(
                model, trip["track"], features, graph, FEATURE_COLUMNS,
            )
        except ValueError as exc:
            summaries.append({
                "trip_id": trip["trip_id"], "status": f"decode_failed: {exc}",
                "decoded_observations": 0, "route_arcs": 0,
                "classifier_only_top1_accuracy": classifier_metrics["top1_accuracy"],
                "classifier_only_macro_f1": classifier_metrics["macro_f1"],
                "hmm_top1_accuracy": float("nan"), "hmm_macro_f1": float("nan"),
                "matched_observations": classifier_metrics["matched_observations"],
                "label_coverage": classifier_metrics["label_coverage"],
            })
            continue

        trip_id = trip["trip_id"]
        decoded.observations.to_csv(
            output_dir / f"{trip_id}_observations.csv", index=False,
        )
        decoded.route.to_csv(output_dir / f"{trip_id}_route.csv", index=False)
        hmm_metrics = _route_metrics(
            decoded.observations, labels, len(trip["track"]),
        )
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
            "classifier_only_top1_accuracy": classifier_metrics["top1_accuracy"],
            "classifier_only_macro_f1": classifier_metrics["macro_f1"],
            "hmm_top1_accuracy": hmm_metrics["top1_accuracy"],
            "hmm_macro_f1": hmm_metrics["macro_f1"],
            "matched_observations": hmm_metrics["matched_observations"],
            "label_coverage": hmm_metrics["label_coverage"],
            "top1_accuracy_delta": hmm_metrics["top1_accuracy"] - classifier_metrics["top1_accuracy"],
            "macro_f1_delta": hmm_metrics["macro_f1"] - classifier_metrics["macro_f1"],
            "log_probability": decoded.log_probability,
        })
        if not map_written:
            create_trip_map(
                trip["track"], trip["nodes"], trip["arcs"],
                trip["route"].arc_id, Path("outputs/maps") / f"{trip_id}_reference_route.html",
            )
            create_trip_map(
                trip["track"], trip["nodes"], trip["arcs"], decoded.route.arc_id,
                Path("outputs/maps") / f"{trip_id}_hmm_route.html",
                route_layer_name="HMM-decoded route", route_color="orange",
            )
            map_written = True

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "decode_summary.csv", index=False)
    plot_route_method_comparison(summary, output_dir / "plots")
    valid = summary[summary.status.eq("ok")]
    report = [
        "# Classifier-only versus HMM route results", "",
        f"Model: **{model_name}**. Evaluation uses the student-testing partition only.",
        "Route predictions are scored against aligned reference-route labels. Label coverage is reported because unaligned observations are excluded from the accuracy and Macro F1 calculations.",
        "", "| Method | Mean per-trip Top-1 Accuracy | Mean per-trip Macro F1 |", "|---|---:|---:|",
    ]
    if not valid.empty:
        report.append(f"| Classifier only | {valid.classifier_only_top1_accuracy.mean():.4f} | {valid.classifier_only_macro_f1.mean():.4f} |")
        report.append(f"| Classifier + HMM | {valid.hmm_top1_accuracy.mean():.4f} | {valid.hmm_macro_f1.mean():.4f} |")
        improved = valid[valid.top1_accuracy_delta > 0].trip_id.astype(str).tolist()
        not_improved = valid[valid.top1_accuracy_delta <= 0].trip_id.astype(str).tolist()
        macro_improved = valid[valid.macro_f1_delta > 0].trip_id.astype(str).tolist()
        macro_not_improved = valid[valid.macro_f1_delta <= 0].trip_id.astype(str).tolist()
        report += ["", f"HMM improved Top-1 accuracy on {len(improved)} of {len(valid)} decoded trips.",
                   f"Improved trips: {', '.join(improved) if improved else 'none'}.",
                   f"No improvement or lower accuracy: {', '.join(not_improved) if not_improved else 'none'}.",
                   f"HMM improved Macro F1 on {len(macro_improved)} of {len(valid)} decoded trips.",
                   f"Macro F1 improved: {', '.join(macro_improved) if macro_improved else 'none'}.",
                   f"Macro F1 unchanged or lower: {', '.join(macro_not_improved) if macro_not_improved else 'none'}.",
                   f"Mean label coverage among decoded trips: {valid.label_coverage.mean():.4f}."]
    else:
        report += ["| Classifier only | unavailable | unavailable |", "| Classifier + HMM | unavailable | unavailable |",
                   "", "No trips decoded successfully; inspect `decode_summary.csv` for reasons."]
    report += ["", "The classifier-only row selects the highest-probability road candidate independently for each observation. The HMM row uses the same emissions with directed-network transition scores and Viterbi decoding. The HMM route is required to be directed-continuous; disconnected transitions are reported as decode failures rather than silently bridged.", ""]
    (output_dir / "route_method_comparison.md").write_text("\n".join(report), encoding="utf-8")
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

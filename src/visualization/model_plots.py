"""Plots for observed model-comparison, tuning, and route-decoding results."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _bar(frame, metric, output, title, ylabel, *, error=None, lower_is_better=False):
    if metric not in frame or frame.empty:
        return None
    data = frame.dropna(subset=[metric]).sort_values(metric, ascending=lower_is_better)
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    yerr = data[error].to_numpy() if error and error in data else None
    ax.bar(data["model"].astype(str), data[metric].astype(float), yerr=yerr,
           capsize=4, color=plt.get_cmap("tab10").colors[:len(data)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_cross_validation(cv_results, output_dir):
    out = Path(output_dir)
    return _bar(cv_results, "cv_macro_f1_mean", out / "cross_validation_macro_f1.png",
                "Group cross-validation Macro F1 by model", "Macro F1",
                error="cv_macro_f1_std")


def plot_student_comparison(results, output_dir):
    out = Path(output_dir)
    plots = {}
    for metric, title, ylabel, lower in (
        ("candidate_macro_f1", "Student-test Candidate Macro F1", "Macro F1", False),
        ("student_top1_accuracy", "Student-test Top-1 Accuracy", "Accuracy", False),
        ("student_top1_macro_f1", "Student-test Top-1 Macro F1", "Macro F1", False),
        ("log_loss", "Student-test Candidate Log Loss", "Log loss (lower is better)", True),
    ):
        path = _bar(results, metric, out / f"{metric}.png", title, ylabel,
                    lower_is_better=lower)
        if path:
            plots[metric] = str(path)
    return plots


def plot_tuned_vs_untuned(comparison, output_dir):
    out = Path(output_dir)
    if comparison.empty:
        return {}
    plots = {}
    metrics = (
        ("cv_macro_f1_mean", "Group cross-validation Macro F1", "Macro F1", False),
        ("candidate_macro_f1", "Student-test Candidate Macro F1", "Macro F1", False),
        ("student_top1_accuracy", "Student-test Top-1 Accuracy", "Accuracy", False),
        ("student_top1_macro_f1", "Student-test Top-1 Macro F1", "Macro F1", False),
        ("log_loss", "Student-test Candidate Log Loss", "Log loss (lower is better)", True),
        ("inference_seconds_per_row", "Student-test Inference Time per Candidate",
         "Seconds per candidate (lower is better)", True),
    )
    for metric, title, ylabel, lower in metrics:
        if metric not in comparison:
            continue
        data = comparison.dropna(subset=[metric])
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        error = data["cv_macro_f1_std"] if metric == "cv_macro_f1_mean" and "cv_macro_f1_std" in data else None
        ax.bar(data["version"], data[metric], yerr=error, capsize=4,
               color=["#718096", "#2b8a3e"][:len(data)])
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        path = out / f"tuned_vs_untuned_{metric}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots[metric] = str(path)
    return plots


def plot_route_method_comparison(results, output_dir):
    out = Path(output_dir)
    metrics = (
        ("classifier_only_top1_accuracy", "hmm_top1_accuracy", "Top-1 Accuracy"),
        ("classifier_only_macro_f1", "hmm_macro_f1", "Macro F1"),
    )
    plots = {}
    for raw_col, hmm_col, label in metrics:
        if raw_col not in results or hmm_col not in results or results.empty:
            continue
        paired = results
        if "status" in paired:
            paired = paired[paired.status.eq("ok")]
        paired = paired.dropna(subset=[raw_col, hmm_col])
        if paired.empty:
            continue
        means = [paired[raw_col].mean(), paired[hmm_col].mean()]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Classifier only", "Classifier + HMM"], means,
               color=["#718096", "#2b8a3e"])
        ax.set(title=f"Student-test route comparison: {label}", ylabel=label)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        path = out / f"route_method_{raw_col.removeprefix('classifier_only_')}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots[label] = str(path)

        # Preserve the per-trip paired results so the chart's aggregate is auditable.
        paired = paired[["trip_id", raw_col, hmm_col]].dropna()
        if not paired.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            x = range(len(paired))
            ax.plot(x, paired[raw_col], "o-", label="Classifier only")
            ax.plot(x, paired[hmm_col], "o-", label="Classifier + HMM")
            ax.set(title=f"Per-trip route comparison: {label}", xlabel="Student-test trip",
                   ylabel=label, xticks=x, xticklabels=paired.trip_id.astype(str))
            ax.tick_params(axis="x", rotation=30)
            ax.legend()
            ax.grid(alpha=.25)
            fig.tight_layout()
            pair_path = out / f"route_method_{raw_col.removeprefix('classifier_only_')}_per_trip.png"
            fig.savefig(pair_path, dpi=160)
            plt.close(fig)
    return plots

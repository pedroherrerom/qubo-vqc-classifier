"""Reporting and documentation visual graph generation protocols."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

import matplotlib.ticker as ticker


def set_academic_style():
    """Enforce strict typographic standards for analytical documentation."""
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 8,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.4, "grid.linestyle": "--",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _save_figure(save_path: Path, logger: logging.Logger) -> Path:
    """Implement graceful fallback structures for I/O graphic persistence collisions."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(save_path, bbox_inches="tight")
        return save_path
    except PermissionError:
        for idx in range(1, 100):
            fallback = save_path.with_name(f"{save_path.stem}_{idx}{save_path.suffix}")
            if not fallback.exists():
                plt.savefig(fallback, bbox_inches="tight")
                logger.warning("Saved fallback: %s", fallback.name)
                return fallback
        raise


def _as_df(x):
    """Normalize input logic targets directly to structural DataFrames."""
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return x
    p = Path(x)
    return pd.read_csv(p) if p.is_file() else None


def _plot_split(df: pd.DataFrame, metric: str, label: str, color: str, linestyle: str = "-") -> int:
    """Plot distributed mapping lines evaluating mean boundaries against standard operational errors."""
    sub = df[df["Metric"] == metric]
    if sub.empty:
        return 0
    grp = sub.groupby("Epoch")["Value"].agg(["mean", "std"]).reset_index()
    grp["std"] = grp["std"].fillna(0)
    plt.plot(grp["Epoch"], grp["mean"], label=f"{label} Mean", color=color, linewidth=2, linestyle=linestyle)
    plt.fill_between(grp["Epoch"], grp["mean"] - grp["std"], grp["mean"] + grp["std"], color=color, alpha=0.15, label=rf"{label} $\pm1 \sigma$")
    return int(sub["Run"].max())


def plot_pso_trajectory(train_csv, outdir, logger):
    """Render optimal boundary convergences tracing system generation curves."""
    set_academic_style()
    df = _as_df(train_csv)
    if df is None or df.empty:
        logger.warning("No historical CSV found; skipping PSO trajectory.")
        return
        
    plt.figure(figsize=(7, 5))
    n_fit = _plot_split(df, "fitness", "Global Best", "indigo", "-")
    n_mean = _plot_split(df, "mean_best", "Swarm Mean", "mediumorchid", "--")
    
    if n_fit > 0 or n_mean > 0:
        n = max(n_fit, n_mean)
        plt.title("PSO Optimization Trajectory", pad=15, fontsize = 13)
        plt.xlabel("Generation", fontsize = 11)
        plt.ylabel("Log-Probability", fontsize = 11)
        plt.legend(loc="lower right")
        
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        path = _save_figure(outdir / "quantum_pso_trajectory.png", logger)
        logger.info("Saved PSO trajectory curve: %s", path.name)
    plt.close()


def plot_final_loss_distribution(train_csv, outdir, logger):
    """Assess and display cross-entropy terminal bounds utilizing boxplot distribution matrices."""
    df = _as_df(train_csv)
    if df is None or df.empty or df[df["Metric"] == "final_loss"].empty:
        logger.warning("No final_loss rows; skipping loss distribution.")
        return
        
    set_academic_style()
    final_losses = df[df["Metric"] == "final_loss"]
    palette = sns.color_palette("Purples", 3)
    fig, ax = plt.subplots(figsize=(5, 5))
    
    sns.boxplot(y="Value", data=final_losses, ax=ax, color=palette[0], width=0.3, showfliers=False, linewidth=1.2, linecolor="#333333")
    sns.stripplot(y="Value", data=final_losses, ax=ax, color=palette[2], alpha=0.8, jitter=0.05, size=7, marker="o", edgecolor="white", linewidth=0.5)
    ax.set_title(f"Final CUNQA Cross-Entropy Loss", pad=15, fontsize=13)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    ax.set_xticks([])
    
    path = _save_figure(outdir / "quantum_final_loss_distribution.png", logger)
    logger.info("Saved final loss distribution: %s", path.name)
    plt.close(fig)


def plot_confusion_matrix_academic(predictions_csv, outdir, logger):
    """Confusion matrix with per-cell mean ± std across runs."""
    df = _as_df(predictions_csv)
    if df is None or df.empty:
        logger.warning("No predictions CSV; skipping confusion matrix.")
        return
    if "True_Label" not in df.columns or "Predicted_Label" not in df.columns:
        logger.error("Predictions CSV missing True_Label / Predicted_Label columns.")
        return
        
    set_academic_style()
    runs = int(df["Run"].max()) if "Run" in df.columns else 1

    if "Run" in df.columns and runs > 1:
        cms = np.array([confusion_matrix(g["True_Label"], g["Predicted_Label"], labels=[0, 1]) for _, g in df.groupby("Run")])
        cm_mean, cm_std = cms.mean(axis=0), cms.std(axis=0)
        annot = np.array([[rf"{cm_mean[i, j]:.0f} $\pm$ {cm_std[i, j]:.0f}" for j in range(cm_mean.shape[1])] for i in range(cm_mean.shape[0])], dtype=object)
        plot_data = cm_mean
    else:
        plot_data = confusion_matrix(df["True_Label"], df["Predicted_Label"], labels=[0, 1])
        annot = np.array([[str(v) for v in row] for row in plot_data], dtype=object)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(plot_data, annot=annot, fmt="", cmap="Purples", vmax=1.05 * np.max(plot_data),
                cbar=True, linewidths=0.5, linecolor="white",
                xticklabels=["Negative Targets (0)", "Positive Classes (1)"], yticklabels=["Negative Targets (0)", "Positive Classes (1)"],
                annot_kws={"size": 14, "family": "serif"}, ax=ax)
                
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.2)

    ax.set_title(f"VQC Confusion Matrix", pad=15, fontsize=13)
    ax.set_xlabel("Predicted Label", labelpad=10, fontsize=11)
    ax.set_ylabel("True Label", labelpad=10, fontsize=11)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    path = _save_figure(outdir / "quantum_confusion_matrix.png", logger)
    logger.info("Saved confusion matrix: %s", path.name)
    plt.close(fig)


def plot_model_comparison_bars(comparison, outdir, logger, runs=None):
    """Grouped bars (mean) with the real per-run values scattered on top.
    No error bars; legend on a single row."""
    comp = _as_df(comparison)
    runs_df = _as_df(runs)
    if comp is None or comp.empty:
        logger.warning("No comparison data; skipping model comparison bar plot.")
        return
        
    set_academic_style()

    models = list(dict.fromkeys(comp["Model"]))
    metrics = list(dict.fromkeys(comp["Metric"]))
    mean_p = comp.pivot(index="Model", columns="Metric", values="Mean").reindex(index=models, columns=metrics)

    n_models, n_metrics = len(models), len(metrics)
    x = np.arange(n_models)
    width = 0.7 / max(n_metrics, 1)
    offsets = [x + (i - n_metrics / 2 + 0.5) * width for i in range(n_metrics)]
    palette = sns.color_palette("Purples", n_metrics)

    if runs_df is None:
        logger.warning("No model_comparison_runs.csv; drawing bars only (no scatter).")

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_models), 5.5))
    for j, metric in enumerate(metrics):
        means = mean_p[metric].to_numpy(dtype=float)
        ax.bar(offsets[j], means, width, label=metric, color=palette[j], edgecolor="#333333", linewidth=0.8, zorder=3)

        if runs_df is not None and {"Model", "Metric", "Value"}.issubset(runs_df.columns):
            for i, model in enumerate(models):
                vals = runs_df[(runs_df["Model"] == model) & (runs_df["Metric"] == metric)]["Value"].to_numpy(dtype=float)
                if vals.size > 1:  
                    xc = float(offsets[j][i])
                    xj = np.random.uniform(xc - width * 0.25, xc + width * 0.25, size=vals.size) 
                    ax.scatter(xj, vals, s=25, color=palette[j], marker=".", zorder=15, alpha=0.6, edgecolors="black", linewidths=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison", pad=15, fontsize=13)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=n_metrics, frameon=False, fontsize = 10)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.grid(axis="x", visible=False)
    
    path = _save_figure(outdir / "model_comparison_bars.png", logger)
    logger.info("Saved model comparison bar chart: %s", path.name)
    plt.close(fig)

def plot_quantum_roc_curves(predictions_csv, outdir, logger: logging.Logger) -> None:
    """Per-run and mean ROC curve for the VQC across all runs."""
    from sklearn.metrics import roc_curve, roc_auc_score

    df = _as_df(predictions_csv)
    if df is None or df.empty:
        logger.warning("No predictions CSV; skipping quantum ROC curves.")
        return
    if "Prob_1" not in df.columns:
        logger.warning("predictions CSV has no Prob_1 column; skipping quantum ROC curves.")
        return

    set_academic_style()
    palette = sns.color_palette("Purples", int(df["Run"].max()) + 2)
    fig, ax = plt.subplots(figsize=(7, 6))

    all_fpr = np.linspace(0, 1, 200)
    tpr_interp_runs = []

    for run_id, grp in df.groupby("Run"):
        y_true = grp["True_Label"].to_numpy(dtype=int)
        y_prob = grp["Prob_1"].to_numpy(dtype=float)
        try:
            auc = roc_auc_score(y_true, y_prob)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax.plot(fpr, tpr, lw=1, alpha=0.35, color=palette[int(run_id)], label=f"Run {int(run_id)} (AUC={auc:.3f})")
            tpr_interp_runs.append(np.interp(all_fpr, fpr, tpr))
        except Exception as exc:
            logger.warning("ROC skipped for run %s: %s", run_id, exc)

    if tpr_interp_runs:
        mean_tpr = np.mean(tpr_interp_runs, axis=0)
        std_tpr  = np.std(tpr_interp_runs, axis=0)
        mean_auc = roc_auc_score(
            df["True_Label"].to_numpy(dtype=int),
            df["Prob_1"].to_numpy(dtype=float),
        )
        ax.plot(all_fpr, mean_tpr, lw=2.5, color=palette[-1],
                label=f"Mean (AUC={mean_auc:.3f})")
        ax.fill_between(all_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color=palette[-1], alpha=0.15, label=r"Mean $\pm1\sigma$")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves - VQC", pad=15, fontsize=13)
    ax.legend(fontsize=8, loc="lower right")

    path = _save_figure(Path(outdir) / "quantum_roc_curves.png", logger)
    logger.info("Saved quantum ROC curves: %s", path.name)
    plt.close(fig)
"""High-level QML pipeline orchestration module.

Executes one of three operational modes dispatched by run_pipeline():

1. SINGLE-NODE (Default):
   Executes the standard sequential workflow. Creates a numbered experiment directory, 
   preprocesses data, selects features, executes all independent runs sequentially, 
   and exports raw metrics alongside cross-model comparison tables.

2. JOB-ARRAY TASK (Triggered by --exp-dir without --aggregate):
   Executes a targeted parallel block. Uses a pre-allocated experiment directory. 
   Re-derives identical deterministic splits and QUBO feature subsets. Executes 
   runs bound by [N, N+C), writing isolated task outputs. Excludes global comparisons.

3. AGGREGATE (Triggered by --exp-dir combined with --aggregate):
   Consolidates output artifacts. Merges all isolated task data into canonical 
   experiment-level structures. Recomputes classical and QSVC baselines globally, 
   and generates the final model-comparison analytics.

Note on Determinism:
In array/aggregate modes, QUBO-QFS is executed independently per node. A fixed 
seed is enforced to guarantee identical feature subset selection across all 
independent processes, preserving cross-run statistical validity.
"""

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA

from .data_processing import load_and_preprocess
from .experiment import make_experiment_dir, setup_logging
from .feature_selection import select_features_qfs
from .serialization import NpEncoder
from .training import run_vqc


# --- Shared Orchestration Helpers ---

def _banner(logger, args, mode: str) -> None:
    """Print the standardized execution preamble."""
    logger.info("\n==============\n== QML Pipeline [%s] ==\n==============\n", mode)
    logger.info("Execution Stages: %s", args.stages)
    logger.info("Total Runs: %d | Global Seed: %d", args.num_runs, args.seed)
    n_workers = getattr(args, "vqc_n_workers", getattr(args, "vqc_n_qpus", 1))
    logger.info("VQC Configuration: Optimizer=%s | Max Iterations=%d | Shots=%d | Workers=%d",
                args.optimizer, args.opt_maxiter, args.vqc_num_shots, n_workers)


def _prepare_data(args, logger):
    """Execute deterministic preprocessing and feature selection bounds.

    Returns:
        tuple: (X_train_sel, X_test_sel, y_train, y_test, selected_features)
    """
    base_seed = args.seed
    logger.info("----------- STAGE 1 - Data Preprocessing -----------")
    X_train_s, X_test_s, y_train_np, y_test_np, all_features = load_and_preprocess(
        data_path=args.data,
        target=args.target,
        test_size=args.test_size,
        seed=base_seed,
        id_cols=args.id_cols,
        train_path=args.train_path,
        test_path=args.test_path,
        logger=logger,
        max_samples=args.max_samples,
    )

    logger.info("----------- STAGE 2 - Feature Dimensionality Reduction -----------")
    if args.pca:
        n_components = args.k
        logger.info("Applying Principal Component Analysis (Target Components: %s)", args.k)
        pca_model = PCA(n_components=n_components, random_state=base_seed)
        X_train_pca = pca_model.fit_transform(X_train_s)
        X_test_pca = pca_model.transform(X_test_s)

        n_comps = getattr(pca_model, "n_components_", pca_model.n_components)
        try:
            n_comps = int(n_comps)
        except Exception:
            n_comps = len(pca_model.explained_variance_)

        pc_cols = [f"PC{i+1}" for i in range(n_comps)]
        X_train_s = pd.DataFrame(X_train_pca, index=X_train_s.index, columns=pc_cols)
        X_test_s = pd.DataFrame(X_test_pca, index=X_test_s.index, columns=pc_cols)

        explained = float(pca_model.explained_variance_ratio_.sum())
        logger.info("PCA converged to %d components (Total explained variance: %.4f).", n_comps, explained)
        selected_features = pc_cols

    elif "annealing" in args.stages:
        selected_features = select_features_qfs(
            X_train_s=X_train_s,
            y_train=y_train_np,
            k=args.k,
            num_reads=args.sa_num_reads,
            B=args.sa_bins,
            epsilon=args.sa_epsilon,
            seed=base_seed,  # Enforces reproducible subset mapping across distributed tasks
            logger=logger,
        )
    else:
        logger.info("Bypassing reduction. Utilizing full feature space (%d total).", len(all_features))
        selected_features = all_features

    X_train_sel = X_train_s[selected_features]
    X_test_sel = X_test_s[selected_features]
    logger.info("Selected Subset (%d features): %s", len(selected_features), selected_features)
    
    return X_train_sel, X_test_sel, y_train_np, y_test_np, selected_features


def _classical_baseline(X_train_sel, X_test_sel, y_train, y_test, logger) -> dict:
    """Evaluate standard classical ML baselines for comparative analytics."""
    logger.info("----------- Classical Baseline Evaluation -----------")
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from .model_comparison import binary_metrics

    classical_metrics = {}
    for name, clf in [("LogReg", LogisticRegression(max_iter=1000)),
                      ("SVM-RBF", SVC(kernel="rbf", probability=True))]:
        clf.fit(X_train_sel, y_train)
        y_pred = clf.predict(X_test_sel)
        y_prob = clf.predict_proba(X_test_sel)[:, 1]
        
        m = binary_metrics(y_test, y_pred, y_prob)
        classical_metrics[name] = m
        logger.info("[Classical Baseline | %s] Acc=%.4f F1=%.4f ROC-AUC=%.4f",
                    name, m["Accuracy"], m["F1"], m["ROC-AUC"])
        
    return classical_metrics


def _run_runs(args, base_seed, X_train_sel, X_test_sel, y_train, y_test,
              num_qubits, run_indices, out_dir, logger) -> list:
    """Execute continuous quantum training across specified target bounds."""
    metrics_list = []
    for g in run_indices:
        args.seed = base_seed + g
        logger.info("\n=================================================="
                    "\n >>> VQC EXECUTION CYCLE %d / %d (Seed Tracker: %d)"
                    "\n==================================================",
                    g + 1, args.num_runs, args.seed)
        logger.info("----------- STAGE 3 - Quantum VQC Optimization -----------")
        
        m = run_vqc(
            args=args,
            X_train=X_train_sel,
            X_test=X_test_sel,
            y_train=y_train,
            y_test=y_test,
            num_qubits=num_qubits,
            outdir=out_dir,
            logger=logger,
            run_idx=g,
        )
        metrics_list.append(m)
        
    return metrics_list


def _write_raw_metrics(out_dir: Path, vqc_runs_metrics: list) -> list:
    """Persist lightweight metric evaluations, discarding dense loss histories."""
    slim = [{k: v for k, v in m.items() if k != "loss_history"} for m in vqc_runs_metrics]
    with open(Path(out_dir) / "quantum_raw_metrics.json", "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=4, cls=NpEncoder)
    return slim


def _write_model_comparison(args, X_train_sel, X_test_sel, y_train, y_test, vqc_runs_metrics, classical_metrics, base_seed, out_dir, logger) -> None:
    """Consolidate standard model tracking into tabular format."""
    try:
        from .model_comparison import build_model_comparison
        comparison_df, comparison_runs_df = build_model_comparison(
            args=args, X_train_sel=X_train_sel, X_test_sel=X_test_sel,
            y_train=y_train, y_test=y_test,
            vqc_runs_metrics=vqc_runs_metrics, classical_metrics=classical_metrics,
            base_seed=base_seed, logger=logger,
        )
        comparison_df.to_csv(Path(out_dir) / "model_comparison.csv", index=False)
        comparison_runs_df.to_csv(Path(out_dir) / "model_comparison_runs.csv", index=False)
        logger.info("Successfully serialized structural model comparison matrices.")
    except Exception:
        logger.exception("Model comparison matrix consolidation failed. Core VQC metrics preserved.")


# --- Execution Mode 1: Single-Node (Sequential Default) ---

def _run_single_node(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    qml_outdir = make_experiment_dir(outdir, args) if "quantum" in args.stages else None
    logger = setup_logging(qml_outdir=qml_outdir)
    _banner(logger, args, "Single-Node Standard Execution")
    logger.info("Workspace Root: %s", outdir.resolve())

    base_seed = args.seed
    X_train_sel, X_test_sel, y_train_np, y_test_np, selected_features = _prepare_data(args, logger)
    classical_metrics = _classical_baseline(X_train_sel, X_test_sel, y_train_np, y_test_np, logger)

    vqc_runs_metrics = []
    t0 = time.time()
    
    if "quantum" in args.stages:
        vqc_runs_metrics = _run_runs(
            args, base_seed, X_train_sel, X_test_sel, y_train_np, y_test_np,
            len(selected_features), range(args.num_runs), qml_outdir, logger,
        )
        
    total = time.time() - t0
    logger.info("\nPipeline total Time: %.2f seconds (%.2f minutes).", total, total / 60)
    logger.info("----------- SERIALIZATION PROTOCOL -----------")

    if qml_outdir and "quantum" in args.stages and vqc_runs_metrics:
        _write_raw_metrics(qml_outdir, vqc_runs_metrics)
        _write_model_comparison(
            args, X_train_sel, X_test_sel, y_train_np, y_test_np,
            vqc_runs_metrics, classical_metrics, base_seed, qml_outdir, logger,
        )

    logger.info(
        "\n============================================================"
        "\n\tPipeline execution complete. Artifacts stored at: %s"
        "\n============================================================",
        outdir.resolve(),
    )


# --- Execution Mode 2: Distributed Job-Array Task ---

def _run_task(args) -> None:
    exp_dir = Path(args.exp_dir)
    offset = int(args.run_offset)
    requested = args.run_count if args.run_count is not None else (args.num_runs - offset)
    count = max(0, min(int(requested), args.num_runs - offset))

    task_dir = exp_dir / "tasks" / f"task_{offset:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(qml_outdir=task_dir)
    _banner(logger, args, f"Distributed Task Chunk (Offset={offset}, Bounds={count})")
    logger.info("Global Experiment directory: %s", exp_dir.resolve())
    logger.info("Isolated Task directory: %s", task_dir.resolve())

    base_seed = args.seed
    X_train_sel, X_test_sel, y_train_np, y_test_np, selected_features = _prepare_data(args, logger)

    if count <= 0:
        logger.warning("Nothing to do: offset %d is beyond number of runs: %d.", offset, args.num_runs)
        return

    t0 = time.time()
    run_indices = range(offset, offset + count)
    
    vqc_runs_metrics = _run_runs(
        args, base_seed, X_train_sel, X_test_sel, y_train_np, y_test_np,
        len(selected_features), run_indices, task_dir, logger,
    )
    
    logger.info("\nTask execution finalized. Number of runs: %d sequence(s) in %.2f seconds (%.2f minutes).",
                count, time.time() - t0, (time.time() - t0) / 60)

    if vqc_runs_metrics:
        _write_raw_metrics(task_dir, vqc_runs_metrics)
        
    logger.info("Task chunk protocol complete -> %s", task_dir.resolve())


# --- Execution Mode 3: Array Aggregation ---

def _run_aggregate(args) -> None:
    exp_dir = Path(args.exp_dir)
    logger = setup_logging(qml_outdir=exp_dir)
    _banner(logger, args, "Global Matrix Aggregation")
    logger.info("Experiment directory Architecture: %s", exp_dir.resolve())

    base_seed = args.seed
    
    # Mirror identical pre-processing states to ensure baselines align perfectly with node computations
    X_train_sel, X_test_sel, y_train_np, y_test_np, _ = _prepare_data(args, logger)
    classical_metrics = _classical_baseline(X_train_sel, X_test_sel, y_train_np, y_test_np, logger)

    tasks_root = exp_dir / "tasks"
    task_dirs = sorted(p for p in tasks_root.iterdir() if p.is_dir()) if tasks_root.exists() else []
    
    if not task_dirs:
        logger.error("No task directories under %s; nothing to aggregate.", tasks_root)
        return
    logger.info("Found %d task director(ies) to merge.", len(task_dirs))

    hist_frames, pred_frames, raw_runs = [], [], []
    for td in task_dirs:
        hp = td / "quantum_train_historical.csv"
        pp = td / "quantum_aggregated_predictions.csv"
        jp = td / "quantum_raw_metrics.json"
        
        if hp.exists():
            hist_frames.append(pd.read_csv(hp))
        if pp.exists():
            pred_frames.append(pd.read_csv(pp))
        if jp.exists():
            with open(jp) as f:
                raw_runs.extend(json.load(f))

    if hist_frames:
        hist = (pd.concat(hist_frames, ignore_index=True)
                  .sort_values(["Run", "Epoch"]).reset_index(drop=True))
        hist.to_csv(exp_dir / "quantum_train_historical.csv", index=False)
        logger.info("Successfully merged global training records (%d indices).", len(hist))
    else:
        logger.warning("Historical training csv block missing.")

    if pred_frames:
        preds = (pd.concat(pred_frames, ignore_index=True)
                   .sort_values(["Run"]).reset_index(drop=True))
        preds.to_csv(exp_dir / "quantum_aggregated_predictions.csv", index=False)
        logger.info("Successfully merged classification bounds predictions (%d targets).", len(preds))
    else:
        logger.warning("Target class prediction tracking frames missing.")

    if raw_runs:
        with open(exp_dir / "quantum_raw_metrics.json", "w", encoding="utf-8") as f:
            json.dump(raw_runs, f, indent=4, cls=NpEncoder)
        logger.info("Successfully integrated raw JSON metrics (%d evaluation targets).", len(raw_runs))
        
        _write_model_comparison(
            args, X_train_sel, X_test_sel, y_train_np, y_test_np,
            raw_runs, classical_metrics, base_seed, exp_dir, logger,
        )
    else:
        logger.error("Raw metric integration missing. Bypassing model comparison array build.")

    logger.info(
        "\n============================================================"
        "\n\tGlobal Distributed Aggregation Finalized."
        "\n\tWorkspace output: %s"
        "\n\tRecommended Visualization Path: python main_plotting.py --dir %s"
        "\n============================================================",
        exp_dir.resolve(), exp_dir.resolve(),
    )


# --- Global Routing Dispatcher ---

def run_pipeline(args) -> None:
    """Interpret CLI directives routing to isolated task, global merge, or standard procedures."""
    if getattr(args, "aggregate", False):
        return _run_aggregate(args)
    if getattr(args, "exp_dir", None):
        return _run_task(args)
    return _run_single_node(args)
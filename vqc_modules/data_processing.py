"""Dataset ingestion and matrix transformation utilities for the QML pipeline."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_and_preprocess(
    target,
    seed,
    logger,
    id_cols=None,
    # --- dual-CSV mode ---
    train_path=None,
    test_path=None,
    # --- legacy single-CSV mode ---
    data_path=None,
    test_size=0.2,
    max_samples=None,):
    """Load and structure a raw CSV dataset into a standardized tensor format.

    Supports two input modes:
    - Dual-CSV mode (recommended): pass train_path and test_path
      pointing to pre-split files.  The internal train/test split is skipped entirely and the official partitioning is
      preserved, which keeps results directly comparable with published benchmarks.
      
    - Legacy single-CSV mode: pass data_path and test_size; the
      function performs a stratified random split as before.

    The pipeline process inherently executing:
    1. Identifier / ID column elimination (id_cols, e.g. SMILES).
    2. Uninformative constant column elimination (fitted on train only).
    3. High-cardinality categorical column elimination.
    4. Categorical variable encoding logic.
    5. Stratified randomized subsampling (legacy mode only).
    6. Median value imputation (fitted on train, applied to test).
    7. Feature bounds standardization mappings to [0, π] (fitted on train).

    Args:
        target (str): Name of the binary target column.
        seed (int): Random seed for reproducibility across cluster nodes.
        logger: Python logger instance.
        id_cols (list[str] | None): Columns to drop unconditionally (identifiers,
            free-text fields, etc.).  SMILES should be listed here for the
            DIA dataset.  Defaults to an empty list.
        train_path (str | None): Path to the training CSV (dual-CSV mode).
        test_path (str | None): Path to the test CSV (dual-CSV mode).
        data_path (str | None): Path to a single CSV (legacy mode).
        test_size (float): Fraction reserved for testing in legacy mode.
        max_samples (int | None): Cap on training samples in legacy mode.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train_array, y_test_array,
                feature_names)
    """
    if id_cols is None:
        id_cols = []

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    dual_mode = train_path is not None and test_path is not None
    legacy_mode = data_path is not None
    if not dual_mode and not legacy_mode:
        raise ValueError(
            "Provide either (train_path + test_path) for dual-CSV mode "
            "or data_path for legacy single-CSV mode."
        )
    if dual_mode and legacy_mode:
        raise ValueError(
            "Ambiguous input: received both data_path and train_path/test_path. "
            "Use one mode at a time."
        )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if dual_mode:
        logger.info("Dual-CSV mode — train: %s | test: %s", train_path, test_path)
        df_train = pd.read_csv(train_path).copy()
        df_test = pd.read_csv(test_path).copy()
        logger.info("Raw shapes — train: %s | test: %s", df_train.shape, df_test.shape)
    else:
        logger.info("Legacy single-CSV mode — path: %s", data_path)
        df_full = pd.read_csv(data_path).copy()
        logger.info("Initial dataset shape: %s", df_full.shape)

    # ------------------------------------------------------------------
    # Target cleaning (drop rows with missing label)
    # ------------------------------------------------------------------
    def _drop_nan_target(df, split_name):
        n_before = len(df)
        df = df.dropna(subset=[target])
        dropped = n_before - len(df)
        if dropped:
            logger.info(
                "[%s] Dropped %d records due to NaN in target '%s'.",
                split_name, dropped, target,
            )
        return df

    if dual_mode:
        df_train = _drop_nan_target(df_train, "train")
        df_test = _drop_nan_target(df_test, "test")
    else:
        df_full = _drop_nan_target(df_full, "full")

    # ------------------------------------------------------------------
    # Drop identifier / irrelevant columns (id_cols)
    # ------------------------------------------------------------------
    if id_cols:
        if dual_mode:
            df_train.drop(columns=id_cols, errors="ignore", inplace=True)
            df_test.drop(columns=id_cols, errors="ignore", inplace=True)
        else:
            df_full.drop(columns=id_cols, errors="ignore", inplace=True)
        logger.info("Dropped ID/irrelevant columns: %s", id_cols)

    # ------------------------------------------------------------------
    # Constant column elimination — fitted on train only
    #    Columns that are constant in train carry no information; they are
    #    removed from both splits to keep the feature space consistent.
    # ------------------------------------------------------------------
    ref_df = df_train if dual_mode else df_full
    constant_cols = [c for c in ref_df.columns if c != target and ref_df[c].nunique() == 1]
    if constant_cols:
        if dual_mode:
            df_train.drop(columns=constant_cols, errors="ignore", inplace=True)
            df_test.drop(columns=constant_cols, errors="ignore", inplace=True)
        else:
            df_full.drop(columns=constant_cols, errors="ignore", inplace=True)
        logger.info("Dropped constant columns (identified on train): %s", constant_cols)

    # ------------------------------------------------------------------
    # High-cardinality categorical column elimination
    # ------------------------------------------------------------------
    ref_df = df_train if dual_mode else df_full
    high_card_cols = [
        c for c in ref_df.select_dtypes("object").columns
        if c != target and ref_df[c].nunique() > 30
    ]
    if high_card_cols:
        if dual_mode:
            df_train.drop(columns=high_card_cols, errors="ignore", inplace=True)
            df_test.drop(columns=high_card_cols, errors="ignore", inplace=True)
        else:
            df_full.drop(columns=high_card_cols, errors="ignore", inplace=True)
        logger.info("Dropped high-cardinality columns: %s", high_card_cols)

    # ------------------------------------------------------------------
    # Categorical encoding
    # ------------------------------------------------------------------
    def _encode_categoricals(df_tr, df_te=None):
        """Fit LabelEncoders on train, apply to both splits."""
        cat_cols = [c for c in df_tr.select_dtypes("object").columns if c != target]
        for col in cat_cols:
            le = LabelEncoder()
            df_tr[col] = le.fit_transform(df_tr[col].astype(str))
            if df_te is not None:
                # Unseen categories become -1 rather than raising an error
                mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
                df_te[col] = df_te[col].astype(str).map(mapping).fillna(-1).astype(int)
        return df_tr, df_te

    if dual_mode:
        df_train, df_test = _encode_categoricals(df_train, df_test)
    else:
        df_full, _ = _encode_categoricals(df_full)

    # ------------------------------------------------------------------
    # Legacy-only: subsampling + train/test split
    # ------------------------------------------------------------------
    if legacy_mode:
        if max_samples is not None and max_samples < len(df_full):
            logger.info("Executing dataset stratification cap at %d entries.", max_samples)
            _, df_full = train_test_split(
                df_full,
                test_size=max_samples,
                random_state=seed,
                stratify=df_full[target],
            )
            logger.info("Subsampled dataset shape: %s", df_full.shape)

        X_full = df_full.drop(columns=[target])
        y_full = df_full[target]
        features = X_full.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full,
            test_size=test_size,
            random_state=seed,
            stratify=y_full,
        )
    else:
        # Dual-CSV: splits are already defined by the two files
        X_train = df_train.drop(columns=[target])
        y_train = df_train[target]
        X_test = df_test.drop(columns=[target])
        y_test = df_test[target]

        # Align columns: test may have columns absent from train after pruning
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        features = X_train.columns.tolist()

    logger.info("Features (%d): %s", len(features), features)
    logger.info(
        "Class balance — train: %s | test: %s",
        y_train.value_counts(normalize=True).round(3).to_dict(),
        y_test.value_counts(normalize=True).round(3).to_dict(),
    )

    # ------------------------------------------------------------------
    # Median imputation — fitted on train, applied to test
    # ------------------------------------------------------------------
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=features, index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=features, index=X_test.index
    )

    # ------------------------------------------------------------------
    # MinMax scaling to [0, π] for rotational angle encoding, Fitted exclusively on train to avoid test information leakage.
    # ------------------------------------------------------------------
    scaler = MinMaxScaler((0, np.pi))
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train_imp), columns=features, index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test_imp), columns=features, index=X_test.index
    )

    logger.info(
        "Final shapes — X_train: %s | X_test: %s",
        X_train_s.shape, X_test_s.shape,
    )

    return (
        X_train_s,
        X_test_s,
        y_train.values.astype(np.float32),
        y_test.values.astype(np.float32),
        features,
    )
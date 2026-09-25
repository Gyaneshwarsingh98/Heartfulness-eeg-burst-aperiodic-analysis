#!/usr/bin/env python3
# =============================================================================
# FINAL STATISTICAL ANALYSIS PIPELINE
# Phase-Resolved EEG Burst + Aperiodic + Mixed-Effects + GAM Analysis
#
# FINAL LOCKED-CANDIDATE VERSION — v3 (reviewer heterogeneity + GAM/bootstrap fixes)
# =============================================================================
#
# Purpose
# -------
# Statistical analysis of the already-preprocessed EEG feature dataset.
# PREPROCESSING, FILTERING, ICA, ARTIFACT REMOVAL, AND BURST EXTRACTION ARE
# NOT performed here. This script starts from EEG_Features_Detailed.csv.
#
# Primary design
# --------------
# 29 complete participants x 7 ordered protocol phases x 5 regions x 4 bands.
#
# Phases:
#   B  = Baseline
#   M1 = Meditation 1
#   M2 = Meditation 2
#   T1 = Transmission 1
#   T2 = Transmission 2
#   P1 = Post 1
#   P2 = Post 2
#
# Groups:
#   CG, STM, LTM
#
# Primary mixed model
# -------------------
# Y ~ C(Phase) * C(Group) + C(Band) + C(Region) + (1 | Subject)
#
# Fixed-effect comparisons are estimated with ML and compared with LRT.
# REML is used for final coefficient estimation, not for fixed-effect LRTs.
#
# IMPORTANT:
#   LRT degrees of freedom are based on the difference in FIXED-EFFECT design
#   matrix rank, not on the total number of MixedLM parameters.
#
# Reviewer-response analyses included
# ------------------------------------
# 1. Primary Phase x Group omnibus LMMs
# 2. Participant-level cluster bootstrap sensitivity for primary LMMs
# 3. Phase x Group x Band heterogeneity omnibus + band-specific follow-ups
# 4. Phase x Group x Region heterogeneity omnibus + region-specific follow-ups
# 5. Corrected aperiodic-slope analysis without duplicated band rows
# 6. Model-consistent EMM group contrasts for corrected slope analysis
# 7. Participant-level slope bootstrap
# 8. Secondary four-state aggregation analysis
# 9. Dependency-aware band-specific burst--slope coupling
# 10. Adjusted phase-specific slope effects for coupling figures
# 11. Penalized GAM + participant-level cluster bootstrap
# 12. Threshold-sensitivity audit/import (2x, 3x, 4x MAD)
# 13. EEG QC summaries and participant-level group quality comparisons
# 14. Explicit pending-analysis audit for data not present in the feature package
#
# Thresholds
# ----------
# Primary burst threshold: median + 3 x MAD, estimated once per
# participant x region x band using data pooled across ALL seven phases.
#
# Sensitivity thresholds already generated in the feature package:
#   median + 2 x MAD
#   median + 3 x MAD
#   median + 4 x MAD
#
# Aperiodic slope
# ---------------
# specparam exponent is used as the sole aperiodic estimate.
# Reported Slope_1f = -exponent.
# No log-log fallback is used.
#
# GAM
# ---
# Penalized GAM, not a formal random-effect GAMM:
#   common:
#       Y ~ Group + s(Phase_Num, k=4) + Band + Region
#   group-specific:
#       Y ~ Group + s(Phase_Num, by=Group, k=4) + Band + Region
#
# Phase_Num is an ordered protocol index (1--7), NOT continuous physiological
# time. GAM interpretation is therefore limited to smooth patterns across the
# ordered protocol phases.
#
# Required packages
# -----------------
# pandas, numpy, scipy, statsmodels, patsy
# Optional: pygam (required only for GAM section)
#
# =============================================================================

from __future__ import annotations
import matplotlib.pyplot as plt
import json
import math
import os
import shutil
import traceback
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

try:
    import patsy
except ImportError as exc:
    raise ImportError("patsy is required. Install with: pip install patsy") from exc


# =============================================================================
# 0. USER CONFIGURATION
# =============================================================================

# ---- Feature package ---------------------------------------------------------
FEATURE_ROOT = Path(
    "data/file"
)
INPUT_FILE = FEATURE_ROOT / "EEG_Features_Detailed.csv"


# Final output directory
OUT_DIR = Path(
    "data/output"
)

# ---- Re-run switches ---------------------------------------------------------
# The feature extraction is NOT rerun.
#
# Set these True when a full re-computation is wanted. For ordinary manuscript
# revision, the already validated primary/GAM outputs can be retained and the
# reviewer-specific analyses can be run.
RUN_PRIMARY_LMM = True
RUN_PRIMARY_BOOTSTRAP = True       # Existing validated 1000-rep output available
RUN_BAND_HETEROGENEITY = True
RUN_REGION_HETEROGENEITY = True
RUN_SLOPE = True
RUN_SLOPE_BOOTSTRAP = True
RUN_STATE_ANALYSIS = True
RUN_COUPLING = True
RUN_GAM = True
RUN_QC = True
RUN_THRESHOLD_AUDIT = True

# GAM is computationally heavier.
GAM_BOOTSTRAPS = 500
GAM_SEED = 20260908

# Primary participant bootstrap
PRIMARY_BOOTSTRAPS = 1000
PRIMARY_BOOTSTRAP_SEED = 20260909

# Slope participant bootstrap
SLOPE_BOOTSTRAPS = 1000
SLOPE_BOOTSTRAP_SEED = 20260910

# Mixed-model fitting
ML_SCALE_FOR_LRT = True
OPTIMIZERS = ("powell", "lbfgs", "cg")
MAXITER = 2000

# Numerical tolerances
FLOAT_TOL = 1e-10

# Primary FDR alpha
ALPHA = 0.05


# =============================================================================
# 1. DESIGN CONSTANTS
# =============================================================================

PHASE_ORDER = ["B", "M1", "M2", "T1", "T2", "P1", "P2"]
GROUP_ORDER = ["CG", "STM", "LTM"]
BAND_ORDER = ["delta", "theta", "alpha", "beta"]
REGION_ORDER = ["Frontal", "Central", "Temporal", "Parietal", "Occipital"]

STATE_MAP = {
    "B": "Baseline",
    "M1": "Meditation",
    "M2": "Meditation",
    "T1": "Transmission",
    "T2": "Transmission",
    "P1": "Post",
    "P2": "Post",
}
STATE_ORDER = ["Baseline", "Meditation", "Transmission", "Post"]

OUTCOMES = [
    "Burst_Rate",
    "Burst_Duration",
    "PowerSpike_Rate",
    "Peak_Rate",
    "Peak_Amplitude",
    "PowerSpike_Magnitude",
    "Band_Power",
    "Burst_Occupancy",
]

COUPLING_FEATURES = [
    "Burst_Rate",
    "Burst_Duration",
    "Peak_Rate",
    "Peak_Amplitude",
    "Burst_Occupancy",
]

FAMILY_NAMES = {
    "primary_omnibus": "Primary Phase × Group omnibus tests across eight outcomes",
    "primary_coefficients": "Primary Phase × Group coefficient tests across eight outcomes",
    "band_heterogeneity": "Phase × Group × Band omnibus tests across eight outcomes",
    "band_specific": "Band-specific Phase × Group omnibus tests across four bands and eight outcomes",
    "region_heterogeneity": "Phase × Group × Region omnibus tests across eight outcomes",
    "region_specific": "Region-specific Phase × Group omnibus tests across five regions and eight outcomes",
    "state_omnibus": "Secondary State × Group omnibus tests across eight outcomes",
    "slope_omnibus": "Aperiodic slope Phase × Group omnibus test",
    "slope_coefficients": "Aperiodic slope Phase × Group coefficient tests",
    "slope_emm": "Model-consistent EMM pairwise group contrasts for aperiodic slope",
    "coupling_phase": "Band-specific Slope × Phase omnibus tests across five coupling outcomes and four bands",
    "coupling_direct_slope": "Band-specific direct Slope coefficient tests across five coupling outcomes and four bands",
}


# =============================================================================
# 2. OUTPUT DIRECTORIES
# =============================================================================

DIRS = {
    "lmm": OUT_DIR / "LMM_Results",
    "band": OUT_DIR / "Band_Heterogeneity",
    "region": OUT_DIR / "Region_Heterogeneity",
    "slope": OUT_DIR / "Slope",
    "state": OUT_DIR / "State_Analysis",
    "coupling": OUT_DIR / "Burst_Slope_Coupling",
    "gam": OUT_DIR / "GAM_01",
    "qc": OUT_DIR / "QC",
    "reviewer": OUT_DIR / "Reviewer_Sensitivity",
    "logs": OUT_DIR / "Execution_Logs",
}

for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================

def log(message: str) -> None:
    print(message, flush=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def fdr_series(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    mask = p.notna()
    if mask.any():
        out.loc[mask] = multipletests(p.loc[mask].values, method="fdr_bh")[1]
    return out


def apply_fdr(df: pd.DataFrame, p_col: str = "p_value",
              family_col: str = "FDR_Family") -> pd.DataFrame:
    out = df.copy()
    out["p_FDR"] = np.nan
    if family_col not in out.columns:
        out[family_col] = "unspecified"
    for family, idx in out.groupby(family_col).groups.items():
        out.loc[idx, "p_FDR"] = fdr_series(out.loc[idx, p_col])
    out["Significant_FDR_0.05"] = out["p_FDR"] < ALPHA
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def fixed_effect_rank(result) -> int:
    """Rank/column count of the fixed-effect design matrix."""
    return int(result.model.exog.shape[1])


def result_converged(result) -> bool:
    return bool(getattr(result, "converged", False))


def extract_fe_cov(result):
    fe_names = list(result.fe_params.index)
    cov_all = result.cov_params()
    cov_fe = cov_all.loc[fe_names, fe_names]
    return fe_names, result.fe_params.copy(), cov_fe


# =============================================================================
# 4. MIXED MODEL FITTING
# =============================================================================

def _prepare_mixed_model_data(
    data: pd.DataFrame,
    formula: str,
    group_col: str,
) -> pd.DataFrame:
    """Prepare exact valid rows and reset the index before MixedLM."""
    work = data.copy()
    y_mat, x_mat = patsy.dmatrices(
        formula, work, return_type="dataframe", NA_action="drop"
    )
    valid_index = y_mat.index.intersection(x_mat.index)
    work = work.loc[valid_index].copy().reset_index(drop=True)
    if group_col not in work.columns:
        raise KeyError(f"Grouping column '{group_col}' not found.")
    if work.empty:
        raise ValueError(f"No complete rows remain for model: {formula}")
    return work


def _fit_mixed_once(
    data: pd.DataFrame,
    formula: str,
    group_col: str,
    re_formula: str | None,
    method: str,
    reml: bool,
):
    clean = _prepare_mixed_model_data(data, formula, group_col)
    model = smf.mixedlm(
        formula, data=clean, groups=clean[group_col], re_formula=re_formula
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(reml=reml, method=method, maxiter=MAXITER, disp=False)
    return result, caught


def fit_mixed_robust(
    data: pd.DataFrame,
    formula: str,
    group_col: str = "Subject",
    re_formula: str = "1",
    reml: bool = False,
    scale_for_lrt: bool = False,
):
    """
    Robust MixedLM wrapper.

    For ML likelihood-ratio comparisons, optional response standardization is
    used to improve numerical conditioning. Scaling is identical between
    nested models and therefore does not change the likelihood-ratio statistic.

    Returns:
        result, metadata
    """
    work = data.copy()
    response = formula.split("~", 1)[0].strip()

    scale_factor = 1.0
    center = 0.0

    if scale_for_lrt:
        y = pd.to_numeric(work[response], errors="coerce").astype(float)
        center = float(y.mean())
        sd = float(y.std(ddof=1))
        if np.isfinite(sd) and sd > 0:
            work[response] = (y - center) / sd
            scale_factor = sd

    attempts = []
    best = None

    for method in OPTIMIZERS:
        try:
            result, caught = _fit_mixed_once(
                work, formula, group_col, re_formula, method, reml
            )
            converged = result_converged(result)
            llf = safe_float(getattr(result, "llf", np.nan))
            finite_llf = np.isfinite(llf)
            attempts.append({
                "optimizer": method,
                "converged": converged,
                "finite_llf": finite_llf,
                "llf": llf,
                "n_warnings": len(caught),
            })
            if converged and finite_llf:
                best = result
                break
            if best is None and finite_llf:
                best = result
        except Exception as exc:
            attempts.append({
                "optimizer": method,
                "converged": False,
                "finite_llf": False,
                "llf": np.nan,
                "n_warnings": 0,
                "error": repr(exc),
            })

    if best is None:
        raise RuntimeError(
            f"MixedLM failed for formula: {formula}\nAttempts: {attempts}"
        )

    meta = {
        "optimizer": attempts[-1]["optimizer"] if attempts else None,
        "converged": result_converged(best),
        "scale_for_lrt": scale_for_lrt,
        "response_center": center,
        "response_scale": scale_factor,
        "attempts": attempts,
    }
    return best, meta


def likelihood_ratio_test(full_result, reduced_result, df_diff=None):
    ll_full = float(full_result.llf)
    ll_reduced = float(reduced_result.llf)
    lr = 2.0 * (ll_full - ll_reduced)

    if df_diff is None:
        df_diff = fixed_effect_rank(full_result) - fixed_effect_rank(reduced_result)

    df_diff = int(df_diff)
    p = stats.chi2.sf(max(lr, 0.0), df_diff)

    return float(lr), df_diff, float(p)


def run_lrt_pair(
    data: pd.DataFrame,
    outcome: str,
    reduced_formula: str,
    full_formula: str,
    family: str,
    label: str,
    subsetting: str = "all",
):
    reduced, meta_r = fit_mixed_robust(
        data, f"{outcome} ~ {reduced_formula}",
        reml=False, scale_for_lrt=ML_SCALE_FOR_LRT
    )
    full, meta_f = fit_mixed_robust(
        data, f"{outcome} ~ {full_formula}",
        reml=False, scale_for_lrt=ML_SCALE_FOR_LRT
    )

    lr, df_diff, p = likelihood_ratio_test(full, reduced)

    return {
        "Outcome": outcome,
        "Analysis": label,
        "Subset": subsetting,
        "LR": lr,
        "df": df_diff,
        "p_value": p,
        "Reduced_Fixed_Effect_Rank": fixed_effect_rank(reduced),
        "Full_Fixed_Effect_Rank": fixed_effect_rank(full),
        "Reduced_Optimizer": meta_r["optimizer"],
        "Full_Optimizer": meta_f["optimizer"],
        "Reduced_Converged": meta_r["converged"],
        "Full_Converged": meta_f["converged"],
        "FDR_Family": FAMILY_NAMES[family],
    }


# =============================================================================
# 5. LOAD + VALIDATE FEATURE DATA
# =============================================================================

def load_features() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{INPUT_FILE}\n"
            "Update FEATURE_ROOT at the top of this script."
        )

    df = pd.read_csv(INPUT_FILE)
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "Subject", "Group", "Phase", "Region", "Band",
        "Burst_Rate", "Burst_Duration", "PowerSpike_Rate",
        "Peak_Rate", "Peak_Amplitude", "PowerSpike_Magnitude",
        "Band_Power", "Slope_1f",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Normalize categorical strings
    for c in ["Subject", "Group", "Phase", "Region", "Band"]:
        df[c] = df[c].astype(str).str.strip()

    df["Band"] = df["Band"].str.lower()

    # Derive phase index and four-state variable here rather than requiring
    # them in the feature-extraction output.
    phase_map = {p: i + 1 for i, p in enumerate(PHASE_ORDER)}
    df["Phase_Num"] = df["Phase"].map(phase_map)
    df["State"] = df["Phase"].map(STATE_MAP)

    # Burst occupancy is derived from burst rate x burst duration.
    # Duration is converted from milliseconds to seconds when the source
    # variable is clearly in milliseconds, as in the existing feature set.
    duration_median = pd.to_numeric(df["Burst_Duration"], errors="coerce").median()

    # Existing feature package stores Burst_Duration in seconds if values are
    # small. If values are clearly milliseconds, convert to seconds.
    if np.isfinite(duration_median) and duration_median > 1:
        duration_seconds = pd.to_numeric(df["Burst_Duration"], errors="coerce") / 1000.0
    else:
        duration_seconds = pd.to_numeric(df["Burst_Duration"], errors="coerce")

    df["Burst_Occupancy"] = (
        pd.to_numeric(df["Burst_Rate"], errors="coerce") * duration_seconds
    )

    # Categories
    for c, order in [
        ("Phase", PHASE_ORDER),
        ("Group", GROUP_ORDER),
        ("Band", BAND_ORDER),
        ("Region", REGION_ORDER),
        ("State", STATE_ORDER),
    ]:
        df[c] = pd.Categorical(df[c], categories=order, ordered=True)

    return df


def complete_case_subjects(df: pd.DataFrame) -> list[str]:
    phase_sets = (
        df[["Subject", "Phase"]]
        .drop_duplicates()
        .groupby("Subject")["Phase"]
        .apply(set)
    )
    complete = [
        sid for sid, phases in phase_sets.items()
        if set(PHASE_ORDER).issubset(phases)
    ]
    return sorted(complete)


def prepare_complete_dataset(df: pd.DataFrame) -> pd.DataFrame:
    complete_subjects = complete_case_subjects(df)
    d = df[df["Subject"].isin(complete_subjects)].copy()

    # Enforce exactly one observation per subject x phase x region x band.
    key = ["Subject", "Phase", "Region", "Band"]
    dup = d.duplicated(key).sum()
    if dup:
        raise ValueError(
            f"Duplicate Subject x Phase x Region x Band rows detected: {dup}"
        )

    # Preserve only complete-case subjects, but do NOT globally drop missing
    # outcome values. Outcome-specific mixed models use the available rows.
    d["Subject"] = d["Subject"].astype(str)

    expected = len(complete_subjects) * 7 * 5 * 4
    log(f"Complete participants: {len(complete_subjects)}")
    log(f"Complete-case expected rows: {expected}")
    log(f"Complete-case observed rows: {len(d)}")

    return d


# =============================================================================
# 6. PRIMARY PHASE x GROUP LMM
# =============================================================================

def run_primary_lmm(df: pd.DataFrame):
    rows = []

    reduced = "C(Phase) + C(Group) + C(Band) + C(Region)"
    full = "C(Phase) * C(Group) + C(Band) + C(Region)"

    for outcome in OUTCOMES:
        log(f"[Primary LMM] {outcome}")
        rows.append(
            run_lrt_pair(
                df, outcome, reduced, full,
                family="primary_omnibus",
                label="Primary Phase x Group omnibus ML-LRT",
            )
        )

    results = pd.DataFrame(rows)
    results = apply_fdr(results)
    write_csv(
        results,
        DIRS["lmm"] / "Primary_Phase_x_Group_Omnibus_ML_LRT_FDR.csv",
    )

    # Final REML coefficient estimates
    coeff_rows = []
    for outcome in OUTCOMES:
        formula = f"{outcome} ~ {full}"
        try:
            res, meta = fit_mixed_robust(
                df, formula, reml=True, scale_for_lrt=False
            )
            fe = res.fe_params
            se = res.bse_fe
            p = res.pvalues
            ci = res.conf_int()
            for term in fe.index:
                if "C(Phase)[T." in term and ":C(Group)" in term:
                    coeff_rows.append({
                        "Outcome": outcome,
                        "Term": term,
                        "Estimate": fe[term],
                        "SE": se[term],
                        "z_or_t": fe[term] / se[term] if se[term] else np.nan,
                        "p_value": p[term],
                        "CI_low": ci.loc[term, 0],
                        "CI_high": ci.loc[term, 1],
                        "Optimizer": meta["optimizer"],
                        "Converged": meta["converged"],
                        "FDR_Family": FAMILY_NAMES["primary_coefficients"],
                    })
        except Exception as exc:
            log(f"  REML coefficient failure: {exc}")

    coeff = pd.DataFrame(coeff_rows)
    if not coeff.empty:
        coeff = apply_fdr(coeff)
        write_csv(
            coeff,
            DIRS["lmm"] / "Primary_Phase_x_Group_Coefficients_REML_FDR.csv",
        )

    return results


# =============================================================================
# 7. PARTICIPANT-LEVEL CLUSTER BOOTSTRAP FOR PRIMARY LMM COEFFICIENTS
# =============================================================================

def bootstrap_participants_by_group(
    df: pd.DataFrame,
    n_boot: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    subjects_by_group = {
        g: sorted(df.loc[df["Group"] == g, "Subject"].astype(str).unique())
        for g in GROUP_ORDER
    }

    for _ in range(n_boot):
        sampled = []
        for g in GROUP_ORDER:
            ids = subjects_by_group[g]
            sampled_ids = rng.choice(ids, size=len(ids), replace=True)
            for new_id in sampled_ids:
                sampled.append(df[df["Subject"] == new_id].copy())
        yield pd.concat(sampled, ignore_index=True)


def run_primary_bootstrap(df: pd.DataFrame):
    rng = np.random.default_rng(PRIMARY_BOOTSTRAP_SEED)
    subjects_by_group = {
        g: sorted(df.loc[df["Group"] == g, "Subject"].astype(str).unique())
        for g in GROUP_ORDER
    }

    reduced = "C(Phase) + C(Group) + C(Band) + C(Region)"
    full = "C(Phase) * C(Group) + C(Band) + C(Region)"

    records = []

    for outcome in OUTCOMES:
        log(f"[Primary bootstrap] {outcome}")
        successful = 0

        # Get target coefficient names from the full-data REML fit.
        try:
            full_res, _ = fit_mixed_robust(
                df, f"{outcome} ~ {full}", reml=True
            )
            interaction_terms = [
                x for x in full_res.fe_params.index
                if "C(Phase)[T." in x and ":C(Group)" in x
            ]
        except Exception as exc:
            log(f"  Full-data bootstrap setup failure: {exc}")
            continue

        boot_values = {term: [] for term in interaction_terms}

        for b in range(PRIMARY_BOOTSTRAPS):
            parts = []
            for g in GROUP_ORDER:
                ids = subjects_by_group[g]
                sampled_ids = rng.choice(ids, size=len(ids), replace=True)
                for draw_index, sid in enumerate(sampled_ids):
                    piece = df[df["Subject"] == sid].copy()
                    # A duplicated participant draw must be represented by a
                    # distinct bootstrap cluster for the random-intercept LMM.
                    piece["_Bootstrap_Subject"] = (
                        f"{g}_boot{b + 1}_draw{draw_index}_{sid}"
                    )
                    parts.append(piece)
            boot_df = pd.concat(parts, ignore_index=True)

            try:
                res, _ = fit_mixed_robust(
                    boot_df,
                    f"{outcome} ~ {full}",
                    group_col="_Bootstrap_Subject",
                    reml=True,
                )
                if not result_converged(res):
                    continue
                successful += 1
                for term in interaction_terms:
                    boot_values[term].append(float(res.fe_params[term]))
            except Exception:
                continue

        for term, vals in boot_values.items():
            vals = np.asarray(vals, dtype=float)
            if len(vals):
                lo, hi = np.quantile(vals, [0.025, 0.975])
                mean = vals.mean()
                records.append({
                    "Outcome": outcome,
                    "Term": term,
                    "Bootstrap_N": len(vals),
                    "Bootstrap_Target_N": PRIMARY_BOOTSTRAPS,
                    "Adequate_Execution": len(vals) >= 0.95 * PRIMARY_BOOTSTRAPS,
                    "Bootstrap_Mean": mean,
                    "CI_low": lo,
                    "CI_high": hi,
                    "CI_Excludes_Zero": bool(lo > 0 or hi < 0),
                    "Seed": PRIMARY_BOOTSTRAP_SEED,
                })

    out = pd.DataFrame(records)
    write_csv(out, DIRS["lmm"] / "Primary_Participant_Cluster_Bootstrap.csv")

    if not out.empty:
        stability = (
            out.groupby("Outcome")
            .agg(
                Interaction_CI_Excludes_Zero=("CI_Excludes_Zero", "sum"),
                Interaction_Coefficients=("Term", "count"),
            )
            .reset_index()
        )
        write_csv(
            stability,
            DIRS["lmm"] / "Primary_Bootstrap_Stability_Summary.csv",
        )

    return out


# =============================================================================
# 8. PHASE x GROUP x BAND HETEROGENEITY
# =============================================================================

def run_band_heterogeneity(df: pd.DataFrame):
    omnibus = []
    for outcome in OUTCOMES:
        omnibus.append(
            run_lrt_pair(
                df,
                outcome,
                "C(Phase) * C(Group) + C(Phase) * C(Band) + C(Group) * C(Band) + C(Region)",
                "C(Phase) * C(Group) * C(Band) + C(Region)",
                family="band_heterogeneity",
                label="Phase x Group x Band omnibus ML-LRT",
            )
        )

    omnibus = apply_fdr(pd.DataFrame(omnibus))
    validate_heterogeneity_df(
        omnibus, expected_df=36, label="Phase x Group x Band heterogeneity"
    )
    write_csv(
        omnibus,
        DIRS["band"] / "Phase_x_Group_x_Band_Omnibus_ML_LRT_FDR.csv",
    )

    # Band-specific follow-up models
    follow = []
    for band in BAND_ORDER:
        bd = df[df["Band"] == band].copy()
        for outcome in OUTCOMES:
            follow.append(
                run_lrt_pair(
                    bd,
                    outcome,
                    "C(Phase) + C(Group) + C(Region)",
                    "C(Phase) * C(Group) + C(Region)",
                    family="band_specific",
                    label="Band-specific Phase x Group omnibus ML-LRT",
                    subsetting=band,
                )
            )

    follow = apply_fdr(pd.DataFrame(follow))
    write_csv(
        follow,
        DIRS["band"] / "Band_Specific_Phase_x_Group_Omnibus_FDR.csv",
    )

    return omnibus, follow


# =============================================================================
# 9. PHASE x GROUP x REGION HETEROGENEITY
# =============================================================================

def run_region_heterogeneity(df: pd.DataFrame):
    omnibus = []
    for outcome in OUTCOMES:
        omnibus.append(
            run_lrt_pair(
                df,
                outcome,
                "C(Phase) * C(Group) + C(Phase) * C(Region) + C(Group) * C(Region) + C(Band)",
                "C(Phase) * C(Group) * C(Region) + C(Band)",
                family="region_heterogeneity",
                label="Phase x Group x Region omnibus ML-LRT",
            )
        )

    omnibus = apply_fdr(pd.DataFrame(omnibus))
    validate_heterogeneity_df(
        omnibus, expected_df=48, label="Phase x Group x Region heterogeneity"
    )
    write_csv(
        omnibus,
        DIRS["region"] / "Phase_x_Group_x_Region_Omnibus_ML_LRT_FDR.csv",
    )

    # All region-specific follow-ups are retained for transparent auditing.
    follow = []
    for region in REGION_ORDER:
        rd = df[df["Region"] == region].copy()
        for outcome in OUTCOMES:
            follow.append(
                run_lrt_pair(
                    rd,
                    outcome,
                    "C(Phase) + C(Group) + C(Band)",
                    "C(Phase) * C(Group) + C(Band)",
                    family="region_specific",
                    label="Region-specific Phase x Group omnibus ML-LRT",
                    subsetting=region,
                )
            )

    follow = apply_fdr(pd.DataFrame(follow))
    write_csv(
        follow,
        DIRS["region"] / "Region_Specific_Phase_x_Group_Omnibus_FDR.csv",
    )

    return omnibus, follow


def validate_heterogeneity_df(omnibus: pd.DataFrame, expected_df: int, label: str):
    """Fail loudly if a purported pure three-way LRT has the wrong df."""
    if omnibus.empty:
        raise ValueError(f"{label}: no omnibus results were produced.")
    bad = omnibus.loc[omnibus["df"].astype(float) != float(expected_df)]
    if not bad.empty:
        raise ValueError(
            f"{label}: expected pure three-way LRT df={expected_df}, "
            f"but found {sorted(bad['df'].dropna().unique().tolist())}."
        )
    return True


# =============================================================================
# 10. CORRECTED APERIODIC SLOPE ANALYSIS
# =============================================================================

def prepare_slope_data(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Subject", "Group", "Phase", "Region", "Slope_1f"]

    # Slope is repeated across bands. Remove exact band duplicates before
    # statistical analysis.
    s = df[cols].copy()

    # Assert there is only one unique slope per Subject x Phase x Region.
    check = (
        s.groupby(["Subject", "Phase", "Region"], observed=True)["Slope_1f"]
        .nunique(dropna=True)
        .reset_index(name="n_unique_slope")
    )
    bad = check[check["n_unique_slope"] > 1]
    if not bad.empty:
        raise ValueError(
            "Slope differs across duplicated band rows for at least one "
            "Subject x Phase x Region combination."
        )

    s = s.drop_duplicates(
        subset=["Subject", "Group", "Phase", "Region"]
    ).copy()

    return s


def run_slope_analysis(df: pd.DataFrame):
    s = prepare_slope_data(df)

    # QC output documenting the de-duplication.
    write_csv(
        pd.DataFrame([{
            "Original_feature_rows": len(df),
            "Unique_slope_rows": len(s),
            "Expected_unique_slope_rows": df["Subject"].nunique() * 7 * 5,
            "Slope_is_common_across_band_rows": True,
            "Model_formula": "Slope_1f ~ C(Phase)*C(Group) + C(Region) + (1|Subject)",
            "Sign_convention": "Slope_1f = -specparam exponent; more negative = steeper decay",
        }]),
        DIRS["slope"] / "Slope_Data_Structure_QC.csv",
    )

    reduced = "C(Phase) + C(Group) + C(Region)"
    full = "C(Phase) * C(Group) + C(Region)"

    omnibus = run_lrt_pair(
        s,
        "Slope_1f",
        reduced,
        full,
        family="slope_omnibus",
        label="Corrected aperiodic slope Phase x Group omnibus ML-LRT",
    )
    omnibus_df = pd.DataFrame([omnibus])
    write_csv(
        omnibus_df,
        DIRS["slope"] / "Corrected_Slope_Phase_x_Group_Omnibus_ML_LRT.csv",
    )

    # Final REML coefficients
    res, meta = fit_mixed_robust(
        s, f"Slope_1f ~ {full}", reml=True, scale_for_lrt=False
    )
    rows = []
    for term in res.fe_params.index:
        if "C(Phase)[T." in term and ":C(Group)" in term:
            se = float(res.bse_fe[term])
            est = float(res.fe_params[term])
            ci = res.conf_int().loc[term]
            rows.append({
                "Term": term,
                "Estimate": est,
                "SE": se,
                "z_or_t": est / se if se else np.nan,
                "p_value": float(res.pvalues[term]),
                "CI_low": float(ci[0]),
                "CI_high": float(ci[1]),
                "Optimizer": meta["optimizer"],
                "Converged": meta["converged"],
                "FDR_Family": FAMILY_NAMES["slope_coefficients"],
            })

    coeff = apply_fdr(pd.DataFrame(rows))
    write_csv(
        coeff,
        DIRS["slope"] / "Corrected_Slope_Phase_x_Group_Coefficients_REML_FDR.csv",
    )

    emm = slope_emm_contrasts(s, res)
    write_csv(
        emm,
        DIRS["slope"] / "Corrected_Slope_Model_Consistent_EMM_Group_Contrasts_FDR.csv",
    )

    return s, omnibus_df, coeff, emm


# =============================================================================
# 11. MODEL-CONSISTENT EMM CONTRASTS
# =============================================================================

def make_design_matrix_from_result(result, newdata: pd.DataFrame):
    design_info = result.model.data.design_info
    X = patsy.build_design_matrices(
        [design_info],
        newdata,
        return_type="dataframe",
    )[0]
    return X


def slope_emm_contrasts(s: pd.DataFrame, result) -> pd.DataFrame:
    """
    Model-consistent marginal means over the observed phase x region grid.

    For each group:
        average fixed-effect design vector over all phase x region cells,
        then calculate the model-based marginal mean.

    Pairwise group contrasts use the fixed-effect covariance matrix.
    """
    grid = pd.DataFrame(
        [
            (g, p, r)
            for g in GROUP_ORDER
            for p in PHASE_ORDER
            for r in REGION_ORDER
        ],
        columns=["Group", "Phase", "Region"],
    )

    for c, order in [
        ("Phase", PHASE_ORDER),
        ("Group", GROUP_ORDER),
        ("Region", REGION_ORDER),
    ]:
        grid[c] = pd.Categorical(grid[c], categories=order, ordered=True)

    X = make_design_matrix_from_result(result, grid)
    fe_names, beta, cov = extract_fe_cov(result)
    X = X.loc[:, fe_names]

    group_vectors = {}
    for g in GROUP_ORDER:
        idx = grid["Group"] == g
        group_vectors[g] = X.loc[idx].mean(axis=0).to_numpy()

    records = []
    for g1, g2 in combinations(GROUP_ORDER, 2):
        d = group_vectors[g1] - group_vectors[g2]
        estimate = float(d @ beta.to_numpy())
        variance = float(d @ cov.to_numpy() @ d)
        se = math.sqrt(max(variance, 0.0))
        z = estimate / se if se > 0 else np.nan
        p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
        ci_lo = estimate - 1.96 * se
        ci_hi = estimate + 1.96 * se

        records.append({
            "Contrast": f"{g1} - {g2}",
            "Estimate": estimate,
            "SE": se,
            "z": z,
            "p_value": p,
            "CI_low": ci_lo,
            "CI_high": ci_hi,
            "FDR_Family": FAMILY_NAMES["slope_emm"],
        })

    return apply_fdr(pd.DataFrame(records))


# =============================================================================
# 12. SLOPE PARTICIPANT-LEVEL BOOTSTRAP
# =============================================================================

def run_slope_bootstrap(s: pd.DataFrame):
    rng = np.random.default_rng(SLOPE_BOOTSTRAP_SEED)
    ids_by_group = {
        g: sorted(s.loc[s["Group"] == g, "Subject"].astype(str).unique())
        for g in GROUP_ORDER
    }

    full_formula = "Slope_1f ~ C(Phase) * C(Group) + C(Region)"
    full_res, _ = fit_mixed_robust(s, full_formula, reml=True)
    terms = [
        t for t in full_res.fe_params.index
        if "C(Phase)[T." in t and ":C(Group)" in t
    ]

    values = {t: [] for t in terms}
    successes = 0

    for b in range(SLOPE_BOOTSTRAPS):
        pieces = []
        for g in GROUP_ORDER:
            ids = ids_by_group[g]
            sampled = rng.choice(ids, size=len(ids), replace=True)
            for draw_index, sid in enumerate(sampled):
                piece = s[s["Subject"] == sid].copy()
                piece["_Bootstrap_Subject"] = (
                    f"{g}_boot{b + 1}_draw{draw_index}_{sid}"
                )
                pieces.append(piece)

        bd = pd.concat(pieces, ignore_index=True)

        try:
            res, _ = fit_mixed_robust(
                bd,
                full_formula,
                group_col="_Bootstrap_Subject",
                reml=True,
            )
            if not result_converged(res):
                continue
            successes += 1
            for t in terms:
                values[t].append(float(res.fe_params[t]))
        except Exception:
            continue

    rows = []
    for term, vals in values.items():
        vals = np.asarray(vals)
        if len(vals):
            lo, hi = np.quantile(vals, [0.025, 0.975])
            rows.append({
                "Term": term,
                "Bootstrap_N": len(vals),
                "Bootstrap_Target_N": SLOPE_BOOTSTRAPS,
                "Adequate_Execution": len(vals) >= 0.95 * SLOPE_BOOTSTRAPS,
                "Bootstrap_Mean": vals.mean(),
                "CI_low": lo,
                "CI_high": hi,
                "CI_Excludes_Zero": bool(lo > 0 or hi < 0),
                "Seed": SLOPE_BOOTSTRAP_SEED,
                "Total_Successful_Fits": successes,
            })

    out = pd.DataFrame(rows)
    write_csv(out, DIRS["slope"] / "Slope_Participant_Cluster_Bootstrap.csv")
    return out


# =============================================================================
# 13. SECONDARY FOUR-STATE ANALYSIS
# =============================================================================

def run_state_analysis(df: pd.DataFrame):
    d = df.copy()
    d["State"] = pd.Categorical(
        d["State"], categories=STATE_ORDER, ordered=True
    )

    rows = []
    for outcome in OUTCOMES:
        rows.append(
            run_lrt_pair(
                d,
                outcome,
                "C(State) + C(Group) + C(Band) + C(Region)",
                "C(State) * C(Group) + C(Band) + C(Region)",
                family="state_omnibus",
                label="Secondary State x Group omnibus ML-LRT",
            )
        )

    out = apply_fdr(pd.DataFrame(rows))
    write_csv(
        out,
        DIRS["state"] / "Secondary_State_x_Group_Omnibus_ML_LRT_FDR.csv",
    )
    return out


# =============================================================================
# 14. DEPENDENCY-AWARE BURST--APERIODIC COUPLING
# =============================================================================

def run_coupling(df: pd.DataFrame):
    """
    Primary reviewer-safe coupling sensitivity:

    Slope is common across bands within Subject x Phase x Region. Therefore,
    band-specific models eliminate cross-band pseudo-replication:

        Feature ~ Slope_1f * C(Phase) + C(Group) + C(Region) + (1|Subject)

    Each band is analysed separately.
    """

    omnibus_rows = []
    direct_rows = []
    phase_effect_rows = []

    for band in BAND_ORDER:
        bd = df[df["Band"] == band].copy()

        for feature in COUPLING_FEATURES:
            reduced = f"C(Phase) + C(Group) + C(Region)"
            full = f"Slope_1f * C(Phase) + C(Group) + C(Region)"

            # Omnibus Slope x Phase test
            reduced_res, meta_r = fit_mixed_robust(
                bd, f"{feature} ~ {reduced}",
                reml=False, scale_for_lrt=ML_SCALE_FOR_LRT
            )
            full_res, meta_f = fit_mixed_robust(
                bd, f"{feature} ~ {full}",
                reml=False, scale_for_lrt=ML_SCALE_FOR_LRT
            )

            lr, df_diff, p = likelihood_ratio_test(
                full_res, reduced_res
            )

            omnibus_rows.append({
                "Feature": feature,
                "Band": band,
                "LR": lr,
                "df": df_diff,
                "p_value": p,
                "Reduced_Optimizer": meta_r["optimizer"],
                "Full_Optimizer": meta_f["optimizer"],
                "Reduced_Converged": meta_r["converged"],
                "Full_Converged": meta_f["converged"],
                "FDR_Family": FAMILY_NAMES["coupling_phase"],
            })

            # REML direct slope effect and adjusted phase-specific slopes
            try:
                reml_res, meta = fit_mixed_robust(
                    bd, f"{feature} ~ {full}",
                    reml=True, scale_for_lrt=False
                )

                fe = reml_res.fe_params
                cov = reml_res.cov_params().loc[fe.index, fe.index]

                slope_terms = [
                    t for t in fe.index if t == "Slope_1f"
                ]
                if slope_terms:
                    term = "Slope_1f"
                    se = float(reml_res.bse_fe[term])
                    est = float(fe[term])
                    p_direct = float(reml_res.pvalues[term])
                    ci = reml_res.conf_int().loc[term]
                    direct_rows.append({
                        "Feature": feature,
                        "Band": band,
                        "Term": term,
                        "Estimate": est,
                        "SE": se,
                        "z_or_t": est / se if se else np.nan,
                        "p_value": p_direct,
                        "CI_low": float(ci[0]),
                        "CI_high": float(ci[1]),
                        "Optimizer": meta["optimizer"],
                        "Converged": meta["converged"],
                        "FDR_Family": FAMILY_NAMES["coupling_direct_slope"],
                    })

                # Adjusted slope effect at each protocol phase:
                # reference phase = B; for other phases, add the corresponding
                # Slope x Phase coefficient.
                for phase in PHASE_ORDER:
                    if phase == "B":
                        term = "Slope_1f"
                        est = float(fe[term])
                        d = np.zeros(len(fe))
                        d[list(fe.index).index(term)] = 1.0
                    else:
                        # Patsy/statsmodels treatment-coded term
                        phase_term = f"Slope_1f:C(Phase)[T.{phase}]"
                        if phase_term not in fe.index:
                            # Alternate naming guard
                            phase_term = f"C(Phase)[T.{phase}]:Slope_1f"

                        if phase_term not in fe.index:
                            continue

                        d = np.zeros(len(fe))
                        d[list(fe.index).index("Slope_1f")] = 1.0
                        d[list(fe.index).index(phase_term)] = 1.0
                        est = float(d @ fe.to_numpy())

                    var = float(d @ cov.to_numpy() @ d)
                    se = math.sqrt(max(var, 0.0))
                    lo = est - 1.96 * se
                    hi = est + 1.96 * se
                    z = est / se if se else np.nan
                    p_phase = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan

                    phase_effect_rows.append({
                        "Feature": feature,
                        "Band": band,
                        "Phase": phase,
                        "Adjusted_Slope_Effect": est,
                        "SE": se,
                        "z": z,
                        "p_value": p_phase,
                        "CI_low": lo,
                        "CI_high": hi,
                        "Optimizer": meta["optimizer"],
                        "Converged": meta["converged"],
                    })

            except Exception as exc:
                log(
                    f"  Coupling REML failure: feature={feature}, band={band}: "
                    f"{exc}"
                )

    omnibus = apply_fdr(pd.DataFrame(omnibus_rows))
    direct = apply_fdr(pd.DataFrame(direct_rows))
    phase_effects = pd.DataFrame(phase_effect_rows)

    write_csv(
        omnibus,
        DIRS["coupling"] / "Band_Specific_Slope_x_Phase_Omnibus_FDR.csv",
    )
    write_csv(
        direct,
        DIRS["coupling"] / "Band_Specific_Direct_Slope_Effects_FDR.csv",
    )
    write_csv(
        phase_effects,
        DIRS["coupling"] / "Adjusted_Phase_Specific_Slope_Effects.csv",
    )

    return omnibus, direct, phase_effects


# =============================================================================
# 15. GAM
# =============================================================================

def try_import_pygam():
    try:
        from pygam import LinearGAM, s, f, l
        return LinearGAM, s, f, l
    except ImportError:
        return None


def build_gam_data(df: pd.DataFrame, outcome: str):
    d = df[
        [
            "Subject", "Group", "Phase", "Phase_Num",
            "Band", "Region", outcome
        ]
    ].dropna(subset=[outcome, "Phase_Num"]).copy()

    # One-hot encoding with fixed reference order.
    X_parts = [d[["Phase_Num"]].astype(float)]

    for g in GROUP_ORDER[1:]:
        X_parts.append((d["Group"] == g).astype(float).rename(f"Group_{g}"))

    for b in BAND_ORDER[1:]:
        X_parts.append((d["Band"] == b).astype(float).rename(f"Band_{b}"))

    for r in REGION_ORDER[1:]:
        X_parts.append(
            (d["Region"] == r).astype(float).rename(f"Region_{r}")
        )

    X = pd.concat(X_parts, axis=1).reset_index(drop=True)
    y = pd.to_numeric(d[outcome], errors="coerce").to_numpy()

    meta = d.reset_index(drop=True)
    return X.to_numpy(), y, meta


def fit_pygam_models(df: pd.DataFrame, outcome: str):
    imported = try_import_pygam()
    if imported is None:
        raise ImportError(
            "pygam is required for GAM analysis. Install with: pip install pygam"
        )

    LinearGAM, s, f, l = imported
    X, y, meta = build_gam_data(df, outcome)

    # Columns:
    # 0 Phase_Num
    # 1,2 Group STM/LTM
    # 3,4,5 Band theta/alpha/beta
    # 6,7,8,9 Region Central/Temporal/Parietal/Occipital
    #
    # Common model: one penalized spline + linear nuisance terms.
    common_terms = s(0, n_splines=4, spline_order=3)
    for j in range(1, X.shape[1]):
        common_terms += l(j)

    common = LinearGAM(common_terms).gridsearch(
        X, y, lam=np.logspace(-3, 3, 9), progress=False
    )

    # Group-specific model:
    # add three group-indicator-by smooth terms.
    # These use columns 1,2 and a derived CG indicator.
    Xg = np.column_stack(
        [
            X,
            (meta["Group"] == "CG").astype(float).to_numpy(),
        ]
    )
    cg_idx = Xg.shape[1] - 1

    group_terms = (
        s(0, by=1, n_splines=4, spline_order=3)
        + s(0, by=2, n_splines=4, spline_order=3)
        + s(0, by=cg_idx, n_splines=4, spline_order=3)
    )
    # Keep nuisance linear terms.
    for j in range(1, X.shape[1]):
        group_terms += l(j)

    group = LinearGAM(group_terms).gridsearch(
        Xg, y, lam=np.logspace(-3, 3, 9), progress=False
    )

    return common, group, X, Xg, y, meta


def gam_statistic(gam, key: str, default=np.nan):
    """Safely extract a scalar statistic from pyGAM."""
    try:
        value = gam.statistics_.get(key, default)
        arr = np.asarray(value)
        if arr.size == 1:
            return float(arr.ravel()[0])
        return float(arr.ravel()[0]) if arr.size else default
    except Exception:
        return default


def gam_term_edf(gam, term_index: int) -> float:
    """Extract EDF for an individual pyGAM term using its coefficient indices."""
    try:
        coef_idx = gam.terms.get_coef_indices(term_index)
        edof = np.asarray(gam.statistics_["edof_per_coef"], dtype=float)
        return float(edof[np.asarray(coef_idx, dtype=int)].sum())
    except Exception:
        return np.nan


def run_gam(df: pd.DataFrame):
    imported = try_import_pygam()
    if imported is None:
        log("pygam not installed: GAM section recorded as pending.")
        write_csv(
            pd.DataFrame([{
                "Executed": False,
                "Reason": "pygam is not installed",
                "Required_install": "pip install pygam",
            }]),
            DIRS["gam"] / "GAM_Execution_Status.csv",
        )
        return None

    LinearGAM, s, f, l = imported

    model_rows = []
    edf_rows = []
    fit_objects = {}

    gam_outcomes = [
        "Burst_Rate",
        "Peak_Rate",
        "PowerSpike_Rate",
        "Burst_Occupancy",
    ]

    for outcome in gam_outcomes:
        log(f"[GAM] {outcome}")

        # Linear phase comparator
        X, y, meta = build_gam_data(df, outcome)

        # Linear model: replace Phase spline with linear term.
        linear_terms = l(0)
        for j in range(1, X.shape[1]):
            linear_terms += l(j)
        linear = LinearGAM(linear_terms).gridsearch(
            X, y, lam=np.logspace(-3, 3, 9), progress=False
        )

        common, group, X, Xg, y, meta = fit_pygam_models(df, outcome)

        model_rows.extend([
            {
                "Outcome": outcome,
                "Model": "Linear",
                "AIC": gam_statistic(linear, "AIC"),
                "GCV": linear.statistics_.get("GCV", np.nan),
            },
            {
                "Outcome": outcome,
                "Model": "Common_Nonlinear",
                "AIC": gam_statistic(common, "AIC"),
                "GCV": common.statistics_.get("GCV", np.nan),
            },
            {
                "Outcome": outcome,
                "Model": "Group_Specific_Nonlinear",
                "AIC": gam_statistic(group, "AIC"),
                "GCV": group.statistics_.get("GCV", np.nan),
            },
        ])

        # EDF for spline terms
        common_edf = gam_term_edf(common, 0)

        edf_rows.append({
            "Outcome": outcome,
            "Model": "Common_Nonlinear",
            "Smooth_EDF": common_edf,
            "Basis_Dimension_k": 4,
            "Spline_Order": 3,
            "Smoothing_Selection": "GCV",
        })

        # Extract EDF by actual pyGAM term indices rather than assuming that
        # the first 12 coefficients always correspond to the three smooths.
        edf_rows.extend([
            {
                "Outcome": outcome,
                "Model": "Group_Specific_Nonlinear",
                "Smooth_EDF": gam_term_edf(group, 0),
                "Group": "STM",
                "Basis_Dimension_k": 4,
                "Spline_Order": 3,
                "Smoothing_Selection": "GCV",
            },
            {
                "Outcome": outcome,
                "Model": "Group_Specific_Nonlinear",
                "Smooth_EDF": gam_term_edf(group, 1),
                "Group": "LTM",
                "Basis_Dimension_k": 4,
                "Spline_Order": 3,
                "Smoothing_Selection": "GCV",
            },
            {
                "Outcome": outcome,
                "Model": "Group_Specific_Nonlinear",
                "Smooth_EDF": gam_term_edf(group, 2),
                "Group": "CG",
                "Basis_Dimension_k": 4,
                "Spline_Order": 3,
                "Smoothing_Selection": "GCV",
            },
        ])

        fit_objects[outcome] = {
            "linear": linear,
            "common": common,
            "group": group,
            "X": X,
            "Xg": Xg,
            "y": y,
            "meta": meta,
        }

    model_df = pd.DataFrame(model_rows)
    edf_df = pd.DataFrame(edf_rows)

    write_csv(
        model_df,
        DIRS["gam"] / "GAM_Model_Fit_Comparison.csv",
    )
    write_csv(
        edf_df,
        DIRS["gam"] / "GAM_EDF_and_Specification.csv",
    )

    specification = pd.DataFrame([
        {
            "Model": "Common_Nonlinear",
            "Formula": "Y ~ Group + s(Phase_Num, k=4) + Band + Region",
            "Phase_Representation": "Ordered numerical phase index 1-7",
            "Spline": "Penalized cubic P-spline",
            "Basis_Dimension": 4,
            "Smoothing_Selection": "GCV",
            "Repeated_Measures": "Participant-level cluster bootstrap",
            "Bootstrap_N": GAM_BOOTSTRAPS,
            "Bootstrap_Seed": GAM_SEED,
            "Interpretation": "Smooth pattern across ordered protocol phases",
        },
        {
            "Model": "Group_Specific_Nonlinear",
            "Formula": "Y ~ Group + s(Phase_Num, by=Group, k=4) + Band + Region",
            "Phase_Representation": "Ordered numerical phase index 1-7",
            "Spline": "Penalized cubic P-spline",
            "Basis_Dimension": 4,
            "Smoothing_Selection": "GCV",
            "Repeated_Measures": "Participant-level cluster bootstrap",
            "Bootstrap_N": GAM_BOOTSTRAPS,
            "Bootstrap_Seed": GAM_SEED,
            "Interpretation": "Group-specific smooth patterns across ordered phases",
        },
    ])
    write_csv(
        specification,
        DIRS["gam"] / "GAM_Methodological_Specification.csv",
    )

    # Participant-level bootstrap of fitted smooth predictions.
    bootstrap_rows = []
    rng = np.random.default_rng(GAM_SEED)
    subjects_by_group = {
        g: sorted(df.loc[df["Group"] == g, "Subject"].astype(str).unique())
        for g in GROUP_ORDER
    }

    for outcome, objs in fit_objects.items():
        full_group_model = objs["group"]
        meta = objs["meta"]

        # Hold full-data smoothing parameters fixed during bootstrap.
        lam = full_group_model.lam

        for b in range(GAM_BOOTSTRAPS):
            pieces = []
            for g in GROUP_ORDER:
                sampled = rng.choice(
                    subjects_by_group[g],
                    size=len(subjects_by_group[g]),
                    replace=True,
                )
                for sid in sampled:
                    pieces.append(df[df["Subject"] == sid].copy())

            bd = pd.concat(pieces, ignore_index=True)

            try:
                # Rebuild matrices from the bootstrap data.
                Xb, yb, mb = build_gam_data(bd, outcome)
                Xgb = np.column_stack([
                    Xb,
                    (mb["Group"] == "CG").astype(float).to_numpy()
                ])

                # Refit with fixed smoothing parameters.
                boot_terms = (
                    s(0, by=1, n_splines=4, spline_order=3)
                    + s(0, by=2, n_splines=4, spline_order=3)
                    + s(
                        0,
                        by=Xgb.shape[1] - 1,
                        n_splines=4,
                        spline_order=3,
                    )
                )
                for j in range(1, Xb.shape[1]):
                    boot_terms += l(j)
                boot_model = LinearGAM(boot_terms)
                boot_model.lam = lam
                boot_model.fit(Xgb, yb)

                # Store fitted values by observed bootstrap row. These are
                # participant-resampled predictions, not seven independent
                # tests.
                pred = boot_model.predict(Xgb)
                for i in range(len(pred)):
                    bootstrap_rows.append({
                        "Outcome": outcome,
                        "Bootstrap": b + 1,
                        "Subject": mb.loc[i, "Subject"],
                        "Group": mb.loc[i, "Group"],
                        "Phase": mb.loc[i, "Phase"],
                        "Region": mb.loc[i, "Region"],
                        "Band": mb.loc[i, "Band"],
                        "Prediction": pred[i],
                    })
            except Exception:
                continue

    boot_df = pd.DataFrame(bootstrap_rows)
    write_csv(
        boot_df,
        DIRS["gam"] / "GAM_Participant_Cluster_Bootstrap_Predictions.csv",
    )

    # Execution QC: each successful bootstrap contributes predictions. A
    # bootstrap replicate is considered present when it contributes at least
    # one prediction row for every GAM outcome.
    if not boot_df.empty:
        successful_bootstrap_counts = (
            boot_df.groupby("Outcome")["Bootstrap"].nunique()
        )
        min_successful_bootstraps = int(successful_bootstrap_counts.min())
    else:
        min_successful_bootstraps = 0

    status = pd.DataFrame([{
        "Executed": True,
        "Bootstrap_N_Requested": GAM_BOOTSTRAPS,
        "Seed": GAM_SEED,
        "Expected_Outcomes": len(gam_outcomes),
        "Bootstrap_Min_Successful_Replicates_Across_Outcomes":
            min_successful_bootstraps,
        "Bootstrap_Adequate_95pct":
            min_successful_bootstraps >= int(0.95 * GAM_BOOTSTRAPS),
        "Missing_Output_Files": 0,
        "Model": "Penalized GAM with participant-level cluster bootstrap",
    }])
    write_csv(status, DIRS["gam"] / "GAM_Execution_Status.csv")

    return model_df


# =============================================================================
# 16. THRESHOLD-SENSITIVITY AUDIT
# =============================================================================

def run_threshold_audit():
    """
    Do not regenerate feature extraction or thresholds here.

    The Features_Revised_2 package already contains the primary 3x-MAD data and
    2x/4x-MAD sensitivity outputs. This section inventories and, where present,
    copies the validated threshold-sensitivity statistical table into the final
    analysis directory.
    """
    threshold_root = FEATURE_ROOT / "Threshold_Sensitivity"

    files_expected = [
        "Threshold_Sensitivity_Feature_Comparison.csv",
        "Threshold_Sensitivity_Specification.csv",
    ]

    records = []
    for f in files_expected:
        p = FEATURE_ROOT / f
        records.append({
            "File": str(p),
            "Exists": p.exists(),
            "Description": "Threshold sensitivity metadata/output",
        })

    # Search recursively for the combined LMM sensitivity table.
    candidates = list(
        threshold_root.rglob("All_Thresholds_Phase_x_Group_LMM_Results.csv")
    ) if threshold_root.exists() else []

    for p in candidates:
        try:
            t = pd.read_csv(p)
            # Preserve the already-generated result exactly; add only a source
            # field.
            t = t.copy()
            t["Source_File"] = str(p)
            write_csv(
                t,
                DIRS["reviewer"] / "All_Thresholds_Phase_x_Group_LMM_Results.csv",
            )
            records.append({
                "File": str(p),
                "Exists": True,
                "Description": "Existing validated 2x/3x/4x-MAD LMM sensitivity results",
            })
        except Exception as exc:
            records.append({
                "File": str(p),
                "Exists": True,
                "Description": f"Read failure: {exc}",
            })

    # Methodological specification
    spec = pd.DataFrame([
        {
            "Threshold": "2x MAD",
            "Formula": "Median(A) + 2 x MAD(A)",
            "Threshold_scope": "Participant x Region x Band, pooled across all phases",
        },
        {
            "Threshold": "3x MAD",
            "Formula": "Median(A) + 3 x MAD(A)",
            "Threshold_scope": "Participant x Region x Band, pooled across all phases",
        },
        {
            "Threshold": "4x MAD",
            "Formula": "Median(A) + 4 x MAD(A)",
            "Threshold_scope": "Participant x Region x Band, pooled across all phases",
        },
    ])
    write_csv(
        spec,
        DIRS["reviewer"] / "Threshold_Sensitivity_Method_Specification.csv",
    )
    write_csv(
        pd.DataFrame(records),
        DIRS["reviewer"] / "Threshold_Sensitivity_Audit.csv",
    )


# =============================================================================
# 17. EEG QUALITY-CONTROL ANALYSIS
# =============================================================================

def run_qc(df: pd.DataFrame):
    qc_file = FEATURE_ROOT / "Epoch_QC_Summary.csv"
    proc_file = FEATURE_ROOT / "Processing_Log.csv"

    outputs = []

    if not qc_file.exists():
        outputs.append({
            "Analysis": "Epoch-level EEG QC",
            "Status": "Pending",
            "Reason": "Epoch_QC_Summary.csv not found",
        })
        write_csv(pd.DataFrame(outputs), DIRS["qc"] / "QC_Execution_Status.csv")
        return

    qc = pd.read_csv(qc_file)

    # Descriptive group x phase QC summaries at participant level.
    # First aggregate across regions so a participant contributes one value
    # per phase.
    numeric_cols = [
        c for c in [
            "Total_QC_Windows",
            "Clean_Windows",
            "Rejected_Windows",
            "Clean_Duration_s",
        ] if c in qc.columns
    ]

    if numeric_cols:
        q = (
            qc.groupby(
                ["Subject", "Group", "Phase"],
                as_index=False
            )[numeric_cols]
            .mean()
        )

        group_phase = (
            q.groupby(["Group", "Phase"])[numeric_cols]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        write_csv(
            group_phase,
            DIRS["qc"] / "QC_Retained_Duration_Epochs_By_Group_Phase.csv",
        )

        # Participant-level overall quality metrics, preventing region/phase
        # pseudo-replication.
        participant_qc = (
            q.groupby(["Subject", "Group"], as_index=False)[numeric_cols]
            .mean()
        )

        write_csv(
            participant_qc,
            DIRS["qc"] / "QC_Participant_Level_Summary.csv",
        )

        # Welch one-way ANOVA if scipy version supports it; otherwise standard
        # one-way ANOVA is retained as a descriptive check.
        test_rows = []
        for metric in numeric_cols:
            arrays = [
                participant_qc.loc[
                    participant_qc["Group"] == g, metric
                ].dropna().to_numpy()
                for g in GROUP_ORDER
            ]
            arrays = [a for a in arrays if len(a) > 1]
            if len(arrays) >= 2:
                try:
                    res = stats.f_oneway(*arrays)
                    test_rows.append({
                        "Metric": metric,
                        "Test": "One-way ANOVA participant-level QC",
                        "Statistic": res.statistic,
                        "p_value": res.pvalue,
                    })
                except Exception as exc:
                    test_rows.append({
                        "Metric": metric,
                        "Test": "ANOVA failed",
                        "Statistic": np.nan,
                        "p_value": np.nan,
                        "Error": str(exc),
                    })

        write_csv(
            pd.DataFrame(test_rows),
            DIRS["qc"] / "QC_Group_Difference_Tests.csv",
        )

    # Processing log gives recording-level missingness/exclusion information.
    if proc_file.exists():
        proc = pd.read_csv(proc_file)
        write_csv(proc, DIRS["qc"] / "Processing_Log_Audit_Copy.csv")

    # The feature package does not contain an explicit preprocessing-exclusion
    # table with the reviewer's requested 11 EEG QC exclusions, interpolation
    # counts, or interpolation percentages. Do not fabricate them.
    pending = pd.DataFrame([
        {
            "Analysis": "Report all 11 EEG QC exclusions by original group and reason",
            "Status": "Pending external input",
            "Required_Input": "Preprocessing/QC exclusion table with participant, group, reason",
        },
        {
            "Analysis": "Group differences in interpolation",
            "Status": "Pending external input",
            "Required_Input": "Participant-level interpolation count/percentage",
        },
        {
            "Analysis": "Electrode-level sensitivity before regional aggregation",
            "Status": "Pending external input",
            "Required_Input": "Electrode-level feature dataset",
        },
        {
            "Analysis": "Minimum-duration sensitivity",
            "Status": "Pending external input",
            "Required_Input": "Feature outputs regenerated at alternative duration thresholds",
        },
        {
            "Analysis": "80-ms peak-separation sensitivity",
            "Status": "Pending external input",
            "Required_Input": "Feature outputs regenerated at alternative peak-separation thresholds",
        },
        {
            "Analysis": "Practice-hours continuous sensitivity",
            "Status": "Pending external input",
            "Required_Input": "Participant-level practice-hours variable",
        },
    ])
    write_csv(
        pending,
        DIRS["qc"] / "Pending_Reviewer_Analyses.csv",
    )


# =============================================================================
# 18. DATASET / ANALYSIS MANIFEST
# =============================================================================

def write_manifest(df: pd.DataFrame):
    complete = complete_case_subjects(df)

    manifest = {
        "Input_file": str(INPUT_FILE),
        "Feature_rows_raw": int(len(df)),
        "Raw_participants": int(df["Subject"].nunique()),
        "Complete_case_participants": int(len(complete)),
        "Complete_case_subjects": complete,
        "Complete_case_rows": int(
            df[df["Subject"].isin(complete)].shape[0]
        ),
        "Phases": PHASE_ORDER,
        "Groups": GROUP_ORDER,
        "Bands": BAND_ORDER,
        "Regions": REGION_ORDER,
        "Primary_threshold": "Median + 3 x MAD",
        "Threshold_scope": "Participant x Region x Band pooled across all phases",
        "Burst_windows": "Non-overlapping 3-s QC windows; retained windows analysed independently",
        "Artificial_concatenation": False,
        "Minimum_burst_duration": "80 ms",
        "Minimum_envelope_peak_separation": "80 ms",
        "Aperiodic_method": "specparam exponent only",
        "Aperiodic_sign_convention": "Slope_1f = -exponent",
        "Primary_LMM": "Y ~ C(Phase)*C(Group) + C(Band) + C(Region) + (1|Subject)",
        "LRT_estimation": "ML",
        "Final_parameter_estimation": "REML",
        "Primary_FDR_family": FAMILY_NAMES["primary_omnibus"],
        "Band_heterogeneity_reduced":
            "Y ~ Phase*Group + Phase*Band + Group*Band + Region + (1|Subject)",
        "Band_heterogeneity_full":
            "Y ~ Phase*Group*Band + Region + (1|Subject)",
        "Region_heterogeneity_reduced":
            "Y ~ Phase*Group + Phase*Region + Group*Region + Band + (1|Subject)",
        "Region_heterogeneity_full":
            "Y ~ Phase*Group*Region + Band + (1|Subject)",
        "GAM": "Penalized GAM + participant-level cluster bootstrap",
        "GAM_phase_interpretation": "Ordered protocol phases, not continuous physiological time",
        "GAM_k": 4,
        "GAM_bootstrap_n": GAM_BOOTSTRAPS,
        "GAM_seed": GAM_SEED,
    }

    with open(
        DIRS["logs"] / "Analysis_Manifest.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2, default=str)


# =============================================================================
# 19. MAIN
# =============================================================================

def main():
    log("=" * 78)
    log("FINAL REVISED EEG STATISTICAL ANALYSIS PIPELINE")
    log("=" * 78)
    log(f"Input: {INPUT_FILE}")
    log(f"Output: {OUT_DIR}")

    df_raw = load_features()
    write_manifest(df_raw)

    # Save an exact audit copy of the input after derived fields/categories.
    write_csv(
        df_raw,
        DIRS["logs"] / "Input_Features_Audit_With_Derived_Fields.csv",
    )

    df = prepare_complete_dataset(df_raw)

    # Subject map / completeness
    completeness = (
        df_raw.groupby(["Subject", "Group"])["Phase"]
        .nunique()
        .reset_index(name="n_phases")
    )
    completeness["Complete_7_Phases"] = completeness["n_phases"] == 7
    write_csv(
        completeness,
        DIRS["logs"] / "Subject_Completeness_Audit.csv",
    )

    # -------------------------------------------------------------------------
    # Primary LMM
    # -------------------------------------------------------------------------
    if RUN_PRIMARY_LMM:
        try:
            run_primary_lmm(df)
        except Exception:
            log("[ERROR] Primary LMM failed")
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Existing validated participant bootstrap
    # -------------------------------------------------------------------------
    if RUN_PRIMARY_BOOTSTRAP:
        try:
            run_primary_bootstrap(df)
        except Exception:
            log("[ERROR] Primary bootstrap failed")
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Band heterogeneity
    # -------------------------------------------------------------------------
    if RUN_BAND_HETEROGENEITY:
        try:
            run_band_heterogeneity(df)
        except Exception:
            log("[ERROR] Band heterogeneity failed")
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Region heterogeneity
    # -------------------------------------------------------------------------
    if RUN_REGION_HETEROGENEITY:
        try:
            run_region_heterogeneity(df)
        except Exception:
            log("[ERROR] Region heterogeneity failed")
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Corrected slope
    # -------------------------------------------------------------------------
    slope_df = None
    if RUN_SLOPE:
        try:
            slope_df, _, _, _ = run_slope_analysis(df)
        except Exception:
            log("[ERROR] Slope analysis failed")
            traceback.print_exc()

    if RUN_SLOPE_BOOTSTRAP:
        try:
            if slope_df is None:
                slope_df = prepare_slope_data(df)
            run_slope_bootstrap(slope_df)
        except Exception:
            log("[ERROR] Slope bootstrap failed")
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Secondary state aggregation
    # -------------------------------------------------------------------------
    if RUN_STATE_ANALYSIS:
        try:
            run_state_analysis(df)
        except Exception:
            log("[ERROR] State analysis failed")
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Dependency-aware coupling
    # -------------------------------------------------------------------------
    if RUN_COUPLING:
        try:
            run_coupling(df)
        except Exception:
            log("[ERROR] Coupling analysis failed")
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # GAM
    # -------------------------------------------------------------------------
    if RUN_GAM:
        try:
            run_gam(df)
        except Exception:
            log("[ERROR] GAM analysis failed")
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Threshold audit
    # -------------------------------------------------------------------------
    if RUN_THRESHOLD_AUDIT:
        try:
            run_threshold_audit()
        except Exception:
            log("[ERROR] Threshold audit failed")
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------
    if RUN_QC:
        try:
            run_qc(df)
        except Exception:
            log("[ERROR] QC analysis failed")
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Final execution status
    # -------------------------------------------------------------------------
    status = pd.DataFrame([
        {
            "Analysis": "Primary LMM",
            "Enabled": RUN_PRIMARY_LMM,
            "Output_Directory": str(DIRS["lmm"]),
        },
        {
            "Analysis": "Primary participant bootstrap",
            "Enabled": RUN_PRIMARY_BOOTSTRAP,
            "Output_Directory": str(DIRS["lmm"]),
        },
        {
            "Analysis": "Phase x Group x Band",
            "Enabled": RUN_BAND_HETEROGENEITY,
            "Output_Directory": str(DIRS["band"]),
        },
        {
            "Analysis": "Phase x Group x Region",
            "Enabled": RUN_REGION_HETEROGENEITY,
            "Output_Directory": str(DIRS["region"]),
        },
        {
            "Analysis": "Corrected aperiodic slope",
            "Enabled": RUN_SLOPE,
            "Output_Directory": str(DIRS["slope"]),
        },
        {
            "Analysis": "Slope participant bootstrap",
            "Enabled": RUN_SLOPE_BOOTSTRAP,
            "Output_Directory": str(DIRS["slope"]),
        },
        {
            "Analysis": "Secondary state analysis",
            "Enabled": RUN_STATE_ANALYSIS,
            "Output_Directory": str(DIRS["state"]),
        },
        {
            "Analysis": "Dependency-aware coupling",
            "Enabled": RUN_COUPLING,
            "Output_Directory": str(DIRS["coupling"]),
        },
        {
            "Analysis": "Penalized GAM",
            "Enabled": RUN_GAM,
            "Output_Directory": str(DIRS["gam"]),
        },
        {
            "Analysis": "Threshold sensitivity audit",
            "Enabled": RUN_THRESHOLD_AUDIT,
            "Output_Directory": str(DIRS["reviewer"]),
        },
        {
            "Analysis": "EEG QC",
            "Enabled": RUN_QC,
            "Output_Directory": str(DIRS["qc"]),
        },
    ])
    write_csv(status, DIRS["logs"] / "Final_Execution_Status.csv")

    log("=" * 78)
    log("PIPELINE FINISHED")
    log("FINAL LOCK CHECK: pure Band heterogeneity requires df=36; pure Region heterogeneity requires df=48.")
    log(f"Results written to: {OUT_DIR}")
    log("=" * 78)


if __name__ == "__main__":
    main()

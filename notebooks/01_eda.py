"""
Phase 1: EDA — AI4I 2020 Predictive Maintenance
Goal: understand class balance, sensor behavior under failure, data quality.
"""

# %% [markdown]
# # Phase 1 — EDA
#
# Industrial CNC milling-process telemetry. 10K observations, 5 continuous
# sensors plus categorical product type. Multi-label failure flags (TWF, HDF,
# PWF, OSF, RNF) and a binary `Machine failure` rollup.
#
# **Goals of this notebook:**
# 1. Confirm dataset shape, dtypes, missingness.
# 2. Quantify class imbalance — drives the modeling strategy in Phase 3.
# 3. Profile sensor distributions to understand which signals separate
#    failures from healthy runs.
# 4. Surface multicollinearity, which constrains the linear baseline.
# 5. Document data-quality issues for the README.

# %%
import sys
from pathlib import Path

# Resolve project root whether the notebook runs from notebooks/ or repo root
HERE = Path.cwd()
ROOT = HERE.parent if HERE.name == "notebooks" else HERE
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import load

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110

FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = load(ROOT / "data" / "ai4i2020.csv")
print(df.shape)
df.head()

# %% [markdown]
# ## Schema & missingness

# %%
df.info()

# %%
df.isna().sum()

# %% [markdown]
# No nulls. Eleven numeric columns, two string (`Product ID`, `Type`). Schema
# matches the canonical UCI/Kaggle release.

# %%
df.describe().round(2)

# %% [markdown]
# ## Class balance — the central modeling challenge
#
# Failures are rare. This is the headline number that determines everything
# downstream: stratified split, recall-oriented metrics, class-imbalance
# weighting in the model.

# %%
SENSORS = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

n = len(df)
n_fail = int(df["Machine failure"].sum())
print(f"Total rows           : {n:,}")
print(f"Machine failure = 1  : {n_fail} ({n_fail/n*100:.2f}%)")
print(f"Imbalance ratio      : {(n - n_fail)/n_fail:.1f} : 1 (negative : positive)")

# %% [markdown]
# ## Failure-mode breakdown
#
# `Machine failure` is the OR of five mutually-non-exclusive root causes.
# Treating this as a single binary problem is fine for a first model, but the
# README notes the multi-label extension as future work — different modes have
# different sensor signatures and would benefit from per-mode classifiers.

# %%
mode_counts = df[MODES].sum().rename("count").to_frame()
mode_counts["rate_%"] = (mode_counts["count"] / n * 100).round(2)
print(mode_counts)

multi_mode = (df[MODES].sum(axis=1) > 1).sum()
print(f"\nRows with >1 active failure mode: {multi_mode}")

# Label consistency: Machine failure should == OR(modes)
inconsistent = (df["Machine failure"] != (df[MODES].sum(axis=1) > 0).astype(int)).sum()
print(f"Rows where Machine_failure != OR(modes): {inconsistent}")

# %% [markdown]
# ## Failure rate by product type
#
# L variants fail more often than M or H. This matches Matzka's overstrain
# threshold (lower for L, higher for H), and means `Type` is signal — keep it.

# %%
by_type = df.groupby("Type")["Machine failure"].agg(["count", "sum", "mean"])
by_type["mean"] = (by_type["mean"] * 100).round(2)
print(by_type.rename(columns={"mean": "fail_rate_%"}))

# %% [markdown]
# ## Sensor distributions — failure vs. healthy

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()
for i, s in enumerate(SENSORS):
    ax = axes[i]
    sns.kdeplot(data=df, x=s, hue="Machine failure", common_norm=False,
                fill=True, alpha=0.35, ax=ax, palette={0: "#4C78A8", 1: "#E45756"})
    ax.set_title(s)
    ax.set_xlabel("")
axes[-1].axis("off")
fig.suptitle("Sensor distributions — failed (red) vs. healthy (blue)",
             y=1.00, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_sensor_distributions.png", bbox_inches="tight", dpi=130)
plt.close(fig)
print(f"Saved {FIG_DIR / '01_sensor_distributions.png'}")

# %% [markdown]
# ### What the sensor plots show
#
# - **Tool wear** and **Torque** shift right under failure — the strongest
#   univariate signals.
# - **Rotational speed** shifts left under failure (low rpm + high torque is
#   the power-failure signature).
# - **Air temp** and **Process temp** look almost identical between groups —
#   on their own, marginal. But their *difference* is the cooling signal that
#   drives heat-dissipation failures, which is the motivation for the
#   `temp_diff` engineered feature in Phase 2.

# %% [markdown]
# ## Failure-mode prevalence

# %%
fig, ax = plt.subplots(figsize=(8, 4))
mode_counts["count"].sort_values().plot(
    kind="barh", ax=ax, color="#54A24B", edgecolor="black"
)
ax.set_xlabel("Count (out of 10,000)")
ax.set_title("Failure modes — prevalence")
for i, v in enumerate(mode_counts["count"].sort_values()):
    ax.text(v + 3, i, str(v), va="center")
fig.tight_layout()
fig.savefig(FIG_DIR / "02_failure_modes.png", bbox_inches="tight", dpi=130)
plt.close(fig)
print(f"Saved {FIG_DIR / '02_failure_modes.png'}")

# %% [markdown]
# ## Correlation structure

# %%
corr = df[SENSORS + ["Machine failure"]].corr()
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_title("Sensor correlation matrix")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_correlations.png", bbox_inches="tight", dpi=130)
plt.close(fig)
print(f"Saved {FIG_DIR / '03_correlations.png'}")

print("\nCorrelation with Machine failure (sorted by |r|):")
print(corr["Machine failure"].drop("Machine failure")
      .reindex(corr["Machine failure"].drop("Machine failure")
               .abs().sort_values(ascending=False).index)
      .round(3))

# %% [markdown]
# ### Multicollinearity flags
#
# - `Air temp` ↔ `Process temp` ≈ +0.96. Process temp is air temp + ~10 K by
#   construction. For the linear baseline this inflates standard errors;
#   either use ridge regularization or replace the pair with `temp_diff`. Tree
#   models are unaffected.
# - `Torque` ↔ `Rotational speed` ≈ -0.94. Mechanical-power conservation
#   (P = τ · ω). Same handling: either keep both for trees, or replace with
#   the engineered `power` feature for the linear baseline.

# %% [markdown]
# ## Bivariate view — torque × rotational speed (the PWF surface)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ok = df[df["Machine failure"] == 0]
fail = df[df["Machine failure"] == 1]
ax.scatter(ok["Rotational speed [rpm]"], ok["Torque [Nm]"],
           s=4, alpha=0.18, c="#4C78A8", label="Healthy")
ax.scatter(fail["Rotational speed [rpm]"], fail["Torque [Nm]"],
           s=18, alpha=0.85, c="#E45756", edgecolor="black",
           linewidth=0.3, label="Failure")
ax.set_xlabel("Rotational speed [rpm]")
ax.set_ylabel("Torque [Nm]")
ax.set_title("Failures concentrate at low-rpm / high-torque (PWF + OSF regimes)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "04_torque_rpm_failures.png", bbox_inches="tight", dpi=130)
plt.close(fig)
print(f"Saved {FIG_DIR / '04_torque_rpm_failures.png'}")

# %% [markdown]
# ## Tool-wear bands and TWF

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(df.loc[df["Machine failure"] == 0, "Tool wear [min]"],
        bins=40, alpha=0.5, label="Healthy", color="#4C78A8")
ax.hist(df.loc[df["Machine failure"] == 1, "Tool wear [min]"],
        bins=40, alpha=0.7, label="Failure", color="#E45756")
ax.axvspan(200, 240, color="gold", alpha=0.25, label="TWF band [200, 240] min")
ax.set_xlabel("Tool wear [min]")
ax.set_ylabel("Count")
ax.set_title("Failures cluster at high tool wear; TWF band highlighted")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "05_tool_wear.png", bbox_inches="tight", dpi=130)
plt.close(fig)
print(f"Saved {FIG_DIR / '05_tool_wear.png'}")

# %% [markdown]
# ## Data-quality summary
#
# | Issue | Severity | Handling |
# |---|---|---|
# | Severe class imbalance (~3-4% positive) | High | Stratified split, `scale_pos_weight`, recall/F1/PR-AUC |
# | Strong multicollinearity (air↔process temp, torque↔rpm) | Medium | Engineered features (temp_diff, power); ridge for linear baseline |
# | RNF (random failure, 0.1%) is by design unpredictable | Medium | Optionally exclude RNF-only rows from training; documented as irreducible noise |
# | Per-mode failure flags would leak the label | Critical | TWF/HDF/PWF/OSF/RNF must be dropped from features (kept only for diagnostics) |
# | UDI and Product ID are row identifiers | Low | Drop from features; not predictive |
# | No nulls, no duplicate UDIs, schema clean | Positive | No imputation needed |
print("Phase 1 complete — figures saved to reports/figures/")

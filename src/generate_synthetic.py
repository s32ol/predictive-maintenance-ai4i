"""
Spec-faithful AI4I 2020 dataset generator.

This is a *development convenience* used only when the canonical ai4i2020.csv
from UCI/Kaggle is not present locally. The canonical dataset is itself
synthetic (Matzka, 2020) — this generator follows the same documented rules
(failure-mode definitions, marginal distributions, Type ratios). Statistics
match within sampling noise; the pipeline runs identically on either file.

Reference:
  S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance
  Applications," 2020 Third International Conference on AI for Industries.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_ROWS = 10_000

# OSF thresholds per product type (per Matzka 2020)
OSF_THRESH = {"L": 11_000, "M": 12_000, "H": 13_000}


def generate(n: int = N_ROWS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Product type — L 50% / M 30% / H 20%
    types = rng.choice(["L", "M", "H"], size=n, p=[0.5, 0.3, 0.2])

    # UDI and Product ID
    udi = np.arange(1, n + 1)
    counters = {"L": 0, "M": 0, "H": 0}
    product_id = []
    for t in types:
        counters[t] += 1
        product_id.append(f"{t}{counters[t]:05d}")

    # Air temperature [K]: random-walk normalized to std=2 K around 300 K
    air_walk = rng.standard_normal(n).cumsum()
    air_temp = (air_walk - air_walk.mean()) / air_walk.std() * 2.0 + 300.0

    # Process temperature [K]: random-walk std=1 K, added to air_temp + 10 K
    proc_walk = rng.standard_normal(n).cumsum()
    proc_temp = (proc_walk - proc_walk.mean()) / proc_walk.std() * 1.0 + air_temp + 10.0

    # Torque [Nm] and Rotational speed [rpm]: jointly Gaussian with negative
    # correlation (canonical AI4I: rpm ~ N(1538, 179), torque ~ N(40, 10),
    # corr ≈ -0.88, reflecting mechanical-power conservation P = τ·ω).
    rpm_mu, rpm_sd = 1538.0, 179.0
    tor_mu, tor_sd = 40.0, 10.0
    rho = -0.88
    cov = np.array([
        [rpm_sd ** 2, rho * rpm_sd * tor_sd],
        [rho * rpm_sd * tor_sd, tor_sd ** 2],
    ])
    rpm_torque = rng.multivariate_normal([rpm_mu, tor_mu], cov, size=n)
    rot_rpm = np.clip(rpm_torque[:, 0].round(), 1168, 2886).astype(int)
    torque = np.clip(rpm_torque[:, 1].round(1), 3.8, 76.6)

    # Tool wear [min] — uniform 0..253 (matches canonical distribution)
    tool_wear = rng.integers(0, 254, size=n)

    # ---- Failure modes per Matzka 2020 ----
    # TWF: tool wear in [200, 240] -> random failure
    # Canonical: ~46/10000. Band covers ~16% of rows -> tune p ≈ 3%
    twf = ((tool_wear >= 200) & (tool_wear <= 240) & (rng.random(n) < 0.03)).astype(int)

    # HDF: process-air diff < 8.6 K AND rotational speed < 1380 rpm
    hdf = (((proc_temp - air_temp) < 8.6) & (rot_rpm < 1380)).astype(int)

    # PWF: power outside [3500 W, 9000 W]
    power = torque * (rot_rpm * 2 * np.pi / 60.0)
    pwf = ((power < 3500) | (power > 9000)).astype(int)

    # OSF: tool_wear * torque exceeds type-specific threshold
    osf_thresh_arr = np.array([OSF_THRESH[t] for t in types])
    osf = ((tool_wear * torque) > osf_thresh_arr).astype(int)

    # RNF: 0.1% random
    rnf = (rng.random(n) < 0.001).astype(int)

    machine_failure = ((twf | hdf | pwf | osf | rnf) > 0).astype(int)

    df = pd.DataFrame({
        "UDI": udi,
        "Product ID": product_id,
        "Type": types,
        "Air temperature [K]": air_temp.round(1),
        "Process temperature [K]": proc_temp.round(1),
        "Rotational speed [rpm]": rot_rpm,
        "Torque [Nm]": torque.round(1),
        "Tool wear [min]": tool_wear,
        "Machine failure": machine_failure,
        "TWF": twf,
        "HDF": hdf,
        "PWF": pwf,
        "OSF": osf,
        "RNF": rnf,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/ai4i2020.csv", index=False)
    print(f"Generated {len(df):,} rows -> data/ai4i2020.csv")
    print(f"Failure rate: {df['Machine failure'].mean():.2%}")

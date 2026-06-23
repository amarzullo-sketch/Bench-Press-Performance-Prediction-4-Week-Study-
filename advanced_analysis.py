from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
VISUALS_DIR = OUTPUT_DIR / "visuals"
INPUT_FILE = OUTPUT_DIR / "bench_analysis_dataset.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE.name}. Run bench_analysis.py first, then run this file."
    )

VISUALS_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(INPUT_FILE)


def clean_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("%", "pct")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


df.columns = [clean_name(c) for c in df.columns]

for col in df.columns:
    if col not in {"subject", "date", "session_type", "week", "bench_day", "notes"}:
        df[col] = pd.to_numeric(df[col], errors="ignore")

OUTCOME = "performance_vs_personal_avg_pct"

if OUTCOME not in df.columns:
    raise KeyError(
        "Could not find performance_vs_personal_avg_pct. "
        "Re-run bench_analysis.py first."
    )

if "subject" not in df.columns:
    raise KeyError("Could not find subject column.")


METRICS = {
    "Sleep performance": "sleep_performance",
    "Recovery score": "recovery_score",
    "HRV": "hrv_rmssd",
    "Activity strain": "activity_strain_total",
    "Fatigue": "fatigue",
    "Soreness": "soreness",
    "Hours slept": "hours_slept",
    "Protein": "protein",
}

available = {
    label: col
    for label, col in METRICS.items()
    if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() >= 4
}

for col in list(available.values()) + [OUTCOME]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def within_subject_z(data, column):
    def z_score(series):
        valid = series.dropna()

        if len(valid) < 3:
            return pd.Series(np.nan, index=series.index)

        sd = valid.std(ddof=0)

        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=series.index)

        return (series - valid.mean()) / sd

    return data.groupby("subject")[column].transform(z_score)


def thirds(series, low_label, middle_label, high_label):
    result = pd.Series(np.nan, index=series.index, dtype="object")
    valid = series.dropna()

    if len(valid) < 6:
        return result

    ranked = valid.rank(method="first")

    result.loc[valid.index] = pd.qcut(
        ranked,
        q=3,
        labels=[low_label, middle_label, high_label],
    ).astype(str)

    return result


# ---------------------------------------------------------
# 1. ATHLETE RESPONSE FINGERPRINT
# ---------------------------------------------------------

fingerprint_rows = []

for athlete, group in df.groupby("subject"):
    for label, col in available.items():
        pair = group[[col, OUTCOME]].dropna()

        if len(pair) >= 4:
            fingerprint_rows.append(
                {
                    "subject": athlete,
                    "predictor": label,
                    "n_sessions": len(pair),
                    "correlation": pair[col].corr(pair[OUTCOME]),
                }
            )

fingerprint = pd.DataFrame(fingerprint_rows)
fingerprint.to_csv(OUTPUT_DIR / "athlete_response_fingerprint.csv", index=False)

if not fingerprint.empty:
    heatmap = fingerprint.pivot(
        index="predictor",
        columns="subject",
        values="correlation",
    )

    order = heatmap.abs().max(axis=1).sort_values(ascending=False).index
    heatmap = heatmap.loc[order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.7 * len(heatmap) + 2)))

    image = ax.imshow(
        heatmap.to_numpy(),
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns)

    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)

    ax.set_title("Athlete Response Fingerprint")
    ax.set_xlabel("Participant")
    ax.set_ylabel("Recovery / Training Variable")

    for i in range(len(heatmap.index)):
        for j in range(len(heatmap.columns)):
            value = heatmap.iloc[i, j]

            if pd.notna(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Correlation With Relative Bench Performance")

    save_figure("athlete_response_fingerprint.png")


# ---------------------------------------------------------
# 2. PERSONAL READINESS TIERS
# ---------------------------------------------------------

readiness_components = {
    "sleep_performance": 1,
    "recovery_score": 1,
    "hrv_rmssd": 1,
    "fatigue": -1,
    "soreness": -1,
    "activity_strain_total": -1,
}

readiness = df[["subject", OUTCOME]].copy()
readiness_z_cols = []

for col, direction in readiness_components.items():
    if col in df.columns and df[col].notna().sum() >= 4:
        z_col = f"z_{col}"

        readiness[z_col] = within_subject_z(df, col) * direction
        readiness_z_cols.append(z_col)

readiness["readiness_index"] = readiness[readiness_z_cols].mean(axis=1)
readiness["readiness_components_used"] = readiness[readiness_z_cols].notna().sum(axis=1)

readiness.loc[
    readiness["readiness_components_used"] < 3,
    "readiness_index",
] = np.nan

readiness["readiness_tier"] = readiness.groupby("subject")[
    "readiness_index"
].transform(
    lambda s: thirds(
        s,
        "Low readiness",
        "Middle readiness",
        "High readiness",
    )
)

readiness.to_csv(OUTPUT_DIR / "session_readiness_index.csv", index=False)

tier_order = [
    "Low readiness",
    "Middle readiness",
    "High readiness",
]

tier_summary = (
    readiness.dropna(subset=["readiness_tier", OUTCOME])
    .groupby(["readiness_tier", "subject"], as_index=False)
    .agg(
        average_relative_performance=(OUTCOME, "mean"),
        sessions=(OUTCOME, "size"),
    )
)

tier_summary.to_csv(OUTPUT_DIR / "readiness_tier_summary.csv", index=False)

if not tier_summary.empty:
    tier_plot = (
        tier_summary.pivot(
            index="readiness_tier",
            columns="subject",
            values="average_relative_performance",
        )
        .reindex(tier_order)
    )

    ax = tier_plot.plot(kind="bar", figsize=(9, 6))

    ax.axhline(100, linewidth=1, linestyle="--")
    ax.set_title("Bench Performance Across Personal Readiness Tiers")
    ax.set_xlabel("")
    ax.set_ylabel("Performance Relative to Personal Average (%)")
    ax.legend(title="Participant")

    save_figure("readiness_tier_performance.png")


# ---------------------------------------------------------
# 3. TOP VS BOTTOM SESSION SIGNATURE
# ---------------------------------------------------------

profile = df[["subject", OUTCOME] + list(available.values())].copy()

profile["performance_tier"] = profile.groupby("subject")[OUTCOME].transform(
    lambda s: thirds(
        s,
        "Bottom sessions",
        "Middle sessions",
        "Top sessions",
    )
)

z_lookup = {}

for label, col in available.items():
    z_col = f"z_{col}"

    profile[z_col] = within_subject_z(df, col)
    z_lookup[label] = z_col

profile_summary = (
    profile[
        profile["performance_tier"].isin(
            ["Bottom sessions", "Top sessions"]
        )
    ]
    .groupby("performance_tier")
    .agg({z_col: "mean" for z_col in z_lookup.values()})
    .T
)

profile_summary.index = [
    label
    for label, z_col in z_lookup.items()
    if z_col in profile_summary.index
]

profile_summary["top_minus_bottom"] = (
    profile_summary.get("Top sessions", np.nan)
    - profile_summary.get("Bottom sessions", np.nan)
)

profile_summary = profile_summary.sort_values("top_minus_bottom")

profile_summary.to_csv(
    OUTPUT_DIR / "top_vs_bottom_session_profile.csv"
)

if not profile_summary.empty:
    ax = profile_summary["top_minus_bottom"].plot(
        kind="barh",
        figsize=(9, 6),
    )

    ax.axvline(0, linewidth=1, linestyle="--")
    ax.set_title("What Separated Top vs Bottom Bench Sessions?")
    ax.set_xlabel(
        "Within-Person Standard-Deviation Difference "
        "(Top Minus Bottom)"
    )
    ax.set_ylabel("")

    save_figure("top_vs_bottom_session_profile.png")


print("\nAdvanced analysis complete.")
print("\nCreated charts:")
print("- athlete_response_fingerprint.png")
print("- readiness_tier_performance.png")
print("- top_vs_bottom_session_profile.png")

print("\nReadiness Tier Summary:")
print(tier_summary.to_string(index=False))
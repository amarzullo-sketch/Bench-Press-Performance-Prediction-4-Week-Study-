from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

PROJECT_FOLDER = Path.cwd()
OUTPUT_FOLDER = PROJECT_FOLDER / "output"
VISUALS_FOLDER = OUTPUT_FOLDER / "visuals"

OUTPUT_FOLDER.mkdir(exist_ok=True)
VISUALS_FOLDER.mkdir(exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_column_name(name: str) -> str:
    """Make column names easier to match."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def clean_subject(value):
    """Standardize subject names."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value.startswith("aless"):
        return "Alessandro"

    if value.startswith("liam"):
        return "Liam"

    return value.title()


def find_csv(required_words):
    """
    Finds a CSV in this folder based on words in the filename.
    Example: find_csv(["daily", "combined"])
    """
    csv_files = list(PROJECT_FOLDER.glob("*.csv"))

    for file in csv_files:
        file_name = file.name.lower()

        if all(word.lower() in file_name for word in required_words):
            return file

    raise FileNotFoundError(
        f"Could not find a CSV containing these words: {required_words}\n"
        f"CSV files found: {[file.name for file in csv_files]}"
    )


def find_column(df, possible_names, required=True):
    """
    Finds a column even if Google Sheets made the wording slightly different.
    """
    cleaned_columns = {clean_column_name(column): column for column in df.columns}

    # First tries exact matches
    for possible_name in possible_names:
        cleaned_name = clean_column_name(possible_name)

        if cleaned_name in cleaned_columns:
            return cleaned_columns[cleaned_name]

    # Then tries partial matches
    for possible_name in possible_names:
        cleaned_name = clean_column_name(possible_name)

        for cleaned_column, original_column in cleaned_columns.items():
            if cleaned_name in cleaned_column:
                return original_column

    if required:
        raise KeyError(
            f"Could not find one of these columns: {possible_names}\n"
            f"Columns found: {list(df.columns)}"
        )

    return None


def convert_numeric(df, columns):
    """Convert selected columns to real numbers."""
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace({"MISSING": np.nan, "": np.nan, "nan": np.nan}),
                errors="coerce",
            )

    return df


def save_plot(file_name):
    """Save a chart cleanly."""
    plt.tight_layout()
    plt.savefig(VISUALS_FOLDER / file_name, dpi=200)
    plt.close()


# ============================================================
# LOAD CSV FILES
# ============================================================

bench_file = find_csv(["bench", "log"])
daily_file = find_csv(["daily", "combined"])

print(f"\nUsing Bench Log file: {bench_file.name}")
print(f"Using Daily Combined file: {daily_file.name}")

bench_raw = pd.read_csv(bench_file)
daily_raw = pd.read_csv(daily_file)

print("\nBench Log columns:")
print(list(bench_raw.columns))

print("\nDaily Combined columns:")
print(list(daily_raw.columns))


# ============================================================
# CLEAN BENCH LOG
# ============================================================

bench = pd.DataFrame()

bench["date"] = pd.to_datetime(
    bench_raw[find_column(bench_raw, ["date", "date all manuel"])],
    errors="coerce",
)

bench["subject"] = bench_raw[
    find_column(bench_raw, ["subject"])
].apply(clean_subject)

bench["session_type"] = bench_raw[
    find_column(bench_raw, ["session type"])
].astype(str).str.strip().str.title()

bench["top_set_weight"] = bench_raw[
    find_column(bench_raw, ["top set weight"])
]

bench["top_set_reps"] = bench_raw[
    find_column(bench_raw, ["top set reps"])
]

bench["estimated_1rm"] = bench_raw[
    find_column(bench_raw, ["estimated 1rm"])
]

bench["total_bench_sets"] = bench_raw[
    find_column(bench_raw, ["total bench sets"])
]

bench["total_bench_reps"] = bench_raw[
    find_column(bench_raw, ["total bench reps"])
]

bench["total_bench_volume"] = bench_raw[
    find_column(bench_raw, ["total bench volume"])
]

bench["top_set_rpe"] = bench_raw[
    find_column(bench_raw, ["top set rpe"])
]

bench["energy_before_lift"] = bench_raw[
    find_column(bench_raw, ["energy before lift"])
]

bench["workout_quality"] = bench_raw[
    find_column(bench_raw, ["workout quality"])
]

week_col_bench = find_column(bench_raw, ["week"], required=False)
if week_col_bench:
    bench["week"] = bench_raw[week_col_bench].astype(str).str.strip()
else:
    bench["week"] = np.nan

bench = convert_numeric(
    bench,
    [
        "top_set_weight",
        "top_set_reps",
        "estimated_1rm",
        "total_bench_sets",
        "total_bench_reps",
        "total_bench_volume",
        "top_set_rpe",
        "energy_before_lift",
        "workout_quality",
    ],
)

# Calculate Estimated 1RM only if it is missing
bench["estimated_1rm"] = bench["estimated_1rm"].fillna(
    bench["top_set_weight"] * (1 + bench["top_set_reps"] / 30)
)

bench = bench.dropna(subset=["date", "subject", "estimated_1rm"])
bench = bench.drop_duplicates(subset=["date", "subject", "session_type"])

print(f"\nClean bench sessions: {len(bench)}")


# ============================================================
# CLEAN DAILY COMBINED
# ============================================================

daily = pd.DataFrame()

daily["date"] = pd.to_datetime(
    daily_raw[find_column(daily_raw, ["date"])],
    errors="coerce",
)

daily["subject"] = daily_raw[
    find_column(daily_raw, ["subject"])
].apply(clean_subject)

daily["recovery_score"] = daily_raw[
    find_column(daily_raw, ["recovery score"], required=False)
] if find_column(daily_raw, ["recovery score"], required=False) else np.nan

daily["resting_hr"] = daily_raw[
    find_column(daily_raw, ["resting hr", "resting heart rate"], required=False)
] if find_column(daily_raw, ["resting hr", "resting heart rate"], required=False) else np.nan

daily["hrv_rmssd"] = daily_raw[
    find_column(daily_raw, ["hrv rmssd", "heart rate variability"], required=False)
] if find_column(daily_raw, ["hrv rmssd", "heart rate variability"], required=False) else np.nan

daily["sleep_performance"] = daily_raw[
    find_column(daily_raw, ["sleep performance"], required=False)
] if find_column(daily_raw, ["sleep performance"], required=False) else np.nan

daily["hours_slept"] = daily_raw[
    find_column(daily_raw, ["hours slept"], required=False)
] if find_column(daily_raw, ["hours slept"], required=False) else np.nan

daily["activity_strain_total"] = daily_raw[
    find_column(
        daily_raw,
        ["activity strain total", "day strain", "activity strain", "avg strain"],
        required=False,
    )
] if find_column(
    daily_raw,
    ["activity strain total", "day strain", "activity strain", "avg strain"],
    required=False,
) else np.nan

daily["body_weight"] = daily_raw[
    find_column(daily_raw, ["body weight"], required=False)
] if find_column(daily_raw, ["body weight"], required=False) else np.nan

daily["calories"] = daily_raw[
    find_column(daily_raw, ["calories"], required=False)
] if find_column(daily_raw, ["calories"], required=False) else np.nan

daily["protein"] = daily_raw[
    find_column(daily_raw, ["protein"], required=False)
] if find_column(daily_raw, ["protein"], required=False) else np.nan

daily["water"] = daily_raw[
    find_column(daily_raw, ["water"], required=False)
] if find_column(daily_raw, ["water"], required=False) else np.nan

daily["stress"] = daily_raw[
    find_column(daily_raw, ["stress"], required=False)
] if find_column(daily_raw, ["stress"], required=False) else np.nan

daily["soreness"] = daily_raw[
    find_column(daily_raw, ["soreness"], required=False)
] if find_column(daily_raw, ["soreness"], required=False) else np.nan

daily["fatigue"] = daily_raw[
    find_column(daily_raw, ["overall fatigue", "fatigue"], required=False)
] if find_column(daily_raw, ["overall fatigue", "fatigue"], required=False) else np.nan

daily["bench_day"] = daily_raw[
    find_column(daily_raw, ["bench day"], required=False)
] if find_column(daily_raw, ["bench day"], required=False) else np.nan

week_col_daily = find_column(daily_raw, ["week"], required=False)
if week_col_daily:
    daily["week"] = daily_raw[week_col_daily].astype(str).str.strip()
else:
    daily["week"] = np.nan

daily = convert_numeric(
    daily,
    [
        "recovery_score",
        "resting_hr",
        "hrv_rmssd",
        "sleep_performance",
        "hours_slept",
        "activity_strain_total",
        "body_weight",
        "calories",
        "protein",
        "water",
        "stress",
        "soreness",
        "fatigue",
    ],
)

daily = daily.dropna(subset=["date", "subject"])

# Keep only one daily row per person/date if accidental duplicates exist
daily = daily.sort_values(["date", "subject"]).drop_duplicates(
    subset=["date", "subject"],
    keep="first",
)

print(f"Clean daily rows: {len(daily)}")


# ============================================================
# MERGE: ONE ROW = ONE BENCH SESSION
# ============================================================

analysis = bench.merge(
    daily,
    how="left",
    on=["date", "subject"],
    suffixes=("_bench", "_daily"),
)

# Prefer week from Bench Log; use Daily Combined week if Bench Log week is blank
analysis["week"] = analysis["week_bench"].replace("nan", np.nan).fillna(
    analysis["week_daily"]
)

analysis = analysis.drop(columns=["week_bench", "week_daily"])

analysis = analysis.sort_values(["subject", "date"]).reset_index(drop=True)

analysis.to_csv(OUTPUT_FOLDER / "bench_analysis_dataset.csv", index=False)

important_columns = [
    "recovery_score",
    "resting_hr",
    "hrv_rmssd",
    "sleep_performance",
    "hours_slept",
    "activity_strain_total",
    "body_weight",
    "protein",
    "fatigue",
    "soreness",
    "stress",
]

missing_bench_days = analysis.loc[
    analysis[important_columns].isna().any(axis=1),
    ["date", "subject", "session_type", "estimated_1rm"] + important_columns,
]

missing_bench_days.to_csv(
    OUTPUT_FOLDER / "missing_bench_day_values.csv",
    index=False,
)

print("Created:")
print(OUTPUT_FOLDER / "missing_bench_day_values.csv")
print("\nCreated:")
print(OUTPUT_FOLDER / "bench_analysis_dataset.csv")


# ============================================================
# DATA QUALITY REPORT
# ============================================================

missing_counts = analysis.isna().sum().sort_values(ascending=False)

suspicious_dates = analysis[
    (analysis["date"].dt.year < 2025) | (analysis["date"].dt.year > 2027)
]

report_lines = [
    "BENCH PROJECT DATA QUALITY REPORT",
    "=" * 40,
    f"Bench sessions in final dataset: {len(analysis)}",
    f"Subjects: {', '.join(analysis['subject'].dropna().unique())}",
    "",
    "Missing values by column:",
    missing_counts.to_string(),
    "",
    "Possible suspicious dates:",
    suspicious_dates[["date", "subject", "estimated_1rm"]].to_string(index=False)
    if not suspicious_dates.empty
    else "None found.",
]

(OUTPUT_FOLDER / "data_quality_report.txt").write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print("Created:")
print(OUTPUT_FOLDER / "data_quality_report.txt")


# ============================================================
# DESCRIPTIVE SUMMARY
# ============================================================

summary_columns = [
    "estimated_1rm",
    "recovery_score",
    "sleep_performance",
    "hours_slept",
    "hrv_rmssd",
    "fatigue",
    "soreness",
]

summary = (
    analysis.groupby("subject")[summary_columns]
    .agg(["count", "mean", "std", "min", "max"])
    .round(2)
)

summary.to_csv(OUTPUT_FOLDER / "descriptive_summary.csv")

print("Created:")
print(OUTPUT_FOLDER / "descriptive_summary.csv")


# ============================================================
# VISUALS
# ============================================================

# Estimated 1RM over time
plt.figure(figsize=(10, 6))

for subject, group in analysis.groupby("subject"):
    group = group.sort_values("date")
    plt.plot(group["date"], group["estimated_1rm"], marker="o", label=subject)

plt.title("Estimated Bench 1RM Over Time")
plt.xlabel("Date")
plt.ylabel("Estimated 1RM")
plt.legend()
save_plot("estimated_1rm_over_time.png")


# Scatter plots for major variables
scatter_variables = [
    ("sleep_performance", "Sleep Performance (%)"),
    ("hours_slept", "Hours Slept"),
    ("recovery_score", "Recovery Score"),
    ("hrv_rmssd", "HRV RMSSD"),
    ("fatigue", "Overall Fatigue"),
    ("soreness", "Chest/Triceps Soreness"),
    ("protein", "Protein (g)"),
]

for column, label in scatter_variables:
    plot_data = analysis.dropna(subset=[column, "estimated_1rm"])

    if len(plot_data) < 2:
        continue

    plt.figure(figsize=(8, 6))

    for subject, group in plot_data.groupby("subject"):
        plt.scatter(group[column], group["estimated_1rm"], label=subject)

    plt.title(f"{label} vs Estimated 1RM")
    plt.xlabel(label)
    plt.ylabel("Estimated 1RM")
    plt.legend()

    clean_file_name = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
    save_plot(f"{clean_file_name}_vs_estimated_1rm.png")


# ============================================================
# CORRELATIONS
# ============================================================

predictors = [
    "recovery_score",
    "hrv_rmssd",
    "resting_hr",
    "sleep_performance",
    "hours_slept",
    "activity_strain_total",
    "body_weight",
    "protein",
    "fatigue",
    "soreness",
    "stress",
]

correlation_rows = []

# Combined correlation results
for predictor in predictors:
    temp = analysis[["estimated_1rm", predictor]].dropna()

    if len(temp) >= 3:
        correlation_rows.append(
            {
                "group": "All Subjects",
                "predictor": predictor,
                "n_sessions": len(temp),
                "correlation_with_estimated_1rm": temp["estimated_1rm"]
                .corr(temp[predictor]),
            }
        )

# Separate correlation results by subject
for subject, group in analysis.groupby("subject"):
    for predictor in predictors:
        temp = group[["estimated_1rm", predictor]].dropna()

        if len(temp) >= 3:
            correlation_rows.append(
                {
                    "group": subject,
                    "predictor": predictor,
                    "n_sessions": len(temp),
                    "correlation_with_estimated_1rm": temp["estimated_1rm"]
                    .corr(temp[predictor]),
                }
            )

correlations = pd.DataFrame(correlation_rows)

if not correlations.empty:
    correlations["abs_correlation"] = correlations[
        "correlation_with_estimated_1rm"
    ].abs()

    correlations = correlations.sort_values(
        ["group", "abs_correlation"],
        ascending=[True, False],
    )

correlations.to_csv(OUTPUT_FOLDER / "correlations.csv", index=False)

print("Created:")
print(OUTPUT_FOLDER / "correlations.csv")


# ============================================================
# SIMPLE REGRESSION MODELS
# ============================================================

regression_rows = []

for group_name, group_data in [("All Subjects", analysis)] + list(analysis.groupby("subject")):
    for predictor in predictors:
        model_data = group_data[["estimated_1rm", predictor]].dropna()

        # Need enough sessions for a small regression
        if len(model_data) < 4:
            continue

        X = model_data[[predictor]]
        y = model_data["estimated_1rm"]

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)

        regression_rows.append(
            {
                "group": group_name,
                "predictor": predictor,
                "n_sessions": len(model_data),
                "intercept": model.intercept_,
                "coefficient": model.coef_[0],
                "r_squared": r2_score(y, predictions),
            }
        )

regressions = pd.DataFrame(regression_rows)

if not regressions.empty:
    regressions = regressions.sort_values(
        ["group", "r_squared"],
        ascending=[True, False],
    )

regressions.to_csv(OUTPUT_FOLDER / "simple_regressions.csv", index=False)

print("Created:")
print(OUTPUT_FOLDER / "simple_regressions.csv")

# ============================================================
# BETTER VISUALS: NORMALIZED PERFORMANCE / FAIR COMPARISON
# ============================================================

# 100 = each person's own average estimated 1RM
analysis["performance_vs_personal_avg_pct"] = (
    analysis["estimated_1rm"]
    / analysis.groupby("subject")["estimated_1rm"].transform("mean")
    * 100
)

analysis.to_csv(OUTPUT_FOLDER / "bench_analysis_dataset.csv", index=False)

# ------------------------------------------------------------
# 1. Performance relative to personal average over time
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for subject, group in analysis.groupby("subject"):
    group = group.sort_values("date")
    plt.plot(
        group["date"],
        group["performance_vs_personal_avg_pct"],
        marker="o",
        label=subject,
    )

plt.axhline(100, linestyle="--", linewidth=1)
plt.title("Bench Performance Relative to Each Person's Average")
plt.xlabel("Date")
plt.ylabel("Performance (% of Personal Average Estimated 1RM)")
plt.legend()
save_plot("performance_vs_personal_average_over_time.png")


# ------------------------------------------------------------
# 2. Weekly average performance relative to personal average
# ------------------------------------------------------------

weekly_performance = (
    analysis.dropna(subset=["week"])
    .groupby(["subject", "week"], as_index=False)["performance_vs_personal_avg_pct"]
    .mean()
)

weekly_performance["week_number"] = (
    weekly_performance["week"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)

weekly_performance = weekly_performance.sort_values(
    ["subject", "week_number"]
)

plt.figure(figsize=(10, 6))

for subject, group in weekly_performance.groupby("subject"):
    plt.plot(
        group["week"],
        group["performance_vs_personal_avg_pct"],
        marker="o",
        label=subject,
    )

plt.axhline(100, linestyle="--", linewidth=1)
plt.title("Weekly Bench Performance Relative to Personal Average")
plt.xlabel("Week")
plt.ylabel("Average Performance (% of Personal Average)")
plt.legend()
save_plot("weekly_performance_relative_to_personal_average.png")


# ------------------------------------------------------------
# 3. Within-person correlations
# This avoids Liam simply looking stronger because his raw 1RM is higher.
# ------------------------------------------------------------

predictor_labels = {
    "recovery_score": "Recovery Score",
    "hrv_rmssd": "HRV",
    "resting_hr": "Resting HR",
    "sleep_performance": "Sleep Performance",
    "hours_slept": "Hours Slept",
    "activity_strain_total": "Activity Strain",
    "body_weight": "Body Weight",
    "protein": "Protein",
    "fatigue": "Fatigue",
    "soreness": "Soreness",
    "stress": "Stress",
}

within_person_rows = []

for predictor, label in predictor_labels.items():
    temp = analysis[
        ["subject", "performance_vs_personal_avg_pct", predictor]
    ].dropna()

    if len(temp) < 5:
        continue

    temp = temp.copy()

    temp["performance_centered"] = (
        temp["performance_vs_personal_avg_pct"]
        - temp.groupby("subject")["performance_vs_personal_avg_pct"].transform("mean")
    )

    temp["predictor_centered"] = (
        temp[predictor]
        - temp.groupby("subject")[predictor].transform("mean")
    )

    correlation = temp["performance_centered"].corr(
        temp["predictor_centered"]
    )

    within_person_rows.append(
        {
            "predictor": predictor,
            "predictor_label": label,
            "n_sessions": len(temp),
            "within_person_correlation": correlation,
            "absolute_correlation": abs(correlation),
        }
    )

within_person_correlations = pd.DataFrame(within_person_rows)

within_person_correlations = within_person_correlations.sort_values(
    "absolute_correlation",
    ascending=False,
)

within_person_correlations.to_csv(
    OUTPUT_FOLDER / "within_person_correlations.csv",
    index=False,
)

print("Created:")
print(OUTPUT_FOLDER / "within_person_correlations.csv")


# ------------------------------------------------------------
# 4. Bar chart: strongest recovery/lifestyle relationships
# ------------------------------------------------------------

top_predictors = within_person_correlations.head(5).copy()
top_predictors = top_predictors.sort_values("within_person_correlation")

plt.figure(figsize=(9, 6))
plt.barh(
    top_predictors["predictor_label"],
    top_predictors["within_person_correlation"],
)

plt.axvline(0, linestyle="--", linewidth=1)
plt.title("Strongest Within-Person Relationships With Bench Performance")
plt.xlabel("Correlation With Performance Relative to Personal Average")
plt.ylabel("Predictor")
save_plot("top_within_person_correlations.png")


# ------------------------------------------------------------
# 5. Scatterplots for the top 3 strongest predictors
# ------------------------------------------------------------

for _, row in within_person_correlations.head(3).iterrows():
    predictor = row["predictor"]
    label = row["predictor_label"]

    plot_data = analysis.dropna(
        subset=[predictor, "performance_vs_personal_avg_pct"]
    )

    if len(plot_data) < 3:
        continue

    plt.figure(figsize=(8, 6))

    for subject, group in plot_data.groupby("subject"):
        plt.scatter(
            group[predictor],
            group["performance_vs_personal_avg_pct"],
            label=subject,
        )

        if len(group) >= 2:
            slope, intercept = np.polyfit(
                group[predictor],
                group["performance_vs_personal_avg_pct"],
                1,
            )

            x_values = np.linspace(
                group[predictor].min(),
                group[predictor].max(),
                100,
            )

            y_values = slope * x_values + intercept

            plt.plot(x_values, y_values)

    plt.axhline(100, linestyle="--", linewidth=1)
    plt.title(f"{label} vs Bench Performance Relative to Personal Average")
    plt.xlabel(label)
    plt.ylabel("Performance (% of Personal Average Estimated 1RM)")
    plt.legend()

    safe_name = re.sub(r"[^a-z0-9]+", "_", predictor.lower()).strip("_")
    save_plot(f"best_predictor_{safe_name}_vs_relative_performance.png")

    # ============================================================
# ACTUAL 1RM: BEFORE VS AFTER
# ============================================================

actual_1rm = pd.DataFrame({
    "subject": ["Alessandro", "Alessandro", "Liam", "Liam"],
    "timepoint": ["Before", "After", "Before", "After"],
    "actual_1rm": [
        225,  # Alessandro real 1RM before project
        245,  # Alessandro real 1RM after project
        280,  # Liam real 1RM before project
        285,  # Liam real 1RM after project
    ],
})

actual_1rm.to_csv(
    OUTPUT_FOLDER / "actual_1rm_before_after_summary.csv",
    index=False,
)

pivot = actual_1rm.pivot(
    index="subject",
    columns="timepoint",
    values="actual_1rm"
).reset_index()

subjects = pivot["subject"].tolist()
x = np.arange(len(subjects))
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(x - width / 2, pivot["Before"], width, label="Before")
plt.bar(x + width / 2, pivot["After"], width, label="After")

plt.xticks(x, subjects)
plt.ylabel("Actual Bench 1RM (lb)")
plt.title("Actual Bench 1RM: Before vs After")
plt.legend()

for i, value in enumerate(pivot["Before"]):
    plt.text(x[i] - width / 2, value, f"{value:.0f}",
             ha="center", va="bottom")

for i, value in enumerate(pivot["After"]):
    plt.text(x[i] + width / 2, value, f"{value:.0f}",
             ha="center", va="bottom")

save_plot("actual_1rm_before_after.png")
# ============================================================
# PRINT QUICK RESULTS
# ============================================================

print("\n" + "=" * 55)
print("PROJECT FINISHED SUCCESSFULLY")
print("=" * 55)

print("\nFinal bench analysis dataset preview:")
print(analysis.head())

print("\nBench sessions by subject:")
print(analysis["subject"].value_counts())

if not correlations.empty:
    print("\nStrongest correlations:")
    print(
        correlations[
            ["group", "predictor", "n_sessions", "correlation_with_estimated_1rm"]
        ]
        .head(12)
        .to_string(index=False)
    )

print("\nOpen the output folder to see:")
print("- bench_analysis_dataset.csv")
print("- descriptive_summary.csv")
print("- correlations.csv")
print("- simple_regressions.csv")
print("- data_quality_report.txt")
print("- visuals folder with charts")
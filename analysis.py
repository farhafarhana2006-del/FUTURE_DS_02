"""Customer churn and retention analysis for a subscription business.

Run from the project root with:
    python src/analysis.py

The script adapts to common churn dataset column names. When signup dates or
churn reasons are absent, it reports that limitation instead of fabricating
those fields.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "customer_data.csv"
VISUALS_DIR = ROOT / "visualizations"
REPORTS_DIR = ROOT / "reports"

ALIASES = {
    "customer_id": {"customerid", "customer_id", "customer"},
    "gender": {"gender", "sex"},
    "age": {"age", "customerage"},
    "senior_citizen": {"seniorcitizen", "senior_citizen"},
    "partner": {"partner", "married"},
    "dependents": {"dependents", "numberofdependents"},
    "tenure": {"tenure", "tenuremonths", "tenureinmonths"},
    "contract": {"contract", "contracttype", "subscriptiontype"},
    "monthly_charges": {"monthlycharges", "monthlycharge", "monthlyrevenue"},
    "total_charges": {"totalcharges", "totalcharge", "totalrevenue"},
    "payment_method": {"paymentmethod", "payment_method"},
    "start_date": {
        "startdate",
        "signupdate",
        "customer_since",
        "customersince",
        "subscriptionstartdate",
    },
    "churn": {"churn", "churnlabel", "churned", "exited"},
    "churn_reason": {"churnreason", "churn_reason"},
    "last_activity_date": {"lastactivitydate", "last_activity_date"},
}

DISPLAY_NAMES = {
    "customer_id": "Customer ID",
    "gender": "Gender",
    "age": "Age",
    "senior_citizen": "Senior citizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure": "Tenure (months)",
    "contract": "Contract",
    "monthly_charges": "Monthly charges",
    "total_charges": "Total charges",
    "payment_method": "Payment method",
    "start_date": "Start date",
    "churn": "Churn",
    "churn_reason": "Churn reason",
    "last_activity_date": "Last activity date",
}


def normalize_name(value: str) -> str:
    """Normalize a column name for alias matching."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename recognized input columns while leaving other fields intact."""
    by_normalized_name = {
        normalize_name(alias): canonical
        for canonical, aliases in ALIASES.items()
        for alias in aliases
    }
    rename: dict[str, str] = {}
    already_used: set[str] = set()
    for column in frame.columns:
        canonical = by_normalized_name.get(normalize_name(str(column)))
        if canonical and canonical not in already_used:
            rename[column] = canonical
            already_used.add(canonical)
    return frame.rename(columns=rename)


def load_and_clean_data(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, inspect, standardize, and derive analysis fields."""
    raw = pd.read_csv(path)
    raw_rows = len(raw)
    raw_columns = list(raw.columns)
    raw_duplicates = int(raw.duplicated().sum())
    blank_normalized = raw.replace(r"^\s*$", pd.NA, regex=True)
    missing_before = blank_normalized.isna().sum().to_dict()

    data = canonicalize_columns(raw)
    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].astype("string").str.strip()
        data[column] = data[column].replace("", pd.NA)

    # Without a stable customer key, identical attribute rows may represent
    # different people. Flag them, but do not remove valid customer records.
    duplicate_rows_removed = 0
    if "customer_id" in data:
        exact_duplicates = data.duplicated().sum()
        data = data.drop_duplicates().copy()
        duplicate_rows_removed = int(exact_duplicates)
    numeric_columns = ["age", "tenure", "monthly_charges", "total_charges"]
    for column in numeric_columns:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in ("start_date", "last_activity_date"):
        if column in data:
            data[column] = pd.to_datetime(data[column], errors="coerce")

    if "senior_citizen" in data:
        senior_map = {
            "1": "Senior citizen",
            "1.0": "Senior citizen",
            "yes": "Senior citizen",
            "true": "Senior citizen",
            "0": "Non-senior",
            "0.0": "Non-senior",
            "no": "Non-senior",
            "false": "Non-senior",
        }
        data["senior_citizen"] = data["senior_citizen"].astype("string").str.lower().map(
            senior_map
        ).fillna(data["senior_citizen"])

    if "churn" in data:
        churn_values = data["churn"].astype("string").str.strip().str.lower()
        churn_map = {
            "yes": "Yes",
            "y": "Yes",
            "true": "Yes",
            "1": "Yes",
            "churned": "Yes",
            "no": "No",
            "n": "No",
            "false": "No",
            "0": "No",
            "retained": "No",
            "active": "No",
        }
        data["churn"] = churn_values.map(churn_map).fillna(
            data["churn"].astype("string").str.strip()
        )

    # Numeric values are not silently discarded: remaining gaps are imputed
    # with a median, while TotalCharges can be derived when duration is known.
    if {"total_charges", "tenure", "monthly_charges"}.issubset(data.columns):
        derivable = data["total_charges"].isna() & data["tenure"].notna()
        data.loc[derivable, "total_charges"] = (
            data.loc[derivable, "tenure"] * data.loc[derivable, "monthly_charges"]
        )
    for column in numeric_columns:
        if column in data and data[column].isna().any():
            median = data[column].median()
            if pd.notna(median):
                data[column] = data[column].fillna(median)

    for column in data.select_dtypes(include=["object", "string"]).columns:
        data[column] = data[column].fillna("Unknown")

    if "tenure" in data:
        data["tenure_group"] = pd.cut(
            data["tenure"],
            bins=[-1, 6, 12, 24, 48, np.inf],
            labels=["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49+ months"],
        )
    if "age" in data:
        data["age_group"] = pd.cut(
            data["age"],
            bins=[0, 25, 35, 45, 55, 65, np.inf],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
            include_lowest=True,
        )
    if "monthly_charges" in data and data["monthly_charges"].nunique() > 1:
        data["monthly_charge_group"] = pd.qcut(
            data["monthly_charges"],
            q=min(4, data["monthly_charges"].nunique()),
            labels=False,
            duplicates="drop",
        )
        data["monthly_charge_group"] = (
            data["monthly_charge_group"].astype("Int64").add(1).astype("string")
            .replace("<NA>", "Unknown")
            .map(lambda value: f"Quartile {value}" if value != "Unknown" else value)
        )
    if "total_charges" in data and data["total_charges"].nunique() > 1:
        data["total_charge_group"] = pd.qcut(
            data["total_charges"],
            q=min(4, data["total_charges"].nunique()),
            labels=False,
            duplicates="drop",
        )
        data["total_charge_group"] = (
            data["total_charge_group"].astype("Int64").add(1).astype("string")
            .replace("<NA>", "Unknown")
            .map(lambda value: f"Quartile {value}" if value != "Unknown" else value)
        )

    quality = {
        "source_rows": raw_rows,
        "rows_after_duplicate_removal": len(data),
        "exact_duplicate_rows_detected": raw_duplicates,
        "duplicate_rows_removed": duplicate_rows_removed,
        "duplicate_handling_note": (
            "Identical rows were retained because no stable customer ID is present."
            if "customer_id" not in data
            else "Exact duplicate records were removed."
        ),
        "source_columns": raw_columns,
        "missing_before_cleaning": missing_before,
        "missing_after_cleaning": data.isna().sum().to_dict(),
        "recognized_fields": sorted(set(data.columns).intersection(ALIASES)),
    }
    return data, quality


def safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def segment_churn(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return customer counts and churn rates for a categorical segment."""
    if column not in data or "churn" not in data:
        return pd.DataFrame()
    result = (
        data.groupby(column, observed=True)["churn"]
        .agg(customers="size", churned=lambda values: (values == "Yes").sum())
        .reset_index()
    )
    result["churn_rate"] = result["churned"] / result["customers"]
    result["retention_rate"] = 1 - result["churn_rate"]
    return result.sort_values("churn_rate", ascending=False)


def kaplan_meier_curve(data: pd.DataFrame) -> pd.DataFrame:
    """Estimate observed-duration survival with churn as event and active as censored."""
    if "tenure" not in data or "churn" not in data or data.empty:
        return pd.DataFrame(columns=["month", "at_risk", "churn_events", "retention"])
    durations = pd.to_numeric(data["tenure"], errors="coerce").fillna(0).clip(lower=0)
    events = data["churn"].eq("Yes")
    at_risk = len(data)
    retention = 1.0
    rows = [{"month": 0, "at_risk": at_risk, "churn_events": 0, "retention": retention}]
    max_month = int(durations.max()) if len(durations) else 0
    for month in range(1, max_month + 1):
        event_count = int((events & durations.eq(month)).sum())
        censor_count = int((~events & durations.eq(month)).sum())
        if at_risk > 0:
            retention *= 1 - event_count / at_risk
        rows.append(
            {
                "month": month,
                "at_risk": at_risk,
                "churn_events": event_count,
                "retention": retention,
            }
        )
        at_risk -= event_count + censor_count
    return pd.DataFrame(rows)


def make_kpis(data: pd.DataFrame) -> dict[str, float]:
    customers = len(data)
    churned = int(data["churn"].eq("Yes").sum()) if "churn" in data else 0
    result = {
        "total_customers": float(customers),
        "churned_customers": float(churned),
        "active_customers": float(customers - churned),
        "churn_rate": safe_rate(churned, customers),
        "retention_rate": safe_rate(customers - churned, customers),
        "average_tenure_months": float(data["tenure"].mean()) if "tenure" in data else float("nan"),
        "average_monthly_revenue": float(data["monthly_charges"].mean())
        if "monthly_charges" in data
        else float("nan"),
        "average_total_revenue": float(data["total_charges"].mean())
        if "total_charges" in data
        else float("nan"),
    }
    if "monthly_charges" in data and "tenure" in data:
        result["simple_estimated_ltv"] = float(
            (data["monthly_charges"] * data["tenure"]).mean()
        )
    else:
        result["simple_estimated_ltv"] = float("nan")
    return result


def _percent(value: float) -> str:
    return f"{value:.1%}" if pd.notna(value) else "not available"


def _currency(value: float) -> str:
    return f"${value:,.2f}" if pd.notna(value) else "not available"


def _save_bar_chart(
    table: pd.DataFrame,
    category: str,
    title: str,
    path: Path,
    *,
    value: str = "churn_rate",
    ylabel: str = "Churn rate",
    percent: bool = True,
) -> None:
    if table.empty:
        return
    ordered = table.sort_values(value)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(ordered))))
    bars = ax.barh(ordered[category].astype(str), ordered[value], color="#2878B5")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(ylabel)
    if percent:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(axis="x", alpha=0.2)
    ax.bar_label(
        bars,
        labels=[f"{x:.1%}" if percent else f"{x:,.0f}" for x in ordered[value]],
        padding=4,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_visualizations(
    data: pd.DataFrame, kpis: dict[str, float], tables: dict[str, pd.DataFrame]
) -> None:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    cards = [
        ("Total customers", f"{int(kpis['total_customers']):,}"),
        ("Churn rate", _percent(kpis["churn_rate"])),
        ("Retention rate", _percent(kpis["retention_rate"])),
        ("Average tenure", f"{kpis['average_tenure_months']:.1f} mo"),
        ("Avg monthly revenue", _currency(kpis["average_monthly_revenue"])),
    ]
    for ax, (label, value) in zip(axes, cards):
        ax.axis("off")
        ax.text(0.5, 0.63, value, ha="center", va="center", fontsize=19, weight="bold", color="#174A6E")
        ax.text(0.5, 0.28, label, ha="center", va="center", fontsize=10, color="#4D5B66")
        ax.add_patch(plt.Rectangle((0.04, 0.08), 0.92, 0.84, fill=False, lw=1.2, ec="#D7E3EA"))
    fig.suptitle("Customer health snapshot", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "kpi_cards.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if "churn" in data:
        counts = data["churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(["Retained", "Churned"], counts.values, color=["#3A9D75", "#D95F59"])
        ax.bar_label(bars, labels=[f"{v:,} ({v / len(data):.1%})" for v in counts], padding=4)
        ax.set(title="Customer churn distribution", ylabel="Customers")
        fig.tight_layout()
        fig.savefig(VISUALS_DIR / "churn_distribution.png", dpi=160)
        plt.close(fig)

    for column, filename, title in [
        ("contract", "churn_by_contract.png", "Churn by contract type"),
        ("payment_method", "churn_by_payment_method.png", "Churn by payment method"),
        ("gender", "churn_by_gender.png", "Churn by gender"),
        ("age_group", "churn_by_age_group.png", "Churn by age group"),
        ("tenure_group", "churn_by_tenure.png", "Churn by tenure"),
        ("monthly_charge_group", "churn_by_monthly_charges.png", "Churn by monthly-charge quartile"),
        ("total_charge_group", "churn_by_total_charges.png", "Churn by total-charge quartile"),
        ("churn_reason", "churn_reasons.png", "Reported churn reasons"),
    ]:
        if column in tables:
            _save_bar_chart(tables[column], column, title, VISUALS_DIR / filename)

    for column, filename, title in [
        ("contract", "retention_by_contract.png", "Retention by contract type"),
        ("tenure_group", "retention_by_tenure.png", "Retention by tenure segment"),
    ]:
        if column in tables:
            _save_bar_chart(
                tables[column],
                column,
                title,
                VISUALS_DIR / filename,
                value="retention_rate",
                ylabel="Retention rate",
            )

    if "tenure" in data:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.histplot(data=data, x="tenure", hue="churn" if "churn" in data else None,
                     bins=24, element="step", stat="count", common_norm=False, ax=ax)
        ax.set(title="Customer tenure distribution", xlabel="Tenure (months)", ylabel="Customers")
        fig.tight_layout()
        fig.savefig(VISUALS_DIR / "customer_lifetime_distribution.png", dpi=160)
        plt.close(fig)

    if "tenure" in data and "churn" in data:
        curve = kaplan_meier_curve(data)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.step(curve["month"], curve["retention"], where="post", color="#2878B5", linewidth=2)
        ax.set(
            title="Estimated customer survival by months since start",
            xlabel="Months since subscription start",
            ylabel="Estimated share retained",
            ylim=(0, 1.04),
        )
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        fig.tight_layout()
        fig.savefig(VISUALS_DIR / "retention_trend.png", dpi=160)
        plt.close(fig)

        # A single cross-section cannot provide true signup-month cohorts.
        # Show survival by available contract segment as a lifecycle proxy.
        milestones = list(range(0, min(int(data["tenure"].max()), 72) + 1, 6))
        if milestones[-1] != int(data["tenure"].max()) and data["tenure"].max() <= 72:
            milestones.append(int(data["tenure"].max()))
        milestones = sorted(set(milestones))
        groups = (
            data.groupby("contract", observed=True)
            if "contract" in data
            else [("All customers", data)]
        )
        heatmap_rows: dict[str, list[float]] = {}
        for group_name, group_data in groups:
            group_curve = kaplan_meier_curve(group_data).set_index("month")["retention"]
            heatmap_rows[str(group_name)] = [
                float(group_curve.loc[group_curve.index <= month].iloc[-1])
                if (group_curve.index <= month).any()
                else 1.0
                for month in milestones
            ]
        heatmap = pd.DataFrame.from_dict(heatmap_rows, orient="index")
        heatmap.columns = [f"Month {month}" for month in milestones]
        heatmap.to_csv(REPORTS_DIR / "tenure_retention_proxy.csv")
        fig, ax = plt.subplots(
            figsize=(max(9, len(milestones) * 0.55), max(2.8, 0.65 * len(heatmap)))
        )
        sns.heatmap(
            heatmap,
            annot=True,
            fmt=".0%",
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            cbar_kws={"label": "Share retained (snapshot proxy)"},
            ax=ax,
        )
        ax.set(
            title="Contract-segment survival proxy (not signup-month cohorts)",
            xlabel="Months since subscription start",
            ylabel="Contract segment",
        )
        fig.tight_layout()
        fig.savefig(VISUALS_DIR / "cohort_retention_heatmap.png", dpi=160)
        plt.close(fig)


def generate_insights(
    data: pd.DataFrame, kpis: dict[str, float], tables: dict[str, pd.DataFrame]
) -> list[str]:
    insights = [
        (
            f"The dataset contains {int(kpis['total_customers']):,} customers; "
            f"{int(kpis['churned_customers']):,} are labeled churned "
            f"({ _percent(kpis['churn_rate'])}), while {_percent(kpis['retention_rate'])} "
            "are labeled retained."
        ),
    ]
    for key, label in [
        ("contract", "contract type"),
        ("payment_method", "payment method"),
        ("tenure_group", "tenure segment"),
        ("monthly_charge_group", "monthly-charge quartile"),
        ("gender", "gender segment"),
        ("age_group", "age segment"),
        ("senior_citizen", "senior-citizen segment"),
    ]:
        table = tables.get(key, pd.DataFrame())
        if len(table) >= 2:
            top = table.iloc[0]
            bottom = table.iloc[-1]
            if pd.notna(top["churn_rate"]) and pd.notna(bottom["churn_rate"]):
                insights.append(
                    f"Churn is highest among {top[key]} customers "
                    f"({top['churn_rate']:.1%}) and lowest among {bottom[key]} customers "
                    f"({bottom['churn_rate']:.1%}); this is an association, not proof of cause."
                )
    if {"churn", "tenure"}.issubset(data.columns):
        means = data.groupby("churn")["tenure"].mean()
        if "Yes" in means and "No" in means:
            insights.append(
                f"Churned customers have average observed tenure of {means['Yes']:.1f} months, "
                f"versus {means['No']:.1f} months for retained customers."
            )
    if {"churn", "monthly_charges"}.issubset(data.columns):
        means = data.groupby("churn")["monthly_charges"].mean()
        if "Yes" in means and "No" in means:
            insights.append(
                f"Average monthly charges are {_currency(means['Yes'])} for churned customers "
                f"and {_currency(means['No'])} for retained customers."
            )
    if {"churn", "total_charges"}.issubset(data.columns):
        means = data.groupby("churn")["total_charges"].mean()
        if "Yes" in means and "No" in means:
            insights.append(
                f"Average accumulated charges are {_currency(means['Yes'])} for churned "
                f"customers and {_currency(means['No'])} for retained customers; shorter "
                "tenure can contribute to this difference."
            )
    if "churn_reason" in tables and not tables["churn_reason"].empty:
        top = tables["churn_reason"].sort_values("customers", ascending=False).iloc[0]
        insights.append(
            f"The most frequently recorded churn reason is {top['churn_reason']} "
            f"({int(top['customers']):,} customers)."
        )
    if len(insights) < 8:
        insights.append(
            "Signup-date cohort comparisons cannot be measured from this extract because "
            "it contains no signup date; collect dated signup and cancellation events."
        )
    return insights[:10]


def generate_recommendations(
    data: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> list[str]:
    recommendations: list[str] = []
    contract = tables.get("contract", pd.DataFrame())
    if not contract.empty:
        top = contract.iloc[0]
        recommendations.append(
            f"Prioritize a contract-specific save test for {top['contract']} customers, "
            f"the highest-churn contract group ({top['churn_rate']:.1%}); compare a flexible "
            "renewal incentive with the current experience."
        )
    tenure = tables.get("tenure_group", pd.DataFrame())
    if not tenure.empty:
        top = tenure.iloc[0]
        recommendations.append(
            f"Strengthen onboarding and proactive check-ins for {top['tenure_group']} "
            f"customers, whose observed churn rate is {top['churn_rate']:.1%}."
        )
    payment = tables.get("payment_method", pd.DataFrame())
    if not payment.empty:
        top = payment.iloc[0]
        recommendations.append(
            f"Review payment friction for customers using {top['payment_method']} "
            f"({top['churn_rate']:.1%} churn); test clearer billing reminders and easy "
            "payment-method updates before changing payment policy."
        )
    monthly = tables.get("monthly_charge_group", pd.DataFrame())
    if not monthly.empty:
        top = monthly.iloc[0]
        recommendations.append(
            f"Test value reviews or transparent plan comparisons for {top['monthly_charge_group']} "
            f"customers ({top['churn_rate']:.1%} churn), and measure incremental saves."
        )
    reason = tables.get("churn_reason", pd.DataFrame())
    if not reason.empty:
        top = reason.sort_values("customers", ascending=False).iloc[0]
        recommendations.append(
            f"Address the leading stated exit reason ({top['churn_reason']}) with a "
            "targeted service/process experiment and track reason-specific churn."
        )
    if "tenure" in data and "churn" in data:
        retained = data[data["churn"] == "No"]
        if not retained.empty:
            recommendations.append(
                f"Pilot recognition or renewal benefits for established customers; retained "
                f"customers average {retained['tenure'].mean():.1f} months of observed tenure. "
                "Evaluate whether benefits improve renewal without eroding margin."
            )
    if "last_activity_date" not in data:
        recommendations.append(
            "Add dated activity/engagement events and cancellation dates to the next data "
            "extract so disengagement and true signup cohorts can be measured before churn."
        )
    recommendations.append(
        "Run small controlled retention experiments and compare incremental retention, "
        "customer lifetime revenue, and offer cost against a holdout group."
    )
    return recommendations[:8]


def write_report(
    data: pd.DataFrame,
    quality: dict[str, Any],
    kpis: dict[str, float],
    tables: dict[str, pd.DataFrame],
    insights: list[str],
    recommendations: list[str],
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if "churn" in data:
        for key, table in tables.items():
            if not table.empty:
                table.to_csv(REPORTS_DIR / f"churn_by_{key}.csv", index=False)
    pd.DataFrame(
        [
            {
                "field": field,
                "missing_before_cleaning": count,
                "missing_after_cleaning": quality["missing_after_cleaning"].get(field, 0),
            }
            for field, count in quality["missing_before_cleaning"].items()
        ]
    ).to_csv(REPORTS_DIR / "data_quality_report.csv", index=False)
    outliers = []
    for column in ("age", "tenure", "monthly_charges", "total_charges"):
        if column in data and pd.api.types.is_numeric_dtype(data[column]):
            q1, q3 = data[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers.append(
                {
                    "field": column,
                    "lower_iqr_bound": lower,
                    "upper_iqr_bound": upper,
                    "outlier_count": int(((data[column] < lower) | (data[column] > upper)).sum()),
                    "outlier_share": float(((data[column] < lower) | (data[column] > upper)).mean()),
                    "treatment": "Flagged for review; not removed",
                }
            )
    pd.DataFrame(outliers).to_csv(REPORTS_DIR / "outlier_check.csv", index=False)
    pd.DataFrame([{"metric": key, "value": value} for key, value in kpis.items()]).to_csv(
        REPORTS_DIR / "kpis.csv", index=False
    )
    pd.DataFrame({"insight": insights}).to_csv(REPORTS_DIR / "business_insights.csv", index=False)
    pd.DataFrame({"recommendation": recommendations}).to_csv(
        REPORTS_DIR / "recommendations.csv", index=False
    )

    contract_table = tables.get("contract", pd.DataFrame())
    payment_table = tables.get("payment_method", pd.DataFrame())
    tenure_table = tables.get("tenure_group", pd.DataFrame())
    top_contract = (
        f"{contract_table.iloc[0]['contract']} ({contract_table.iloc[0]['churn_rate']:.1%})"
        if not contract_table.empty
        else "not available"
    )
    top_payment = (
        f"{payment_table.iloc[0]['payment_method']} "
        f"({payment_table.iloc[0]['churn_rate']:.1%})"
        if not payment_table.empty
        else "not available"
    )
    short_tenure = (
        f"{tenure_table.iloc[0]['tenure_group']} "
        f"({tenure_table.iloc[0]['churn_rate']:.1%})"
        if not tenure_table.empty
        else "not available"
    )
    tenure_note = (
        "A true signup-month cohort matrix is not identifiable from this extract because "
        "no signup/start date or dated cancellation history is present. The heatmap is an "
        "explicitly labeled tenure/survival proxy, not a calendar cohort result."
    )
    lines = [
        "# Customer Churn & Retention — Final Report",
        "",
        "## Executive summary",
        f"- **Customers:** {int(kpis['total_customers']):,}",
        f"- **Churned:** {int(kpis['churned_customers']):,}",
        f"- **Churn rate:** {_percent(kpis['churn_rate'])}",
        f"- **Retention rate:** {_percent(kpis['retention_rate'])}",
        f"- **Average observed tenure:** {kpis['average_tenure_months']:.1f} months",
        f"- **Average monthly charges:** {_currency(kpis['average_monthly_revenue'])}",
        f"- **Average total charges:** {_currency(kpis['average_total_revenue'])}",
        f"- **Simple estimated lifetime revenue:** {_currency(kpis['simple_estimated_ltv'])} "
        "(average monthly charges × observed tenure; descriptive proxy, not a discounted CLV model)",
        "",
        "## Main patterns",
        f"- Highest-churn contract category: {top_contract}.",
        f"- Highest-churn payment method: {top_payment}.",
        f"- Highest-churn tenure segment: {short_tenure}.",
        f"- Age, signup cohorts, customer activity, and churn reasons are not available in this data extract.",
        "",
        "## Cohort and lifetime interpretation",
        f"- {tenure_note}",
        "- Tenure is a cross-sectional observed duration. Churned customers have an observed exit duration; currently retained customers are right-censored at the snapshot.",
        "",
        "## Top business insights",
    ]
    lines.extend(f"{index}. {insight}" for index, insight in enumerate(insights, start=1))
    lines.extend(["", "## Recommended actions"])
    lines.extend(f"{index}. {item}" for index, item in enumerate(recommendations, start=1))
    lines.extend(
        [
            "",
            "## Data quality and limitations",
            f"- Source rows: {quality['source_rows']:,}; duplicate rows removed: {quality['duplicate_rows_removed']:,}.",
            f"- Fields detected: {', '.join(quality['recognized_fields'])}.",
            f"- Exact duplicate rows detected: {quality['exact_duplicate_rows_detected']:,}; removed: {quality['duplicate_rows_removed']:,}. {quality['duplicate_handling_note']}",
            "- Missing numeric values were derived when possible (TotalCharges = tenure × monthly charges) and otherwise median-imputed; missing categories are labeled Unknown.",
            "- Numeric outliers are screened with the 1.5×IQR rule and flagged, not automatically removed.",
            "- These are descriptive associations, not causal effects. Validate retention actions using controlled experiments.",
        ]
    )
    (REPORTS_DIR / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(data_path: Path) -> dict[str, Any]:
    data, quality = load_and_clean_data(data_path)
    if "churn" not in data:
        raise ValueError("The dataset needs a churn outcome column (for example Churn).")
    kpis = make_kpis(data)
    segment_columns = [
        "contract",
        "payment_method",
        "gender",
        "age_group",
        "tenure_group",
        "monthly_charge_group",
        "total_charge_group",
        "senior_citizen",
        "churn_reason",
    ]
    tables = {column: segment_churn(data, column) for column in segment_columns if column in data}
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(REPORTS_DIR / "cleaned_customer_data.csv", index=False)
    make_visualizations(data, kpis, tables)
    insights = generate_insights(data, kpis, tables)
    recommendations = generate_recommendations(data, tables)
    write_report(data, quality, kpis, tables, insights, recommendations)

    print("Analysis complete.")
    print(f"Customers: {int(kpis['total_customers']):,}")
    print(f"Churn rate: {_percent(kpis['churn_rate'])}")
    print(f"Retention rate: {_percent(kpis['retention_rate'])}")
    print(f"Average tenure: {kpis['average_tenure_months']:.1f} months")
    print(f"Final report: {REPORTS_DIR / 'FINAL_REPORT.md'}")
    print(f"Visualizations: {VISUALS_DIR}")
    return {"data": data, "quality": quality, "kpis": kpis, "tables": tables}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to a customer churn CSV file (default: data/customer_data.csv)",
    )
    args = parser.parse_args()
    analyze(args.data)


if __name__ == "__main__":
    main()

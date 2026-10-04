"""Build one standalone HTML page from the existing analysis outputs."""

from __future__ import annotations

import base64
import csv
import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
VISUALIZATIONS = ROOT / "visualizations"
OUTPUT = REPORTS / "customer_churn_dashboard.html"
USD_INR_RATE = 96.34955
RATE_DATE = "4 Oct 2026"

CHARTS = [
    ("churn_distribution.png", "Churn distribution"),
    ("retention_trend.png", "Estimated retention trend"),
    ("churn_by_contract.png", "Churn by contract"),
    ("retention_by_contract.png", "Retention by contract"),
    ("churn_by_payment_method.png", "Churn by payment method"),
    ("churn_by_tenure.png", "Churn by tenure"),
    ("retention_by_tenure.png", "Retention by tenure"),
    ("churn_by_gender.png", "Churn by gender"),
    ("churn_by_monthly_charges.png", "Churn by monthly charges"),
    ("churn_by_total_charges.png", "Churn by total charges"),
    ("customer_lifetime_distribution.png", "Customer tenure distribution"),
    ("cohort_retention_heatmap.png", "Contract survival proxy (not signup cohorts)"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def convert_dollars_to_rupees(text: str) -> str:
    """Convert dollar-denominated insight amounts to INR for this report."""
    return re.sub(
        r"\$([\d,]+\.\d{2})",
        lambda match: f"₹{float(match.group(1).replace(',', '')) * USD_INR_RATE:,.2f}",
        text,
    )


def build_page() -> None:
    kpis = {row["metric"]: float(row["value"]) for row in read_csv(REPORTS / "kpis.csv")}
    insights = [
        convert_dollars_to_rupees(row["insight"])
        for row in read_csv(REPORTS / "business_insights.csv")
    ]
    recommendations = [
        row["recommendation"] for row in read_csv(REPORTS / "recommendations.csv")
    ]
    charts_html = []
    for filename, title in CHARTS:
        path = VISUALIZATIONS / filename
        if not path.exists():
            continue
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        charts_html.append(
            f'<figure><img src="data:image/png;base64,{encoded}" '
            f'alt="{escape(title)}"><figcaption>{escape(title)}</figcaption></figure>'
        )

    cards = [
        ("Total customers", f"{int(kpis['total_customers']):,}"),
        ("Churned customers", f"{int(kpis['churned_customers']):,}"),
        ("Retained customers", f"{int(kpis['active_customers']):,}"),
        ("Churn rate", f"{kpis['churn_rate']:.1%}"),
        ("Retention rate", f"{kpis['retention_rate']:.1%}"),
        ("Average tenure", f"{kpis['average_tenure_months']:.1f} months"),
        ("Average monthly charges", f"₹{kpis['average_monthly_revenue'] * USD_INR_RATE:,.2f}"),
        (
            "Simple estimated lifetime revenue",
            f"₹{kpis['simple_estimated_ltv'] * USD_INR_RATE:,.2f}",
        ),
    ]
    cards_html = "".join(
        f'<div class="card"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
        for label, value in cards
    )
    insight_html = "".join(f"<li>{escape(item)}</li>" for item in insights)
    recommendation_html = "".join(
        f"<li>{escape(item)}</li>" for item in recommendations
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Customer Churn &amp; Retention | Analysis</title>
<style>
:root {{ color-scheme: light; --ink:#17324d; --muted:#536779; --accent:#2878b5; --line:#dce5ec; --surface:#fff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#f3f6f9; color:var(--ink); font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ padding:38px max(5vw,24px); color:white; background:linear-gradient(120deg,#123454,#2878b5); }}
header h1 {{ margin:0 0 8px; font-size:clamp(1.8rem,4vw,2.8rem); }}
header p {{ max-width:850px; margin:0; color:#e1edf6; }}
nav {{ display:flex; flex-wrap:wrap; gap:18px; margin-top:20px; }}
nav a {{ color:#fff; }}
main {{ width:min(1200px,92vw); margin:28px auto 64px; }}
section {{ margin:32px 0; }}
h2 {{ margin:0 0 14px; font-size:1.5rem; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; }}
.rate {{ margin:10px 0 0; color:var(--muted); font-size:.9rem; }}
.card, figure, .note {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; box-shadow:0 4px 16px #16324d0a; }}
.card {{ padding:18px; }}
.card span {{ display:block; color:var(--muted); font-size:.9rem; }}
.card strong {{ display:block; margin-top:5px; font-size:1.55rem; }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr)); gap:16px; }}
figure {{ overflow:hidden; margin:0; padding:12px; }}
figure img {{ display:block; width:100%; height:auto; }}
figcaption {{ padding:8px 4px 2px; color:var(--muted); font-weight:600; }}
.lists {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr)); gap:22px; }}
ol {{ padding-left:24px; }}
li {{ margin:0 0 10px; }}
.note {{ padding:18px 20px; border-left:5px solid #e4a93b; }}
.note strong {{ display:block; margin-bottom:5px; }}
footer {{ margin-top:38px; color:var(--muted); font-size:.9rem; }}
@media print {{ body {{ background:white; }} header {{ print-color-adjust:exact; }} .card,figure,.note {{ break-inside:avoid; box-shadow:none; }} }}
</style>
</head>
<body>
<header>
  <h1>Customer Churn &amp; Retention Analysis</h1>
  <p>Portfolio-ready snapshot of the Telco customer dataset. Findings describe associations and should not be interpreted as causal effects.</p>
  <nav><a href="#summary">KPI summary</a><a href="#charts">Visualizations</a><a href="#insights">Insights</a><a href="#actions">Recommendations</a><a href="#limits">Limitations</a></nav>
</header>
<main>
  <section id="summary"><h2>KPI summary</h2><div class="cards">{cards_html}</div><p class="rate">Currency conversion: USD 1 = INR ₹{USD_INR_RATE:.5f} (ExchangeRate-API reference rate, updated {RATE_DATE}).</p></section>
  <section id="charts"><h2>Existing visualizations</h2><div class="charts">{''.join(charts_html)}</div></section>
  <section class="lists">
    <div id="insights"><h2>Business insights</h2><ol>{insight_html}</ol></div>
    <div id="actions"><h2>Recommended actions</h2><ol>{recommendation_html}</ol></div>
  </section>
  <section id="limits">
    <h2>Data limitations &amp; interpretation</h2>
    <div class="note">
      <strong>No true signup-month cohort data</strong>
      The source file has no signup date, dated cancellation history, churn reason, customer ID, age, or activity events. The heatmap is a contract-segment survival proxy—not a cohort-retention matrix. Retention estimates use observed tenure and a snapshot of current churn status.
    </div>
    <p>Estimated lifetime revenue is a simple descriptive proxy (average monthly charges × observed tenure); it is not discounted, margin-adjusted, or predictive. Validate retention actions through controlled experiments.</p>
  </section>
  <footer>Charts are embedded from the existing project outputs, except the original USD KPI graphic, which is replaced by the INR-converted KPI cards above. This is a standalone HTML file and can be opened locally or printed to PDF.</footer>
</main>
</body>
</html>
"""
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"Created {OUTPUT}")
    print(f"Embedded {len(charts_html)} existing charts.")


if __name__ == "__main__":
    build_page()

import os
import zipfile
from io import BytesIO
from datetime import datetime

import pandas as pd
import streamlit as st
import altair as alt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# ================== App config ==================
st.set_page_config(page_title="CredWyn — Rent & Bills Detector", layout="wide")

# ================== Sidebar (user + settings) ==================
st.sidebar.header("CredWyn")
st.sidebar.caption("Build Trust. Build Credit. Build Your Future.")

# Personal details for PDF
st.sidebar.subheader("Your details (for PDF)")
USER_NAME = st.sidebar.text_input("Full name", value="")
USER_ADDRESS = st.sidebar.text_area("Address (optional)", value="", height=70)

# Detection knobs
st.sidebar.subheader("Detection settings")
AMOUNT_TOLERANCE = st.sidebar.slider("Amount tolerance (±%)", 5, 25, 10)
DUE_WINDOW = st.sidebar.slider("On-time window (± days around due day)", 2, 7, 3)

# Samples
st.sidebar.subheader("Samples")
sample_choice = st.sidebar.selectbox(
    "Choose a sample (or upload your own below):",
    ["None", "Perfect sample", "Late sample", "Partial sample"],
    index=0
)
SAMPLES = {
    "Perfect sample": "data/sample_transactions.csv",
    "Late sample": "data/sample_late.csv",
    "Partial sample": "data/sample_partial.csv",
}

# ================== Helper functions ==================
def load_df_from_csv_like(file_or_buf) -> pd.DataFrame:
    """Normalize to required columns: date, amount, description, counterparty."""
    df = pd.read_csv(file_or_buf)
    df.columns = [c.strip() for c in df.columns]
    if "date" not in df.columns:
        st.error("CSV must include a 'date' column after mapping.")
        st.stop()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["amount", "description", "counterparty"]:
        if col not in df.columns:
            if col == "amount":
                df[col] = 0.0
            else:
                df[col] = ""
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df[["date", "amount", "description", "counterparty"]]

def _monthly_candidates(
    out_df: pd.DataFrame,
    keyword_list=None,
    prefer_keywords=True,
    exclude_cps: set[str] | None = None,
    require_keyword: bool = False
) -> pd.DataFrame:
    """
    Return candidate series for obligations: keyword hits first (if any),
    else 'best' monthly pattern per counterparty (~30d + low amount variance).
    Exclude counterparties in exclude_cps. If require_keyword=True and no keyword hit, return empty.
    """
    out = out_df.copy()
    if exclude_cps:
        out = out[~out["counterparty"].isin(exclude_cps)]

    hit = pd.DataFrame()
    if keyword_list:
        kw = "|".join([k.lower() for k in keyword_list])
        mask = out["description"].str.lower().str.contains(kw, na=False)
        hit = out[mask].copy()
        if prefer_keywords and not hit.empty:
            return hit
        if require_keyword and hit.empty:
            return pd.DataFrame(columns=out.columns)

    # Fallback: best counterparty by (periodicity + low variance)
    best_cp, best_score = None, -1.0
    for cp, g in out.groupby("counterparty"):
        if len(g) < 3:
            continue
        g = g.sort_values("date")
        gaps = g["date"].diff().dt.days.dropna()
        if gaps.empty:
            continue
        median_gap = gaps.median()
        amt_med = g["amount"].abs().median()
        mad_like = (g["amount"].abs() - amt_med).abs().mean()  # manual MAD replacement
        periodicity = max(0.0, 1.0 - abs(median_gap - 30) / 30)        # 1 when exactly 30d
        variance_term = max(0.0, 1.0 - (mad_like / max(1.0, amt_med)))  # 1 when amounts identical
        score = periodicity + variance_term
        if score > best_score:
            best_score, best_cp = score, cp

    if best_cp:
        return out[out["counterparty"] == best_cp].copy()
    return pd.DataFrame(columns=out.columns)

def _status_table(
    cand: pd.DataFrame,
    due_dom: int,
    base_amt: float,
    amount_tol_pct: float,
    due_window_days: int
) -> pd.DataFrame:
    """Mark each payment On-time / Partial / Late based on due-day window and amount tolerance."""
    def status_for(row):
        dom = row["date"].day
        # Late if outside [due_day - window, due_day + window + 2] (slight grace)
        if dom < due_dom - due_window_days or dom > due_dom + (due_window_days + 2):
            return "Late"
        # Partial if below tolerance threshold
        if abs(row["amount"]) < (1 - amount_tol_pct / 100.0) * base_amt:
            return "Partial"
        return "On-time"

    payments = cand[["date", "amount", "description", "counterparty"]].copy()
    payments["status"] = payments.apply(status_for, axis=1)
    return payments.sort_values("date").reset_index(drop=True)

def detect_obligation(
    df: pd.DataFrame,
    label: str,
    keywords,
    exclude_cps: set[str] | None = None,
    require_keyword: bool = False
) -> dict | None:
    """
    Detect a recurring outgoing obligation (rent/council/utility),
    excluding counterparties in exclude_cps.
    If require_keyword=True, must match a keyword to qualify (no fallback).
    """
    out = df[df["amount"] < 0].copy()
    if out.empty:
        return None

    cand = _monthly_candidates(
        out,
        keyword_list=keywords,
        prefer_keywords=True,
        exclude_cps=exclude_cps,
        require_keyword=require_keyword,
    )
    if cand.empty:
        return None

    base_amt = float(cand["amount"].abs().median())
    due_dom = int(cand["date"].dt.day.mode()[0])
    payments = _status_table(cand, due_dom, base_amt, AMOUNT_TOLERANCE, DUE_WINDOW)

    return {
        "type": label,
        "counterparty": str(cand["counterparty"].iloc[0]),
        "baseline_amount": round(base_amt, 2),
        "due_day": due_dom,
        "payments": payments
    }

def compute_credwyn_score(obligations: list[dict]):
    """
    Simple, explainable 300–850 score across all detected obligations.
    70% on-time (weighted by #payments), 20% stability, 10% history length (cap 12m).
    """
    total_payments = 0
    on_time_sum = 0
    stability_terms = []
    months_terms = 0

    for ob in obligations:
        pay = ob["payments"]
        if pay.empty:
            continue
        total_payments += len(pay)
        on_time_sum += (pay["status"].eq("On-time").sum())

        amounts = pay["amount"].abs()
        med = float(amounts.median()) if len(amounts) else 0.0
        if med > 0:
            mad_like = float((amounts - med).abs().mean())  # mean abs deviation (manual)
            stability = 1.0 - (mad_like / max(1.0, med))
            stability_terms.append(max(0.0, min(1.0, stability)))

        months_terms += min(12, len(pay))

    if total_payments == 0:
        return 300, ["No obligations detected yet."], 0.0

    on_time_ratio = on_time_sum / total_payments
    stability_avg = sum(stability_terms) / len(stability_terms) if stability_terms else 0.0
    # Normalize by number of obligation types
    history_factor = min(1.0, months_terms / (12.0 * max(1, len(obligations))))

    w_on_time, w_stability, w_history = 0.7, 0.2, 0.1
    raw = (w_on_time * on_time_ratio) + (w_stability * stability_avg) + (w_history * history_factor)
    score = int(300 + raw * (850 - 300))

    explain = [
        f"On-time across all obligations: {int(on_time_ratio * 100)}%",
        f"Payment stability (amount variance): {int(stability_avg * 100)}%",
        f"History length factor: {int(history_factor * 100)}%"
    ]
    return score, explain, on_time_ratio

def build_pdf(
    obligations: list[dict],
    score: int,
    explain: list[str],
    user_name: str = "",
    user_address: str = ""
) -> BytesIO:
    """Generate Proof of Payments PDF."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = H - 20 * mm

    def write(line, size=12, bold=False, extra_gap=0):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(20 * mm, y, line)
        y -= (7 + extra_gap) * mm

    write("CredWyn — Proof of Payments", size=16, bold=True)
    write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if user_name:
        write(f"Name: {user_name}")
    if user_address:
        for i, line in enumerate(user_address.splitlines()):
            write(("Address: " if i == 0 else "         ") + line)
    write(f"CredWyn Score (300–850): {score}", bold=True, extra_gap=1)
    write("Why this score:", bold=True)
    for e in explain:
        write(f"• {e}")

    for ob in obligations:
        y -= 4 * mm
        write(f"{ob['type'].upper()} — {ob['counterparty']}", bold=True)
        write(f"Baseline amount: £{ob['baseline_amount']:.2f} | Typical due day: {ob['due_day']}")
        write("Month        Date         Amount     Status", bold=True)
        for _, r in ob["payments"].iterrows():
            line = f"{r['date'].strftime('%b %Y'): <12} {r['date'].strftime('%Y-%m-%d'): <12} £{abs(r['amount']):<9.2f} {r['status']}"
            write(line)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

def build_landlord_pack(obligations, score, explain, user_name, user_address, df_normalized, monthly_df):
    """Return an in-memory ZIP containing PDF + normalized transactions + monthly summary."""
    # 1) PDF bytes
    pdf_bytes = build_pdf(obligations, score, explain, user_name, user_address)

    # 2) CSV bytes
    tx_csv = df_normalized.to_csv(index=False).encode("utf-8")
    monthly_csv = monthly_df.to_csv(index=False).encode("utf-8")

    # 3) Pack ZIP in memory
    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"credwyn_proof_{datetime.now().date()}.pdf", pdf_bytes.getvalue())
        z.writestr("transactions_normalized.csv", tx_csv)
        z.writestr("monthly_summary.csv", monthly_csv)
        z.writestr(
            "README.txt",
            "CredWyn Landlord Pack\n"
            "Includes: proof PDF, normalized transactions, monthly summary.\n"
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        )
    buf.seek(0)
    return buf

def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly inflow/outflow/net by calendar month."""
    m = df.copy()
    m["period"] = m["date"].dt.to_period("M").dt.to_timestamp()
    grp = m.groupby("period", as_index=False)
    inflow = grp.apply(lambda g: g.loc[g["amount"] > 0, "amount"].sum()).rename(columns={None: "inflow"})
    outflow = grp.apply(lambda g: g.loc[g["amount"] < 0, "amount"].abs().sum()).rename(columns={None: "outflow"})
    out = pd.merge(inflow, outflow, on="period", how="outer").fillna(0.0)
    out["net"] = out["inflow"] - out["outflow"]
    return out.sort_values("period")

def ontime_trend(obligations: list[dict]) -> pd.DataFrame:
    """Monthly on-time ratio across all detected obligations."""
    rows = []
    for ob in obligations:
        p = ob["payments"].copy()
        if p.empty:
            continue
        p["period"] = p["date"].dt.to_period("M").dt.to_timestamp()
        p["on_time"] = (p["status"] == "On-time").astype(int)
        rows.append(p[["period", "on_time"]])
    if not rows:
        return pd.DataFrame(columns=["period", "on_time_ratio"])
    allp = pd.concat(rows)
    trend = allp.groupby("period")["on_time"].mean().reset_index().rename(columns={"on_time": "on_time_ratio"})
    return trend.sort_values("period")

# ================== UI ==================
st.title("💳 CredWyn — Build Credit from Everyday Life")
st.write(
    "Upload bank transactions (CSV), map your columns, and CredWyn will detect **Rent, Council Tax, and Utilities**. "
    "You’ll get an explainable **CredWyn Score** and a downloadable **Proof PDF**. "
    "Open the **Dashboard** tab for monthly summaries and trends."
)

# Choose sample or upload
uploaded = None
if sample_choice != "None":
    path = SAMPLES.get(sample_choice)
    if path and os.path.exists(path):
        uploaded = path
    elif path:
        st.warning(f"Sample '{sample_choice}' not found at `{path}`. Upload your CSV below instead.")

upl = st.file_uploader("Upload CSV (any schema — we’ll help you map columns)", type=["csv"])
if upl is not None:
    uploaded = upl

if uploaded is None:
    st.info("➡️ Choose a sample in the sidebar **or** upload your CSV.")
    st.stop()

# Tabs
tab_analyze, tab_dashboard = st.tabs(["🔎 Analyze", "📊 Dashboard"])

with tab_analyze:
    # Raw preview
    raw_df = pd.read_csv(uploaded)
    st.caption("Raw CSV preview (first 20 rows):")
    st.dataframe(raw_df.head(20), width="stretch", hide_index=True)

    # Column mapping wizard
    st.subheader("Map your columns")
    cols = list(raw_df.columns)

    def _guess(cands):
        for col in cols:
            low = col.lower()
            if any(k in low for k in cands):
                return col
        return cols[0] if cols else None

    default_date = _guess(["date", "posted", "transaction date", "txn date"])
    default_amount = _guess(["amount", "value", "debit", "credit", "amt"])
    default_desc = _guess(["description", "narrative", "details", "reference", "memo"])
    default_cp = _guess(["counterparty", "merchant", "payee", "beneficiary", "name"])

    c1, c2, c3, c4 = st.columns(4)
    map_date = c1.selectbox("Date column", cols, index=cols.index(default_date) if default_date in cols else 0)
    map_amount = c2.selectbox("Amount column (negative=out)", cols, index=cols.index(default_amount) if default_amount in cols else 0)
    map_desc = c3.selectbox("Description column", cols, index=cols.index(default_desc) if default_desc in cols else 0)
    map_cp = c4.selectbox("Counterparty / Merchant", cols, index=cols.index(default_cp) if default_cp in cols else 0)

    norm = raw_df.rename(columns={
        map_date: "date",
        map_amount: "amount",
        map_desc: "description",
        map_cp: "counterparty"
    })[["date", "amount", "description", "counterparty"]]

    df = load_df_from_csv_like(BytesIO(norm.to_csv(index=False).encode("utf-8")))

    st.subheader("Normalized transactions")
    st.dataframe(df, width="stretch", hide_index=True)

    # Detect obligations (keywords)
    RENT_KW = ["rent", "tenancy", "landlord", "letting"]
    COUNCIL_KW = [
        "council tax", "council", "borough council", "city council",
        "hackney", "islington", "camden", "westminster", "newham", "tower hamlets",
        "waltham forest", "southwark", "lambeth", "haringey", "enfield", "redbridge",
        "lewisham", "greenwich", "barnet", "brent", "harrow", "hammersmith", "fulham",
        "kensingt", "chelsea", "croydon", "bromley", "ealing"
    ]
    UTILITY_KW = [
        "octopus", "edf", "e.on", "british gas", "thames water", "severn trent",
        "ovo", "shell energy", "bulb", "npower", "scottish power", "southern water",
        "yorkshire water", "anglian water", "united utilities", "ee", "vodafone", "o2", "three"
    ]

    # Enforce unique counterparties via exclusions; Council requires keyword
    rent = detect_obligation(df, "Rent", RENT_KW, exclude_cps=None, require_keyword=False)
    used = set([rent["counterparty"]]) if rent else set()

    council = detect_obligation(df, "Council tax", COUNCIL_KW, exclude_cps=used, require_keyword=True)
    if council:
        used.add(council["counterparty"])

    utilities = detect_obligation(df, "Utility", UTILITY_KW, exclude_cps=used, require_keyword=False)

    detected = [ob for ob in [rent, council, utilities] if ob is not None]

    st.markdown("---")
    if not detected:
        st.warning("No clear recurring obligations detected yet. Try adjusting tolerance/window in the sidebar, "
                   "or include the keyword **RENT** in one of the descriptions.")
        st.stop()

    # Summary metrics
    cols_top = st.columns(len(detected))
    for col, ob in zip(cols_top, detected):
        col.metric(f"{ob['type']} — {ob['counterparty']}", f"£{ob['baseline_amount']:.2f}", f"Due day ~ {ob['due_day']}")

    # Detailed tables
    for ob in detected:
        st.subheader(f"📅 {ob['type']} payment history")
        show = ob["payments"].copy()
        show["amount (£)"] = show["amount"].abs().round(2)
        show = show[["date", "amount (£)", "status", "description"]]
        st.dataframe(show, width="stretch", hide_index=True)

    # Score + insights + PDF
    score, explain, on_time_ratio = compute_credwyn_score(detected)
    st.markdown("### ⭐ CredWyn Score")
    st.metric("Score (300–850)", score)

    st.markdown("#### Insights")
    if on_time_ratio == 1:
        st.write("• Perfect on-time history across detected obligations — keep it up!")
    elif on_time_ratio >= 0.9:
        st.write("• Strong payment reliability. One or two delays spotted — consider a standing order on the due date.")
    else:
        st.write("• Payment reliability is inconsistent. Automating payments could improve your score quickly.")
    for e in explain:
        st.write("• " + e)

    if st.button("Generate proof PDF"):
        pdf_bytes = build_pdf(detected, score, explain, USER_NAME, USER_ADDRESS)
        st.download_button(
            label="Download proof (PDF)",
            data=pdf_bytes,
            file_name=f"credwyn_proof_{datetime.now().date()}.pdf",
            mime="application/pdf"
        )

    # --- Landlord Pack ZIP (PDF + CSVs) ---
    st.markdown("#### 📦 Landlord Pack")
    monthly_here = monthly_summary(df)
    if st.button("Build Landlord Pack (ZIP)"):
        zip_buf = build_landlord_pack(
            obligations=detected,
            score=score,
            explain=explain,
            user_name=USER_NAME,
            user_address=USER_ADDRESS,
            df_normalized=df,
            monthly_df=monthly_here
        )
        st.download_button(
            label="Download Landlord Pack",
            data=zip_buf,
            file_name=f"credwyn_landlord_pack_{datetime.now().date()}.zip",
            mime="application/zip"
        )

    # --- Feedback form ---
    st.markdown("---")
    st.markdown("### 📨 Feedback")
    with st.form("feedback"):
        fb_email = st.text_input("Email (optional)")
        fb_text = st.text_area("What didn’t work or what’s missing?")
        submitted = st.form_submit_button("Send")
        if submitted:
            row = pd.DataFrame([{
                "ts": datetime.now().isoformat(timespec="seconds"),
                "email": fb_email,
                "feedback": fb_text
            }])
            try:
                existing = pd.read_csv("feedback.csv")
                allf = pd.concat([existing, row], ignore_index=True)
            except Exception:
                allf = row
            allf.to_csv("feedback.csv", index=False)
            st.success("Thanks! Your feedback was recorded locally.")

with tab_dashboard:
    st.subheader("Monthly summary")
    monthly = monthly_summary(df)
    st.dataframe(monthly, width="stretch", hide_index=True)

    # Charts
    st.markdown("#### Inflow / Outflow")
    chart_io = alt.Chart(monthly).mark_bar().encode(
        x=alt.X('period:T', title='Month'),
        y=alt.Y('inflow:Q', title='Inflow')
    )
    chart_oo = alt.Chart(monthly).mark_bar().encode(
        x=alt.X('period:T', title='Month'),
        y=alt.Y('outflow:Q', title='Outflow')
    )
    st.altair_chart((chart_io | chart_oo).resolve_scale(y='independent'), use_container_width=True)

    st.markdown("#### Net flow")
    chart_net = alt.Chart(monthly).mark_line(point=True).encode(
        x=alt.X('period:T', title='Month'),
        y=alt.Y('net:Q', title='Net flow')
    )
    st.altair_chart(chart_net, use_container_width=True)

    st.markdown("#### On-time trend")
    trend = ontime_trend(detected)
    if not trend.empty:
        chart_ot = alt.Chart(trend).mark_line(point=True).encode(
            x=alt.X('period:T', title='Month'),
            y=alt.Y('on_time_ratio:Q', title='On-time ratio', scale=alt.Scale(domain=[0,1]))
        )
        st.altair_chart(chart_ot, use_container_width=True)
    else:
        st.info("On-time trend will appear once obligations have multiple months of payments.")

    # Exports
    st.markdown("#### Export")
    st.download_button(
        "Download normalized transactions (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="credwyn_transactions_normalized.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download monthly summary (CSV)",
        data=monthly.to_csv(index=False).encode("utf-8"),
        file_name="credwyn_monthly_summary.csv",
        mime="text/csv"
    )

st.caption("Local demo only. No data leaves your machine.")

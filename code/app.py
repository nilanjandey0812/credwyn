import streamlit as st
import pandas as pd

st.set_page_config(page_title="CredWyn — Rent & Bills Detector", layout="wide")

# ---------- Helpers ----------
def load_df(file):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in ["amount", "description", "counterparty"]:
        if col not in df.columns:
            df[col] = ""
    return df

def detect_rent(df):
    out = df[df["amount"] < 0].copy()
    rent_kw = out["description"].str.lower().str.contains("rent|tenancy|landlord|letting", na=False)
    candidates = out[rent_kw].copy()

    if candidates.empty and not out.empty:
        best_cp, best_score = None, -1
        for cp, g in out.groupby("counterparty"):
            if len(g) < 3:
                continue
            g = g.sort_values("date")
            gaps = g["date"].diff().dt.days.dropna()
            if gaps.empty:
                continue
            median_gap = gaps.median()
            amt_med = g["amount"].abs().median()
            amt_mad = (g["amount"].abs() - amt_med).abs().median()
            score = (max(0, 1 - abs(median_gap - 30)/30)) + max(0, 1 - (amt_mad / max(1, amt_med)))
            if score > best_score:
                best_score, best_cp = score, cp
        if best_cp:
            candidates = out[out["counterparty"] == best_cp].copy()

    if candidates.empty:
        return None

    base_amt = candidates["amount"].abs().median()
    due_dom = int(candidates["date"].dt.day.mode()[0])

    def status_for(row):
        dom = row["date"].day
        if dom < due_dom - 3 or dom > due_dom + 5:
            return "Late"
        if abs(row["amount"]) < 0.9 * base_amt:
            return "Partial"
        return "On-time"

    payments = candidates[["date","amount","description"]].copy()
    payments["status"] = payments.apply(status_for, axis=1)

    return {
        "counterparty": str(candidates["counterparty"].iloc[0]),
        "baseline_amount": round(float(base_amt), 2),
        "due_day": due_dom,
        "payments": payments.sort_values("date").reset_index(drop=True)
    }

# ---------- UI ----------
st.title("💳 CredWyn — Build Credit from Everyday Life")
st.write("Upload bank transactions (CSV) to detect **rent** automatically and view a reliability summary.")

sample_path = "data/sample_transactions.csv"
st.sidebar.header("Demo")
use_sample = st.sidebar.button("Load sample data")

uploaded = st.file_uploader("Upload CSV with columns: date, amount, description, counterparty", type=["csv"])

if use_sample and uploaded is None:
    uploaded = sample_path

if uploaded is None:
    st.info("➡️ Upload a CSV to continue, or click **Load sample data** in the sidebar.")
    st.stop()

df = load_df(uploaded)
st.subheader("Raw transactions")
st.dataframe(df, use_container_width=True, hide_index=True)

det = detect_rent(df)
st.markdown("---")

if det is None:
    st.warning("Couldn’t confidently detect rent yet. Try adding more months or include the word **RENT** in the description.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Likely landlord / counterparty", det["counterparty"])
    col2.metric("Baseline rent (median)", f"£{det['baseline_amount']:.2f}")
    col3.metric("Typical due day", f"{det['due_day']}")

    st.subheader("📅 Rent payment history")
    show = det["payments"].copy()
    show["amount (£)"] = show["amount"].abs().round(2)
    show = show[["date","amount (£)","status","description"]]
    st.dataframe(show, use_container_width=True, hide_index=True)

    on_time_ratio = (show["status"] == "On-time").mean()
    st.success(f"On-time ratio: **{on_time_ratio:.0%}**")

st.caption("Local demo only. No data leaves your machine.")

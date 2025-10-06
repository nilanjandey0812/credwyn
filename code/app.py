# app.py — CredWyn (Streamlit MVP, all-fixed + viewer)
# Features:
# - CSV column mapping (any bank export)
# - Detect Rent / Council Tax / Utilities (monthly pattern + keywords)
# - On-time / Partial / Late labelling with tunable tolerance/window
# - CredWyn Score (300–850) with explainers
# - Proof PDF + Landlord Pack (ZIP)
# - Cloud save to Supabase (users, score snapshots, feedback)
# - Dashboard: monthly inflow/outflow/net + on-time trend + CSV exports
# - Recent snapshots viewer (pulls last saves for your email)

import os
import json
import zipfile
from io import BytesIO
from datetime import datetime

import pandas as pd
import streamlit as st
import altair as alt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# ---------------- Supabase (secrets.toml OR .env) ----------------
SUPABASE_AVAILABLE = True
try:
    from supabase import create_client
except Exception:
    SUPABASE_AVAILABLE = False

# robust .env support (searches up the tree)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

def init_supabase():
    """Prefer Streamlit secrets (.streamlit/secrets.toml). Fallback to .env.
       Return supabase client or None (no crash)."""
    if not SUPABASE_AVAILABLE:
        return None
    url = None
    key = None
    # Try secrets (guarded)
    try:
        _ = st.secrets
        url = st.secrets.get("SUPABASE_URL", None)
        key = st.secrets.get("SUPABASE_KEY", None)
    except Exception:
        pass
    # Fallback to environment
    url = url or os.getenv("SUPABASE_URL", "")
    key = key or os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None

supabase = init_supabase()

# ---------------- Streamlit config ----------------
st.set_page_config(page_title="CredWyn — Rent & Bills Detector", layout="wide")
st.markdown("<style>.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)

# ---------------- Sidebar (user + settings) ----------------
st.sidebar.header("CredWyn")
st.sidebar.caption("Build Trust. Build Credit. Build Your Future.")
st.sidebar.markdown(f"**Supabase:** {'🟢 Connected' if supabase is not None else '🔴 Not configured'}")

st.sidebar.subheader("Your details")
USER_NAME = st.sidebar.text_input("Full name", value="")
USER_EMAIL = st.sidebar.text_input("Email (for cloud save / feedback)", value="")
USER_ADDRESS = st.sidebar.text_area("Address (for PDF, optional)", value="", height=70)

st.sidebar.subheader("Detection settings")
AMOUNT_TOLERANCE = st.sidebar.slider("Amount tolerance (±%)", 5, 25, 10)
DUE_WINDOW = st.sidebar.slider("On-time window (± days around due day)", 2, 7, 3)

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

# ---------------- Helpers ----------------
def sanitize_email(s: str) -> str:
    """Strip normal + invisible whitespace (NBSP, zero-width spaces, thin spaces)."""
    if s is None:
        return ""
    bad = ["\u00a0", "\u200b", "\u2009", "\u200a", "\u2008", "\u2007", "\u202f"]
    out = s.strip()
    for ch in bad:
        out = out.replace(ch, "")
    return out

def load_normalized_df(file_or_buf) -> pd.DataFrame:
    """Ensure required columns exist: date, amount, description, counterparty."""
    df = pd.read_csv(file_or_buf)
    df.columns = [c.strip() for c in df.columns]
    if "date" not in df.columns:
        st.error("CSV must include a 'date' column after mapping."); st.stop()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["amount", "description", "counterparty"]:
        if col not in df.columns:
            df[col] = 0.0 if col == "amount" else ""
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df[["date", "amount", "description", "counterparty"]]

def monthly_candidates(out_df: pd.DataFrame, keyword_list=None,
                       prefer_keywords=True, exclude_cps: set | None = None,
                       require_keyword: bool = False) -> pd.DataFrame:
    """Pick candidate series: prefer keyword hits; else best monthly (~30d + low variance)."""
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

    # fallback: best cp by periodicity + low amount MAD-like
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
        mad_like = (g["amount"].abs() - amt_med).abs().mean()  # mean abs deviation
        periodicity = max(0.0, 1.0 - abs(median_gap - 30) / 30)
        variance_term = max(0.0, 1.0 - (mad_like / max(1.0, amt_med)))
        score = periodicity + variance_term
        if score > best_score:
            best_score, best_cp = score, cp

    if best_cp:
        return out[out["counterparty"] == best_cp].copy()
    return pd.DataFrame(columns=out.columns)

def status_table(cand: pd.DataFrame, due_dom: int, base_amt: float,
                 amount_tol_pct: float, due_window_days: int) -> pd.DataFrame:
    """Label each payment On-time / Partial / Late."""
    def status_for(row):
        dom = row["date"].day
        if dom < due_dom - due_window_days or dom > due_dom + (due_window_days + 2):
            return "Late"
        if abs(row["amount"]) < (1 - amount_tol_pct / 100.0) * base_amt:
            return "Partial"
        return "On-time"
    payments = cand[["date", "amount", "description", "counterparty"]].copy()
    payments["status"] = payments.apply(status_for, axis=1)
    return payments.sort_values("date").reset_index(drop=True)

def detect_obligation(df: pd.DataFrame, label: str, keywords,
                      exclude_cps: set | None = None, require_keyword: bool = False) -> dict | None:
    """Detect recurring outgoing obligation (rent/council/utility)."""
    out = df[df["amount"] < 0].copy()
    if out.empty:
        return None
    cand = monthly_candidates(out, keyword_list=keywords, prefer_keywords=True,
                              exclude_cps=exclude_cps, require_keyword=require_keyword)
    if cand.empty:
        return None
    base_amt = float(cand["amount"].abs().median())
    due_dom = int(cand["date"].dt.day.mode()[0])
    payments = status_table(cand, due_dom, base_amt, AMOUNT_TOLERANCE, DUE_WINDOW)
    return {
        "type": label,
        "counterparty": str(cand["counterparty"].iloc[0]),
        "baseline_amount": round(base_amt, 2),
        "due_day": due_dom,
        "payments": payments
    }

def compute_credwyn_score(obligations: list[dict]):
    """Explainable 300–850 score: 70% on-time, 20% stability, 10% history."""
    total_payments, on_time_sum, months_terms = 0, 0, 0
    stability_terms = []
    for ob in obligations:
        pay = ob["payments"]
        if pay.empty: continue
        total_payments += len(pay)
        on_time_sum += pay["status"].eq("On-time").sum()
        amounts = pay["amount"].abs()
        med = float(amounts.median()) if len(amounts) else 0.0
        if med > 0:
            mad_like = float((amounts - med).abs().mean())
            stability = 1.0 - (mad_like / max(1.0, med))
            stability_terms.append(max(0.0, min(1.0, stability)))
        months_terms += min(12, len(pay))
    if total_payments == 0:
        return 300, ["No obligations detected yet."], 0.0
    on_time_ratio = on_time_sum / total_payments
    stability_avg = sum(stability_terms) / len(stability_terms) if stability_terms else 0.0
    history_factor = min(1.0, months_terms / (12.0 * max(1, len(obligations))))
    raw = 0.7*on_time_ratio + 0.2*stability_avg + 0.1*history_factor
    score = int(300 + raw * (850 - 300))
    explain = [
        f"On-time across all obligations: {int(on_time_ratio*100)}%",
        f"Payment stability (amount variance): {int(stability_avg*100)}%",
        f"History length factor: {int(history_factor*100)}%",
    ]
    return score, explain, on_time_ratio

def build_pdf(obligations, score, explain, user_name="", user_address="") -> BytesIO:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = H - 20*mm
    def write(line, size=12, bold=False, extra_gap=0):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(20*mm, y, line); y -= (7+extra_gap)*mm
    write("CredWyn — Proof of Payments", size=16, bold=True)
    write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if user_name: write(f"Name: {user_name}")
    if user_address:
        for i, line in enumerate(user_address.splitlines()):
            write(("Address: " if i==0 else "         ")+line)
    write(f"CredWyn Score (300–850): {score}", bold=True, extra_gap=1)
    write("Why this score:", bold=True)
    for e in explain: write("• "+e)
    for ob in obligations:
        y -= 4*mm
        write(f"{ob['type'].upper()} — {ob['counterparty']}", bold=True)
        write(f"Baseline amount: £{ob['baseline_amount']:.2f} | Typical due day: {ob['due_day']}")
        write("Month        Date         Amount     Status", bold=True)
        for _, r in ob["payments"].iterrows():
            line = f"{r['date'].strftime('%b %Y'): <12} {r['date'].strftime('%Y-%m-%d'): <12} £{abs(r['amount']):<9.2f} {r['status']}"
            write(line)
    c.showPage(); c.save(); buf.seek(0)
    return buf

def build_landlord_pack(obligations, score, explain, user_name, user_address, df_normalized, monthly_df):
    pdf_bytes = build_pdf(obligations, score, explain, user_name, user_address)
    tx_csv = df_normalized.to_csv(index=False).encode("utf-8")
    monthly_csv = monthly_df.to_csv(index=False).encode("utf-8")
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"credwyn_proof_{datetime.now().date()}.pdf", pdf_bytes.getvalue())
        z.writestr("transactions_normalized.csv", tx_csv)
        z.writestr("monthly_summary.csv", monthly_csv)
        z.writestr("README.txt",
                   "CredWyn Landlord Pack\n"
                   "Includes: proof PDF, normalized transactions, monthly summary.\n"
                   f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    buf.seek(0); return buf

def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    m = df.copy()
    m["period"] = m["date"].dt.to_period("M").dt.to_timestamp()
    grp = m.groupby("period", as_index=False)
    inflow  = grp.apply(lambda g: g.loc[g["amount"]>0, "amount"].sum()).rename(columns={None:"inflow"})
    outflow = grp.apply(lambda g: g.loc[g["amount"]<0, "amount"].abs().sum()).rename(columns={None:"outflow"})
    out = pd.merge(inflow, outflow, on="period", how="outer").fillna(0.0)
    out["net"] = out["inflow"] - out["outflow"]
    return out.sort_values("period")

def ontime_trend(obligations: list[dict]) -> pd.DataFrame:
    rows = []
    for ob in obligations:
        p = ob["payments"].copy()
        if p.empty: continue
        p["period"] = p["date"].dt.to_period("M").dt.to_timestamp()
        p["on_time"] = (p["status"] == "On-time").astype(int)
        rows.append(p[["period","on_time"]])
    if not rows: return pd.DataFrame(columns=["period","on_time_ratio"])
    allp = pd.concat(rows)
    return (allp.groupby("period")["on_time"].mean()
            .reset_index().rename(columns={"on_time":"on_time_ratio"})
            .sort_values("period"))

# ---------------- Supabase helpers ----------------
def supabase_enabled() -> bool:
    return supabase is not None

def upsert_user(email: str, name: str):
    email = sanitize_email(email)
    if not supabase_enabled() or not email:
        return False, "Supabase not configured or email missing."
    try:
        try:
            supabase.table("users").upsert(
                {"email": email, "name": (name or '').strip()}, on_conflict="email"
            ).execute()
            return True, "User saved (upsert)."
        except Exception:
            existing = supabase.table("users").select("id").eq("email", email).limit(1).execute()
            if existing.data:
                supabase.table("users").update({"name": (name or '').strip()}).eq("email", email).execute()
                return True, "User updated."
            else:
                supabase.table("users").insert({"email": email, "name": (name or '').strip()}).execute()
                return True, "User created."
    except Exception as e:
        return False, f"{e}"

def insert_score_snapshot(email: str, score: int, on_time_ratio: float, payload: dict | None = None):
    email = sanitize_email(email)
    if not supabase_enabled() or not email:
        return False, "Supabase not configured or email missing."
    try:
        row = {
            "user_email": email,
            "score": int(score),
            "on_time_ratio": float(on_time_ratio),
            "meta": (payload or {}),   # JSONB (dict) — not string
        }
        res = supabase.table("score_snapshots").insert(row).execute()
        inserted = res.data[0] if res and res.data else {}
        return True, f"Score snapshot saved. (id: {inserted.get('id','?')})"
    except Exception as e:
        return False, f"{e}"

def insert_feedback(email: str, text: str):
    if not supabase_enabled():
        return False, "Supabase not configured."
    try:
        supabase.table("feedback").insert({"email": sanitize_email(email), "text": text}).execute()
        return True, "Thanks! Your feedback was recorded."
    except Exception as e:
        return False, f"{e}"

# ---------------- UI ----------------
st.title("💳 CredWyn — Build Credit from Everyday Life")
st.write(
    "Upload your bank transactions (CSV), map your columns, and CredWyn will detect **Rent, Council Tax, and Utilities**. "
    "You’ll get an explainable **CredWyn Score** and a downloadable **Proof PDF**. "
    "Use the **Dashboard** tab for monthly summaries and on-time trends."
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
    st.info("➡️ Choose a sample in the sidebar **or** upload your CSV."); st.stop()

tab_analyze, tab_dashboard = st.tabs(["🔎 Analyze", "📊 Dashboard"])

with tab_analyze:
    raw_df = pd.read_csv(uploaded)
    st.caption("Raw CSV preview (first 20 rows):")
    st.dataframe(raw_df.head(20), width="stretch", hide_index=True)

    # Column mapping wizard
    st.subheader("Map your columns")
    cols = list(raw_df.columns)

    def guess(cands):
        for col in cols:
            low = col.lower()
            if any(k in low for k in cands):
                return col
        return cols[0] if cols else None

    default_date = guess(["date","posted","transaction date","txn date"])
    default_amount = guess(["amount","value","debit","credit","amt"])
    default_desc = guess(["description","narrative","details","reference","memo"])
    default_cp = guess(["counterparty","merchant","payee","beneficiary","name"])

    c1,c2,c3,c4 = st.columns(4)
    map_date = c1.selectbox("Date column", cols, index=cols.index(default_date) if default_date in cols else 0)
    map_amount = c2.selectbox("Amount column (negative=out)", cols, index=cols.index(default_amount) if default_amount in cols else 0)
    map_desc = c3.selectbox("Description column", cols, index=cols.index(default_desc) if default_desc in cols else 0)
    map_cp = c4.selectbox("Counterparty / Merchant", cols, index=cols.index(default_cp) if default_cp in cols else 0)

    norm = raw_df.rename(columns={
        map_date: "date", map_amount: "amount", map_desc: "description", map_cp: "counterparty"
    })[["date","amount","description","counterparty"]]

    df = load_normalized_df(BytesIO(norm.to_csv(index=False).encode("utf-8")))

    st.subheader("Normalized transactions")
    st.dataframe(df, width="stretch", hide_index=True)

    # Detection keywords
    RENT_KW = ["rent","tenancy","landlord","letting"]
    COUNCIL_KW = [
        "council tax","council","borough council","city council",
        "hackney","islington","camden","westminster","newham","tower hamlets",
        "waltham forest","southwark","lambeth","haringey","enfield","redbridge",
        "lewisham","greenwich","barnet","brent","harrow","hammersmith","fulham",
        "kensingt","chelsea","croydon","bromley","ealing"
    ]
    UTILITY_KW = [
        "octopus","edf","e.on","british gas","thames water","severn trent",
        "ovo","shell energy","bulb","npower","scottish power","southern water",
        "yorkshire water","anglian water","united utilities","ee","vodafone","o2","three"
    ]

    rent = detect_obligation(df, "Rent", RENT_KW)
    used = set([rent["counterparty"]]) if rent else set()
    council = detect_obligation(df, "Council tax", COUNCIL_KW, exclude_cps=used, require_keyword=True)
    if council: used.add(council["counterparty"])
    utilities = detect_obligation(df, "Utility", UTILITY_KW, exclude_cps=used)

    detected = [ob for ob in [rent, council, utilities] if ob]

    st.markdown("---")
    if not detected:
        st.warning("No clear recurring obligations detected. Try adjusting tolerance/window in the sidebar, "
                   "or include the keyword **RENT** in a description."); st.stop()

    # Summary metrics
    cols_top = st.columns(len(detected))
    for col, ob in zip(cols_top, detected):
        col.metric(f"{ob['type']} — {ob['counterparty']}",
                   f"£{ob['baseline_amount']:.2f}", f"Due day ~ {ob['due_day']}")

    # Detailed tables
    for ob in detected:
        st.subheader(f"📅 {ob['type']} payment history")
        show = ob["payments"].copy()
        show["amount (£)"] = show["amount"].abs().round(2)
        st.dataframe(show[["date","amount (£)","status","description"]],
                     width="stretch", hide_index=True)

    # Score + insights
    score, explain, on_time_ratio = compute_credwyn_score(detected)
    st.markdown("### ⭐ CredWyn Score")
    st.metric("Score (300–850)", score)
    st.markdown("#### Insights")
    st.write("• Perfect on-time history — keep it up!" if on_time_ratio==1
             else "• Strong reliability. Consider standing orders." if on_time_ratio>=0.9
             else "• Inconsistent reliability — automate payments to improve.")
    for e in explain: st.write("• "+e)

    # Proof PDF
    if st.button("Generate proof PDF"):
        pdf_bytes = build_pdf(detected, score, explain, USER_NAME, USER_ADDRESS)
        st.download_button("Download proof (PDF)", pdf_bytes,
                           file_name=f"credwyn_proof_{datetime.now().date()}.pdf",
                           mime="application/pdf")

    # Landlord Pack
    st.markdown("#### 📦 Landlord Pack")
    monthly_here = monthly_summary(df)
    if st.button("Build Landlord Pack (ZIP)"):
        zip_buf = build_landlord_pack(detected, score, explain, USER_NAME, USER_ADDRESS, df, monthly_here)
        st.download_button("Download Landlord Pack", zip_buf,
                           file_name=f"credwyn_landlord_pack_{datetime.now().date()}.zip",
                           mime="application/zip")

    # Cloud save (uses sanitized email)
    st.markdown("---")
    st.markdown("### ☁️ Save to Cloud (Supabase)")
    if supabase_enabled():
        _raw = USER_EMAIL
        _clean = sanitize_email(_raw)
        st.caption(f"Email entered → raw: {repr(_raw)} | clean: {repr(_clean)}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save / Update User"):
                email = sanitize_email(USER_EMAIL)
                name = (USER_NAME or "").strip()
                if not email:
                    st.error("Email is empty after cleaning. Please type it manually (don’t paste).")
                else:
                    ok, msg = upsert_user(email, name)
                    (st.success if ok else st.error)(msg)
        with c2:
            payload = {"detected_types": [ob["type"] for ob in detected]}
            if st.button("Save Score Snapshot"):
                email = sanitize_email(USER_EMAIL)
                if not email:
                    st.error("Email is empty after cleaning. Please type it in the sidebar.")
                else:
                    ok, msg = insert_score_snapshot(email, score, on_time_ratio, payload)
                    (st.success if ok else st.error)(msg)

        # ---- Recent snapshots viewer -------------------------------------------------
        email_view = sanitize_email(USER_EMAIL)
        if email_view:
            try:
                recent = (supabase.table("score_snapshots")
                          .select("created_at, score, on_time_ratio, meta")
                          .eq("user_email", email_view)
                          .order("created_at", desc=True)
                          .limit(5)
                          .execute())
                rows = recent.data or []
                if rows:
                    st.markdown("#### Recent saved score snapshots")
                    df_recent = pd.DataFrame(rows)
                    st.dataframe(df_recent, use_container_width=True, hide_index=True)
                else:
                    st.caption("No snapshots yet for this email. Click **Save Score Snapshot** above.")
            except Exception as e:
                st.caption(f"Could not read snapshots: {e}")
    else:
        st.info("Supabase not configured. Put SUPABASE_URL and SUPABASE_KEY in .streamlit/secrets.toml or in a .env at project root.")

    # Feedback
    st.markdown("---")
    st.markdown("### 📨 Feedback")
    with st.form("feedback"):
        fb_text = st.text_area("What didn’t work or what’s missing?")
        submitted = st.form_submit_button("Send")
        if submitted:
            if not fb_text.strip():
                st.warning("Please write a short message before sending.")
            else:
                if supabase_enabled():
                    ok, msg = insert_feedback(sanitize_email(USER_EMAIL), fb_text.strip())
                    (st.success if ok else st.error)(msg)
                else:
                    # local fallback
                    row = pd.DataFrame([{
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "email": USER_EMAIL, "feedback": fb_text
                    }])
                    try:
                        existing = pd.read_csv("feedback.csv")
                        allf = pd.concat([existing, row], ignore_index=True)
                    except Exception:
                        allf = row
                    allf.to_csv("feedback.csv", index=False)
                    st.success("Thanks! Feedback saved locally (feedback.csv).")

with tab_dashboard:
    st.subheader("Monthly summary")
    monthly = monthly_summary(df)
    st.dataframe(monthly, width="stretch", hide_index=True)

    st.markdown("#### Inflow / Outflow")
    chart_io = alt.Chart(monthly).mark_bar().encode(
        x=alt.X('period:T', title='Month'), y=alt.Y('inflow:Q', title='Inflow'))
    chart_oo = alt.Chart(monthly).mark_bar().encode(
        x=alt.X('period:T', title='Month'), y=alt.Y('outflow:Q', title='Outflow'))
    st.altair_chart((chart_io | chart_oo).resolve_scale(y='independent'), use_container_width=True)

    st.markdown("#### Net flow")
    chart_net = alt.Chart(monthly).mark_line(point=True).encode(
        x=alt.X('period:T', title='Month'), y=alt.Y('net:Q', title='Net flow'))
    st.altair_chart(chart_net, use_container_width=True)

    st.markdown("#### On-time trend")
    # re-detect to build trend
    RENT_KW = ["rent","tenancy","landlord","letting"]
    COUNCIL_KW = ["council tax","council","borough council","city council","hackney","islington","camden",
                  "westminster","newham","tower hamlets","waltham forest","southwark","lambeth","haringey",
                  "enfield","redbridge","lewisham","greenwich","barnet","brent","harrow","hammersmith",
                  "fulham","kensingt","chelsea","croydon","bromley","ealing"]
    UTILITY_KW = ["octopus","edf","e.on","british gas","thames water","severn trent","ovo","shell energy","bulb",
                  "npower","scottish power","southern water","yorkshire water","anglian water","united utilities",
                  "ee","vodafone","o2","three"]
    r = detect_obligation(df, "Rent", RENT_KW)
    used = set([r["counterparty"]]) if r else set()
    c = detect_obligation(df, "Council tax", COUNCIL_KW, exclude_cps=used, require_keyword=True)
    if c: used.add(c["counterparty"])
    u = detect_obligation(df, "Utility", UTILITY_KW, exclude_cps=used)
    trend = ontime_trend([x for x in [r,c,u] if x])

    if not trend.empty:
        chart_ot = alt.Chart(trend).mark_line(point=True).encode(
            x=alt.X('period:T', title='Month'),
            y=alt.Y('on_time_ratio:Q', title='On-time ratio', scale=alt.Scale(domain=[0,1])))
        st.altair_chart(chart_ot, use_container_width=True)
    else:
        st.info("On-time trend will appear once obligations have multiple months of payments.")

    st.markdown("#### Export")
    st.download_button("Download normalized transactions (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       file_name="credwyn_transactions_normalized.csv", mime="text/csv")
    st.download_button("Download monthly summary (CSV)",
                       monthly.to_csv(index=False).encode("utf-8"),
                       file_name="credwyn_monthly_summary.csv", mime="text/csv")

st.caption("Local demo. If Supabase is configured, user & score data save to your project.")

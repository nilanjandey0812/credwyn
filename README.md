# CredWyn (Local Demo)

Upload a bank CSV, map columns, and CredWyn detects *Rent, Council Tax, Utilities*, computes an **explainable Credit Health Score (300–850)**, and generates a **Proof of Payments** PDF.

**Local only**: no data leaves your device in this demo.

## Run locally
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd code
streamlit run app.py

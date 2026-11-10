"""Analyst dashboard (Streamlit):  streamlit run dashboard/app.py"""
import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Fraud Analyst Console", layout="wide")


def get(path, **kw):
    try:
        return requests.get(f"{API}{path}", timeout=5, **kw).json()
    except Exception as e:
        st.error(f"API unreachable at {API}: {e}")
        return None


st.title("Real-Time Fraud Analyst Console")
auto = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
status = st.sidebar.selectbox("Case status", ["open", "closed", "all"])
analyst = st.sidebar.text_input("Analyst name", "analyst")

s = get("/stats")
if s:
    c = st.columns(6)
    c[0].metric("Scored", f"{s['scored']:,}")
    c[1].metric("Alerts", s["alerts"])
    c[2].metric("Open cases", s["open_cases"])
    c[3].metric("Confirmed fraud", s["confirmed_fraud"])
    c[4].metric("p95 scoring (ms)", f"{s['latency_ms']['p95']:.1f}" if s["latency_ms"] else "-")
    c[5].metric("p95 end-to-end (ms)", f"{s['e2e_ms']['p95']:.0f}" if s["e2e_ms"] else "-")
    if s.get("sim_alert_precision") is not None:
        st.caption(f"Simulation ground truth: alert precision {s['sim_alert_precision']:.1%}, recall {s['sim_recall']:.1%}")

left, right = st.columns([3, 2])
with left:
    st.subheader("Alert queue (highest risk first)")
    rows = get("/alerts", params={} if status == "all" else {"status": status})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df[["case_id", "txn_id", "user_id", "amount", "score", "xgb_score", "gnn_score", "status", "label"]],
                     use_container_width=True, hide_index=True)
        ids = df.loc[df.status == "open", "case_id"].tolist()
        if ids:
            cid = st.selectbox("Review case", ids)
            note = st.text_input("Note")
            b1, b2 = st.columns(2)
            for col, lab in ((b1, "fraud"), (b2, "legit")):
                if col.button(f"Mark {lab}", key=lab):
                    r = requests.post(f"{API}/alerts/{cid}/feedback",
                                      json={"label": lab, "analyst": analyst, "note": note}, timeout=5)
                    st.success(r.json()) if r.ok else st.error(r.text)
                    time.sleep(0.3)
                    st.rerun()
    else:
        st.info("No cases yet - start the Kafka producer.")
with right:
    st.subheader("Data drift")
    d = get("/drift")
    if d and "share_used" in d:
        st.metric("Share of drifted features", f"{d['share_used']:.0%}", delta="RETRAIN TRIGGERED" if d["retrain_triggered"] else None,
                  delta_color="inverse")
        st.write("Drifted (PSI > 0.2):", ", ".join(d["psi_drifted_features"]) or "none")
        st.bar_chart(pd.Series(d["psi"], name="PSI"))
    else:
        st.info("No drift report yet (run `make drift`).")
if auto:
    time.sleep(5)
    st.rerun()

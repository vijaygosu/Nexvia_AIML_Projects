import React, { useCallback, useEffect, useState } from "react";
import { api, heatmapUrl, imageUrl } from "./api.js";

const DISCLAIMER = "FOR EDUCATION ONLY - NOT FOR CLINICAL USE. Research prototype; not clinically validated.";

function StudyList({ studies, selected, onSelect }) {
  if (!studies.length) return <p className="muted">No studies yet. Upload to Orthanc and wait a few seconds.</p>;
  return (
    <ul className="studies">
      {studies.map((s) => (
        <li key={s.id} className={s.id === selected ? "active" : ""} onClick={() => onSelect(s.id)}>
          <div><strong>{s.patient_ref || "unknown"}</strong> <span className={`badge ${s.status}`}>{s.status}</span></div>
          <div className="muted">{s.top_label} {(s.top_prob * 100).toFixed(0)}%</div>
          <div className="muted small">{new Date(s.received_at).toLocaleString()}</div>
        </li>
      ))}
    </ul>
  );
}

function Predictions({ predictions, activeLabel, onPick }) {
  const rows = Object.entries(predictions).sort((a, b) => b[1] - a[1]);
  return (
    <div className="preds">
      {rows.map(([label, p]) => (
        <div key={label} className={`pred ${label === activeLabel ? "active" : ""}`} onClick={() => onPick(label)}>
          <span>{label}</span>
          <div className="bar"><div style={{ width: `${p * 100}%` }} /></div>
          <span className="pct">{(p * 100).toFixed(1)}%</span>
        </div>
      ))}
      <p className="muted small">Raw model scores (not calibrated probabilities). Click a row to see its Grad-CAM.</p>
    </div>
  );
}

function Review({ id, user, onDecided }) {
  const [study, setStudy] = useState(null);
  const [label, setLabel] = useState(null);
  const [showMap, setShowMap] = useState(true);
  const [opacity, setOpacity] = useState(0.6);
  const [comment, setComment] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setStudy(null); setErr(""); setComment("");
    api(`/studies/${id}`, { user }).then((s) => { setStudy(s); setLabel(s.top_label); }).catch((e) => setErr(e.message));
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  const decide = async (decision) => {
    if (!user.trim()) { setErr("Enter your reviewer name (top right) before deciding."); return; }
    setBusy(true); setErr("");
    try {
      await api(`/studies/${id}/decision`, { user, method: "POST", body: { decision, comment } });
      setStudy(await api(`/studies/${id}`, { user }));
      onDecided();
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };

  if (err && !study) return <p className="error">{err}</p>;
  if (!study) return <p className="muted">Loading…</p>;
  const pending = study.status === "pending";
  return (
    <div className="review">
      <div className="viewport">
        <img src={imageUrl(id)} alt="chest x-ray" />
        {showMap && label && <img className="overlay" style={{ opacity }} src={heatmapUrl(id, label === study.top_label ? "" : label)} alt={`Grad-CAM for ${label}`} />}
        <div className="watermark">NOT FOR CLINICAL USE</div>
      </div>
      <div className="side">
        <h3>AI findings <span className="muted small">model {study.model_version}</span></h3>
        {study.model_version.startsWith("UNTRAINED") && (
          <p className="error">This is an UNTRAINED demo model: its outputs are meaningless.</p>
        )}
        <div className="confidence">Top finding: <strong>{study.top_label}</strong> — {(study.top_prob * 100).toFixed(1)}%</div>
        <Predictions predictions={study.predictions} activeLabel={label} onPick={setLabel} />
        <label className="row"><input type="checkbox" checked={showMap} onChange={(e) => setShowMap(e.target.checked)} /> Show Grad-CAM ({label})</label>
        <label className="row">Opacity <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(+e.target.value)} /></label>
        <hr />
        {pending ? (
          <>
            <textarea placeholder="Optional comment for the audit log" value={comment} onChange={(e) => setComment(e.target.value)} />
            <div className="buttons">
              <button className="approve" disabled={busy} onClick={() => decide("approve")}>Approve AI finding</button>
              <button className="reject" disabled={busy} onClick={() => decide("reject")}>Reject</button>
            </div>
          </>
        ) : (
          <p>Decision: <span className={`badge ${study.status}`}>{study.status}</span> by <strong>{study.decided_by}</strong><br /><span className="muted small">{study.comment}</span></p>
        )}
        {err && <p className="error">{err}</p>}
      </div>
    </div>
  );
}

function AuditTab() {
  const [rows, setRows] = useState([]);
  const [verify, setVerify] = useState(null);
  const load = () => api("/audit?limit=200").then((r) => setRows(r.data));
  useEffect(() => { load(); }, []);
  return (
    <div>
      <button onClick={() => api("/audit/verify").then(setVerify)}>Verify hash chain</button>{" "}
      <button onClick={load}>Refresh</button>
      {verify && <span className={verify.ok ? "ok" : "error"}> {verify.ok ? `Chain intact (${verify.checked} entries)` : `TAMPERING DETECTED at entry ${verify.first_bad_seq}`}</span>}
      <table>
        <thead><tr><th>#</th><th>Time (UTC)</th><th>User</th><th>Action</th><th>Study</th><th>Hash</th></tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={r.seq}><td>{r.seq}</td><td>{r.ts.slice(0, 19)}</td><td>{r.user}</td><td>{r.action}</td><td>{r.study_id}</td><td className="mono">{r.hash.slice(0, 10)}…</td></tr>
        ))}</tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [user, setUser] = useState("");
  const [studies, setStudies] = useState([]);
  const [selected, setSelected] = useState(null);
  const [tab, setTab] = useState("review");
  const [err, setErr] = useState("");

  const refresh = useCallback(() => api("/studies").then((r) => { setStudies(r.data); setErr(""); }).catch((e) => setErr(e.message)), []);
  useEffect(() => { refresh(); const t = setInterval(refresh, 5000); return () => clearInterval(t); }, [refresh]);

  return (
    <div className="app">
      <div className="banner">{DISCLAIMER}</div>
      <header>
        <h1>AI-Assisted Imaging Review</h1>
        <nav>
          <button className={tab === "review" ? "on" : ""} onClick={() => setTab("review")}>Review</button>
          <button className={tab === "audit" ? "on" : ""} onClick={() => setTab("audit")}>Audit log</button>
        </nav>
        <input className="user" placeholder="Reviewer name (required to decide)" value={user} onChange={(e) => setUser(e.target.value)} />
      </header>
      {err && <p className="error pad">Cannot reach service: {err}</p>}
      {tab === "audit" ? <div className="pad"><AuditTab /></div> : (
        <main>
          <aside><StudyList studies={studies} selected={selected} onSelect={setSelected} /></aside>
          <section>{selected ? <Review id={selected} user={user} onDecided={refresh} /> : <p className="muted pad">Select a study.</p>}</section>
        </main>
      )}
      <footer>{DISCLAIMER}</footer>
    </div>
  );
}

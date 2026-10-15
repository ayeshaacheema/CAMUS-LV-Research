"""
ASCEND EchoLV - application layer
=================================
Presentation layer for the frozen LV-analysis pipeline (Phase 5C).

It does NOT change any model, threshold or measurement logic. It only
  * calls  frozen.load_nifti_2d / frozen.analyze_a4c_pair
  * renders Clinical View and ML Research View
  * stores every analysis as a patient case (SQLite + npz, on Google Drive)
  * builds a downloadable PDF report

Usage (Colab):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("echolv_app", APP_PATH)
    echolv_app = importlib.util.module_from_spec(spec); sys.modules["echolv_app"] = echolv_app
    spec.loader.exec_module(echolv_app)
    demo, launch_kw = echolv_app.build_app(frozen, RECORDS_DIR, demo_paths=(ED_PATH, ES_PATH))
    demo.launch(share=True, **launch_kw)
"""
from __future__ import annotations

import html
import inspect
import io
import json
import re
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import cm  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

APP_NAME = "ASCEND EchoLV"
BAND_ORDER = ("Lower_EF", "Intermediate_EF", "Higher_EF")
BAND_PRETTY = {"Lower_EF": "Lower EF", "Intermediate_EF": "Intermediate EF", "Higher_EF": "Higher EF"}
TEAL = (31, 214, 196)
DISCLAIMER = (
    "Research prototype. Outputs are assistive and non-diagnostic. Volumes are single-plane "
    "(A4C) geometric estimates and the classifier scores are uncalibrated. Independent expert "
    "review is required before any clinical use."
)


def esc(x) -> str:
    return html.escape(str(x))


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #
def ef_band_from_value(ef: float) -> str:
    """Geometry-derived EF band (Lower <=40, Intermediate 40-50, Higher >=50)."""
    if ef <= 40:
        return "Lower_EF"
    if ef < 50:
        return "Intermediate_EF"
    return "Higher_EF"


def _arr(x) -> np.ndarray:
    return np.asarray(x).squeeze().astype(np.float32)


def _gray01(img) -> np.ndarray:
    a = _arr(img)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def fig_to_np(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def to_png_bytes(arr: np.ndarray) -> io.BytesIO:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf


# --------------------------------------------------------------------------- #
# Image composition (Original / Contour / Mask / Probability + opacity)
# --------------------------------------------------------------------------- #
def compose_view(frame: dict, mode: str = "Contour", opacity: float = 70, size: int = 512) -> np.ndarray:
    a = float(np.clip(opacity, 0, 100)) / 100.0
    base = cv2.resize(_gray01(frame["image"]), (size, size), interpolation=cv2.INTER_CUBIC)
    rgb = np.clip(np.stack([base] * 3, -1), 0, 1) * 255.0
    mask_up = cv2.resize(_arr(frame["mask"]), (size, size), interpolation=cv2.INTER_LINEAR) > 0.5
    prob_up = np.clip(cv2.resize(_arr(frame["prob"]), (size, size), interpolation=cv2.INTER_LINEAR), 0, 1)
    colour = np.array(TEAL, dtype=np.float32)

    if mode == "Contour" and mask_up.any():
        cnts, _ = cv2.findContours(mask_up.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        line = np.zeros((size, size), np.uint8)
        cv2.drawContours(line, cnts, -1, 255, 2)
        lm = line > 0
        rgb[lm] = (1 - a) * rgb[lm] + a * colour
    elif mode == "Mask":
        rgb[mask_up] = (1 - a) * rgb[mask_up] + a * colour
    elif mode == "Probability":
        heat = cm.viridis(prob_up)[..., :3] * 255.0
        rgb = (1 - a) * rgb + a * heat
    return np.clip(rgb, 0, 255).astype(np.uint8)


def render_panel(frame: dict, kind: str, size: int = 384) -> np.ndarray:
    """Images for the ML explorer row."""
    if kind == "pre":
        return compose_view(frame, "Original", 100, size)
    if kind == "contour":
        return compose_view(frame, "Contour", 100, size)
    if kind == "mask":
        m = cv2.resize(_arr(frame["mask"]), (size, size), interpolation=cv2.INTER_NEAREST) > 0.5
        out = np.zeros((size, size, 3), np.uint8)
        out[m] = 255
        return out
    if kind == "prob":
        return compose_view(frame, "Probability", 100, size)
    raise ValueError(kind)


def hist_figure(prob, title: str) -> np.ndarray:
    p = _arr(prob).ravel()
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    bg, fg = "#102c45", "#cfe3f2"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.hist(p, bins=30, range=(0, 1), color="#1fc8b4", edgecolor=bg)
    ax.set_yscale("log")
    ax.set_title(title, color=fg, fontsize=10)
    ax.set_xlabel("Probability", color=fg, fontsize=8)
    ax.set_ylabel("Pixels (log)", color=fg, fontsize=8)
    ax.tick_params(colors=fg, labelsize=7)
    for s in ax.spines.values():
        s.set_color("#2b5478")
    return fig_to_np(fig)


# --------------------------------------------------------------------------- #
# Result packing, technical flags, ground-truth comparison
# --------------------------------------------------------------------------- #
def pack_result(result: dict) -> dict:
    seg = result["segmentation"]
    c = result["classification"]
    frames = {}
    for ph in ("ED", "ES"):
        d = seg[ph]
        frames[ph] = {
            "image": _arr(d["processed_image"]),
            "mask": (_arr(d["predicted_mask"]) > 0.5).astype(np.float32),
            "prob": _arr(d["probability_mask"]),
        }
    rec = {
        "measurements": {k: float(v) for k, v in result["quantitative_measurements"].items()},
        "classification": {
            "band": str(c["estimated_EF_band"]),
            "top_score": float(c["model_confidence_score"]),
            "scores": {k: float(v) for k, v in c["class_scores"].items()},
        },
        "frames": frames,
        "gt": None,
    }
    rec["geom_band"] = ef_band_from_value(rec["measurements"]["EF_pct"])
    return rec


def compute_flags(rec: dict, quality_fn=None) -> list[str]:
    """Exploratory technical checks - NOT a clinical quality judgement."""
    m, flags = rec["measurements"], []
    for ph in ("ED", "ES"):
        mask = rec["frames"][ph]["mask"] > 0.5
        if mask.sum() == 0:
            flags.append(f"{ph}: no LV region was predicted.")
            continue
        n_cc = ndi.label(mask)[1]
        if n_cc > 1:
            flags.append(f"{ph}: predicted mask is split into {n_cc} disconnected regions.")
        cov = 100.0 * mask.mean()
        if cov < 1.5:
            flags.append(f"{ph}: predicted LV region is very small ({cov:.2f}% of the frame).")
        if cov > 35:
            flags.append(f"{ph}: predicted LV region is unusually large ({cov:.1f}% of the frame).")
    if m["ESV_ml"] >= m["EDV_ml"]:
        flags.append("ESV is not smaller than EDV (non-positive stroke volume).")
    if not 10 <= m["EF_pct"] <= 85:
        flags.append(f"EF of {m['EF_pct']:.1f}% is outside the 10-85% range expected for A4C estimates.")
    if rec["classification"]["band"] != rec["geom_band"]:
        flags.append("ML classifier band differs from the geometry-derived EF band.")
    if rec["classification"]["top_score"] < 0.5:
        flags.append("Top classifier score is below 0.50 (uncalibrated) - interpret with caution.")
    if quality_fn is not None:
        try:
            extra = quality_fn(rec) or []
            flags.extend(str(e) for e in extra)
        except Exception as exc:  # never let an optional hook break the app
            flags.append(f"Quality module hook failed: {exc}")
    return flags


def _dice_iou(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = a > 0.5, b > 0.5
    inter = float((a & b).sum())
    s = float(a.sum() + b.sum())
    union = float((a | b).sum())
    return (2 * inter / s if s else 1.0), (inter / union if union else 1.0)


def load_gt_mask(frozen, image_path: str, mask_path: str, target_shape, label: int = 1) -> np.ndarray:
    """
    Load a CAMUS ground-truth mask and bring it into the orientation used for inference.
    Orientation is decided by comparing the *image* with frozen.load_nifti_2d's output
    (never by looking at the mask), and refused if they cannot be aligned.
    """
    import nibabel as nib

    raw_img = np.squeeze(np.asarray(nib.load(image_path).dataobj)).astype(np.float32)
    raw_msk = np.squeeze(np.asarray(nib.load(mask_path).dataobj))
    ref = _arr(frozen.load_nifti_2d(image_path)[0])
    if raw_img.shape != raw_msk.shape:
        raise ValueError("mask and image have different shapes")

    def corr(x, y):
        x, y = x.ravel() - x.mean(), y.ravel() - y.mean()
        return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8))

    best, best_c = None, -1.0
    for tf in (lambda z: z, lambda z: z.T):
        cand = tf(raw_img)
        if cand.shape == ref.shape:
            c = corr(cand, ref)
            if c > best_c:
                best, best_c = tf, c
    if best is None or best_c < 0.9:
        raise ValueError("could not align the mask orientation with the image")
    gt = (best(raw_msk) == label).astype(np.float32)
    return (cv2.resize(gt, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.float32)


# --------------------------------------------------------------------------- #
# Patient / case store  (SQLite + one npz per case)
# --------------------------------------------------------------------------- #
class CaseStore:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "cases").mkdir(parents=True, exist_ok=True)
        self.db = self.root / "echolv_cases.sqlite"
        self._run(
            """CREATE TABLE IF NOT EXISTS cases(
                case_id TEXT PRIMARY KEY, patient_id TEXT, created_at TEXT,
                ef REAL, edv REAL, esv REAL, sv REAL,
                band_geom TEXT, band_ml TEXT, n_flags INTEGER,
                notes TEXT, payload TEXT)"""
        )

    def _run(self, sql, args=(), fetch=False):
        with closing(sqlite3.connect(self.db, timeout=30)) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql, args)
            rows = cur.fetchall() if fetch else None
            conn.commit()
            return rows

    # -- write --------------------------------------------------------------
    def add(self, rec: dict, notes: str = "") -> str:
        cid = datetime.now().strftime("C%Y%m%d%H%M%S") + uuid.uuid4().hex[:3]
        arrays = {}
        for ph in ("ED", "ES"):
            f = rec["frames"][ph]
            arrays[f"{ph}_image"] = f["image"].astype(np.float16)
            arrays[f"{ph}_mask"] = f["mask"].astype(np.uint8)
            arrays[f"{ph}_prob"] = f["prob"].astype(np.float16)
        if rec.get("gt"):
            for ph in ("ED", "ES"):
                if rec["gt"].get(ph) is not None:
                    arrays[f"{ph}_gt"] = rec["gt"][ph]["mask"].astype(np.uint8)
        np.savez_compressed(self.root / "cases" / f"{cid}.npz", **arrays)
        m = rec["measurements"]
        payload = {
            "measurements": m,
            "classification": rec["classification"],
            "geom_band": rec["geom_band"],
            "flags": rec["flags"],
            "ed_file": rec.get("ed_file", ""),
            "es_file": rec.get("es_file", ""),
            "gt_metrics": {ph: rec["gt"][ph]["metrics"] for ph in ("ED", "ES") if rec.get("gt") and rec["gt"].get(ph)}
            if rec.get("gt")
            else None,
        }
        self._run(
            "INSERT INTO cases VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cid, rec["patient_id"], datetime.now().isoformat(timespec="seconds"),
                m["EF_pct"], m["EDV_ml"], m["ESV_ml"], m["SV_ml"],
                rec["geom_band"], rec["classification"]["band"], len(rec["flags"]),
                notes or "", json.dumps(payload),
            ),
        )
        return cid

    def update_notes(self, cid: str, notes: str):
        self._run("UPDATE cases SET notes=? WHERE case_id=?", (notes or "", cid))

    def delete(self, cid: str):
        self._run("DELETE FROM cases WHERE case_id=?", (cid,))
        p = self.root / "cases" / f"{cid}.npz"
        if p.exists():
            p.unlink()

    # -- read ---------------------------------------------------------------
    def patients(self) -> list[str]:
        return [r["patient_id"] for r in self._run("SELECT DISTINCT patient_id FROM cases ORDER BY patient_id", fetch=True)]

    def rows(self, patient: str | None = None):
        if patient:
            return self._run("SELECT * FROM cases WHERE patient_id=? ORDER BY created_at DESC", (patient,), fetch=True)
        return self._run("SELECT * FROM cases ORDER BY created_at DESC", fetch=True)

    def table(self, patient: str | None = None) -> pd.DataFrame:
        rows = self.rows(patient)
        cols = ["Case ID", "Patient", "Date", "EF %", "EDV mL", "ESV mL", "SV mL", "Geometry band", "ML band", "Flags"]
        data = [
            [r["case_id"], r["patient_id"], r["created_at"].replace("T", " "), round(r["ef"], 2), round(r["edv"], 2),
             round(r["esv"], 2), round(r["sv"], 2), BAND_PRETTY.get(r["band_geom"], r["band_geom"]),
             BAND_PRETTY.get(r["band_ml"], r["band_ml"]), r["n_flags"]]
            for r in rows
        ]
        return pd.DataFrame(data, columns=cols)

    def choices(self, patient: str | None = None):
        return [
            (f"{r['patient_id']} | {r['created_at'][:16].replace('T', ' ')} | EF {r['ef']:.1f}% | {r['case_id']}", r["case_id"])
            for r in self.rows(patient)
        ]

    def get(self, cid: str) -> dict:
        rows = self._run("SELECT * FROM cases WHERE case_id=?", (cid,), fetch=True)
        if not rows:
            raise KeyError(cid)
        r = rows[0]
        payload = json.loads(r["payload"])
        z = np.load(self.root / "cases" / f"{cid}.npz")
        frames = {
            ph: {
                "image": z[f"{ph}_image"].astype(np.float32),
                "mask": z[f"{ph}_mask"].astype(np.float32),
                "prob": z[f"{ph}_prob"].astype(np.float32),
            }
            for ph in ("ED", "ES")
        }
        gt = None
        if payload.get("gt_metrics"):
            gt = {}
            for ph in ("ED", "ES"):
                if payload["gt_metrics"].get(ph) is not None and f"{ph}_gt" in z.files:
                    gt[ph] = {"mask": z[f"{ph}_gt"].astype(np.float32), "metrics": payload["gt_metrics"][ph]}
        return {
            "case_id": cid, "patient_id": r["patient_id"], "created_at": r["created_at"], "notes": r["notes"],
            "measurements": payload["measurements"], "classification": payload["classification"],
            "geom_band": payload["geom_band"], "flags": payload["flags"],
            "ed_file": payload.get("ed_file", ""), "es_file": payload.get("es_file", ""),
            "frames": frames, "gt": gt,
        }

    def export_csv(self, path) -> str:
        df = self.table()
        df.to_csv(path, index=False)
        return str(path)


# --------------------------------------------------------------------------- #
# HTML builders
# --------------------------------------------------------------------------- #
def page_head(title: str, subtitle: str, patient: str | None = None) -> str:
    chip = f'<div class="chip">Current Patient<b>{esc(patient)}</b>A4C View</div>' if patient else ""
    return (
        f'<div class="page-head"><div><h2>{title}</h2><p>{subtitle}</p></div>'
        f'<div class="head-right">{chip}<div class="badge">Research Only<small>Not for clinical diagnosis</small></div></div></div>'
    )


def kpi_html(m: dict) -> str:
    items = [
        ("Ejection Fraction (EF)", f"{m['EF_pct']:.2f}%", "Single-plane geometric estimate", "ef"),
        ("End-Diastolic Volume (EDV)", f"{m['EDV_ml']:.2f} mL", "Estimated at ED", ""),
        ("End-Systolic Volume (ESV)", f"{m['ESV_ml']:.2f} mL", "Estimated at ES", ""),
        ("Stroke Volume (SV)", f"{m['SV_ml']:.2f} mL", "EDV minus ESV", ""),
    ]
    cells = "".join(
        f'<div class="kpi {c}"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
        for l, v, s, c in items
    )
    return f'<div class="section-title">Key LV Functional Measurements</div><div class="kpis">{cells}</div>'


def volume_html(m: dict) -> str:
    mx = max(m["EDV_ml"], m["ESV_ml"], 1e-6)
    h = lambda v: max(6.0, 150.0 * v / mx)  # noqa: E731
    return (
        '<div class="section-title">LV Volume Comparison</div><div class="vol">'
        f'<div class="bar-col"><div class="bar-val">{m["EDV_ml"]:.2f}</div><div class="bar ed" style="height:{h(m["EDV_ml"]):.0f}px"></div><div class="bar-lbl">End-Diastole (ED)</div></div>'
        f'<div class="bar-col"><div class="bar-val">{m["ESV_ml"]:.2f}</div><div class="bar es" style="height:{h(m["ESV_ml"]):.0f}px"></div><div class="bar-lbl">End-Systole (ES)</div></div>'
        "</div>"
    )


def classification_html(rec: dict) -> str:
    g, c = rec["geom_band"], rec["classification"]["band"]
    agree = g == c
    banner = (
        '<div class="agree ok">&#10003; Categories agree <span>The geometry-derived and ML classifier EF bands are the same.</span></div>'
        if agree
        else '<div class="agree warn">&#9888; Categories differ <span>The geometry-derived and ML classifier EF bands are not the same - review the case.</span></div>'
    )
    return (
        '<div class="section-title">EF Assessment and Classification</div>'
        '<div class="cats">'
        f'<div class="cat"><div class="l">Geometry-Derived Category</div><div class="v">{BAND_PRETTY.get(g, g)}</div><div class="s">Based on A4C geometric EF</div></div>'
        f'<div class="cat"><div class="l">ML Classifier Category</div><div class="v">{BAND_PRETTY.get(c, c)}</div><div class="s">Frozen ML model prediction</div></div>'
        f"</div>{banner}"
    )


def notices_html(flags: list[str]) -> str:
    if flags:
        body = "<ul>" + "".join(f"<li>{esc(f)}</li>" for f in flags) + "</ul>"
        title = f"{len(flags)} Technical Flag{'s' if len(flags) != 1 else ''} Raised"
    else:
        body = "<p>No predefined technical inconsistencies were detected for this case.</p>"
        title = "No Technical Flags Detected"
    return (
        '<div class="notice"><div class="notice-h">Analysis Review Notices</div>'
        f'<div class="notice-b"><div><b>{title}</b>{body}</div>'
        "<div class='notice-note'>These technical checks are exploratory and do not establish measurement accuracy, EF accuracy, "
        "or clinical reliability. Independent expert review is required.</div></div></div>"
    )


def details_html(m: dict) -> str:
    rows = [
        ("ED LV Cavity Area", f"{m['ED_area_mm2']:.2f} mm&sup2;", "End-Diastolic Volume (EDV)", f"{m['EDV_ml']:.2f} mL"),
        ("ES LV Cavity Area", f"{m['ES_area_mm2']:.2f} mm&sup2;", "End-Systolic Volume (ESV)", f"{m['ESV_ml']:.2f} mL"),
        ("ED LV Long-Axis Length", f"{m['ED_length_mm']:.2f} mm", "Stroke Volume (SV)", f"{m['SV_ml']:.2f} mL"),
        ("ES LV Long-Axis Length", f"{m['ES_length_mm']:.2f} mm", "Ejection Fraction (EF)", f"{m['EF_pct']:.2f} %"),
    ]
    body = "".join(f"<tr><td>{a}</td><td class='n'>{b}</td><td>{c}</td><td class='n'>{d}</td></tr>" for a, b, c, d in rows)
    return f'<div class="section-title">Detailed LV Geometric Measurements</div><table class="tbl">{body}</table>'






def ml_quantitative_dashboard_html(rec: dict) -> str:
    """Display frozen pipeline measurements in a research dashboard."""

    m = rec["measurements"]

    def metric(label, value, unit):
        return (
            '<div class="research-kpi">'
            f'<div class="research-kpi-label">{label}</div>'
            f'<div class="research-kpi-value">{value:.2f}'
            f'<span>{unit}</span></div>'
            '</div>'
        )

    kpis = "".join([
        metric("Ejection Fraction", m["EF_pct"], "%"),
        metric("End-Diastolic Volume", m["EDV_ml"], "mL"),
        metric("End-Systolic Volume", m["ESV_ml"], "mL"),
        metric("Stroke Volume", m["SV_ml"], "mL"),
    ])

    geometry_rows = [
        ("LV Area", m["ED_area_mm2"], m["ES_area_mm2"], "mm²"),
        ("LV Long-Axis Length", m["ED_length_mm"], m["ES_length_mm"], "mm"),
    ]

    rows = "".join(
        f"<tr><td>{label}</td>"
        f"<td>{ed:.2f} {unit}</td>"
        f"<td>{es:.2f} {unit}</td></tr>"
        for label, ed, es, unit in geometry_rows
    )

    return (
        '<div class="research-intro">'
        '<h3>Quantitative LV Analysis</h3>'
        '<p>Geometric and functional measurements derived from '
        'the frozen segmentation pipeline.</p>'
        '</div>'
        '<div class="research-kpi-grid">' + kpis + '</div>'
        '<div class="dcard">'
        '<div class="dtitle">ED vs ES Geometry</div>'
        '<table class="tbl dark">'
        '<tr><th>Measurement</th><th>ED</th><th>ES</th></tr>'
        + rows +
        '</table></div>'
        '<div class="dsub" style="margin-top:12px">'
        'Volumes are single-plane A4C estimates. These outputs are '
        'research-only and must not be interpreted as clinically validated.'
        '</div>'
    )

def ml_classification_details_html(rec: dict) -> str:
    """Research-only EF classification and frozen feature vector."""

    m = rec["measurements"]
    c = rec["classification"]

    features = [
        ("ED LV area", m["ED_area_mm2"], "mm²"),
        ("ES LV area", m["ES_area_mm2"], "mm²"),
        ("ED long-axis length", m["ED_length_mm"], "mm"),
        ("ES long-axis length", m["ES_length_mm"], "mm"),
        ("EDV", m["EDV_ml"], "mL"),
        ("ESV", m["ESV_ml"], "mL"),
        ("Stroke volume", m["SV_ml"], "mL"),
        ("Ejection fraction", m["EF_pct"], "%"),
    ]

    feature_rows = "".join(
        f"<tr><td>{i}</td><td>{name}</td>"
        f"<td>{value:.4f} {unit}</td></tr>"
        for i, (name, value, unit) in enumerate(features, 1)
    )

    scores = c.get("class_scores", {})
    score_rows = "".join(
        f"<tr><td>{label}</td><td>{float(score):.4f}</td></tr>"
        for label, score in scores.items()
    )

    return (
        '<div class="dcard">'
        '<div class="dtitle">Frozen EF Classification</div>'
        + classification_html(rec)
        + '<div class="dtitle">Classifier Class Scores</div>'
        + '<table class="tbl dark"><tr><th>Class</th>'
        '<th>Uncalibrated score</th></tr>'
        + score_rows
        + '</table>'
        + '<div class="dtitle" style="margin-top:20px">'
        'Eight Frozen Classifier Input Features</div>'
        + '<table class="tbl dark">'
        '<tr><th>#</th><th>Feature</th><th>Value</th></tr>'
        + feature_rows
        + '</table>'
        + notices_html(rec['flags'])
        + '<div class="dsub">'
        'Feature order is preserved from the frozen Phase 3 classifier. '
        'Scores are uncalibrated and do not indicate clinical reliability.'
        '</div></div>'
    )

def ml_metrics_html(rec: dict, ph: str) -> str:
    m = rec["measurements"]
    cov = 100.0 * float((rec["frames"][ph]["mask"] > 0.5).mean())
    return (
        f'<div class="dcard"><div class="dtitle">Segmentation Metrics ({ph})</div>'
        f'<div class="mrow"><span>Mask Coverage</span><b>{cov:.2f}%</b></div>'
        f'<div class="mrow"><span>Predicted LV Area</span><b>{m[ph + "_area_mm2"]:.2f} mm&sup2;</b></div>'
        f'<div class="mrow"><span>LV Long-Axis Length</span><b>{m[ph + "_length_mm"]:.2f} mm</b></div></div>'
    )


def class_scores_html(rec: dict) -> str:
    scores = rec["classification"]["scores"]
    rows = ""
    for k in ("Lower_EF", "Intermediate_EF", "Higher_EF"):
        v = scores.get(k, 0.0)
        rows += (
            f'<div class="srow"><span>{BAND_PRETTY[k].replace(" EF", "_EF")}</span>'
            f'<div class="strack"><div class="sfill" style="width:{100 * v:.1f}%"></div></div><b>{v:.3f}</b></div>'
        )
    return (
        '<div class="dcard"><div class="dtitle">Classifier Output (Frozen Model)</div>'
        f'<div class="dsub">Predicted EF-Band Class Scores</div>{rows}'
        '<div class="dsub">Uncalibrated classifier score - not a clinical reliability estimate.</div></div>'
    )


def all_measurements_html(rec: dict) -> str:
    body = "".join(f"<tr><td>{esc(k)}</td><td class='n'>{v:.4f}</td></tr>" for k, v in rec["measurements"].items())
    return f'<div class="dcard"><div class="dtitle">All Frozen Measurements</div><table class="tbl dark">{body}</table></div>'


def gt_html(rec: dict, note: str = "") -> str:
    gt = rec.get("gt")
    if not gt:
        msg = note or "No ground-truth masks were supplied for this case. Upload ED/ES masks on the New Analysis page to enable this tab."
        return f'<div class="dcard"><div class="dtitle">Ground Truth Comparison</div><div class="dsub">{esc(msg)}</div></div>'
    rows = "".join(
        f"<tr><td>{ph}</td><td class='n'>{gt[ph]['metrics']['dice']:.4f}</td><td class='n'>{gt[ph]['metrics']['iou']:.4f}</td></tr>"
        for ph in ("ED", "ES") if gt.get(ph)
    )
    return (
        '<div class="dcard"><div class="dtitle">Ground Truth Comparison (LV endocardium, label 1)</div>'
        f'<table class="tbl dark"><tr><th>Phase</th><th>Dice</th><th>IoU</th></tr>{rows}</table>'
        '<div class="dsub">Prediction vs. expert mask in the 256x256 model space. Teal = prediction, amber = expert mask.</div></div>'
    )


def gt_overlay(rec: dict, ph: str):
    gt = rec.get("gt")
    if not gt or not gt.get(ph):
        return None
    size = 384
    base = compose_view(rec["frames"][ph], "Original", 100, size).astype(np.float32)

    def draw(mask, colour):
        m = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR) > 0.5
        if not m.any():
            return
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        line = np.zeros((size, size), np.uint8)
        cv2.drawContours(line, cnts, -1, 255, 2)
        base[line > 0] = colour

    draw(gt[ph]["mask"], (255, 176, 32))
    draw(rec["frames"][ph]["mask"], TEAL)
    return base.astype(np.uint8)


def model_info_html(frozen) -> str:
    try:
        n_par = f"{frozen.frozen_segmentation_model.count_params():,}"
    except Exception:
        n_par = "n/a"
    try:
        thr = f"{float(frozen.SEGMENTATION_THRESHOLD):.2f}"
    except Exception:
        thr = "n/a"
    return f"""
<div class="dcard"><div class="dtitle">Segmentation model</div>
<div class="mrow"><span>Architecture</span><b>U-Net (selected "Large" configuration)</b></div>
<div class="mrow"><span>Input</span><b>256 x 256 grayscale, A4C view</b></div>
<div class="mrow"><span>Parameters</span><b>{n_par}</b></div>
<div class="mrow"><span>Binarisation threshold</span><b>{thr}</b></div>
<div class="mrow"><span>Training data</span><b>CAMUS (patient-level 400/50/50 split)</b></div>
<div class="mrow"><span>Held-out test Dice</span><b>0.932</b></div></div>
<div class="dcard"><div class="dtitle">EF-band classifier</div>
<div class="mrow"><span>Model</span><b>StandardScaler + Logistic Regression</b></div>
<div class="mrow"><span>Classes</span><b>Lower (&le;40) / Intermediate (40-50) / Higher (&ge;50)</b></div>
<div class="mrow"><span>Held-out accuracy</span><b>70% (35/50), macro-F1 0.663</b></div>
<div class="dsub">Class scores are uncalibrated and must not be read as clinical confidence.</div></div>
<div class="dcard"><div class="dtitle">Pipeline</div>
<div class="dsub">ED/ES frame &rarr; resize &rarr; frozen U-Net &rarr; LV mask &rarr; area &amp; long-axis length &rarr; single-plane
(area-length) EDV/ESV &rarr; SV, EF &rarr; EF-band classifier. No model is retrained inside this application.</div></div>"""


ABOUT_HTML = f"""
<div class="card-html">
<h3>About {APP_NAME}</h3>
<p><b>{APP_NAME}</b> is a research prototype developed at the NUST ASCEND Lab. It segments the left ventricle (LV) in paired
apical four-chamber (A4C) end-diastolic (ED) and end-systolic (ES) echocardiography frames and derives single-plane
volumes, stroke volume and ejection fraction (EF).</p>
<h4>How to use it</h4>
<ol>
<li><b>New Analysis</b> - enter a patient ID, upload the ED and ES NIfTI frames (optionally the expert masks), press <i>Run LV Analysis</i>.</li>
<li><b>Clinical View</b> - contours, key measurements, EF category and review notices.</li>
<li><b>ML Research View</b> - masks, probability maps, classifier scores, ground-truth Dice/IoU, model information.</li>
<li><b>Case Management</b> - every analysis is saved; reopen, compare two cases, or follow a patient's EF trend.</li>
<li><b>Download Report</b> - generate a PDF for the current case, or export the case table as CSV.</li>
</ol>
<h4>Limitations</h4>
<ul>
<li>Non-diagnostic research use only. Not validated for clinical decision making.</li>
<li>Volumes are single-plane A4C estimates, not biplane Simpson volumes.</li>
<li>Technical flags are exploratory checks and do not establish accuracy.</li>
<li>Classifier scores are uncalibrated.</li>
<li>Trained and evaluated on one public dataset (CAMUS); performance on other scanners or populations is unknown.</li>
<li>Use only de-identified data. Do not enter real patient names or identifiers.</li>
</ul></div>"""


# --------------------------------------------------------------------------- #
# PDF report
# --------------------------------------------------------------------------- #
def build_pdf(rec: dict, path, include_research: bool = True, notes: str = "", model_info: dict | None = None) -> str:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    navy, teal, grey = colors.HexColor("#0f2a43"), colors.HexColor("#0f9d8e"), colors.HexColor("#587086")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=navy, fontSize=20, leading=24, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=navy, fontSize=13, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10.5, textColor=grey)

    m, c = rec["measurements"], rec["classification"]
    created = rec.get("created_at") or datetime.now().isoformat(timespec="seconds")
    story = [
        Paragraph(f"{APP_NAME} - LV Analysis Report", h1),
        Paragraph("AI-assisted left ventricular functional assessment (A4C) - research prototype", small),
        Spacer(1, 6),
    ]

    def table(data, widths, header=False, zebra=True):
        t = Table(data, colWidths=widths)
        style = [
            ("FONTSIZE", (0, 0), (-1, -1), 9), ("TEXTCOLOR", (0, 0), (-1, -1), navy),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d3dfea")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        if zebra:
            style += [("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")])]
        t.setStyle(TableStyle(style))
        return t

    story.append(table(
        [["Patient ID", esc(rec["patient_id"]), "Case ID", esc(rec.get("case_id") or "not saved")],
         ["Report date", datetime.now().strftime("%Y-%m-%d %H:%M"), "Analysed", esc(created.replace("T", " "))],
         ["ED file", esc(rec.get("ed_file") or "-"), "ES file", esc(rec.get("es_file") or "-")]],
        [26 * mm, 62 * mm, 22 * mm, 62 * mm]))

    story.append(Paragraph("Key LV functional measurements", h2))
    story.append(table(
        [["Ejection fraction", "End-diastolic volume", "End-systolic volume", "Stroke volume"],
         [f"{m['EF_pct']:.2f} %", f"{m['EDV_ml']:.2f} mL", f"{m['ESV_ml']:.2f} mL", f"{m['SV_ml']:.2f} mL"]],
        [43 * mm] * 4, header=True, zebra=False))
    story.append(Spacer(1, 4))
    agree = "Categories agree" if rec["geom_band"] == c["band"] else "Categories DIFFER - review"
    story.append(table(
        [["Geometry-derived EF category", BAND_PRETTY.get(rec["geom_band"], rec["geom_band"])],
         ["ML classifier EF category", BAND_PRETTY.get(c["band"], c["band"])],
         ["Agreement", agree]], [70 * mm, 102 * mm]))

    story.append(Paragraph("Segmentation (predicted LV contour)", h2))
    imgs = []
    for ph in ("ED", "ES"):
        imgs.append(RLImage(to_png_bytes(compose_view(rec["frames"][ph], "Contour", 100, 480)), width=78 * mm, height=78 * mm))
    story.append(Table([imgs, [Paragraph("End-diastole (ED)", small), Paragraph("End-systole (ES)", small)]],
                       colWidths=[86 * mm, 86 * mm]))

    story.append(Paragraph("Geometric measurements", h2))
    story.append(table(
        [["ED cavity area", f"{m['ED_area_mm2']:.2f} mm2", "ES cavity area", f"{m['ES_area_mm2']:.2f} mm2"],
         ["ED long-axis length", f"{m['ED_length_mm']:.2f} mm", "ES long-axis length", f"{m['ES_length_mm']:.2f} mm"]],
        [40 * mm, 46 * mm, 40 * mm, 46 * mm]))

    story.append(Paragraph("Analysis review notices", h2))
    if rec["flags"]:
        for f in rec["flags"]:
            story.append(Paragraph(f"&bull; {esc(f)}", body))
    else:
        story.append(Paragraph("No predefined technical inconsistencies were detected for this case.", body))
    story.append(Paragraph("These exploratory checks do not establish measurement accuracy, EF accuracy or clinical reliability.", small))

    if notes and notes.strip():
        story.append(Paragraph("Notes", h2))
        story.append(Paragraph(esc(notes).replace("\n", "<br/>"), body))

    if include_research:
        story.append(PageBreak())
        story.append(Paragraph("Research appendix - ML view", h1))
        probs = [RLImage(to_png_bytes(compose_view(rec["frames"][ph], "Probability", 100, 480)), width=70 * mm, height=70 * mm)
                 for ph in ("ED", "ES")]
        story.append(Table([probs, [Paragraph("ED probability map", small), Paragraph("ES probability map", small)]],
                           colWidths=[86 * mm, 86 * mm]))
        cov = {ph: 100.0 * float((rec["frames"][ph]["mask"] > 0.5).mean()) for ph in ("ED", "ES")}
        story.append(Paragraph("Segmentation metrics", h2))
        story.append(table(
            [["Phase", "Mask coverage", "Predicted LV area", "Long-axis length"],
             ["ED", f"{cov['ED']:.2f} %", f"{m['ED_area_mm2']:.2f} mm2", f"{m['ED_length_mm']:.2f} mm"],
             ["ES", f"{cov['ES']:.2f} %", f"{m['ES_area_mm2']:.2f} mm2", f"{m['ES_length_mm']:.2f} mm"]],
            [30 * mm, 40 * mm, 51 * mm, 51 * mm], header=True))
        story.append(Paragraph("Classifier output (uncalibrated)", h2))
        sc = c["scores"]
        story.append(table(
            [["Lower_EF", "Intermediate_EF", "Higher_EF"],
             [f"{sc.get('Lower_EF', 0):.3f}", f"{sc.get('Intermediate_EF', 0):.3f}", f"{sc.get('Higher_EF', 0):.3f}"]],
            [57 * mm] * 3, header=True, zebra=False))
        if rec.get("gt"):
            story.append(Paragraph("Ground-truth comparison (expert mask, LV endocardium)", h2))
            rows = [["Phase", "Dice", "IoU"]] + [
                [ph, f"{rec['gt'][ph]['metrics']['dice']:.4f}", f"{rec['gt'][ph]['metrics']['iou']:.4f}"]
                for ph in ("ED", "ES") if rec["gt"].get(ph)]
            story.append(table(rows, [40 * mm, 40 * mm, 40 * mm], header=True))
        if model_info:
            story.append(Paragraph("Model information", h2))
            for k, v in model_info.items():
                story.append(Paragraph(f"<b>{esc(k)}:</b> {esc(v)}", body))

    story.append(Spacer(1, 10))
    story.append(Paragraph(DISCLAIMER, small))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(grey)
        canvas.drawString(18 * mm, 10 * mm, f"{APP_NAME} | Research prototype - non-diagnostic")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm, title=f"{APP_NAME} report - {rec['patient_id']}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return str(path)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = '\n:root{color-scheme:light}\n.gradio-container{max-width:1560px!important;background:#e9eff6!important;font-family:Inter,\'Segoe UI\',system-ui,sans-serif!important}\nfooter{display:none!important}\n.flat,.flat>*{background:transparent!important;border:none!important;box-shadow:none!important}\n.flat{padding:0!important}\n.card{background:#fff!important;border:1px solid #d9e4ee!important;border-radius:14px!important;padding:14px 16px!important;box-shadow:0 1px 3px rgba(15,40,70,.06)!important}\n#sidebar{background:linear-gradient(180deg,#09192b 0%,#0d2c47 100%)!important;border-radius:16px!important;padding:18px 14px!important;gap:6px!important;min-height:86vh}\n.brand{display:flex;gap:10px;align-items:center;margin-bottom:14px;color:#fff}\n.brand .logo{width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,#14b8a6,#2563eb);display:flex;align-items:center;justify-content:center;font-size:22px}\n.brand b{font-size:17px;display:block}.brand span{font-size:11px;color:#9db6cb}\n.nav-btn{justify-content:flex-start!important;text-align:left!important;background:transparent!important;color:#bcd0e0!important;border:1px solid transparent!important;border-radius:10px!important;font-weight:500!important;box-shadow:none!important;min-height:44px}\n.nav-btn:hover{background:rgba(255,255,255,.08)!important}\nbutton.nav-btn.primary{background:rgba(31,200,180,.17)!important;border-color:rgba(31,200,180,.6)!important;color:#fff!important}\n.side-foot{margin-top:auto;color:#8ea8bd;font-size:11px;line-height:1.5;border-top:1px solid rgba(255,255,255,.1);padding-top:10px}\n.page-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin:2px 0 6px}\n.page-head h2{margin:0;font-size:23px;color:#0f2a43}.page-head p{margin:2px 0 0;color:#587086;font-size:13px}\n.head-right{display:flex;gap:10px;align-items:center}\n.chip{border:1px solid #d3dfea;background:#fff;border-radius:10px;padding:6px 14px;font-size:11px;color:#587086;line-height:1.25}\n.chip b{display:block;color:#0f2a43;font-size:14px}\n.badge{background:#13b3a2;color:#fff;border-radius:10px;padding:7px 14px;font-size:12px;font-weight:700;line-height:1.2}\n.badge small{display:block;font-weight:400;font-size:10px;opacity:.9}\n.section-title{font-weight:700;color:#0f2a43;font-size:15px;margin:2px 0 10px}\n.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}\n.kpi{background:#f7fafd;border:1px solid #dde7f0;border-radius:12px;padding:12px 14px}\n.kpi .l{font-size:12px;color:#587086}.kpi .v{font-size:30px;font-weight:800;color:#0f2a43;line-height:1.15}.kpi .s{font-size:11px;color:#8aa0b2}\n.kpi.ef .v{color:#0f9d8e}\n.vol{display:flex;gap:46px;justify-content:center;align-items:flex-end;height:215px;padding-top:6px;border-bottom:1px solid #dde7f0}\n.bar-col{display:flex;flex-direction:column;align-items:center;justify-content:flex-end}\n.bar{width:84px;border-radius:6px 6px 0 0}.bar.ed{background:#0f9d8e}.bar.es{background:#2f7fc4}\n.bar-val{font-weight:700;color:#0f2a43;margin-bottom:4px}.bar-lbl{font-size:11px;color:#587086;margin-top:8px;position:relative;top:26px}\n.cats{display:grid;grid-template-columns:1fr 1fr;gap:10px}\n.cat{border:1px solid #dde7f0;border-radius:12px;padding:10px 14px;background:#f7fafd}\n.cat .l{font-size:11px;color:#587086}.cat .v{font-size:22px;font-weight:800;color:#0f9d8e}.cat .s{font-size:10.5px;color:#8aa0b2}\n.agree{margin-top:10px;border-radius:10px;padding:9px 12px;font-weight:700;font-size:13px}.agree span{font-weight:400;font-size:12px;margin-left:6px}\n.agree.ok{background:#e6f6f1;color:#0b7a64;border:1px solid #b8e5d8}.agree.warn{background:#fff4dc;color:#8a5a00;border:1px solid #f1d28a}\n.notice{background:#fff6e5;border:1px solid #f0d9a6;border-radius:14px;overflow:hidden}\n.notice-h{background:#fbe7bd;color:#6a4a00;font-weight:700;padding:8px 14px}\n.notice-b{display:grid;grid-template-columns:1fr 1.2fr;gap:14px;padding:12px 14px;color:#5d4a1a;font-size:13px}\n.notice-b ul{margin:6px 0 0 16px;padding:0}.notice-b p{margin:4px 0 0}.notice-note{font-size:12px;color:#7a6a40}\n.tbl{width:100%;border-collapse:collapse;font-size:13px;color:#16324b}\n.tbl td,.tbl th{padding:8px 10px;border-bottom:1px solid #e3ebf3;text-align:left}.tbl td.n{font-weight:700;text-align:right}\n.tbl.dark{color:#dcebf6}.tbl.dark td,.tbl.dark th{border-bottom:1px solid #22496b}\n.ml-page{background:linear-gradient(180deg,#081a2c,#0c2740)!important;border-radius:16px!important;padding:16px!important}\n.ml-page .page-head h2{color:#fff}.ml-page .page-head p{color:#9db6cb}\n.ml-page .card{background:#102c45!important;border-color:#1d4466!important}\n.ml-page .chip{background:#102c45;border-color:#1d4466;color:#9db6cb}.ml-page .chip b{color:#fff}\n.dcard{background:#102c45;border:1px solid #1d4466;border-radius:12px;padding:12px 14px;color:#dcebf6;margin-bottom:10px}\n.dtitle{font-weight:700;color:#fff;margin-bottom:8px}.dsub{font-size:12px;color:#9db6cb;margin-top:6px}\n.mrow{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #1f4262;font-size:13px}.mrow b{color:#4fe3cf}\n.srow{display:grid;grid-template-columns:130px 1fr 52px;gap:8px;align-items:center;margin:8px 0;font-size:12px;color:#dcebf6}\n.strack{background:#0a2236;border-radius:6px;height:14px}.sfill{background:#2fd0bd;height:14px;border-radius:6px}\n.phase-h{color:#fff;font-weight:700;margin:6px 0}\n.card-html{background:#fff;border:1px solid #d9e4ee;border-radius:14px;padding:18px 22px;color:#16324b;line-height:1.55}\n.card-html h3{margin-top:0;color:#0f2a43}.card-html h4{color:#0f2a43;margin-bottom:4px}\n.status{background:#e6f6f1;border:1px solid #b8e5d8;color:#0b7a64;border-radius:10px;padding:10px 14px;font-weight:600}\n.status.err{background:#fdeaea;border-color:#f3b8b8;color:#8a1f1f}\n@media(max-width:900px){.kpis{grid-template-columns:1fr 1fr}.notice-b{grid-template-columns:1fr}.cats{grid-template-columns:1fr}}\n\n\n\n/* ==========================================================\n   ASCEND EchoLV — VISUAL REFINEMENT\n   ========================================================== */\n\n/* Sidebar branding */\n#sidebar .brand,\n#sidebar .brand b,\n#sidebar .brand b *,\n#sidebar .brand > div {\n    color: #ffffff !important;\n}\n\n#sidebar .brand b {\n    font-size: 19px !important;\n    font-weight: 800 !important;\n    line-height: 1.25 !important;\n    letter-spacing: -0.3px !important;\n}\n\n#sidebar .brand span {\n    color: #b9d5e9 !important;\n    font-size: 12px !important;\n}\n\n/* Cleaner sidebar buttons */\n#sidebar button.nav-btn {\n    border-radius: 11px !important;\n    font-size: 14px !important;\n    font-weight: 600 !important;\n    min-height: 46px !important;\n}\n\n/* Clinical ultrasound image components */\n#clinical-images .image-container,\n#clinical-images .image-frame,\n#clinical-images .image-container > div {\n    background: #071523 !important;\n}\n\n#clinical-images img {\n    object-fit: contain !important;\n    background: #071523 !important;\n}\n\n/* Dark research image panels */\n.ml-page .image-container,\n.ml-page .image-frame,\n.ml-page .image-container > div {\n    background: #071523 !important;\n}\n\n.ml-page img {\n    object-fit: contain !important;\n    background: #071523 !important;\n}\n\n/* Research image card styling */\n.ml-page .card {\n    border-radius: 12px !important;\n    overflow: hidden !important;\n}\n\n/* Improve text contrast in ML Research View */\n.ml-page .mrow,\n.ml-page .mrow span,\n.ml-page .dsub,\n.ml-page .phase-h {\n    color: #d5e7f4 !important;\n}\n\n.ml-page .mrow b {\n    color: #4fe3cf !important;\n}\n\n/* More consistent page typography */\n.page-head h2 {\n    font-weight: 800 !important;\n    letter-spacing: -0.5px !important;\n}\n\n.section-title {\n    font-weight: 750 !important;\n}\n\n/* Smaller screens */\n@media (max-width: 1100px) {\n    .kpis {\n        grid-template-columns: repeat(2, minmax(0, 1fr));\n    }\n}\n\n\n/* ASCEND Clinical View Refinement V2 */\n\n#sidebar .brand b,\n#sidebar .brand b * {\n    color: #ffffff !important;\n    opacity: 1 !important;\n    -webkit-text-fill-color: #ffffff !important;\n}\n\n#sidebar .brand span {\n    color: #bdd9eb !important;\n}\n\n#clinical-images {\n    gap: 12px !important;\n}\n\n#clinical-images .image-container,\n#clinical-images .image-frame,\n#clinical-images .image-container > div {\n    background: #071523 !important;\n    border-radius: 12px !important;\n}\n\n#clinical-images img {\n    object-fit: contain !important;\n    background: #071523 !important;\n    border-radius: 12px !important;\n}\n\n.clinical-image-captions {\n    display: flex;\n    justify-content: space-around;\n    gap: 12px;\n    margin: 6px 0 12px;\n    color: #24465f;\n    font-size: 13px;\n    font-weight: 700;\n}\n\n#clinical-controls {\n    align-items: center !important;\n    gap: 16px !important;\n}\n\n#clinical-controls .wrap {\n    gap: 8px !important;\n}\n\n/* Keep clinical measurement cards readable */\n.kpis .kpi {\n    min-width: 0 !important;\n}\n\n\n\n/* ==========================================================\n   ASCEND EchoLV — ML RESEARCH VIEW V3\n   ========================================================== */\n\n.ml-page {\n    background: linear-gradient(\n        180deg, #071728 0%, #0b2238 100%\n    ) !important;\n    border: 1px solid #1c405b !important;\n    border-radius: 18px !important;\n    padding: 20px !important;\n    color: #e7f2fa !important;\n}\n\n/* Research tab navigation */\n.ml-page .tab-nav {\n    background: #102b43 !important;\n    border: 1px solid #25445d !important;\n    border-radius: 12px !important;\n    padding: 5px !important;\n    gap: 5px !important;\n}\n\n.ml-page .tab-nav button {\n    color: #b7d1e3 !important;\n    background: transparent !important;\n    border-radius: 8px !important;\n    font-weight: 650 !important;\n    font-size: 13px !important;\n    padding: 11px 14px !important;\n}\n\n.ml-page .tab-nav button.selected {\n    background: #176f78 !important;\n    color: #ffffff !important;\n    border-bottom: none !important;\n}\n\n/* Research text */\n.ml-page h3,\n.ml-page h4,\n.ml-page .phase-h,\n.ml-page .dtitle,\n.ml-page .section-title {\n    color: #f2f8fc !important;\n}\n\n.ml-page p,\n.ml-page .dsub,\n.ml-page .mrow,\n.ml-page label {\n    color: #c4d8e8 !important;\n}\n\n/* Image components */\n.ml-page .image-container,\n.ml-page .image-frame,\n.ml-page .image-container > div {\n    background: #061522 !important;\n    border-radius: 12px !important;\n}\n\n.ml-page img {\n    object-fit: contain !important;\n    background: #061522 !important;\n    border-radius: 10px !important;\n}\n\n/* Segmentation metrics */\n.ml-page .dcard {\n    background: #102e47 !important;\n    border: 1px solid #25516e !important;\n    border-radius: 12px !important;\n    padding: 16px !important;\n    color: #e7f2fa !important;\n}\n\n.ml-page .mrow {\n    border-bottom: 1px solid #274962 !important;\n    padding: 10px 0 !important;\n}\n\n.ml-page .mrow b {\n    color: #50e2cc !important;\n}\n\n/* Research tables */\n.ml-page table,\n.ml-page .tbl {\n    width: 100% !important;\n    border-collapse: collapse !important;\n    color: #e3f0f9 !important;\n}\n\n.ml-page .tbl td,\n.ml-page .tbl th {\n    color: #e3f0f9 !important;\n    border-bottom: 1px solid #25465e !important;\n    padding: 11px !important;\n}\n\n.ml-page .tbl th {\n    color: #8bded4 !important;\n    font-weight: 750 !important;\n}\n\n/* Classification comparison */\n.ml-page .cats {\n    display: grid !important;\n    grid-template-columns: repeat(2, minmax(0, 1fr));\n    gap: 12px !important;\n}\n\n.ml-page .cat {\n    background: #143650 !important;\n    border: 1px solid #2a5872 !important;\n    border-radius: 12px !important;\n    padding: 16px !important;\n}\n\n.ml-page .cat .l,\n.ml-page .cat .s {\n    color: #b9d6e7 !important;\n}\n\n.ml-page .cat .v {\n    color: #ffffff !important;\n}\n\n.ml-page .agree {\n    margin-top: 14px !important;\n}\n\n/* Research spacing */\n.ml-page .phase-h {\n    margin: 18px 0 12px !important;\n    font-size: 17px !important;\n    font-weight: 750 !important;\n}\n\n/* Smaller screens */\n@media (max-width: 1100px) {\n    .ml-page {\n        padding: 12px !important;\n    }\n\n    .ml-page .cats {\n        grid-template-columns: 1fr !important;\n    }\n\n    .ml-page .tab-nav {\n        flex-wrap: wrap !important;\n    }\n}\n\n\n/* ML QUANTITATIVE DASHBOARD V5 */\n\n.ml-page .research-kpi-grid {\n    display: grid;\n    grid-template-columns: repeat(4, minmax(0, 1fr));\n    gap: 12px;\n    margin: 16px 0 22px;\n}\n\n.ml-page .research-kpi {\n    background: #12334d;\n    border: 1px solid #28516d;\n    border-radius: 12px;\n    padding: 18px 14px;\n}\n\n.ml-page .research-kpi-label {\n    color: #b4d0e2;\n    font-size: 12px;\n    font-weight: 600;\n    margin-bottom: 12px;\n}\n\n.ml-page .research-kpi-value {\n    color: #50e2cc;\n    font-size: 27px;\n    font-weight: 800;\n}\n\n.ml-page .research-kpi-value span {\n    color: #a9c5d8;\n    font-size: 12px;\n    font-weight: 500;\n    margin-left: 5px;\n}\n\n@media (max-width: 1100px) {\n    .ml-page .research-kpi-grid {\n        grid-template-columns: repeat(2, minmax(0, 1fr));\n    }\n}\n\n\n\n/* SEGMENTATION EXPLORER — CLEAN LAYOUT V4 */\n\n.ml-page .research-intro {\n    padding: 4px 2px 14px;\n}\n\n.ml-page .research-intro h3 {\n    color: #f3f8fc !important;\n    font-size: 21px !important;\n    font-weight: 750 !important;\n    margin: 0 0 6px !important;\n}\n\n.ml-page .research-intro p {\n    color: #a9c5d8 !important;\n    font-size: 13px !important;\n    margin: 0 !important;\n}\n\n.ml-page .research-phase {\n    background: #0b243a !important;\n    border: 1px solid #24465e !important;\n    border-radius: 14px !important;\n    padding: 16px !important;\n    margin: 8px 0 20px !important;\n    gap: 12px !important;\n}\n\n.ml-page .research-phase-title {\n    color: #ffffff !important;\n    font-size: 18px !important;\n    font-weight: 750 !important;\n    padding-bottom: 8px !important;\n}\n\n.ml-page .research-image-card {\n    background: #102e45 !important;\n    border: 1px solid #254d68 !important;\n    border-radius: 12px !important;\n    padding: 12px !important;\n    min-width: 0 !important;\n    gap: 8px !important;\n}\n\n.ml-page .research-image-card img {\n    background: #061522 !important;\n    object-fit: contain !important;\n}\n\n.ml-page .research-image-card p {\n    color: #dcecf6 !important;\n    font-size: 13px !important;\n    font-weight: 650 !important;\n}\n\n.ml-page .research-metrics .dcard {\n    background: #12334d !important;\n    margin-top: 4px !important;\n}\n\n\n/* ML TAB HEADING CONTRAST FIX */\n.ml-page .tab-nav button,\n.ml-page .tab-nav button span,\n.ml-page button[role="tab"],\n.ml-page button[role="tab"] span {\n    color: #DCEAF5 !important;\n    opacity: 1 !important;\n}\n\n.ml-page .tab-nav button.selected,\n.ml-page .tab-nav button.selected span,\n.ml-page button[role="tab"][aria-selected="true"],\n.ml-page button[role="tab"][aria-selected="true"] span {\n    color: #21D4C4 !important;\n    opacity: 1 !important;\n}\n\n.ml-page .tab-nav button:hover,\n.ml-page button[role="tab"]:hover {\n    color: #FFFFFF !important;\n}\n'

HEAD = ("<script>if(!location.search.includes('__theme')){var u=new URL(location.href);"
        "u.searchParams.set('__theme','light');location.replace(u.toString());}</script>")


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
PAGES = ["new", "clinical", "ml", "cases", "report", "about"]
NAV_LABELS = {
    "new": "\u2795  New Analysis", "clinical": "\U0001FA7A  Clinical View", "ml": "\U0001F9E0  ML Research View",
    "cases": "\U0001F4C1  Case Management", "report": "\U0001F4C4  Download Report", "about": "\u2139\uFE0F  About",
}


def _path_of(f):
    if f is None:
        return None
    return f if isinstance(f, str) else getattr(f, "name", str(f))


def build_app(frozen, records_dir, demo_paths=None, quality_fn=None):
    """
    frozen       : the imported frozen_inference module
    records_dir  : folder on Google Drive that keeps the case database (persistent)
    demo_paths   : optional (ED_PATH, ES_PATH) for the one-click demo button
    quality_fn   : optional callable(rec) -> list[str] of extra flags (e.g. from quality_checks.py)
    Returns (demo, launch_kwargs). Call demo.launch(share=True, **launch_kwargs).
    """
    store = CaseStore(records_dir)
    reports_dir = Path(records_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    try:
        info = {
            "Segmentation model": "U-Net (Large), 256x256 input",
            "Parameters": f"{frozen.frozen_segmentation_model.count_params():,}",
            "Threshold": str(frozen.SEGMENTATION_THRESHOLD),
            "Classifier": "StandardScaler + Logistic Regression (EF bands)",
        }
    except Exception:
        info = None

    # ------------------------------------------------------------------ core
    def run_pipeline(patient_id, ed_path, es_path, ed_gt_path, es_gt_path, notes, save, progress=None):
        pid = re.sub(r"[^\w\-]+", "_", (patient_id or "").strip())
        if not pid:
            m = re.search(r"(patient\d+)", Path(ed_path).name, re.I)
            pid = m.group(1).lower() if m else "unlabelled"
        if progress:
            progress(0.15, desc="Loading ED / ES images")
        ed_img, ed_sp = frozen.load_nifti_2d(ed_path)
        es_img, es_sp = frozen.load_nifti_2d(es_path)
        if progress:
            progress(0.45, desc="Running LV segmentation and analysis")
        result = frozen.analyze_a4c_pair(ed_image=ed_img, es_image=es_img, ed_native_spacing=ed_sp, es_native_spacing=es_sp)
        rec = pack_result(result)
        rec.update(patient_id=pid, ed_file=Path(ed_path).name, es_file=Path(es_path).name, notes=notes or "")
        # optional ground truth
        gt_note = ""
        if ed_gt_path or es_gt_path:
            gt = {}
            for ph, img_p, gt_p in (("ED", ed_path, ed_gt_path), ("ES", es_path, es_gt_path)):
                if not gt_p:
                    continue
                try:
                    mk = load_gt_mask(frozen, img_p, gt_p, rec["frames"][ph]["mask"].shape)
                    d, i = _dice_iou(rec["frames"][ph]["mask"], mk)
                    gt[ph] = {"mask": mk, "metrics": {"dice": d, "iou": i}}
                except Exception as exc:
                    gt_note += f"{ph} ground truth skipped: {exc}. "
            rec["gt"] = gt or None
        rec["gt_note"] = gt_note
        rec["flags"] = compute_flags(rec, quality_fn)
        rec["case_id"], rec["created_at"] = None, datetime.now().isoformat(timespec="seconds")
        if save:
            if progress:
                progress(0.85, desc="Saving case")
            rec["case_id"] = store.add(rec, notes)
        return rec

    def render(rec, mode="Contour", opacity=70):
        """Everything every page needs, in the exact order of VIEW_OUTPUTS."""
        m, F = rec["measurements"], rec["frames"]
        pid = rec["patient_id"]
        ml_imgs = []
        for ph in ("ED", "ES"):
            ml_imgs += [render_panel(F[ph], k) for k in ("pre", "contour", "mask", "prob")]
        report_txt = (
            f'<div class="card-html"><b>Current case:</b> {esc(pid)} &nbsp;|&nbsp; <b>Case ID:</b> {esc(rec.get("case_id") or "not saved")}'
            f" &nbsp;|&nbsp; <b>EF:</b> {m['EF_pct']:.2f}% &nbsp;|&nbsp; <b>Flags:</b> {len(rec['flags'])}</div>"
        )
        return [
            page_head("Clinical View", "Left Ventricular Functional Assessment<br>Review the AI-derived segmentation, quantitative measurements and functional interpretation.", pid),
            compose_view(F["ED"], mode, opacity), compose_view(F["ES"], mode, opacity),
            kpi_html(m), volume_html(m), classification_html(rec), notices_html(rec["flags"]), details_html(m),
            page_head("ML Research View", "Segmentation Analysis, Quantitative Evaluation and Model Insights<br>Explore the model outputs, intermediate results and performance metrics.", pid),
            *ml_imgs,
            ml_metrics_html(rec, "ED"), ml_metrics_html(rec, "ES"),
            class_scores_html(rec), hist_figure(F["ED"]["prob"], "ED probability distribution"),
            hist_figure(F["ES"]["prob"], "ES probability distribution"), ml_quantitative_dashboard_html(rec),
            gt_html(rec, rec.get("gt_note", "")), gt_overlay(rec, "ED"), gt_overlay(rec, "ES"),
            report_txt, rec.get("notes", ""),
            ml_classification_details_html(rec),
        ]

    # ---------------------------------------------------------------- layout
    ui_kwargs = dict(css=CSS, theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate"), head=HEAD)
    blocks_params = inspect.signature(gr.Blocks.__init__).parameters
    blocks_kw = {k: v for k, v in ui_kwargs.items() if k in blocks_params}
    launch_kw = {k: v for k, v in ui_kwargs.items() if k not in blocks_params}
    if "title" in blocks_params:
        blocks_kw["title"] = APP_NAME

    with gr.Blocks(**blocks_kw) as demo:
        state = gr.State(None)

        with gr.Row(equal_height=False):
            # ------------------------------------------------ sidebar
            with gr.Column(scale=1, min_width=230, elem_id="sidebar"):
                gr.HTML(
                    '<div class="brand"><div class="logo">&#128154;</div><div><b>ASCEND EchoLV</b>'
                    "<span>AI-Assisted LV Analysis</span></div></div>", elem_classes="flat")
                nav = {p: gr.Button(NAV_LABELS[p], elem_classes="nav-btn", variant="primary" if p == "new" else "secondary")
                       for p in PAGES}
                gr.HTML('<div class="side-foot">Research Prototype<br>Non-Diagnostic<br>A4C Echocardiography</div>', elem_classes="flat")

            # ------------------------------------------------ main area
            with gr.Column(scale=5):
                # ============ NEW ANALYSIS
                with gr.Column(visible=True) as pg_new:
                    gr.HTML(page_head("New Analysis", "Upload a paired A4C end-diastolic (ED) and end-systolic (ES) frame."), elem_classes="flat")
                    with gr.Column(elem_classes="card"):
                        with gr.Row():
                            patient_in = gr.Textbox(label="Patient ID (de-identified)", placeholder="e.g. patient0173 - leave empty to infer from filename")
                            save_in = gr.Checkbox(label="Save this analysis to the patient record", value=True)
                        with gr.Row():
                            ed_in = gr.File(label="End-Diastolic (ED) NIfTI", file_types=[".nii", ".gz"], type="filepath")
                            es_in = gr.File(label="End-Systolic (ES) NIfTI", file_types=[".nii", ".gz"], type="filepath")
                        with gr.Accordion("Optional: expert ground-truth masks (enables Dice / IoU)", open=False):
                            with gr.Row():
                                ed_gt_in = gr.File(label="ED mask NIfTI (CAMUS, label 1 = LV)", file_types=[".nii", ".gz"], type="filepath")
                                es_gt_in = gr.File(label="ES mask NIfTI (CAMUS, label 1 = LV)", file_types=[".nii", ".gz"], type="filepath")
                        notes_in = gr.Textbox(label="Notes (optional, appear in the report)", lines=2)
                        with gr.Row():
                            run_btn = gr.Button("Run LV Analysis", variant="primary", scale=3)
                            demo_btn = gr.Button("Use demo patient", scale=1, visible=bool(demo_paths))
                    status = gr.HTML("", elem_classes="flat")

                # ============ CLINICAL VIEW
                with gr.Column(visible=False) as pg_clin:
                    clin_head = gr.HTML(elem_classes="flat")
                    with gr.Row():
                        with gr.Column(elem_classes="card", scale=3):
                            gr.HTML('<div class="section-title">End-Diastole (ED) &nbsp;|&nbsp; End-Systole (ES)</div>', elem_classes="flat")
                            with gr.Row(elem_id="clinical-images"):
                                ed_view = gr.Image(
                                    label="End-Diastole (ED)",
                                    interactive=False,
                                    height=300,
                                    show_label=False,
                                    container=False
                                )
                                es_view = gr.Image(
                                    label="End-Systole (ES)",
                                    interactive=False,
                                    height=300,
                                    show_label=False,
                                    container=False
                                )
                            gr.HTML(
                                '<div class="clinical-image-captions">'
                                '<span>End-Diastole (ED)</span>'
                                '<span>End-Systole (ES)</span>'
                                '</div>',
                                elem_classes="flat"
                            )
                            with gr.Row(elem_id="clinical-controls"):
                                mode_in = gr.Radio(
                                    ["Original", "Contour"],
                                    value="Contour",
                                    label="Image display"
                                )
                                opacity_in = gr.Slider(
                                    0, 100, value=70, step=5,
                                    label="Contour opacity (%)"
                                )
                        with gr.Column(elem_classes="card", scale=4):
                            kpi_out = gr.HTML(elem_classes="flat")
                            vol_out = gr.HTML(elem_classes="flat")

                    # Retained as hidden outputs for callback compatibility.
                    cls_out = gr.HTML(visible=False)
                    notice_out = gr.HTML(visible=False)

                    with gr.Column(elem_classes="card"):
                        details_out = gr.HTML(elem_classes="flat")

                # ============ ML RESEARCH VIEW
                with gr.Column(visible=False, elem_classes="ml-page") as pg_ml:
                    ml_head = gr.HTML(elem_classes="flat")
                    with gr.Tabs():

                        # ===== SEGMENTATION EXPLORER =====
                        with gr.Tab("Segmentation Explorer"):
                            gr.HTML(
                                '<div class="research-intro">'
                                '<h3 style="color:#fff !important;">'
                                'Frozen U-Net Segmentation Explorer</h3>'
                                '<p style="color:#b9d2e3 !important;">'
                                'Inspect ultrasound frames, contours, '
                                'binary masks and probability maps.</p>'
                                '</div>'
                            )

                            ml_imgs, ml_met = [], []

                            for ph in ("ED", "ES"):
                                with gr.Column(elem_classes="research-phase"):
                                    gr.HTML(
                                        '<div class="research-phase-title" '
                                        'style="color:#fff !important;">'
                                        + (
                                            "ED — End-Diastole"
                                            if ph == "ED"
                                            else "ES — End-Systole"
                                        )
                                        + '</div>'
                                    )

                                    with gr.Row(equal_height=False):
                                        for title in (
                                            "Original ultrasound",
                                            "Predicted LV contour",
                                            "Binary LV mask",
                                            "Probability map",
                                        ):
                                            with gr.Column(
                                                min_width=130,
                                                elem_classes="research-image-card"
                                            ):
                                                gr.HTML(
                                                    '<div style="color:#e8f4fc;'
                                                    'font-size:13px;'
                                                    'font-weight:700;'
                                                    'padding:4px 0 10px;">'
                                                    + title +
                                                    '</div>'
                                                )
                                                ml_imgs.append(
                                                    gr.Image(
                                                        interactive=False,
                                                        height=190,
                                                        container=False,
                                                        show_label=False
                                                    )
                                                )

                                    ml_met.append(
                                        gr.HTML(elem_classes="research-metrics")
                                    )

# ===== QUANTITATIVE ANALYSIS =====
                        with gr.Tab("Quantitative Analysis"):
                            gr.Markdown(
                                "### Quantitative LV Analysis\\n"
                                "Physical LV geometry, volume estimates and "
                                "pixel probability distributions."
                            )

                            allm_out = gr.HTML(elem_classes="flat")

                            with gr.Row():
                                hist_ed = gr.Image(interactive=False, height=290, show_label=False)
                                hist_es = gr.Image(interactive=False, height=290, show_label=False)

                        # ===== EF CLASSIFICATION =====
                        with gr.Tab("EF Classification"):
                            gr.Markdown(
                                "### Frozen EF Classification\\n"
                                "Compare geometry-derived and ML-predicted EF bands."
                            )

                            ml_cls_out = gr.HTML(elem_classes="flat")

                            # Retained for compatibility with the existing
                            # render() output mapping.
                            scores_out = gr.HTML(visible=False)

                        # ===== GROUND TRUTH =====
                        with gr.Tab("Ground Truth Comparison"):
                            gr.Markdown(
                                "### Expert Annotation Comparison\\n"
                                "Dice and IoU are available only when expert "
                                "ground-truth masks have been provided."
                            )

                            gt_out = gr.HTML(elem_classes="flat")

                            with gr.Row():
                                gt_ed = gr.Image(interactive=False, height=340, show_label=False)
                                gt_es = gr.Image(interactive=False, height=340, show_label=False)

                        # ===== MODEL INFORMATION =====
                        with gr.Tab("Model Information"):
                            gr.HTML(
                                model_info_html(frozen),
                                elem_classes="flat"
                            )

                # ============ CASE MANAGEMENT
                with gr.Column(visible=False) as pg_cases:
                    gr.HTML(page_head("Case Management", "Load, compare and track saved cases."), elem_classes="flat")
                    with gr.Column(elem_classes="card"):
                        with gr.Row():
                            filt = gr.Dropdown(["All patients"], value="All patients", label="Filter by patient")
                            refresh_btn = gr.Button("Refresh", scale=0)
                        table_out = gr.Dataframe(interactive=False, wrap=True, label="Saved cases")
                    with gr.Row():
                        with gr.Column(elem_classes="card"):
                            gr.HTML('<div class="section-title">Open or delete a case</div>', elem_classes="flat")
                            case_pick = gr.Dropdown([], label="Case")
                            with gr.Row():
                                open_btn = gr.Button("Open case", variant="primary")
                                del_btn = gr.Button("Delete case", variant="stop")
                            case_msg = gr.HTML(elem_classes="flat")
                        with gr.Column(elem_classes="card"):
                            gr.HTML('<div class="section-title">Compare two cases</div>', elem_classes="flat")
                            cmp_a = gr.Dropdown([], label="Case A (baseline)")
                            cmp_b = gr.Dropdown([], label="Case B (follow-up)")
                            cmp_btn = gr.Button("Compare")
                            cmp_out = gr.HTML(elem_classes="flat")
                    with gr.Column(elem_classes="card"):
                        gr.HTML('<div class="section-title">EF trend for the selected patient</div>', elem_classes="flat")
                        trend_out = gr.Image(label="EF over saved cases", interactive=False, height=320)

                # ============ DOWNLOAD REPORT
                with gr.Column(visible=False) as pg_report:
                    gr.HTML(page_head("Download Report", "Generate a PDF report for the current case, or export all cases."), elem_classes="flat")
                    with gr.Column(elem_classes="card"):
                        report_info = gr.HTML(elem_classes="flat")
                        report_notes = gr.Textbox(label="Notes to include in the report", lines=3)
                        include_res = gr.Checkbox(label="Include research appendix (probability maps, classifier scores, Dice/IoU, model info)", value=True)
                        with gr.Row():
                            pdf_btn = gr.Button("Generate PDF report", variant="primary")
                            csv_btn = gr.Button("Export all cases (CSV)")
                        with gr.Row():
                            pdf_file = gr.File(label="PDF report", interactive=False)
                            csv_file = gr.File(label="Case table (CSV)", interactive=False)

                # ============ ABOUT
                with gr.Column(visible=False) as pg_about:
                    gr.HTML(page_head("About", "Model, usage and limitations."), elem_classes="flat")
                    gr.HTML(ABOUT_HTML, elem_classes="flat")

        # ---------------------------------------------------------- wiring
        page_cols = {"new": pg_new, "clinical": pg_clin, "ml": pg_ml, "cases": pg_cases, "report": pg_report, "about": pg_about}
        nav_outputs = [page_cols[p] for p in PAGES] + [nav[p] for p in PAGES]

        def go(page):
            return [gr.update(visible=(p == page)) for p in PAGES] + [
                gr.update(variant="primary" if p == page else "secondary") for p in PAGES]

        view_outputs = [
            clin_head, ed_view, es_view, kpi_out, vol_out, cls_out, notice_out, details_out, ml_head,
            *ml_imgs, ml_met[0], ml_met[1], scores_out, hist_ed, hist_es, allm_out,
            gt_out, gt_ed, gt_es, report_info, report_notes,
            ml_cls_out,
        ]

        def finish(rec, extra_msg=""):
            saved = f"Saved as case <b>{esc(rec['case_id'])}</b>." if rec.get("case_id") else "Not saved to the patient record."
            msg = f'<div class="status">Analysis complete for <b>{esc(rec["patient_id"])}</b>. {saved} {esc(extra_msg)}</div>'
            return [rec] + render(rec) + [msg]

        def analyze(patient_id, ed, es, ed_gt, es_gt, notes, save, progress=gr.Progress()):
            ed, es = _path_of(ed), _path_of(es)
            if not ed or not es:
                raise gr.Error("Please upload both the ED and ES NIfTI files.")
            try:
                rec = run_pipeline(patient_id, ed, es, _path_of(ed_gt), _path_of(es_gt), notes, save, progress)
            except gr.Error:
                raise
            except Exception as exc:
                raise gr.Error(f"Analysis could not be completed: {exc}") from exc
            return finish(rec, rec.get("gt_note", ""))

        def analyze_demo(save, notes, progress=gr.Progress()):
            try:
                rec = run_pipeline("patient0173", str(demo_paths[0]), str(demo_paths[1]), None, None, notes, save, progress)
            except Exception as exc:
                raise gr.Error(f"Demo analysis failed: {exc}") from exc
            return finish(rec)

        run_outputs = [state] + view_outputs + [status]
        run_btn.click(analyze, [patient_in, ed_in, es_in, ed_gt_in, es_gt_in, notes_in, save_in], run_outputs).success(
            lambda: go("clinical"), None, nav_outputs)
        if demo_paths:
            demo_btn.click(analyze_demo, [save_in, notes_in], run_outputs).success(lambda: go("clinical"), None, nav_outputs)

        def update_views(rec, mode, opacity):
            if not rec:
                return None, None
            return compose_view(rec["frames"]["ED"], mode, opacity), compose_view(rec["frames"]["ES"], mode, opacity)

        mode_in.change(update_views, [state, mode_in, opacity_in], [ed_view, es_view])
        opacity_in.change(update_views, [state, mode_in, opacity_in], [ed_view, es_view])

        # ---- case management
        def refresh_cases(patient):
            p = None if (not patient or patient == "All patients") else patient
            ch = store.choices(p)
            return (
                store.table(p),
                gr.update(choices=["All patients"] + store.patients(), value=patient or "All patients"),
                gr.update(choices=ch, value=None), gr.update(choices=ch, value=None), gr.update(choices=ch, value=None),
                trend_figure(p),
            )

        def trend_figure(patient):
            if not patient:
                return None
            rows = list(reversed(store.rows(patient)))
            if len(rows) < 2:
                return None
            x = [datetime.fromisoformat(r["created_at"]) for r in rows]
            ef = [r["ef"] for r in rows]
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.axhspan(0, 40, color="#fde8e8", alpha=0.6)
            ax.axhspan(40, 50, color="#fff4dc", alpha=0.6)
            ax.axhspan(50, 100, color="#e6f6f1", alpha=0.6)
            ax.plot(range(len(x)), ef, marker="o", color="#0f9d8e", lw=2)
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels([d.strftime("%d %b %H:%M") for d in x], rotation=20, fontsize=7)
            ax.set_ylim(max(0, min(ef) - 10), min(100, max(ef) + 10))
            ax.set_ylabel("EF (%)")
            ax.set_title(f"{patient} - EF across saved cases (bands: <=40 / 40-50 / >=50)", fontsize=9)
            fig.patch.set_facecolor("white")
            return fig_to_np(fig)

        def on_filter(patient):
            p = None if (not patient or patient == "All patients") else patient
            ch = store.choices(p)
            return store.table(p), gr.update(choices=ch, value=None), gr.update(choices=ch, value=None), gr.update(choices=ch, value=None), trend_figure(p)

        case_outputs = [table_out, filt, case_pick, cmp_a, cmp_b, trend_out]
        refresh_btn.click(refresh_cases, [filt], case_outputs)
        filt.change(on_filter, [filt], [table_out, case_pick, cmp_a, cmp_b, trend_out])

        def open_case(cid):
            if not cid:
                raise gr.Error("Select a case first.")
            rec = store.get(cid)
            return finish(rec)

        open_btn.click(open_case, [case_pick], run_outputs).success(lambda: go("clinical"), None, nav_outputs)

        def delete_case(cid, patient):
            if not cid:
                raise gr.Error("Select a case first.")
            store.delete(cid)
            return (*refresh_cases(patient), f'<div class="status">Case {esc(cid)} deleted.</div>')

        del_btn.click(delete_case, [case_pick, filt], case_outputs + [case_msg])

        def compare(a, b):
            if not a or not b:
                raise gr.Error("Select both cases.")
            ra, rb = store.get(a), store.get(b)
            keys = [("EF_pct", "EF (%)"), ("EDV_ml", "EDV (mL)"), ("ESV_ml", "ESV (mL)"), ("SV_ml", "SV (mL)"),
                    ("ED_area_mm2", "ED area (mm2)"), ("ES_area_mm2", "ES area (mm2)")]
            rows = "".join(
                f"<tr><td>{lab}</td><td class='n'>{ra['measurements'][k]:.2f}</td><td class='n'>{rb['measurements'][k]:.2f}</td>"
                f"<td class='n'>{rb['measurements'][k] - ra['measurements'][k]:+.2f}</td></tr>" for k, lab in keys)
            head = f"<tr><th></th><th>A: {esc(ra['patient_id'])}<br>{esc(ra['created_at'][:16])}</th><th>B: {esc(rb['patient_id'])}<br>{esc(rb['created_at'][:16])}</th><th>B - A</th></tr>"
            note = '<div class="dsub" style="color:#587086">Differences are descriptive; single-plane estimates carry measurement uncertainty.</div>'
            return f'<table class="tbl">{head}{rows}</table>{note}'

        cmp_btn.click(compare, [cmp_a, cmp_b], [cmp_out])

        # ---- report
        def make_pdf(rec, notes, include):
            if not rec:
                raise gr.Error("Run or open a case first.")
            rec = dict(rec)
            if rec.get("case_id") and notes != rec.get("notes", ""):
                store.update_notes(rec["case_id"], notes)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = reports_dir / f"EchoLV_{rec['patient_id']}_{stamp}.pdf"
            return build_pdf(rec, path, include_research=include, notes=notes, model_info=info)

        pdf_btn.click(make_pdf, [state, report_notes, include_res], [pdf_file])
        csv_btn.click(lambda: store.export_csv(reports_dir / "EchoLV_all_cases.csv"), None, [csv_file])

        # ---- navigation
        for p in PAGES:
            ev = nav[p].click(lambda p=p: go(p), None, nav_outputs)
            if p == "cases":
                ev.then(refresh_cases, [filt], case_outputs)

        demo.load(refresh_cases, [filt], case_outputs)

    return demo, launch_kw

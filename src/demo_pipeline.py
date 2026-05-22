"""
Demo pipeline — runs the full SCAS compliance pipeline on synthetic video
using color-based detection (no trained models required).

Produces:
  - Annotated output video
  - SQLite violations database
  - JPEG evidence frames
"""

import os
import sys
import math
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── ensure deep_sort_realtime is importable ───────────────────────────────────
_CAPSTONE = Path(__file__).parent.parent
sys.path.insert(0, str(_CAPSTONE / "Lib" / "site-packages"))


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC VIDEO GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def create_synthetic_test_video(output_path: str, duration_sec: int = 20,
                                fps: int = 30):
    """
    Generates a synthetic traffic video with colored rectangle vehicles.
    Red = emergency vehicle, Green = non-emergency vehicles.
    """
    W, H = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    vehicles = [
        {"x": -120, "y": 300, "w": 120, "h": 80, "vx": 14, "vy": 0,  "emergency": True},
        {"x": 400,  "y": 290, "w": 100, "h": 70, "vx": 2,  "vy": 0,  "emergency": False},
        {"x": 700,  "y": 280, "w": 100, "h": 70, "vx": 3,  "vy": -5, "emergency": False},
        {"x": 200,  "y": 500, "w": 90,  "h": 65, "vx": 8,  "vy": 0,  "emergency": False},
        {"x": 900,  "y": 420, "w": 95,  "h": 68, "vx": 1,  "vy": 0,  "emergency": False},
    ]

    total_frames = duration_sec * fps
    for frame_idx in range(total_frames):
        frame = np.ones((H, W, 3), dtype=np.uint8) * 70
        cv2.line(frame, (0, H // 3),   (W, H // 3),   (200, 200, 200), 2)
        cv2.line(frame, (0, 2 * H // 3), (W, 2 * H // 3), (200, 200, 200), 2)

        for v in vehicles:
            v["x"] += v["vx"]
            v["y"] += v["vy"]
            if v["x"] > W + v["w"]:
                v["x"] = -v["w"]

            x, y, w, h = int(v["x"]), int(v["y"]), v["w"], v["h"]
            color = (0, 0, 220) if v["emergency"] else (0, 200, 50)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            label = "EV" if v["emergency"] else "VEH"
            cv2.putText(frame, label, (x + 4, y + h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        writer.write(frame)

    writer.release()
    print(f"[demo_pipeline] Synthetic video saved: {output_path} "
          f"({total_frames} frames @ {fps}fps)")


# ─────────────────────────────────────────────────────────────────────────────
# COMPLIANCE ENGINE  (same logic as emergency_vehicle_system.py)
# ─────────────────────────────────────────────────────────────────────────────

class ComplianceEngine:
    def __init__(self, proximity_r=250, reaction_w=20, yield_d=30):
        self.R = proximity_r
        self.W = reaction_w
        self.D = yield_d
        self.distance_history = defaultdict(list)
        self.already_flagged  = set()

    @staticmethod
    def _centre(x1, y1, x2, y2):
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _dist(cx1, cy1, cx2, cy2):
        return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def check(self, tracks, frame_id, frame):
        violations = []
        ev_tracks = [(tid, x1, y1, x2, y2)
                     for tid, x1, y1, x2, y2, label in tracks
                     if label == "emergency"]
        non_ev_tracks = [(tid, x1, y1, x2, y2)
                         for tid, x1, y1, x2, y2, label in tracks
                         if label == "non-emergency"]

        if not ev_tracks:
            self.distance_history.clear()
            self.already_flagged.clear()
            return violations

        for non_tid, nx1, ny1, nx2, ny2 in non_ev_tracks:
            ncx, ncy = self._centre(nx1, ny1, nx2, ny2)
            for ev_tid, ex1, ey1, ex2, ey2 in ev_tracks:
                ecx, ecy = self._centre(ex1, ey1, ex2, ey2)
                dist = self._dist(ncx, ncy, ecx, ecy)
                pair = (non_tid, ev_tid)

                if dist > self.R:
                    self.distance_history.pop(pair, None)
                    self.already_flagged.discard(pair)
                    continue

                hist = self.distance_history[pair]
                hist.append(dist)
                if len(hist) > self.W:
                    hist.pop(0)
                if len(hist) < self.W:
                    continue

                yield_amount = dist - hist[0]
                if yield_amount <= self.D:
                    if pair in self.already_flagged:
                        continue
                    severity = "Major" if yield_amount < 0 else "Minor"
                    violations.append({
                        "vehicle_id"  : non_tid,
                        "ev_id"       : ev_tid,
                        "frame_id"    : frame_id,
                        "distance_px" : round(dist, 2),
                        "yield_amount": round(yield_amount, 2),
                        "severity"    : severity,
                        "timestamp"   : datetime.now().isoformat(),
                        "frame"       : frame,
                        "bbox"        : (int(nx1), int(ny1), int(nx2), int(ny2)),
                    })
                    self.already_flagged.add(pair)
                else:
                    self.already_flagged.discard(pair)
        return violations


# ─────────────────────────────────────────────────────────────────────────────
# VIOLATION LOGGER  (same schema as emergency_vehicle_system.py)
# ─────────────────────────────────────────────────────────────────────────────

class ViolationLogger:
    def __init__(self, db_path, evidence_dir):
        self.db_path      = db_path
        self.evidence_dir = evidence_dir
        os.makedirs(evidence_dir, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS Vehicle (
            vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id   INTEGER NOT NULL UNIQUE,
            type TEXT, class TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS VideoFrame (
            frame_id  INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL, source TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS Violation (
            violation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id    INTEGER NOT NULL,
            frame_id      INTEGER NOT NULL,
            distance_px   REAL,
            severity      TEXT,
            evidence_path TEXT,
            timestamp     TEXT,
            FOREIGN KEY (vehicle_id) REFERENCES Vehicle(vehicle_id),
            FOREIGN KEY (frame_id)   REFERENCES VideoFrame(frame_id))""")
        self.conn.commit()

    def log(self, v):
        cur = self.conn.cursor()
        cur.execute("INSERT OR IGNORE INTO Vehicle (track_id, type, class) VALUES (?,?,?)",
                    (v["vehicle_id"], "non-emergency", "unknown"))
        cur.execute("SELECT vehicle_id FROM Vehicle WHERE track_id=?", (v["vehicle_id"],))
        vid = cur.fetchone()[0]
        cur.execute("INSERT OR IGNORE INTO VideoFrame (frame_id, timestamp, source) VALUES (?,?,?)",
                    (v["frame_id"], v["timestamp"], "synthetic_video"))
        path = self._save_evidence(v["frame"], v["bbox"], v["vehicle_id"], v["frame_id"])
        cur.execute("""INSERT INTO Violation
            (vehicle_id, frame_id, distance_px, severity, evidence_path, timestamp)
            VALUES (?,?,?,?,?,?)""",
            (vid, v["frame_id"], v["distance_px"], v["severity"], path, v["timestamp"]))
        self.conn.commit()

    def log_many(self, violations, frame):
        for v in violations:
            self.log(v)

    def _save_evidence(self, frame, bbox, vehicle_id, frame_id):
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        fname = f"viol_track{int(vehicle_id):03d}_frame{int(frame_id):05d}.jpg"
        fpath = os.path.join(self.evidence_dir, fname)
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imwrite(fpath, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return fpath

    def get_violation_count(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Violation")
        return cur.fetchone()[0]

    def close(self):
        self.conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: RUN DEMO PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_demo_pipeline(input_video: str, output_video: str,
                      db_path: str, evidence_dir: str,
                      proximity_r: int = 250, reaction_w: int = 20,
                      yield_d: int = 30,
                      progress_callback=None) -> int:
    """
    Runs the full compliance pipeline on the synthetic video using
    color-based detection (no trained models required).

    Returns the total number of violations logged.
    """
    import torch
    from deep_sort_realtime.deepsort_tracker import DeepSort

    tracker = DeepSort(
        max_age      = 30,
        embedder     = "mobilenet",
        half         = False,
        embedder_gpu = torch.cuda.is_available(),
    )
    label_history = defaultdict(list)

    compliance = ComplianceEngine(proximity_r, reaction_w, yield_d)

    if os.path.exists(db_path):
        os.remove(db_path)
    for f in Path(evidence_dir).glob("*.jpg"):
        f.unlink()
    logger = ViolationLogger(db_path, evidence_dir)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {input_video}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(output_video,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Color-based detection ─────────────────────────────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        r1  = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        r2  = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        red_mask   = cv2.bitwise_or(r1, r2)
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))

        raw = []
        for mask, lbl in [(red_mask, "emergency"), (green_mask, "non-emergency")]:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h > 1000:
                    conf = 0.9 if lbl == "emergency" else 0.85
                    raw.append([x, y, x + w, y + h, conf, lbl])

        # ── DeepSORT tracking ─────────────────────────────────────────────
        ds_input = [([x1, y1, x2-x1, y2-y1], conf, lbl)
                    for x1, y1, x2, y2, conf, lbl in raw]
        tracks_out = tracker.update_tracks(ds_input, frame=frame)

        tracks = []
        for tr in tracks_out:
            if not tr.is_confirmed():
                continue
            tid   = tr.track_id
            ltrb  = tr.to_ltrb()
            label = tr.get_det_class() or "unknown"
            label_history[tid].append(label)
            recent = label_history[tid][-15:]
            stable = max(set(recent), key=recent.count)
            tracks.append((tid, *ltrb, stable))

        # ── Compliance check ──────────────────────────────────────────────
        violations = compliance.check(tracks, frame_id, frame)
        logger.log_many(violations, frame)

        # ── Annotate frame ────────────────────────────────────────────────
        annotated = frame.copy()
        for tid, x1, y1, x2, y2, label in tracks:
            color = (0, 0, 220) if label == "emergency" \
                    else (0, 200, 50) if label == "non-emergency" \
                    else (200, 200, 50)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated, f"T{tid} {label[:3].upper()}",
                        (int(x1), max(int(y1) - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        for v in violations:
            bx1, by1, bx2, by2 = v["bbox"]
            cv2.putText(annotated, "VIOLATION", (bx1, by2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if any(l == "emergency" for _, *_, l in tracks):
            cv2.putText(annotated, "! EMERGENCY VEHICLE DETECTED !",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(annotated, f"Violations: {logger.get_violation_count()}",
                    (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        writer.write(annotated)

        if progress_callback and total > 0:
            progress_callback(frame_id / total)

        frame_id += 1

    cap.release()
    writer.release()
    vcount = logger.get_violation_count()
    logger.close()

    print(f"[demo_pipeline] Done — {frame_id} frames, {vcount} violations")
    return vcount


# ─────────────────────────────────────────────────────────────────────────────
# DB HELPERS  (used by Streamlit pages)
# ─────────────────────────────────────────────────────────────────────────────

def read_violations(db_path: str):
    """Returns DataFrame of all violations from the SQLite DB."""
    import pandas as pd
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("""
            SELECT v.violation_id, v.timestamp, vh.track_id,
                   v.distance_px, v.severity, v.evidence_path, v.frame_id
            FROM Violation v
            JOIN Vehicle vh ON v.vehicle_id = vh.vehicle_id
            ORDER BY v.frame_id ASC
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def read_summary(db_path: str) -> dict:
    """Returns a summary dict: total, major, minor, unique_vehicles."""
    if not os.path.exists(db_path):
        return {"total": 0, "major": 0, "minor": 0, "unique_vehicles": 0}
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Violation")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM Violation WHERE severity='Major'")
        major = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM Violation WHERE severity='Minor'")
        minor = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT vehicle_id) FROM Violation")
        uveh  = cur.fetchone()[0]
        conn.close()
        return {"total": total, "major": major, "minor": minor,
                "unique_vehicles": uveh}
    except Exception:
        return {"total": 0, "major": 0, "minor": 0, "unique_vehicles": 0}

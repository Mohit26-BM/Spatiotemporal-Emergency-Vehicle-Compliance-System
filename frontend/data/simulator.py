import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# ── Seed for reproducibility ──
random.seed(42)
np.random.seed(42)

ZONES = ["Zone A – Downtown Core", "Zone B – Midtown Grid", "Zone C – Highway Corridor",
         "Zone D – Residential North", "Zone E – Industrial South", "Zone F – Airport Sector"]

VEHICLE_TYPES = {
    "AMB": ("🚑", "Ambulance"),
    "FIR": ("🚒", "Fire Truck"),
    "POL": ("🚓", "Police Car"),
    "HZM": ("🚐", "Hazmat Unit"),
    "MED": ("🚁", "Medevac"),
}

INTERSECTIONS = [
    "Main St & 5th Ave", "Oak Blvd & Highway 101", "Central Park West & 72nd",
    "Industrial Rd & Route 9", "Airport Connector & Terminal Blvd",
    "Riverside Dr & Bridge Rd", "Downtown Loop & Commerce St",
    "North Ave & Medical Center Dr", "South Corridor & Freight Terminal",
    "University Blvd & Research Pkwy"
]

def generate_vehicle_data(n=18):
    vehicles = []
    for i in range(n):
        v_type = random.choice(list(VEHICLE_TYPES.keys()))
        icon, name = VEHICLE_TYPES[v_type]
        compliance = round(random.gauss(91, 8), 1)
        compliance = max(60, min(100, compliance))
        status = "ACTIVE" if random.random() > 0.25 else ("VIOLATION" if random.random() > 0.6 else "IDLE")
        vehicles.append({
            "id": f"{v_type}-{100 + i}",
            "type": v_type,
            "icon": icon,
            "name": name,
            "zone": random.choice(ZONES),
            "intersection": random.choice(INTERSECTIONS),
            "compliance_pct": compliance,
            "status": status,
            "speed_kmh": round(random.uniform(40, 110), 1),
            "response_time_s": round(random.uniform(8, 45), 1),
            "priority": random.choice(["P1", "P2", "P3"]),
            "last_seen": datetime.now() - timedelta(seconds=random.randint(5, 300)),
            "lat": round(random.uniform(28.5, 28.8), 5),
            "lon": round(random.uniform(77.0, 77.4), 5),
        })
    return pd.DataFrame(vehicles)

def generate_compliance_timeseries(days=30):
    dates = [datetime.now() - timedelta(days=d) for d in range(days, 0, -1)]
    compliance = []
    val = 90
    for _ in dates:
        val += random.gauss(0, 1.5)
        val = max(75, min(99, val))
        compliance.append(round(val, 2))
    violations = [max(0, int(random.gauss(8, 4))) for _ in dates]
    incidents = [max(0, int(random.gauss(3, 2))) for _ in dates]
    return pd.DataFrame({
        "date": dates,
        "compliance_pct": compliance,
        "violations": violations,
        "incidents": incidents,
    })

def generate_zone_compliance():
    data = []
    for zone in ZONES:
        data.append({
            "zone": zone,
            "compliance": round(random.uniform(78, 98), 1),
            "violations_today": random.randint(0, 8),
            "vehicles": random.randint(2, 7),
            "avg_response": round(random.uniform(10, 35), 1),
        })
    return pd.DataFrame(data)

def generate_alerts(n=10):
    severities = ["critical", "critical", "warning", "warning", "warning", "info"]
    alert_types = [
        ("Signal Non-Compliance", "Vehicle failed to clear intersection within threshold"),
        ("Wrong-Way Incident", "Vehicle detected on incorrect lane during priority corridor"),
        ("Delayed Yield", "Civilian vehicle delayed yield by >3 seconds"),
        ("Sensor Anomaly", "GPS drift detected — position accuracy degraded"),
        ("Speed Threshold Breach", "Emergency vehicle exceeded safe priority speed"),
        ("Corridor Blocked", "Priority corridor obstruction detected"),
        ("Signal Override Failure", "Traffic signal preemption system fault"),
        ("Zone Entry Unauthorized", "Non-authorized entry into priority zone"),
    ]
    alerts = []
    for i in range(n):
        atype, desc = random.choice(alert_types)
        sev = random.choice(severities)
        v_type = random.choice(list(VEHICLE_TYPES.keys()))
        icon, vname = VEHICLE_TYPES[v_type]
        alerts.append({
            "id": f"ALT-{1000+i}",
            "severity": sev,
            "type": atype,
            "description": desc,
            "vehicle": f"{v_type}-{100 + random.randint(0, 17)}",
            "intersection": random.choice(INTERSECTIONS),
            "time": datetime.now() - timedelta(minutes=random.randint(1, 240)),
        })
    alerts.sort(key=lambda x: x["time"], reverse=True)
    return alerts

def generate_heatmap_data():
    hours = list(range(24))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    data = []
    for d in days:
        row = []
        for h in hours:
            # Rush hours get more violations
            base = 2
            if h in [7, 8, 9, 17, 18, 19]:
                base = 12
            elif h in [10, 11, 12, 13, 14, 15, 16]:
                base = 6
            row.append(max(0, int(random.gauss(base, base * 0.5))))
        data.append(row)
    return days, hours, data

def generate_incident_log(n=30):
    rows = []
    for i in range(n):
        v_type = random.choice(list(VEHICLE_TYPES.keys()))
        icon, vname = VEHICLE_TYPES[v_type]
        comp = round(random.uniform(60, 100), 1)
        severity = "Critical" if comp < 75 else ("Warning" if comp < 88 else "Pass")
        rows.append({
            "Timestamp": (datetime.now() - timedelta(hours=random.randint(0, 72))).strftime("%Y-%m-%d %H:%M:%S"),
            "Vehicle ID": f"{v_type}-{100 + random.randint(0, 17)}",
            "Type": vname,
            "Zone": random.choice(ZONES),
            "Intersection": random.choice(INTERSECTIONS),
            "Compliance (%)": comp,
            "Speed (km/h)": round(random.uniform(40, 110), 1),
            "Response Time (s)": round(random.uniform(8, 45), 1),
            "Status": severity,
            "Priority": random.choice(["P1", "P2", "P3"]),
        })
    return pd.DataFrame(rows).sort_values("Timestamp", ascending=False)

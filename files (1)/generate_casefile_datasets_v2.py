"""
CASEFILE — Linked Synthetic Dataset Generator (v2: 3-table schema)
=====================================================================
*** ALL DATA GENERATED HERE IS 100% SYNTHETIC / FICTIONAL. ***
No real missing-person records, real identities, or real GPS logs are
used. Coordinates are simulation-only, centered on an artificial point.

Produces THREE linked, reproducible CSV files:

  1. data/synthetic_v2/gps_trajectory_raw.csv     (raw, has missing values)
  2. data/synthetic_v2/geographical_poi.csv       (fictional POI reference)
  3. data/synthetic_v2/synthetic_case_data.csv    (fictional case records)

Run:
    python src/generate_casefile_datasets_v2.py
"""

import os
import math
import json
import datetime as dt

import numpy as np
import pandas as pd

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

OUT_DIR = os.path.join("data", "synthetic_v2")
os.makedirs(OUT_DIR, exist_ok=True)

STUDY_CENTER_LAT = 23.25
STUDY_CENTER_LON = 77.41

# =============================================================================
# SHARED GEOGRAPHY (same fictional region used across all 3 tables so they
# can be joined on Area_ID / Area_Name)
# =============================================================================

AREA_NAMES = [f"Area_{c}" for c in "ABCDEFGHIJ"]
AREA_IDS = [f"AR{str(i+1).zfill(2)}" for i in range(len(AREA_NAMES))]
AREA_ID_OF = dict(zip(AREA_NAMES, AREA_IDS))
N_AREAS = len(AREA_NAMES)

AREA_BASE_PROB = np.array([0.18, 0.16, 0.13, 0.12, 0.11,
                            0.10, 0.08, 0.07, 0.03, 0.02])

AREA_TYPE_MAP = {
    "Area_A": "Residential", "Area_B": "Commercial", "Area_C": "Educational",
    "Area_D": "Transportation", "Area_E": "Mixed", "Area_F": "Healthcare",
    "Area_G": "Recreational", "Area_H": "Industrial", "Area_I": "Residential",
    "Area_J": "Mixed",
}


def build_area_centers():
    step = 0.07
    rows = {0: step, 1: 0.0, 2: -step}
    cols = {0: -step, 1: 0.0, 2: step}
    layout = {
        "Area_A": (0, 0), "Area_B": (0, 1), "Area_C": (0, 2),
        "Area_D": (1, 0), "Area_E": (1, 1), "Area_F": (1, 2),
        "Area_G": (2, 0), "Area_H": (2, 1), "Area_I": (2, 2),
    }
    centers = {a: (round(STUDY_CENTER_LAT + rows[r], 6), round(STUDY_CENTER_LON + cols[c], 6))
               for a, (r, c) in layout.items()}
    centers["Area_J"] = (round(STUDY_CENTER_LAT - 0.18, 6), round(STUDY_CENTER_LON + 0.20, 6))
    return centers


AREA_CENTERS = build_area_centers()


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_distance_matrix():
    dist = np.zeros((N_AREAS, N_AREAS))
    for i, a in enumerate(AREA_NAMES):
        for j, b in enumerate(AREA_NAMES):
            if i != j:
                lat1, lon1 = AREA_CENTERS[a]
                lat2, lon2 = AREA_CENTERS[b]
                dist[i, j] = haversine_km(lat1, lon1, lat2, lon2)
    return dist


DIST_MATRIX = build_distance_matrix()
TRANSITION_SCALE_KM = 9.0
SELF_TRANSITION_WEIGHT = 0.35


def build_transition_matrix():
    T = np.zeros((N_AREAS, N_AREAS))
    for i in range(N_AREAS):
        w = np.exp(-DIST_MATRIX[i] / TRANSITION_SCALE_KM)
        w[i] = SELF_TRANSITION_WEIGHT * w.max()
        T[i] = w / w.sum()
    return T


TRANSITION_MATRIX = build_transition_matrix()
AREA_IDX = {a: i for i, a in enumerate(AREA_NAMES)}

AREA_PEAK_HOUR = {a: int(rng.integers(6, 22)) for a in AREA_NAMES}
HOUR_WEIGHTS = np.array([1, 1, 1, 1, 1, 2, 4, 7, 8, 6, 5, 5,
                          6, 6, 5, 5, 6, 8, 9, 7, 5, 3, 2, 1], dtype=float)
HOUR_PROB = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()
WEATHERS = ["Clear", "Cloudy", "Rain", "Fog"]
WEATHER_PROB = np.array([0.55, 0.25, 0.15, 0.05])
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def hour_affinity(area, hour):
    peak = AREA_PEAK_HOUR[area]
    diff = min(abs(hour - peak), 24 - abs(hour - peak))
    return math.exp(-(diff ** 2) / (2 * 3.5 ** 2))


def softmax(x):
    z = np.asarray(x, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# =============================================================================
# DATASET 2 — geographical_poi.csv
# =============================================================================

POI_TYPES = ["Road", "Intersection", "Railway Station", "Bus Stop", "Hospital",
             "Market", "School", "Park", "Residential Area", "Shopping Area",
             "Office Area", "Restaurant", "College", "Metro Station", "Public Place"]
POI_BASE_WEIGHTS = np.array([0.12, 0.08, 0.02, 0.09, 0.04, 0.07, 0.06, 0.07,
                              0.12, 0.08, 0.07, 0.09, 0.03, 0.02, 0.04])

# Which POI types get boosted per Area_Type, to keep POIs consistent with
# the area's declared character (e.g. Healthcare areas get more hospitals).
TYPE_BOOST = {
    "Residential": ["Residential Area", "Park", "School", "Bus Stop"],
    "Commercial": ["Shopping Area", "Market", "Office Area", "Restaurant"],
    "Educational": ["School", "College", "Public Place"],
    "Transportation": ["Bus Stop", "Railway Station", "Metro Station", "Road", "Intersection"],
    "Healthcare": ["Hospital", "Public Place"],
    "Recreational": ["Park", "Public Place", "Restaurant"],
    "Industrial": ["Office Area", "Road"],
    "Mixed": [],
}

NAME_PREFIXES = ["North", "South", "East", "West", "Central", "New", "Old",
                  "Green", "Lake", "Hill", "River", "Sunrise", "Silver",
                  "Golden", "Union", "Palm", "Star", "Garden"]
ROAD_TYPES = ["Road", "Street", "Avenue", "Lane", "Highway", "Boulevard"]


def area_poi_weights(area_type):
    w = POI_BASE_WEIGHTS.copy()
    boosted = TYPE_BOOST.get(area_type, [])
    for t in boosted:
        idx = POI_TYPES.index(t)
        w[idx] *= 3.0
    return w / w.sum()


def generate_poi_dataset():
    rows = []
    for area in AREA_NAMES:
        area_id = AREA_ID_OF[area]
        area_type = AREA_TYPE_MAP[area]
        c_lat, c_lon = AREA_CENTERS[area]
        n_poi = int(rng.integers(15, 31))
        weights = area_poi_weights(area_type)

        # Small fictional road pool for this area, reused across its POIs
        n_roads = int(rng.integers(4, 8))
        road_pool = [
            f"{rng.choice(NAME_PREFIXES)} {rng.choice(ROAD_TYPES)} {int(rng.integers(1, 40))}"
            for _ in range(n_roads)
        ]

        for k in range(n_poi):
            poi_type = rng.choice(POI_TYPES, p=weights)
            lat = round(c_lat + rng.normal(0, 0.010), 6)
            lon = round(c_lon + rng.normal(0, 0.010), 6)
            poi_id = f"POI-{area_id}-{k+1:03d}"
            poi_name = f"{rng.choice(NAME_PREFIXES)} {poi_type} {k+1}"
            road_name = rng.choice(road_pool)
            other_roads = [r for r in road_pool if r != road_name]
            nearby_road = rng.choice(other_roads) if other_roads else road_name
            dist_center = round(haversine_km(lat, lon, STUDY_CENTER_LAT, STUDY_CENTER_LON), 2)

            rows.append({
                "Area_ID": area_id, "Area_Name": area, "Latitude": lat, "Longitude": lon,
                "POI_ID": poi_id, "POI_Type": poi_type, "POI_Name": poi_name,
                "Road_Name": road_name, "Nearby_Road": nearby_road,
                "Distance_From_City_Center": dist_center, "Area_Type": area_type,
            })

    return pd.DataFrame(rows)


# =============================================================================
# DATASET 1 — gps_trajectory_raw.csv
# =============================================================================

N_USERS_TOTAL = 1200
N_OVERLAP_WITH_CASES = 700      # users who also appear as case Person_IDs
N_CASE_PERSONS = 5000            # must match generate_case_dataset()

ACTIVITY_BY_AREA_TYPE = {
    "Residential": [("Social", 0.35), ("Unknown", 0.25), ("Leisure", 0.20), ("Commute", 0.20)],
    "Commercial": [("Shopping", 0.35), ("Work", 0.30), ("Commute", 0.20), ("Leisure", 0.15)],
    "Educational": [("Education", 0.55), ("Commute", 0.25), ("Social", 0.20)],
    "Transportation": [("Commute", 0.70), ("Unknown", 0.30)],
    "Healthcare": [("Medical", 0.60), ("Commute", 0.25), ("Unknown", 0.15)],
    "Recreational": [("Leisure", 0.55), ("Social", 0.25), ("Unknown", 0.20)],
    "Industrial": [("Work", 0.65), ("Commute", 0.35)],
    "Mixed": [("Work", 0.25), ("Shopping", 0.20), ("Commute", 0.25), ("Leisure", 0.15), ("Social", 0.15)],
}

TRAJ_DATE_START = dt.date(2025, 11, 1)
TRAJ_DATE_END = dt.date(2026, 2, 28)
TRAJ_DATE_RANGE_DAYS = (TRAJ_DATE_END - TRAJ_DATE_START).days


def sample_regime_and_speed(rush_hour_bias):
    if rush_hour_bias:
        regime_probs = [0.15, 0.20, 0.65]
    else:
        regime_probs = [0.35, 0.30, 0.35]
    regime = rng.choice(["walk", "cycle", "vehicle"], p=regime_probs)
    if regime == "walk":
        speed = rng.uniform(1, 8)
        mode = "Walking"
    elif regime == "cycle":
        speed = rng.uniform(8, 25)
        mode = "Bicycle"
    else:
        speed = rng.uniform(20, 80)
        mode = rng.choice(["Car", "Bus", "Motorbike", "Train"], p=[0.45, 0.30, 0.20, 0.05])
    return float(np.clip(speed, 1.0, 80.0)), mode


def generate_trajectory_dataset():
    case_person_ids = [f"P{i+1:04d}" for i in range(N_CASE_PERSONS)]
    overlap_ids = list(rng.choice(case_person_ids, size=N_OVERLAP_WITH_CASES, replace=False))
    poponly_ids = [f"P{N_CASE_PERSONS + i + 1:04d}" for i in range(N_USERS_TOTAL - N_OVERLAP_WITH_CASES)]
    user_ids = overlap_ids + poponly_ids
    rng.shuffle(user_ids)

    rows = []
    for user_id in user_ids:
        home_idx = rng.choice(N_AREAS, p=AREA_BASE_PROB)
        home_area = AREA_NAMES[home_idx]
        work_probs = TRANSITION_MATRIX[home_idx]
        work_area = rng.choice(AREA_NAMES, p=work_probs)

        n_points = int(rng.integers(8, 46))
        for _ in range(n_points):
            offset = int(rng.integers(0, TRAJ_DATE_RANGE_DAYS + 1))
            date_obj = TRAJ_DATE_START + dt.timedelta(days=offset)
            weekday_num = date_obj.weekday()
            is_weekday = weekday_num < 5

            hour = int(rng.choice(np.arange(24), p=HOUR_PROB))
            minute = int(rng.integers(0, 60))

            rush_hour = is_weekday and (7 <= hour <= 10 or 17 <= hour <= 20)
            if is_weekday and 8 <= hour <= 18:
                area = rng.choice([work_area, home_area, "other"], p=[0.75, 0.15, 0.10])
            else:
                area = rng.choice([home_area, work_area, "other"], p=[0.65, 0.10, 0.25])
            if area == "other":
                area = rng.choice(AREA_NAMES, p=TRANSITION_MATRIX[home_idx])

            area_type = AREA_TYPE_MAP[area]
            c_lat, c_lon = AREA_CENTERS[area]
            lat = round(c_lat + rng.normal(0, 0.012), 6)
            lon = round(c_lon + rng.normal(0, 0.012), 6)

            speed, transport_mode = sample_regime_and_speed(rush_hour)
            duration = float(np.clip(rng.lognormal(mean=math.log(20), sigma=0.6), 2, 180))
            distance = float(np.clip(speed * (duration / 60.0) * rng.uniform(0.7, 1.1), 0.1, 50))

            acts, act_w = zip(*ACTIVITY_BY_AREA_TYPE[area_type])
            activity = rng.choice(acts, p=np.array(act_w) / sum(act_w))
            weather = rng.choice(WEATHERS, p=WEATHER_PROB)

            rows.append({
                "User_ID": user_id,
                "Latitude": lat, "Longitude": lon,
                "Date": date_obj.strftime("%Y-%m-%d"),
                "Time": f"{hour:02d}:{minute:02d}",
                "Movement_Duration": round(duration, 2),
                "Distance": round(distance, 2),
                "Speed": round(speed, 2),
                "Area_ID": AREA_ID_OF[area], "Area_Name": area,
                "Transport_Mode": transport_mode, "Activity_Type": activity,
                "Weather": weather, "Day": DAYS[weekday_num], "Hour": hour,
                "Weekday": weekday_num,
                "Weekend_Weekday": "Weekend" if weekday_num >= 5 else "Weekday",
            })

    df = pd.DataFrame(rows)

    # ---- Intentional RAW-data missingness (sensor gaps / unlogged fields) ---
    n = len(df)

    def blank(col, frac):
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan

    blank("Weather", 0.08)
    blank("Transport_Mode", 0.06)
    blank("Activity_Type", 0.10)
    blank("Movement_Duration", 0.03)
    blank("Distance", 0.02)
    blank("Area_Name", 0.02)   # Area_ID retained even when name lookup failed

    return df, overlap_ids, poponly_ids


# =============================================================================
# DATASET 3 — synthetic_case_data.csv
# =============================================================================

N_CASES = N_CASE_PERSONS
AGE_GROUPS = ["18-25", "26-35", "36-45", "46-55", "56+"]
AGE_GROUP_PROB = np.array([0.20, 0.30, 0.25, 0.15, 0.10])
GENDERS = ["Male", "Female", "Other"]
GENDER_PROB = np.array([0.47, 0.47, 0.06])
DAY_PROB = np.array([0.145, 0.145, 0.145, 0.145, 0.145, 0.1375, 0.1375])
AGE_DISTANCE_FACTOR = {"18-25": 1.15, "26-35": 1.08, "36-45": 1.00, "46-55": 0.90, "56+": 0.80}
AREA_DISTANCE_FACTOR = {a: round(rng.uniform(0.85, 1.15), 3) for a in AREA_NAMES}
EPS = 1e-6

TARGET_WEIGHTS = {
    "w_prev_transition": 1.8, "w_current_transition": 1.5, "w_usual_match": 1.0,
    "w_hist_visit_bonus": 1.2, "w_hour_affinity": 1.5, "w_weather_affinity": 0.5,
    "w_area_popularity": 1.0, "w_distance_penalty": 2.0, "noise_std": 1.5,
}
WEATHER_AFFINITY = rng.uniform(-0.3, 0.3, size=(N_AREAS, len(WEATHERS)))


def generate_case_dataset():
    case_ids = [f"MP-2026-{i+1:05d}" for i in range(N_CASES)]
    person_ids = [f"P{i+1:04d}" for i in range(N_CASES)]
    age_groups = rng.choice(AGE_GROUPS, size=N_CASES, p=AGE_GROUP_PROB)
    genders = rng.choice(GENDERS, size=N_CASES, p=GENDER_PROB)
    days = rng.choice(DAYS, size=N_CASES, p=DAY_PROB)
    hours = rng.choice(np.arange(24), size=N_CASES, p=HOUR_PROB)
    minutes = rng.integers(0, 60, size=N_CASES)
    weathers = rng.choice(WEATHERS, size=N_CASES, p=WEATHER_PROB)
    usual_areas = rng.choice(AREA_NAMES, size=N_CASES, p=AREA_BASE_PROB)

    prev_areas, last_lats, last_lons, target_areas = [], [], [], []
    avg_distances, avg_speeds, tsls_list = [], [], []

    for i in range(N_CASES):
        usual_area = usual_areas[i]
        usual_idx = AREA_IDX[usual_area]
        hour = int(hours[i])
        weather_idx = WEATHERS.index(weathers[i])
        age_group = age_groups[i]

        prev_area = rng.choice(AREA_NAMES, p=TRANSITION_MATRIX[usual_idx])
        prev_idx = AREA_IDX[prev_area]
        current_area = rng.choice(AREA_NAMES, p=TRANSITION_MATRIX[prev_idx])
        current_idx = AREA_IDX[current_area]

        c_lat, c_lon = AREA_CENTERS[current_area]
        lat = float(np.clip(c_lat + rng.normal(0, 0.009), -90, 90))
        lon = float(np.clip(c_lon + rng.normal(0, 0.009), -180, 180))

        freq = float(np.clip(AREA_BASE_PROB[usual_idx] / AREA_BASE_PROB.max() + rng.normal(0, 0.05), 0.01, 1.0))

        base_dist = rng.lognormal(mean=math.log(9.0), sigma=0.5)
        dist = float(np.clip(base_dist * AGE_DISTANCE_FACTOR[age_group] * AREA_DISTANCE_FACTOR[usual_area], 1.0, 30.0))

        if dist < 4:
            regime_probs = [0.65, 0.25, 0.10]
        elif dist < 12:
            regime_probs = [0.20, 0.40, 0.40]
        else:
            regime_probs = [0.05, 0.20, 0.75]
        regime = rng.choice(["walk", "cycle", "vehicle"], p=regime_probs)
        speed = rng.uniform(1, 8) if regime == "walk" else (rng.uniform(8, 25) if regime == "cycle" else rng.uniform(20, 80))
        speed = float(np.clip(speed, 1.0, 80.0))

        tsls = float(np.clip(rng.exponential(scale=4.0), 0.1, 24.0))

        logits = np.zeros(N_AREAS)
        for k, cand in enumerate(AREA_NAMES):
            logit = 0.0
            logit += TARGET_WEIGHTS["w_prev_transition"] * math.log(TRANSITION_MATRIX[prev_idx, k] + EPS)
            logit += TARGET_WEIGHTS["w_current_transition"] * math.log(TRANSITION_MATRIX[current_idx, k] + EPS)
            if cand == usual_area:
                logit += TARGET_WEIGHTS["w_usual_match"] + TARGET_WEIGHTS["w_hist_visit_bonus"] * freq
            logit += TARGET_WEIGHTS["w_hour_affinity"] * hour_affinity(cand, hour)
            logit += TARGET_WEIGHTS["w_weather_affinity"] * WEATHER_AFFINITY[k, weather_idx]
            logit += TARGET_WEIGHTS["w_area_popularity"] * AREA_BASE_PROB[k]
            norm_penalty = DIST_MATRIX[current_idx, k] / (dist + 1.0)
            logit -= TARGET_WEIGHTS["w_distance_penalty"] * (norm_penalty / 10.0)
            logits[k] = logit
        logits += rng.normal(0, TARGET_WEIGHTS["noise_std"], size=N_AREAS)
        target_area = rng.choice(AREA_NAMES, p=softmax(logits))

        prev_areas.append(prev_area)
        last_lats.append(round(lat, 6))
        last_lons.append(round(lon, 6))
        target_areas.append(target_area)
        avg_distances.append(round(dist, 2))
        avg_speeds.append(round(speed, 2))
        tsls_list.append(round(tsls, 2))

    last_seen_time = [f"{h:02d}:{m:02d}" for h, m in zip(hours, minutes)]

    df = pd.DataFrame({
        "Case_ID": case_ids, "Person_ID": person_ids, "Age_Group": age_groups,
        "Gender": genders, "Last_Latitude": last_lats, "Last_Longitude": last_lons,
        "Last_Seen_Time": last_seen_time, "Day": days, "Weather": weathers,
        "Usual_Area": usual_areas, "Average_Distance": avg_distances,
        "Average_Speed": avg_speeds, "Previous_Area": prev_areas,
        "Time_Since_Last_Seen": tsls_list, "Target_Area": target_areas,
    })
    return df


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("Generating DATASET 2: geographical_poi.csv ...")
    poi_df = generate_poi_dataset()
    poi_path = os.path.join(OUT_DIR, "geographical_poi.csv")
    poi_df.to_csv(poi_path, index=False)
    print(f"  -> {poi_path}  ({len(poi_df)} rows, {poi_df.shape[1]} cols)")

    print("Generating DATASET 1: gps_trajectory_raw.csv ...")
    traj_df, overlap_ids, poponly_ids = generate_trajectory_dataset()
    traj_path = os.path.join(OUT_DIR, "gps_trajectory_raw.csv")
    traj_df.to_csv(traj_path, index=False)
    print(f"  -> {traj_path}  ({len(traj_df)} rows, {traj_df.shape[1]} cols)")
    print(f"  -> unique users: {traj_df['User_ID'].nunique()} "
          f"({len(overlap_ids)} overlap with case Person_IDs, {len(poponly_ids)} population-only)")
    print("  -> missing values per column:")
    print(traj_df.isna().sum()[traj_df.isna().sum() > 0])

    print("Generating DATASET 3: synthetic_case_data.csv ...")
    case_df = generate_case_dataset()
    case_path = os.path.join(OUT_DIR, "synthetic_case_data.csv")
    case_df.to_csv(case_path, index=False)
    print(f"  -> {case_path}  ({len(case_df)} rows, {case_df.shape[1]} cols)")
    print(f"  -> duplicate Case_ID: {case_df['Case_ID'].duplicated().sum()}, "
          f"duplicate Person_ID: {case_df['Person_ID'].duplicated().sum()}")
    print(f"  -> Target_Area distribution:\n{case_df['Target_Area'].value_counts(normalize=True).round(3)}")

    linkable = case_df["Person_ID"].isin(overlap_ids).sum()
    print(f"\nLINKAGE SUMMARY: {linkable}/{len(case_df)} case Person_IDs "
          f"({linkable/len(case_df):.1%}) have prior GPS history in gps_trajectory_raw.csv")

    print("\nAll three datasets are fully synthetic and intended only for academic experimentation.")


if __name__ == "__main__":
    main()

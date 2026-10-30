# Step 2: find the Sentinel-2 images of every TSS record (Google Earth Engine).
#
# Input : data/step1/WQP_TSS_filtered.parquet
# Output: data/step2/matchups.parquet  (one row per record and image)
#
# Images: COPERNICUS/S2_SR_HARMONIZED, taken on the day before, the same day or the
# day after sampling. Cloud Score+ (cs_cdf) is joined to every image.
# All bands, the SCL and Cloud Score+ are put on the 10 m grid of B2 and a 4 x 4
# pixel window (40 m) is read around each station (the station is pixel 6).
# Answers are cached in data/step2/cache, so the script can be restarted.
#
# Needs an Earth Engine account:  earthengine authenticate

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import ee
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_FILE = ROOT / "data" / "step1" / "WQP_TSS_filtered.parquet"
CACHE = ROOT / "data" / "step2" / "cache"
OUT_FILE = ROOT / "data" / "step2" / "matchups.parquet"

EE_PROJECT = "your-ee-project"      # write your own Earth Engine (Google Cloud) project here
WINDOW_DAYS = 1
WINDOW_PX = 4
MAX_POINTS = 400                    # points per request (Earth Engine answers are limited to 5,000 features)
FILL = -9999.0
DAY_MS = 86_400_000
BANDS_10M = ["B2", "B3", "B4", "B8"]
BANDS_20M = ["B5", "B6", "B7", "B8A", "B11", "B12"]
BANDS_60M = ["B1", "B9"]
LAYERS = BANDS_10M + BANDS_20M + ["SCL", "CS"]
IMAGE_PROPS = ["Granule_ID", "MGRS_TILE", "Spacecraft", "Cloud_Coverage", "SZA", "SAA", "VZA", "VAA"]
POINT_COLUMNS = ["RowID", "StationID", "Latitude", "Longitude", "LocalDate", "DateTimeUTC_ForMatch",
                 "Time_Source", "AnalyteClass", "ResultValue_mgL", "Depth_m", "LocationType", "HUC8",
                 "OrgID", "Provider"]


def window_values(image, kernel):
    """All layers on the 10 m grid of B2, as 4 x 4 arrays around each pixel."""
    grid = image.select("B2").projection()
    stack = image.select(BANDS_10M).toFloat().divide(10000)
    stack = stack.addBands(image.select(BANDS_20M).toFloat().divide(10000).reproject(grid))
    stack = stack.addBands(image.select(BANDS_60M).toFloat().divide(10000).reproject(grid))
    stack = stack.addBands(image.select("SCL").toFloat().reproject(grid))
    cs = ee.Image(image.get("cs_img")).select("cs_cdf").toFloat().rename("CS")
    stack = stack.addBands(cs.reproject(grid))
    return stack.unmask(FILL).neighborhoodToArray(kernel)


def request(points, kernel):
    """One Earth Engine request for a group of records sampled on the same date."""
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(p.Longitude), float(p.Latitude)]),
                   {"rid": int(p.RowID), "t_ms": int(p.t_ms), "day": int(p.day)})
        for p in points.itertuples()])
    start = ee.Date((int(points.day.min()) - WINDOW_DAYS) * DAY_MS)
    end = ee.Date((int(points.day.max()) + WINDOW_DAYS + 1) * DAY_MS)
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(start, end).filterBounds(fc.geometry())
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filterDate(start, end).filterBounds(fc.geometry())
    images = ee.ImageCollection(ee.Join.saveFirst("cs_img").apply(
        primary=s2, secondary=cs, condition=ee.Filter.equals(leftField="system:index", rightField="system:index")))

    def sample(img):
        img = ee.Image(img)
        t = img.date().millis()
        day = t.divide(DAY_MS).floor()
        near = fc.filter(ee.Filter.And(ee.Filter.gte("day", day.subtract(WINDOW_DAYS)),
                                       ee.Filter.lte("day", day.add(WINDOW_DAYS)))).filterBounds(img.geometry())
        return window_values(img, kernel).sampleRegions(
            collection=near, properties=["rid", "t_ms"], scale=10, tileScale=4, geometries=False
        ).map(lambda f: f.set({
            "Granule_ID": img.get("system:index"), "MGRS_TILE": img.get("MGRS_TILE"),
            "Spacecraft": img.get("SPACECRAFT_NAME"), "Cloud_Coverage": img.get("CLOUDY_PIXEL_PERCENTAGE"),
            "SZA": img.get("MEAN_SOLAR_ZENITH_ANGLE"), "SAA": img.get("MEAN_SOLAR_AZIMUTH_ANGLE"),
            "VZA": img.get("MEAN_INCIDENCE_ZENITH_ANGLE_B8A"), "VAA": img.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B8A"),
            "S2_ms": t}))

    image_list = images.toList(images.size().max(1))
    return ee.FeatureCollection(image_list.map(sample)).flatten()


def get_info(req):
    for attempt in range(4):
        try:
            return req.getInfo()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * 2 ** attempt + random.random())


def to_rows(answer, points):
    """Earth Engine answer -> one row per record and image (16 values per layer)."""
    info = points.set_index("RowID")[POINT_COLUMNS[1:]].to_dict("index")
    rows = []
    for f in answer.get("features", []):
        p = f["properties"]
        row = {"RowID": int(p["rid"]), **info[int(p["rid"])]}
        for name in IMAGE_PROPS:
            row[name] = p.get(name)
        row["S2_DateTimeUTC"] = datetime.fromtimestamp(p["S2_ms"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        row["TimeDiff_Hours"] = round((p["S2_ms"] - p["t_ms"]) / 3_600_000, 4)
        empty = [[None] * WINDOW_PX] * WINDOW_PX
        for layer in LAYERS:
            values = [v for line in p.get(layer, empty) for v in line]
            for i, v in enumerate(values, start=1):
                row[f"{layer}_{i}"] = np.nan if v is None or v == FILL else float(v)
        for band in BANDS_60M:                          # 60 m bands: centre pixel only
            values = [v for line in p.get(band, empty) for v in line]
            v = values[5] if len(values) > 5 else None
            row[band] = np.nan if v is None or v == FILL else float(v)
        rows.append(row)
    return rows


def date_groups(points):
    """Records of one sampling date, in groups of at most MAX_POINTS (sorted by place)."""
    groups = []
    for date, same in points.groupby("LocalDate", sort=True):
        idx = same.index.to_numpy()
        if len(idx) > MAX_POINTS:
            idx = idx[np.lexsort((points.Longitude.to_numpy()[idx], points.Latitude.to_numpy()[idx] // 1.0))]
        for k, start in enumerate(range(0, len(idx), MAX_POINTS)):
            groups.append((f"{date}__{k:03d}", idx[start:start + MAX_POINTS]))
    return groups


def load_points():
    df = pd.read_parquet(IN_FILE, columns=POINT_COLUMNS + ["RecordKey"])
    t = pd.to_datetime(df["DateTimeUTC_ForMatch"], errors="coerce", utc=True)
    df = df[t.notna()].copy()
    df["t_ms"] = ((t[t.notna()] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")).astype("int64")
    df["day"] = (pd.to_datetime(df["LocalDate"]) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")
    return df.reset_index(drop=True)


def main():
    ee.Initialize(project=EE_PROJECT, opt_url="https://earthengine-highvolume.googleapis.com")
    kernel = ee.Kernel.fixed(WINDOW_PX, WINDOW_PX, [[1] * WINDOW_PX] * WINDOW_PX, 1, 1, False)
    points = load_points()
    CACHE.mkdir(parents=True, exist_ok=True)
    done = {f.stem for f in CACHE.glob("*.parquet")}
    todo = [g for g in date_groups(points) if g[0] not in done]
    print(f"{len(points):,} records, {len(todo):,} date groups to query")

    def run(group):
        key, idx = group
        rows = to_rows(get_info(request(points.loc[idx], kernel)), points.loc[idx])
        pd.DataFrame(rows, columns=None if rows else ["RowID"]).to_parquet(CACHE / f"{key}.parquet", index=False)
        return key

    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(run, g) for g in todo]
        for n, job in enumerate(as_completed(jobs), start=1):
            try:
                job.result()
            except Exception as error:          # failed groups are simply run again next time
                print("failed:", error)
            if n % 100 == 0:
                print(f"  {n:,} / {len(todo):,}")

    parts = [pd.read_parquet(f) for f in sorted(CACHE.glob("*.parquet"))]
    m = pd.concat([p for p in parts if len(p)], ignore_index=True).drop_duplicates(["RowID", "Granule_ID"])
    gap = (pd.to_datetime(m["S2_DateTimeUTC"]).dt.normalize() - pd.to_datetime(m["LocalDate"])).dt.days
    m = m[gap.abs() <= WINDOW_DAYS].copy()
    m["DayDiff"] = gap[gap.abs() <= WINDOW_DAYS].astype("int64")
    m = m.merge(points[["RowID", "RecordKey"]], on="RowID", how="left")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(OUT_FILE, index=False)
    print(f"saved {len(m):,} record-image pairs for {m.RowID.nunique():,} records -> {OUT_FILE}")


if __name__ == "__main__":
    main()

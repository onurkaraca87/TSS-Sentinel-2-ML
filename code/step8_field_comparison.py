# Step 8: field TSS compared with the storm maps (Table S20 of the supplement).
#
#   python step8_field_comparison.py okeechobee
#   python step8_field_comparison.py houston
#
# Field samples: quality-controlled WQP TSS in the case box within 30 days of landfall.
# A sample is compared with the map of the closest image when it was taken within +-3 days of it;
# the map value is the median of the 3 x 3 cells (120 m) around the station (at least 3 cells mapped).
# For samples that are also training matchups, the out-of-fold predictions of the temporal
# (leave-one-year-out) and kNNDM schemes are compared as well (from step 4).

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from step4_train_models import RESULTS, model_table
from step7_storm_maps import CASES

ROOT = Path(__file__).resolve().parents[1]
WQP_FILE = ROOT / "data" / "step1" / "WQP_TSS_filtered.parquet"
TOL_DAYS = 3
SPECIAL = {"okeechobee": ("2022-10-19", "2022-10-22")}   # samples of 19 Oct, taken between two images


def scores(obs, pred):
    o, p = np.maximum(np.asarray(obs, float), 0.1), np.maximum(np.asarray(pred, float), 0.1)
    q = np.log(p / o)
    if q.size == 0:
        return {"n": 0}
    lo = np.log10(o)
    m = np.median(q)
    return {"n": int(q.size), "field_median": float(np.median(o)), "map_median": float(np.median(p)),
            "R2": float(1 - np.sum((np.log10(p) - lo) ** 2) / np.sum((lo - lo.mean()) ** 2)) if q.size > 2 else np.nan,
            "MdSA": float(100 * (np.exp(np.median(np.abs(q))) - 1)),
            "SSPB": float(100 * np.sign(m) * (np.exp(abs(m)) - 1)),
            "within_x2": float(100 * np.mean(np.abs(q) <= np.log(2)))}


def map_value(path, lon, lat):
    with rasterio.open(path) as src:
        x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
        r, c = rasterio.transform.rowcol(src.transform, x, y)
        a = src.read(1)
    if not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]):
        return np.nan
    block = a[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
    return float(np.nanmedian(block)) if np.isfinite(block).sum() >= 3 else np.nan


def main(case):
    c = CASES[case]
    maps = ROOT / "results" / "storms" / case
    images = sorted(p.stem[4:] for p in maps.glob("TSS_*.tif"))
    landfall = pd.Timestamp(c["landfall"])

    f = pd.read_parquet(WQP_FILE)
    f["LocalDate"] = pd.to_datetime(f["LocalDate"])
    b = c["box"]
    f = f[f.Longitude.between(b[0], b[2]) & f.Latitude.between(b[1], b[3])
          & ((f.LocalDate - landfall).dt.days.abs() <= 30)].copy()
    train = model_table()
    f["training"] = f.RecordKey.isin(set(train.RecordKey))
    f["after"] = f.LocalDate >= landfall
    print(f"{len(f)} field samples at {f.StationID.nunique()} stations within 30 days of landfall")

    img = pd.to_datetime(pd.Series(images))
    f["image"], f["map"] = None, np.nan
    for i, r in f.iterrows():
        gap = (img - r.LocalDate).dt.days
        j = gap.abs().idxmin()
        if abs(gap[j]) <= TOL_DAYS:
            f.loc[i, "image"] = images[j]
            f.loc[i, "map"] = map_value(maps / f"TSS_{images[j]}.tif", r.Longitude, r.Latitude)
    f["gap_days"] = (pd.to_datetime(f["image"]) - f.LocalDate).dt.days

    # out-of-fold predictions of the training samples (models that did not see them)
    model = c["model"].split("_")[-1].replace(".pkl", "")
    rec = train.set_index("RowID")["RecordKey"]
    for scheme in ["temporal", "knndm"]:
        p = RESULTS / "CatBoost" / scheme / model / "predictions.parquet"
        if p.exists():
            oof = pd.read_parquet(p)
            f[scheme] = f.RecordKey.map(dict(zip(oof.RowID.map(rec), oof.TSS_pred)))

    m = f[f["map"].notna()]
    rows = [("before landfall", m[~m.after]), ("after landfall", m[m.after])]
    if case in SPECIAL:
        day, other = SPECIAL[case]
        d = m[m.LocalDate == day]
        rows += [(f"after landfall, without {day}", m[m.after & (m.LocalDate != day)]),
                 (f"{day} samples vs map of {d.image.iloc[0] if len(d) else '-'}", d)]
        d2 = d.copy()
        d2["map"] = [map_value(maps / f"TSS_{other}.tif", r.Longitude, r.Latitude) for r in d2.itertuples()]
        rows.append((f"{day} samples vs map of {other}", d2))
    rows += [("within +-1 day of the image", m[m.gap_days.abs() <= 1]), ("all samples (+-3 days)", m)]
    table = [{"samples": name, **scores(g.ResultValue_mgL, g["map"])} for name, g in rows]
    t = m[m.training]
    table.append({"samples": "training samples: map of the final model", **scores(t.ResultValue_mgL, t["map"])})
    for scheme, label in [("temporal", "leave-one-year-out"), ("knndm", "kNNDM out-of-fold")]:
        if scheme in t:
            g = t[t[scheme].notna()]
            table.append({"samples": f"training samples: {label} prediction", **scores(g.ResultValue_mgL, g[scheme])})
    out = pd.DataFrame(table).round(2)
    out.to_csv(maps / "field_vs_map_scores.csv", index=False)
    f.to_csv(maps / "field_vs_map.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "okeechobee")

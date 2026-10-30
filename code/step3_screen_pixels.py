# Step 3: pixel screening, duplicate removal and removal of ocean sites.
#
# Input : data/step2/matchups.parquet
# Output: data/final_TSS.parquet  (final matchup dataset, 69,287 rows)
#
# 1. A window pixel is valid when Sen2Cor calls it water (SCL = 6) and Cloud Score+ >= 0.60.
#    The band value of a record-image pair is the mean of its valid pixels; pairs with
#    fewer than 4 valid pixels are removed.
# 2. Duplicates are removed in three steps (one existing row is always kept):
#    a) same record seen in overlapping tiles of the same pass -> clearest tile
#    b) several samples in the same pixel and pass -> shallowest, closest in time,
#       closest to the median, smallest RowID
#    c) same record matched to images of different days -> closest in time
# 3. Sites of WQP type "Ocean" are removed.

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
IN_FILE = ROOT / "data" / "step2" / "matchups.parquet"
OUT_FILE = ROOT / "data" / "final_TSS.parquet"

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
CS_MIN = 0.60
MIN_PIXELS = 4
RECORD_COLUMNS = ["RowID", "RecordKey", "StationID", "Latitude", "Longitude", "LocalDate", "DateTimeUTC_ForMatch",
                  "Time_Source", "AnalyteClass", "ResultValue_mgL", "Depth_m", "LocationType", "HUC8", "OrgID",
                  "Provider", "Granule_ID", "MGRS_TILE", "Spacecraft", "S2_DateTimeUTC", "TimeDiff_Hours",
                  "Cloud_Coverage"]
FINAL_COLUMNS = ["RowID", "RecordKey", "StationID", "Latitude", "Longitude", "LocalDate", "AnalyteClass",
                 "ResultValue_mgL", "Depth_m", "LocationType", "HUC8", "OrgID", "Provider", "Granule_ID",
                 "MGRS_TILE", "Spacecraft", "S2_DateTimeUTC", "TimeDiff_Hours", "Time_Source", "N_Water_Px",
                 "N_Cloud_Px", "N_Land_Px", "Water_Frac", "CS_mean_water", "CS_min_water", "Cloud_Coverage",
                 "NDWI_mean_water", "N_Tiles", "Tile_Spread_Max", "N_Samples", "Target_Spread_Pct",
                 "Dropped_Values", "N_Passes", "SZA", "SAA", "VZA", "VAA"]


def pixels(df, layer):
    return df[[f"{layer}_{i}" for i in range(1, 17)]].to_numpy(dtype="float32")


def screen(m):
    """Mean of the valid water pixels of every record-image pair."""
    green, nir, scl, cs = pixels(m, "B3"), pixels(m, "B8"), pixels(m, "SCL"), pixels(m, "CS")
    with np.errstate(all="ignore"):
        ndwi = (green - nir) / (green + nir + 1e-6)
    has_values = np.isfinite(green) & np.isfinite(nir) & np.isfinite(scl) & np.isfinite(cs)
    clear = cs >= CS_MIN
    valid = has_values & clear & (scl == 6)
    n_valid = valid.sum(axis=1)

    out = m[[c for c in RECORD_COLUMNS if c in m.columns]].copy()
    for b in BANDS:
        values = np.where(valid, pixels(m, b), np.nan)
        out[f"{b}_mean"] = np.nanmean(values, axis=1)
        out[f"{b}_std"] = np.nanstd(values, axis=1)
    out["N_Water_Px"] = n_valid.astype("int8")
    out["N_Cloud_Px"] = (has_values & ~clear).sum(axis=1).astype("int8")
    out["N_Land_Px"] = (has_values & clear & (scl != 6)).sum(axis=1).astype("int8")
    out["Water_Frac"] = (n_valid / 16).astype("float32")
    out["CS_mean_water"] = np.nanmean(np.where(valid, cs, np.nan), axis=1)
    out["CS_min_water"] = np.nanmin(np.where(valid, cs, np.nan), axis=1)
    out["NDWI_mean_water"] = np.nanmean(np.where(valid, ndwi, np.nan), axis=1)
    for c in ["SZA", "SAA", "VZA", "VAA"]:
        out[c] = m[c]
    return out[n_valid >= MIN_PIXELS].copy()


def num(df, column, fill):
    return pd.to_numeric(df[column], errors="coerce").fillna(fill)


def pick_one(df, group, rules):
    """Keep one row per group: apply the rules one after another until one row is left."""
    left = pd.Series(True, index=df.index)
    for values, how in rules:
        if not pd.api.types.is_numeric_dtype(values):
            values = values.rank(method="dense")
        candidate = values.where(left)
        left &= candidate.eq(candidate.groupby(group).transform(how))
    return left


def same_pass_tiles(d):
    group = d.groupby(["RowID", d["Granule_ID"].astype(str).str[:15]], sort=False).ngroup()
    keep = pick_one(d, group, [(num(d, "CS_mean_water", -1.0), "max"), (num(d, "CS_min_water", -1.0), "max"),
                               (num(d, "N_Water_Px", 0.0), "max"), (num(d, "Cloud_Coverage", 999.0), "min"),
                               (d["Granule_ID"].astype(str), "min")])
    spread = pd.concat([d[f"{b}_mean"].groupby(group).transform("max") - d[f"{b}_mean"].groupby(group).transform("min")
                        for b in BANDS], axis=1).max(axis=1)
    d = d.assign(N_Tiles=group.map(group.value_counts()).astype("int16"), Tile_Spread_Max=spread.astype("float32"))
    return d[keep].copy()


def same_pixel(d):
    spectrum = d[[f"{b}_mean" for b in BANDS]].round(7).astype(str).agg("|".join, axis=1)
    group = d.assign(spectrum=spectrum).groupby(["Granule_ID", "spectrum"], sort=False).ngroup()
    tss = pd.to_numeric(d["ResultValue_mgL"], errors="coerce")
    depth = num(d, "Depth_m", np.inf)
    gap = num(d, "TimeDiff_Hours", 9e9).abs()
    shallow = depth.eq(depth.groupby(group).transform("min"))
    gap_shallow = gap.where(shallow)
    closest = shallow & gap_shallow.eq(gap_shallow.groupby(group).transform("min"))
    to_median = (tss - tss.where(closest).groupby(group).transform("median")).abs()
    keep = pick_one(d, group, [(depth, "min"), (gap, "min"), (to_median, "min"), (num(d, "RowID", 9e18), "min")])

    vmax, vmin = tss.groupby(group).transform("max"), tss.groupby(group).transform("min")
    dropped = d.assign(g=group).loc[~keep].groupby("g")["ResultValue_mgL"].apply(lambda v: ", ".join(f"{x:g}" for x in v))
    d = d.assign(N_Samples=group.map(group.value_counts()).astype("int16"),
                 Target_Spread_Pct=((vmax - vmin) / vmax.replace(0, np.nan) * 100).fillna(0).astype("float32"),
                 Dropped_Values=group.map(dropped).fillna(""))
    return d[keep].copy()


def several_days(d):
    group = d.groupby("RowID", sort=False).ngroup()
    keep = pick_one(d, group, [(num(d, "TimeDiff_Hours", 9e9).abs(), "min"), (num(d, "CS_mean_water", -1.0), "max"),
                               (d["Granule_ID"].astype(str), "min")])
    d = d.assign(N_Passes=group.map(group.value_counts()).astype("int16"))
    return d[keep].copy()


def water_type(location_type):
    text = str(location_type)
    if "Estuary" in text:
        return "Estuary"
    if "Ocean" in text:
        return "Ocean"
    if "River" in text or "Stream" in text or "Canal" in text:
        return "River"
    if "Lake" in text or "Impoundment" in text:
        return "Lake"
    return "Other"


def main():
    m = pd.read_parquet(IN_FILE)
    print(f"record-image pairs           {len(m):>9,}")
    d = screen(m)
    print(f"with >= {MIN_PIXELS} valid water pixels  {len(d):>9,}")
    d = d[d["AnalyteClass"].astype(str).str.upper() == "TSS"]
    d = same_pass_tiles(d)
    print(f"after tile overlaps          {len(d):>9,}")
    d = same_pixel(d)
    print(f"after same pixel and pass    {len(d):>9,}")
    d = several_days(d)
    print(f"after several days           {len(d):>9,}")

    columns = [c for c in d.columns if c in FINAL_COLUMNS or c.endswith(("_mean", "_std"))]
    d = d[list(dict.fromkeys(columns))].copy()
    d["WATER_TYPE"] = d["LocationType"].map(water_type)
    final = d[d["WATER_TYPE"] != "Ocean"].reset_index(drop=True)
    print(f"without ocean sites          {len(final):>9,}  at {final.StationID.nunique():,} sites")
    final.to_parquet(OUT_FILE, index=False)


if __name__ == "__main__":
    main()

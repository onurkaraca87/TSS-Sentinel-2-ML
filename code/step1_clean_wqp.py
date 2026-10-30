# Step 1: clean the Water Quality Portal (WQP) download.
#
# Input : data/raw/WQP_TSS_records.parquet   (WQP download, TSS + SSC, 1,780,020 records)
# Output: data/step1/WQP_TSS_filtered.parquet (quality-controlled TSS records)
#
# Units are converted to mg/L, the UTC time is computed from the local time,
# depths are converted to metres and NAD27 coordinates to WGS84.
# Then 13 filters are applied (Table S3 of the supplement).

from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "WQP_TSS_records.parquet"
OUT_FILE = ROOT / "data" / "step1" / "WQP_TSS_filtered.parquet"

UNIT_TO_MGL = {"mg/l": 1.0, "ppm": 1.0, "g/l": 1000.0, "ppt": 1000.0,
               "ug/l": 0.001, "ppb": 0.001, "µg/l": 0.001}
DEPTH_TO_M = {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "ft": 0.3048, "feet": 0.3048,
              "foot": 0.3048, "in": 0.0254, "inch": 0.0254, "inches": 0.0254, "cm": 0.01,
              "centimeter": 0.01, "centimeters": 0.01, "mm": 0.001}
TZ_OFFSET = {"EST": -5, "EDT": -4, "CST": -6, "CDT": -5, "MST": -7, "MDT": -6, "PST": -8, "PDT": -7,
             "AKST": -9, "AKDT": -8, "HST": -10, "HDT": -9, "AST": -4, "ADT": -3, "GMT": 0, "UTC": 0,
             "GST": 10, "CHST": 10, "SST": -11, "DST": np.nan}
NON_DETECT = {"Not Detected", "Present Below Quantification Limit", "Below Reporting Limit",
              "Between Inst Detect and Quant Limit", "Below Method Detection Limit",
              "Detected Not Quantified", "Not Detected at Detection Limit",
              "Not Detected at Reporting Limit", "Below Detection Limit"}
NON_DETECT_CODES = {"U", "LT", "<"}
LOWER_48 = {"AL", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "ID", "IL", "IN", "IA", "KS",
            "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
            "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA",
            "WA", "WV", "WI", "WY"}
S2_START = "2017-03-28"     # start of the Sentinel-2 L2A archive in Earth Engine
SOLAR_HOUR = 10.75          # Sentinel-2 overpass, used when only a date is reported
MAX_DEPTH = 1.0             # m; samples without a depth are kept

KEEP_COLUMNS = ["RowID", "RecordKey", "StationID", "Latitude", "Longitude", "LocalDate", "DateTimeUTC",
                "DateTimeUTC_ForMatch", "Time_Source", "AnalyteClass", "ResultValue_mgL", "Unit",
                "Depth_m", "LocationType", "HUC8", "OrgID", "Provider"]


def col(raw, *names):
    for name in names:
        if name in raw.columns:
            return raw[name]
    return pd.Series(pd.NA, index=raw.index)


def add_time(raw, df):
    date = pd.to_datetime(col(raw, "Date", "ActivityStartDate"), errors="coerce")
    time_text = col(raw, "Time", "ActivityStartTime")
    tz = col(raw, "TimeZone").astype(str).str.strip().str.upper()
    offset = pd.to_numeric(col(raw, "TZOffsetHours"), errors="coerce")
    has_date = date.notna()
    has_time = (time_text.notna() & (time_text.astype(str).str.strip() != "")
                & (time_text.astype(str).str.lower() != "nat"))

    # missing offsets are taken from the time-zone code
    fill = offset.isna() & has_time
    offset.loc[fill] = tz[fill].map(TZ_OFFSET).to_numpy()

    time_text = time_text.astype(str).str.strip()
    local = pd.to_datetime((date.dt.strftime("%Y-%m-%d") + " " + time_text).where(has_date & has_time),
                           errors="coerce", format="mixed")
    utc = local - pd.to_timedelta(offset.fillna(0), unit="h")
    utc = utc.where(local.notna() & offset.notna())

    # records with a date only get the local solar time of the overpass
    lon = pd.to_numeric(col(raw, "Longitude"), errors="coerce")
    anchored = (date + pd.to_timedelta(SOLAR_HOUR - lon / 15.0, unit="h")).where(has_date & utc.isna() & lon.notna())

    df["LocalDate"] = date.dt.strftime("%Y-%m-%d")
    df["DateTimeUTC"] = utc.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["DateTimeUTC_ForMatch"] = utc.fillna(anchored).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["Time_Source"] = np.where(utc.notna(), "measured",
                                 np.where(anchored.notna(), "date_only_anchored",
                                          np.where(has_date, "date_only_no_lon", "no_date")))
    df["has_date"] = has_date


def add_value(raw, df):
    value = pd.to_numeric(col(raw, "ResultValueNum", "ResultValue"), errors="coerce")
    unit = col(raw, "Unit").astype(str).str.strip()
    factor = unit.str.lower().str.replace(" ", "", regex=False).map(UNIT_TO_MGL)
    df["AnalyteClass"] = col(raw, "AnalyteClass")
    df["ResultValue_mgL"] = value * factor
    df["Unit"] = np.where(factor.notna(), "mg/L", unit)
    df["has_value"] = value.notna()
    df["unit_ok"] = factor.notna()

    condition = col(raw, "DetectionCondition").astype(str).str.strip()
    codes = col(raw, "Qualifier").astype(str).str.upper().str.split(";")
    code_nd = codes.map(lambda c: isinstance(c, list) and any(x.strip() in NON_DETECT_CODES for x in c))
    df["non_detect"] = condition.isin(NON_DETECT) | code_nd
    df["rejected"] = col(raw, "Status").astype(str).str.strip().eq("Rejected")


def add_depth(raw, df):
    d1 = pd.to_numeric(col(raw, "DepthValue"), errors="coerce") \
        * col(raw, "DepthUnit").astype(str).str.strip().str.lower().map(DEPTH_TO_M)
    d2 = pd.to_numeric(col(raw, "ResultDepthValue"), errors="coerce") \
        * col(raw, "ResultDepthUnit").astype(str).str.strip().str.lower().map(DEPTH_TO_M)
    df["Depth_m"] = d1.fillna(d2)


def add_location(raw, df):
    lat = pd.to_numeric(col(raw, "Latitude"), errors="coerce")
    lon = pd.to_numeric(col(raw, "Longitude"), errors="coerce")
    nad27 = col(raw, "Datum").astype(str).str.upper().str.contains("NAD27", na=False) & lat.notna() & lon.notna()
    if nad27.any():
        pyproj.network.set_network_enabled(True)
        to_wgs84 = Transformer.from_crs("EPSG:4267", "EPSG:4326", always_xy=True)
        x, y = to_wgs84.transform(lon[nad27].to_numpy(), lat[nad27].to_numpy())
        lon.loc[nad27], lat.loc[nad27] = x, y
    df["Latitude"], df["Longitude"] = lat, lon
    df["LocationType"] = col(raw, "LocationType")
    huc = col(raw, "HUC8").astype(str).str.replace(r"\.0+$", "", regex=True).str.strip()
    df["HUC8"] = huc.where(huc.str.fullmatch(r"\d{1,8}"), "").str.zfill(8).replace("00000000", "")

    state = col(raw, "StateQueried").astype(str).str.strip().str.upper()
    in_box = lat.between(24.0, 50.0) & lon.between(-125.5, -66.0)
    df["coords_ok"] = lat.between(-90, 90) & lon.between(-180, 180)
    df["lower48"] = state.isin(LOWER_48) & in_box


def main():
    raw = pd.read_parquet(RAW_FILE)
    df = pd.DataFrame(index=raw.index)
    for c in ["StationID", "OrgID", "Provider"]:
        df[c] = col(raw, c)
    add_time(raw, df)
    add_value(raw, df)
    add_depth(raw, df)
    add_location(raw, df)
    key_cols = [c for c in ["SampleID", "ActivityID", "ResultID", "StationID", "CharName", "Date", "Time",
                            "ResultValue", "Unit"] if c in raw.columns]
    df["RecordKey"] = pd.util.hash_pandas_object(raw[key_cols].astype(str), index=False).astype("uint64")
    df = df.reset_index(drop=True)
    df["RowID"] = np.arange(1, len(df) + 1, dtype=np.int64)

    value, depth = df["ResultValue_mgL"], df["Depth_m"]
    filters = [
        ("TSS only", df["AnalyteClass"].astype(str).str.strip().str.upper() == "TSS"),
        ("has a date", df["has_date"]),
        ("valid coordinates", df["coords_ok"]),
        ("numeric value", df["has_value"]),
        ("unit converts to mg/L", df["unit_ok"]),
        ("value > 0", value > 0),
        ("on/after " + S2_START, pd.to_datetime(df["LocalDate"], errors="coerce") >= pd.Timestamp(S2_START)),
        ("lower 48 states + DC", df["lower48"]),
        ("not a non-detect", ~df["non_detect"]),
        ("not rejected by provider", ~df["rejected"]),
        ("value 0.1-50,000 mg/L", value.between(0.1, 50_000)),
        ("depth 0-100 m or unknown", depth.isna() | depth.between(0, 100)),
        ("depth <= 1 m (unknown kept)", (depth <= MAX_DEPTH) | depth.isna()),
    ]
    keep = pd.Series(True, index=df.index)
    print(f"{'all records':<32}{len(df):>12,}")
    for name, ok in filters:
        removed = int((keep & ~ok.fillna(False)).sum())
        keep &= ok.fillna(False)
        print(f"{name:<32}{int(keep.sum()):>12,}   (-{removed:,})")

    out = df.loc[keep, KEEP_COLUMNS]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_FILE, index=False)
    print(f"saved {len(out):,} records at {out.StationID.nunique():,} stations -> {OUT_FILE}")


if __name__ == "__main__":
    main()

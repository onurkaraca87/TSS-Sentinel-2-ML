# Apply a trained TSS model (models/*.pkl) to Sentinel-2 data.
#
# A model file is a dictionary saved with joblib: the CatBoost model ("estimator"), the 36 input
# features, the location-type levels, the smearing factor S and the training limits.
# Inputs must be prepared like the training data: L2A reflectance (0-1), all bands on the 10 m
# grid, 4 x 4 pixel cells (40 m), valid pixel = SCL 6 (water) and Cloud Score+ >= 0.60.

import warnings

import joblib
import numpy as np
import pandas as pd

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
CELL = 4


def load(path):
    return joblib.load(path)


def add_features(t):
    t = t.copy()
    doy = pd.to_datetime(t["LocalDate"], errors="coerce").dt.dayofyear.fillna(180)
    t["SIN_DOY"] = np.sin(2 * np.pi * doy / 365)
    t["COS_DOY"] = np.cos(2 * np.pi * doy / 365)
    c = {b: t[b].astype(float).clip(lower=1e-4) for b in BANDS}
    for a, b in [("B4", "B3"), ("B5", "B4"), ("B5", "B3"), ("B6", "B5"), ("B8", "B4"),
                 ("B3", "B2"), ("B4", "B2"), ("B8A", "B4"), ("B11", "B4")]:
        t[f"LR_{a}{b}"] = np.log(c[a] / c[b])
    t["NDTI"] = (c["B4"] - c["B3"]) / (c["B4"] + c["B3"])
    t["NDSSI"] = (c["B2"] - c["B8"]) / (c["B2"] + c["B8"])
    t["SUM_VIS"] = c["B2"] + c["B3"] + c["B4"]
    t["NDCI"] = (c["B5"] - c["B4"]) / (c["B5"] + c["B4"])
    t["B4_B3_DIF"] = c["B4"] - c["B3"]
    t["SUM_RE"] = c["B5"] + c["B6"] + c["B7"]
    t["LocationType"] = t["LocationType"].astype(str)
    return t


def cells_from_image(bands, scl, cs, lon, lat, date, location_type, scene_cloud_pct):
    """10 m arrays -> one row per 40 m cell (mean of the valid water pixels)."""
    def cells(a):
        a = np.asarray(a, dtype="float64")
        h, w = a.shape[0] // CELL * CELL, a.shape[1] // CELL * CELL
        return a[:h, :w].reshape(h // CELL, CELL, w // CELL, CELL).transpose(0, 2, 1, 3).reshape(h // CELL, w // CELL, CELL * CELL)

    cs = np.asarray(cs, dtype="float64")
    valid = cells((np.asarray(scl) == 6) & (cs >= 0.60)).astype(bool)

    def mean(a):
        return np.nanmean(np.where(valid, cells(a), np.nan), axis=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t = pd.DataFrame({b: mean(bands[b]).ravel() for b in BANDS})
        b3, b8 = np.asarray(bands["B3"], float), np.asarray(bands["B8"], float)
        t["NDWI_mean_water"] = mean((b3 - b8) / (b3 + b8 + 1e-6)).ravel()
        t["CS_mean_water"] = mean(cs).ravel()
        t["CS_min_water"] = np.nanmin(np.where(valid, cells(cs), np.nan), axis=2).ravel()
    t["N_Water_Px"] = valid.sum(axis=2).ravel()
    t["Water_Frac"] = t.N_Water_Px / (CELL * CELL)
    t["Cloud_Coverage"] = float(scene_cloud_pct)
    t["LocationType"] = location_type
    t["LocalDate"] = date
    t["Longitude"] = np.asarray(lon, float).ravel()
    t["Latitude"] = np.asarray(lat, float).ravel()
    return t


def predict_tss(bundle, table):
    """TSS (mg/L) per row; NaN where a cell is outside the limits of the training data."""
    if "SIN_DOY" not in table.columns:
        table = add_features(table)
    lim = bundle["limits"]
    ok = ((table["N_Water_Px"] >= lim["min_water_pixels"]) & (table["B11"] < lim["b11_max"])
          & (table[["B2", "B3", "B4", "B5"]] > 0).all(axis=1)
          & table[[c for c in bundle["features"] if c != "LocationType"]].notna().all(axis=1)
          & table["LocationType"].astype(str).isin(bundle["levels"]["LocationType"])).to_numpy()
    tss = np.full(len(table), np.nan)
    if ok.any():
        X = table.loc[ok, bundle["features"]].copy()
        X["LocationType"] = X["LocationType"].astype(str)
        pred = np.exp(bundle["estimator"].predict(X)) * bundle["smearing_S"] - 1
        tss[ok] = np.clip(pred, 0, lim["tss_max_mgL"])
    return tss

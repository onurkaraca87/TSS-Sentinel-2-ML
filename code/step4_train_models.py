# Step 4: train and test the TSS models.
#
# Input : data/final_TSS.parquet, data/step1/WQP_TSS_filtered.parquet (for kNNDM)
# Output: results/<algorithm>/<scheme>/<model>/  predictions.parquet, scores.json, model.pkl
#
# 5 algorithms (CatBoost, LightGBM, XGBoost, RandomForest, MLP)
# x 4 validation schemes (random, knndm, temporal, huc8)
# x 4 models (Combined, Lake, River, Estuary).
# Every model predicts log(1 + TSS). For each scheme the hyper-parameters are chosen by a
# randomized search (8 candidates) on that scheme's folds; every matchup is then predicted
# once by a model that did not see it (pooled out-of-fold predictions), and a final model
# is trained on all matchups.
#
# Examples:
#   python step4_train_models.py                                   (everything, takes days)
#   python step4_train_models.py --algorithm CatBoost --scheme knndm --model Lake

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import loguniform, wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "final_TSS.parquet"
WQP_FILE = ROOT / "data" / "step1" / "WQP_TSS_filtered.parquet"
RESULTS = ROOT / "results"

SEED = 42
N_FOLDS = 5
N_CANDIDATES = 8
CAP = 250.0                     # mg/L; largest TSS of the model data
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
FEATURES = BANDS + ["NDWI_mean_water", "Latitude", "Longitude", "LocationType", "SIN_DOY", "COS_DOY",
                    "LR_B4B3", "LR_B5B4", "LR_B5B3", "LR_B6B5", "LR_B8B4", "NDTI", "NDSSI", "SUM_VIS",
                    "LR_B3B2", "LR_B4B2", "LR_B8AB4", "LR_B11B4", "NDCI", "B4_B3_DIF", "SUM_RE",
                    "N_Water_Px", "Water_Frac", "CS_mean_water", "CS_min_water", "Cloud_Coverage"]
CATEGORICAL = ["LocationType"]
MONOTONE = ["B4", "B5", "B8A"]  # TSS may only increase with these bands

SEARCH = {
    "CatBoost": {"iterations": [1000, 1500, 2000], "learning_rate": loguniform(0.003, 0.05),
                 "depth": [4, 5, 6, 7, 8], "l2_leaf_reg": loguniform(1, 20),
                 "random_strength": loguniform(0.1, 10), "bagging_temperature": [0, 0.25, 0.5, 1, 2],
                 "border_count": [32, 64, 128, 254]},
    "LightGBM": {"n_estimators": [1000, 1500, 2000], "learning_rate": loguniform(0.003, 0.05),
                 "num_leaves": [15, 31, 63, 127, 255], "min_child_samples": [5, 10, 20, 50],
                 "reg_lambda": loguniform(1, 20), "subsample": [0.7, 0.85, 1.0],
                 "colsample_bytree": [0.5, 0.7, 1.0]},
    "XGBoost": {"n_estimators": [1000, 1500, 2000], "learning_rate": loguniform(0.003, 0.05),
                "max_depth": [4, 5, 6, 7, 8], "min_child_weight": [1, 3, 5, 10],
                "reg_lambda": loguniform(1, 20), "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.5, 0.7, 1.0]},
    "RandomForest": {"n_estimators": [300, 500], "max_features": [0.33, 0.5, "sqrt", 1.0],
                     "min_samples_leaf": [1, 2, 5, 10], "max_depth": [None, 20, 30]},
    "MLP": {"hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (256, 128), (128, 64, 32)],
            "alpha": loguniform(1e-5, 1e-1), "learning_rate_init": loguniform(1e-4, 3e-3),
            "batch_size": [64, 128, 256]},
}


# ------------------------------------------------------------------ data
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


def model_table():
    """Final dataset -> the 48,222 model matchups with their 36 features."""
    d = pd.read_parquet(DATA_FILE).rename(columns={f"{b}_mean": b for b in BANDS})
    tss = pd.to_numeric(d["ResultValue_mgL"], errors="coerce")
    keep = ((pd.to_numeric(d["B11"], errors="coerce") < 0.05) & (tss >= 0) & (tss <= CAP)
            & (pd.to_numeric(d["N_Water_Px"], errors="coerce") >= 12) & (d[["B2", "B3", "B4", "B5"]] > 0).all(axis=1))
    d = d[keep].copy()

    d["LocationType"] = d["LocationType"].astype(str)
    doy = pd.to_datetime(d["LocalDate"], errors="coerce").dt.dayofyear.fillna(180)
    d["SIN_DOY"] = np.sin(2 * np.pi * doy / 365)
    d["COS_DOY"] = np.cos(2 * np.pi * doy / 365)
    c = {b: d[b].clip(lower=1e-4) for b in BANDS}             # bands >= 0.0001, so ratios stay finite
    for name, (a, b) in {"LR_B4B3": ("B4", "B3"), "LR_B5B4": ("B5", "B4"), "LR_B5B3": ("B5", "B3"),
                         "LR_B6B5": ("B6", "B5"), "LR_B8B4": ("B8", "B4")}.items():
        d[name] = np.log(c[a] / c[b])
    d["NDTI"] = (c["B4"] - c["B3"]) / (c["B4"] + c["B3"])
    d["NDSSI"] = (c["B2"] - c["B8"]) / (c["B2"] + c["B8"])
    d["SUM_VIS"] = c["B2"] + c["B3"] + c["B4"]
    for name, (a, b) in {"LR_B3B2": ("B3", "B2"), "LR_B4B2": ("B4", "B2"), "LR_B8AB4": ("B8A", "B4"),
                         "LR_B11B4": ("B11", "B4")}.items():
        d[name] = np.log(c[a] / c[b])
    d["NDCI"] = (c["B5"] - c["B4"]) / (c["B5"] + c["B4"])
    d["B4_B3_DIF"] = c["B4"] - c["B3"]
    d["SUM_RE"] = c["B5"] + c["B6"] + c["B7"]

    d = d.dropna(subset=FEATURES + ["ResultValue_mgL"]).reset_index(drop=True)
    d["WATER_TYPE"] = d["LocationType"].map(water_type)
    d["Year"] = pd.to_datetime(d["LocalDate"]).dt.year
    return d


def km(lat, lon):
    """Coordinates in km (CONUS Albers, EPSG:5070)."""
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True).transform(
        np.asarray(lon, float), np.asarray(lat, float))
    return np.c_[np.asarray(x) / 1000, np.asarray(y) / 1000]


# ------------------------------------------------------------------ folds
def knndm_folds(train, target, k=N_FOLDS, max_share=0.5, max_q=3000):
    """kNNDM (Linnenbrink et al. 2024): choose the k-means clustering whose test-to-training
    distances best match the distances from the target locations to the training sites."""
    n = len(train)
    tree = cKDTree(train)
    g_target = tree.query(target, k=1)[0]

    def g_test(folds):
        out = np.empty(n)
        for f in range(k):
            inside = folds == f
            out[inside] = cKDTree(train[~inside]).query(train[inside], k=1)[0]
        return out

    rng = np.random.default_rng(SEED)
    best = None
    for q in sorted(set(np.round(np.geomspace(k, min(n // 2, max_q), 25)).astype(int))) + [n]:
        if q == n:
            folds = rng.permutation(np.arange(n) % k)                 # random folds
        else:
            clusters = KMeans(n_clusters=q, random_state=SEED, n_init=1).fit_predict(train)
            fold_of, size = {}, np.zeros(k, dtype=int)
            for i, (cl, cnt) in enumerate(pd.Series(clusters).value_counts().items()):
                f = i if i < k else int(np.argmin(size))              # biggest clusters first, then smallest fold
                fold_of[cl] = f
                size[f] += cnt
            if size.max() > max_share * n:
                continue
            folds = np.vectorize(fold_of.get)(clusters)
        w = wasserstein_distance(g_test(folds), g_target)
        if best is None or w < best[0]:
            best = (w, q, folds)
    print(f"  kNNDM: q = {best[1]}{' (random folds)' if best[1] == n else ''}, W = {best[0]:.3f} km")
    return best[2]


def make_folds(scheme, sub, model):
    n = len(sub)
    folds = np.empty(n, dtype=int)
    if scheme == "random":
        for f, (_, test) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(sub)):
            folds[test] = f
    elif scheme == "huc8":
        code = sub["HUC8"].astype(str).str.replace(r"\.0+$", "", regex=True).str.zfill(8).str[:8]
        basins = code.where(code.str.fullmatch(r"\d{8}").fillna(False), "INVALID").to_numpy()
        for f, (_, test) in enumerate(GroupKFold(N_FOLDS).split(sub, groups=basins)):
            folds[test] = f
    elif scheme == "temporal":                                     # leave one year out
        years = sorted(sub["Year"].unique())
        folds = sub["Year"].map({y: i for i, y in enumerate(years)}).to_numpy()
    elif scheme == "knndm":
        # target locations: all WQP TSS stations of the model's water type(s)
        st = pd.read_parquet(WQP_FILE, columns=["Latitude", "Longitude", "LocationType"])
        wt = st["LocationType"].map(water_type)
        st = st[wt != "Ocean"] if model == "Combined" else st[wt == model]
        st = st[["Latitude", "Longitude"]].round(4).drop_duplicates()
        folds = knndm_folds(km(sub.Latitude, sub.Longitude), km(st.Latitude, st.Longitude))
    return folds


# ------------------------------------------------------------------ algorithms
def encode(X, levels, one_hot):
    """Category columns as pandas categories (LightGBM, XGBoost) or as 0/1 columns (RF, MLP)."""
    out = X.drop(columns=CATEGORICAL).astype(float) if one_hot else X.copy()
    for c in CATEGORICAL:
        if one_hot:
            for level in levels[c]:
                out[f"{c}={level}"] = (X[c].astype(str) == level).astype(float)
        else:
            out[c] = pd.Categorical(X[c].astype(str), categories=levels[c])
    if not one_hot:
        for c in out.columns:
            if c not in CATEGORICAL:
                out[c] = out[c].astype(float)
    return out


def estimator(algorithm, params, columns, n_jobs, for_search=False):
    mono = [1 if c in MONOTONE else 0 for c in columns]
    if algorithm == "CatBoost":
        from catboost import CatBoostRegressor
        extra = {} if for_search else {"monotone_constraints": {columns.index(c): 1 for c in MONOTONE}}
        return CatBoostRegressor(**params, loss_function="RMSE", random_seed=SEED, verbose=0,
                                 thread_count=n_jobs, allow_writing_files=False, **extra)
    if algorithm == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params, objective="regression", subsample_freq=1, random_state=SEED, n_jobs=n_jobs,
                             verbose=-1, monotone_constraints=mono, monotone_constraints_method="intermediate")
    if algorithm == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(**params, objective="reg:squarederror", tree_method="hist", enable_categorical=True,
                            max_cat_to_onehot=1, random_state=SEED, n_jobs=n_jobs, verbosity=0,
                            monotone_constraints=tuple(mono))
    if algorithm == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params, random_state=SEED, n_jobs=n_jobs)    # no monotone constraint
    if algorithm == "MLP":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        p = {k: (tuple(v) if k == "hidden_layer_sizes" else v) for k, v in params.items()}
        return Pipeline([("scale", StandardScaler()),
                         ("mlp", MLPRegressor(**p, activation="relu", solver="adam", early_stopping=True,
                                              validation_fraction=0.1, n_iter_no_change=20, max_iter=500,
                                              random_state=SEED))])


class Model:
    """One fitted model that takes the feature table and returns log(1 + TSS)."""

    def __init__(self, algorithm, params, levels):
        self.algorithm, self.params, self.levels = algorithm, params, levels
        self.one_hot = algorithm in ("RandomForest", "MLP")

    def matrix(self, X):
        if self.algorithm == "CatBoost":
            return X[FEATURES]
        E = encode(X[FEATURES], self.levels, self.one_hot)
        return E.values if self.algorithm == "MLP" else E

    def fit(self, X, y):
        E = self.matrix(X)
        columns = FEATURES if self.algorithm == "CatBoost" else list(encode(X[FEATURES].head(1), self.levels, self.one_hot).columns)
        self.est = estimator(self.algorithm, self.params, columns, n_jobs=-1)
        if self.algorithm == "CatBoost":
            self.est.fit(E, np.asarray(y), cat_features=CATEGORICAL)
        elif self.algorithm == "LightGBM":
            self.est.fit(E, np.asarray(y), categorical_feature=CATEGORICAL)
        else:
            self.est.fit(E, np.asarray(y))
        return self

    def predict(self, X):
        return self.est.predict(self.matrix(X))


def tune(algorithm, sub, y, folds, levels):
    """Randomized search (8 candidates) on the folds of this scheme, scored by R2."""
    cv = [(np.where(folds != f)[0], np.where(folds == f)[0]) for f in sorted(set(folds) - {-1})]
    model = Model(algorithm, {}, levels)
    E = model.matrix(sub)
    columns = FEATURES if algorithm == "CatBoost" else list(encode(sub[FEATURES].head(1), levels, model.one_hot).columns)
    space = SEARCH[algorithm]
    if algorithm == "MLP":
        space = {f"mlp__{k}": v for k, v in space.items()}
    search = RandomizedSearchCV(estimator(algorithm, {}, columns, n_jobs=1, for_search=True), space,
                                n_iter=N_CANDIDATES, scoring="r2", cv=cv, random_state=SEED, n_jobs=-1)
    if algorithm == "CatBoost":
        search.fit(E, y, cat_features=CATEGORICAL)
    elif algorithm == "LightGBM":
        search.fit(E, y, categorical_feature=CATEGORICAL)
    else:
        search.fit(E, y)
    params = {k.replace("mlp__", ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    print(f"  best parameters: {params}")
    return params


# ------------------------------------------------------------------ scores
def scores(y_log, pred_log):
    """Scores of the pooled out-of-fold predictions (mg/L values limited to 0-250 mg/L)."""
    s = float(np.mean(np.exp(y_log - pred_log)))                  # Duan smearing factor
    obs = np.expm1(y_log)
    pred = np.clip(np.exp(pred_log) * s - 1, 0, CAP)
    ratio = np.clip(pred, 1e-3, None) / np.clip(obs, 1e-3, None)
    m = np.median(np.log(ratio))
    ok = (obs > 0) & (pred > 0)
    r2 = lambda a, b: 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)
    return {"n": int(len(obs)), "S": s, "R2_log": r2(y_log, pred_log), "R2_mgL": r2(obs, pred),
            "RMSE_mgL": float(np.sqrt(np.mean((obs - pred) ** 2))), "MAE_mgL": float(np.mean(np.abs(obs - pred))),
            "MdSA": float(100 * (np.exp(np.median(np.abs(np.log(ratio)))) - 1)),
            "SSPB": float(100 * np.sign(m) * (np.exp(abs(m)) - 1)),
            "slope": float(np.polyfit(np.log10(obs[ok]), np.log10(pred[ok]), 1)[0])}


def run(table, levels, algorithm, scheme, model):
    sub = (table if model == "Combined" else table[table.WATER_TYPE == model]).reset_index(drop=True)
    print(f"\n{algorithm} | {scheme} | {model} ({len(sub):,} matchups)")
    y = np.log1p(sub["ResultValue_mgL"].astype(float)).to_numpy()
    folds = make_folds(scheme, sub, model)
    params = tune(algorithm, sub, y, folds, levels)

    oof = np.full(len(sub), np.nan)
    for f in sorted(set(folds) - {-1}):
        test = folds == f
        oof[test] = Model(algorithm, params, levels).fit(sub[~test], y[~test]).predict(sub[test])
    final = Model(algorithm, params, levels).fit(sub, y)

    sc = scores(y, oof)
    sc["train_R2_log"] = scores(y, final.predict(sub))["R2_log"]
    print("  " + ", ".join(f"{k} {v:.3f}" for k, v in sc.items() if k != "n"))

    out = RESULTS / algorithm / scheme / model
    out.mkdir(parents=True, exist_ok=True)
    pred = sub[["RowID", "StationID", "Latitude", "Longitude", "LocalDate", "WATER_TYPE", "ResultValue_mgL"]].copy()
    pred["fold"], pred["pred_log"] = folds + 1, oof
    pred["TSS_pred"] = np.maximum(np.exp(oof) * sc["S"] - 1, 0)
    pred.to_parquet(out / "predictions.parquet", index=False)
    (out / "scores.json").write_text(json.dumps({**sc, "params": params}, indent=2))
    joblib.dump({"model": final, "features": FEATURES, "levels": levels, "S": sc["S"], "max_mgL": CAP},
                out / "model.pkl", compress=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", nargs="*", default=list(SEARCH))
    ap.add_argument("--scheme", nargs="*", default=["random", "knndm", "temporal", "huc8"])
    ap.add_argument("--model", nargs="*", default=["Combined", "Lake", "River", "Estuary"])
    args = ap.parse_args()
    table = model_table()
    levels = {c: sorted(table[c].astype(str).unique()) for c in CATEGORICAL}
    print(f"model data: {len(table):,} matchups at {table.StationID.nunique():,} sites")
    for algorithm in args.algorithm:
        for scheme in args.scheme:
            for model in args.model:
                run(table, levels, algorithm, scheme, model)


if __name__ == "__main__":
    main()

# Step 6: SHAP values of the final models.
#
# Input : results/<algorithm>/<scheme>/<model>/model.pkl
# Output: results/<algorithm>/<scheme>/<model>/shap_values.parquet and shap_importance.csv
#
# CatBoost, LightGBM, XGBoost: exact TreeSHAP of each package, all matchups.
# Random Forest: TreeSHAP of the shap package for 500 random matchups (slow for deep forests).
# MLP: permutation SHAP with 50 background matchups.
# Importance of a feature = mean |SHAP| as a share of the sum over all features (%).

import argparse

import joblib
import numpy as np
import pandas as pd

from step4_train_models import CATEGORICAL, FEATURES, RESULTS, SEED, encode, model_table


def shap_values(algorithm, bundle, X):
    m = bundle["model"]
    if algorithm == "CatBoost":
        from catboost import Pool
        v = m.est.get_feature_importance(Pool(X[FEATURES], cat_features=CATEGORICAL), type="ShapValues")
        return pd.DataFrame(v[:, :-1], columns=FEATURES)
    if algorithm == "LightGBM":
        v = m.est.predict(m.matrix(X), pred_contrib=True)
        return pd.DataFrame(v[:, :-1], columns=FEATURES)
    if algorithm == "XGBoost":
        import xgboost as xgb
        v = m.est.get_booster().predict(xgb.DMatrix(m.matrix(X), enable_categorical=True), pred_contribs=True)
        return pd.DataFrame(v[:, :-1], columns=FEATURES)
    if algorithm == "RandomForest":
        import shap
        E = m.matrix(X)
        v = pd.DataFrame(shap.TreeExplainer(m.est).shap_values(E, check_additivity=False), columns=E.columns)
        return pd.DataFrame({f: v[[c for c in E.columns if c == f or c.startswith(f + "=")]].sum(axis=1)
                             for f in FEATURES})
    if algorithm == "MLP":
        import shap
        levels = bundle["levels"]
        Z = X[FEATURES].copy()
        for c in CATEGORICAL:                       # the location type is ONE feature here
            Z[c] = pd.Categorical(Z[c].astype(str), categories=levels[c]).codes
        Z = Z.astype(float).values

        def predict(z):
            frame = pd.DataFrame(z, columns=FEATURES)
            for c in CATEGORICAL:
                codes = np.clip(np.rint(frame[c]).astype(int), 0, len(levels[c]) - 1)
                frame[c] = np.asarray(levels[c])[codes]
            return m.predict(frame)

        background = Z[np.random.default_rng(SEED).choice(len(Z), min(50, len(Z)), replace=False)]
        explainer = shap.PermutationExplainer(predict, shap.maskers.Independent(background), seed=SEED)
        result = explainer(Z, max_evals=2 * len(FEATURES) + 1, silent=True)
        return pd.DataFrame(result.values, columns=FEATURES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", nargs="*", default=["CatBoost", "LightGBM", "XGBoost", "RandomForest", "MLP"])
    ap.add_argument("--scheme", nargs="*", default=["knndm", "random", "temporal", "huc8"])
    ap.add_argument("--model", nargs="*", default=["Combined", "Lake", "River", "Estuary"])
    args = ap.parse_args()
    table = model_table()
    for algorithm in args.algorithm:
        for scheme in args.scheme:
            for model in args.model:
                folder = RESULTS / algorithm / scheme / model
                if not (folder / "model.pkl").exists():
                    continue
                X = (table if model == "Combined" else table[table.WATER_TYPE == model]).reset_index(drop=True)
                if algorithm == "RandomForest" and len(X) > 500:
                    X = X.sample(500, random_state=SEED).reset_index(drop=True)
                values = shap_values(algorithm, joblib.load(folder / "model.pkl"), X)
                values.to_parquet(folder / "shap_values.parquet", index=False)
                imp = values.abs().mean().sort_values(ascending=False)
                share = (100 * imp / imp.sum()).round(2)
                share.rename("share_pct").to_csv(folder / "shap_importance.csv", index_label="feature")
                print(f"{algorithm:<12} {scheme:<8} {model:<8} top: " +
                      ", ".join(f"{f} {p:.1f}%" for f, p in share.head(3).items()))


if __name__ == "__main__":
    main()

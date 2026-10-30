# Step 5: three standard TSS algorithms, refitted on the same folds as CatBoost.
#
# Input : data/final_TSS.parquet, results/CatBoost/<scheme>/<model>/predictions.parquet (folds)
# Output: results/<method>/<scheme>/<model>/predictions.parquet and scores.json
#
#   LogLinear : log(1 + TSS) = a + b ln(B4) + c ln(B8A)
#   Nechad    : TSS = A rho / (1 - rho / C), rho = B4 (Nechad et al. 2010)
#   Dogliotti : Nechad with B4 below 0.05, with B8A above 0.07, blended in between (Dogliotti et al. 2015)

import json

import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression

from step4_train_models import RESULTS, model_table, scores

SCHEMES = ["random", "knndm", "temporal", "huc8"]
MODELS = ["Combined", "Lake", "River", "Estuary"]


class LogLinear:
    def fit(self, d, y):
        self.m = LinearRegression().fit(self.x(d), y)
        return self

    def predict(self, d):
        return self.m.predict(self.x(d))

    @staticmethod
    def x(d):
        return np.c_[np.log(d["B4"].clip(lower=1e-4)), np.log(d["B8A"].clip(lower=1e-4))]


class Nechad:
    def __init__(self, band="B4"):
        self.band = band

    def fit(self, d, y):
        rho = d[self.band].clip(lower=1e-4).to_numpy()
        y = np.asarray(y)
        c_min = rho.max() * 1.01

        def residuals(p):
            a, c = p
            return np.log1p(np.maximum(a * rho / (1 - rho / c), 0)) - y

        self.a, self.c = least_squares(residuals, x0=[300.0, max(0.2, c_min * 1.5)],
                                       bounds=([0.001, c_min], [1e6, 10.0])).x
        return self

    def mgl(self, d):
        rho = d[self.band].clip(lower=1e-4).to_numpy()
        return np.maximum(self.a * rho / (1 - np.minimum(rho / self.c, 0.999)), 0)

    def predict(self, d):
        return np.log1p(self.mgl(d))


class Dogliotti:
    LOW, HIGH = 0.05, 0.07

    def fit(self, d, y):
        y = np.asarray(y)
        red = d["B4"].to_numpy()
        clear, turbid = red < self.HIGH, red > self.LOW
        self.red = Nechad("B4").fit(d[clear], y[clear]) if clear.sum() >= 30 else Nechad("B4").fit(d, y)
        self.nir = Nechad("B8A").fit(d[turbid], y[turbid]) if turbid.sum() >= 30 else Nechad("B8A").fit(d, y)
        return self

    def predict(self, d):
        w = np.clip((d["B4"].to_numpy() - self.LOW) / (self.HIGH - self.LOW), 0, 1)
        return np.log1p((1 - w) * self.red.mgl(d) + w * self.nir.mgl(d))


def main():
    import pandas as pd
    table = model_table()
    for scheme in SCHEMES:
        for model in MODELS:
            sub = (table if model == "Combined" else table[table.WATER_TYPE == model]).reset_index(drop=True)
            ref = pd.read_parquet(RESULTS / "CatBoost" / scheme / model / "predictions.parquet")
            assert (ref.RowID.values == sub.RowID.values).all()
            folds = ref["fold"].to_numpy()
            y = np.log1p(sub["ResultValue_mgL"].astype(float)).to_numpy()
            for method in [LogLinear, Nechad, Dogliotti]:
                oof = np.full(len(sub), np.nan)
                for f in np.unique(folds):
                    test = folds == f
                    oof[test] = method().fit(sub[~test], y[~test]).predict(sub[test])
                sc = scores(y, oof)
                sc["train_R2_log"] = scores(y, method().fit(sub, y).predict(sub))["R2_log"]
                out = RESULTS / method.__name__ / scheme / model
                out.mkdir(parents=True, exist_ok=True)
                pred = ref[["RowID", "StationID", "WATER_TYPE", "ResultValue_mgL", "fold"]].copy()
                pred["pred_log"], pred["TSS_pred"] = oof, np.maximum(np.exp(oof) * sc["S"] - 1, 0)
                pred.to_parquet(out / "predictions.parquet", index=False)
                (out / "scores.json").write_text(json.dumps(sc, indent=2))
                print(f"{method.__name__:<10} {scheme:<8} {model:<8} R2_log {sc['R2_log']:.3f}  MdSA {sc['MdSA']:.1f}%")


if __name__ == "__main__":
    main()

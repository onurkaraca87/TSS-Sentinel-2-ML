# Step 7: TSS maps before and after a tropical cyclone.
#
#   python step7_storm_maps.py okeechobee     Lake Okeechobee, Hurricane Ian (2022), Lake model
#   python step7_storm_maps.py houston        Houston Ship Channel, Tropical Storm Imelda (2019), River model
#
# 1. find the Sentinel-2 L2A scenes of the case box (Microsoft Planetary Computer) and keep the dates
#    with < 30% cloud (SCL 3, 8, 9, 10) in the box
# 2. download the bands, SCL and metadata of each scene, cut to the box + about 2 km (values unchanged)
# 3. put the scenes of one date on one 10 m grid whose 40 m cells line up with the tile grid
#    (reflectance = (DN + BOA_ADD_OFFSET) / 10,000, as in the Earth Engine collection)
# 4. download Cloud Score+ (cs_cdf) for the same grid from Earth Engine
# 5. apply the model to every 40 m cell and write a GeoTIFF
# 6. statistics of every date over a fixed water mask
#
# Output: data/storms/<case>/ (scenes, raw data) and results/storms/<case>/ (TSS_<date>.tif, map_statistics.csv)

import json
import math
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds
from scipy import ndimage

import tss_model

ROOT = Path(__file__).resolve().parents[1]
EE_PROJECT = "your-ee-project"           # your Earth Engine (Google Cloud) project
CASES = {
    "okeechobee": {"box": [-81.13, 26.68, -80.60, 27.21], "start": "2022-08-29", "end": "2022-10-28",
                   "landfall": "2022-09-28", "model": "CatBoost_kNNDM_Lake.pkl",
                   "location_type": "Lake, Reservoir, Impoundment", "extra_dates": [],
                   "mask_min_cells": None},                  # None = largest connected water body
    "houston": {"box": [-95.32, 29.68, -94.97, 29.92], "start": "2019-08-18", "end": "2019-10-17",
                "landfall": "2019-09-17", "model": "CatBoost_kNNDM_River.pkl",
                "location_type": "River/Stream", "extra_dates": ["2019-09-21", "2019-10-01"],
                "mask_min_cells": 50},                       # every water body of >= 50 cells
}
MARGIN = 0.02            # degrees around the box
MAX_CLOUD_PCT = 30
ASSETS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
_token = {}


# ------------------------------------------------------------------ download
def token():
    if "t" not in _token or time.time() - _token["time"] > 1200:
        out = subprocess.run(["curl", "-s", "-m", "60", TOKEN_URL], capture_output=True)
        _token["t"], _token["time"] = json.loads(out.stdout)["token"], time.time()
    return _token["t"]


def curl(url, target=None):
    args = ["curl", "-s", "-f", "-L", "-m", "600", "--retry", "6", "--retry-delay", "10"]
    if target:
        subprocess.run(args + ["-o", str(target), url], check=True)
        return None
    return subprocess.run(args + [url], capture_output=True, check=True).stdout


def box_window(src, box):
    tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    xs, ys = tr.transform([box[0], box[2], box[0], box[2]], [box[1], box[1], box[3], box[3]])
    return from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform)


def find_scenes(case, folder):
    c = CASES[case]
    body = {"collections": ["sentinel-2-l2a"], "bbox": c["box"], "limit": 500,
            "datetime": f"{c['start']}T00:00:00Z/{c['end']}T23:59:59Z"}
    items = json.loads(subprocess.run(["curl", "-s", "-m", "120", "-X", "POST", STAC + "/search", "-H",
                                       "Content-Type: application/json", "-d", json.dumps(body)],
                                      capture_output=True).stdout)["features"]
    rows = []
    for it in sorted(items, key=lambda i: i["properties"]["datetime"]):
        p = it["properties"]
        with rasterio.open(it["assets"]["SCL"]["href"] + "?" + token()) as src:
            s = src.read(1, window=box_window(src, c["box"]), boundless=True, fill_value=0)
        rows.append({"date": p["datetime"][:10], "tile": p["s2:mgrs_tile"], "item": it["id"],
                     "tile_cloud_pct": p.get("eo:cloud_cover"), "box_water_px": int((s == 6).sum()),
                     "box_cloud_px": int(np.isin(s, [3, 8, 9, 10]).sum()), "box_valid_px": int((s > 0).sum())})
    scenes = pd.DataFrame(rows)
    scenes.to_csv(folder / "scene_list.csv", index=False)
    d = scenes.groupby("date")[["box_water_px", "box_cloud_px", "box_valid_px"]].sum()
    d["box_cloud_pct"] = (100 * d.box_cloud_px / d.box_valid_px).round(1)
    d["kept"] = (d.box_cloud_pct < MAX_CLOUD_PCT) | d.index.isin(c["extra_dates"])
    d.to_csv(folder / "dates.csv")
    # earliest processing of every tile and kept date (closest to the processing of the training images)
    s = scenes[scenes.date.isin(d.index[d.kept])].copy()
    s["processed"] = s["item"].str.rsplit("_", n=1).str[1]
    used = s.sort_values("processed").groupby(["date", "tile"], as_index=False).first()
    used.to_csv(folder / "items_used.csv", index=False)
    print(f"{len(scenes)} scenes, {int(d.kept.sum())} dates kept")


def download(case, folder):
    c = CASES[case]
    box = [c["box"][0] - MARGIN, c["box"][1] - MARGIN, c["box"][2] + MARGIN, c["box"][3] + MARGIN]
    for _, r in pd.read_csv(folder / "items_used.csv").iterrows():
        out = folder / "raw" / r["date"] / r["item"]
        if (out / "stac_item.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        item = json.loads(curl(f"{STAC}/collections/sentinel-2-l2a/items/{r['item']}"))
        for key, asset in item["assets"].items():
            if key in {"tilejson", "rendered_preview", "preview"}:
                continue
            name = asset["href"].split("?")[0].rsplit("/", 1)[1]
            if name.lower().endswith((".tif", ".jp2")):
                with rasterio.open(asset["href"] + "?" + token()) as src:
                    win = box_window(src, box).round_offsets().round_lengths().intersection(
                        Window(0, 0, src.width, src.height))
                    data = src.read(window=win)
                    prof = src.profile.copy()
                    prof.update(width=win.width, height=win.height, transform=src.window_transform(win),
                                compress="deflate", tiled=True, blockxsize=256, blockysize=256, driver="GTiff")
                    prof.pop("photometric", None)
                with rasterio.open(out / Path(name).with_suffix(".tif").name, "w", **prof) as dst:
                    dst.write(data)
            else:
                curl(asset["href"] + "?" + token(), out / name)
        (out / "stac_item.json").write_text(json.dumps(item, indent=1), encoding="utf-8")
        print("downloaded", r["item"])


# ------------------------------------------------------------------ product
def boa_offsets(mtd_xml):
    root = ET.parse(mtd_xml).getroot()
    ids = {e.attrib.get("bandId"): e.attrib["physicalBand"] for e in root.iter()
           if e.tag.endswith("Spectral_Information") and "physicalBand" in e.attrib}
    off = {ids.get(e.attrib.get("band_id"), e.attrib.get("band_id")): int(float(e.text))
           for e in root.iter() if e.tag.endswith("BOA_ADD_OFFSET")}
    quant = next((float(e.text) for e in root.iter() if e.tag.endswith("BOA_QUANTIFICATION_VALUE")), 10000.0)
    return off, quant


def scenes_of_date(item_dirs):
    rows = []
    for d in item_dirs:
        stac = json.loads((d / "stac_item.json").read_text(encoding="utf-8"))
        files = {k: d / Path(a["href"].split("?")[0].rsplit("/", 1)[1]).with_suffix(".tif").name
                 for k, a in stac["assets"].items()}
        rows.append({"dir": d, "tile": stac["properties"]["s2:mgrs_tile"], "files": files,
                     "cloud": stac["properties"].get("eo:cloud_cover"), "epsg": stac["properties"].get("proj:epsg"),
                     "origin": (stac["assets"]["B02"]["proj:transform"][2], stac["assets"]["B02"]["proj:transform"][5])})
    return rows


def make_grid(case, all_scenes, scene_list):
    """10 m grid over box + margin; its corner lies on the 40 m lattice of the tile with the most water."""
    c = CASES[case]
    order = list(scene_list.groupby("tile").box_water_px.mean().sort_values(ascending=False).index)
    first = next(s for s in all_scenes if s["tile"] == order[0])
    crs = f"EPSG:{int(first['epsg'])}"
    ax, ay = first["origin"]
    b = [c["box"][0] - MARGIN, c["box"][1] - MARGIN, c["box"][2] + MARGIN, c["box"][3] + MARGIN]
    xs, ys = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]])
    x0 = ax + math.floor((min(xs) - ax) / 40) * 40
    y1 = ay + math.ceil((max(ys) - ay) / 40) * 40
    return {"crs": crs, "transform": from_origin(x0, y1, 10, 10), "width": int(math.ceil((max(xs) - x0) / 40)) * 4,
            "height": int(math.ceil((y1 - min(ys)) / 40)) * 4, "order": order}


def on_grid(path, grid):
    out = np.zeros((grid["height"], grid["width"]), dtype="float32")
    with rasterio.open(path) as src:
        reproject(src.read(1).astype("float32"), out, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=grid["transform"], dst_crs=grid["crs"], resampling=Resampling.nearest,
                  src_nodata=0, dst_nodata=0)
    return out


def mosaic(scenes, grid):
    """Scenes of one date on the grid; where tiles overlap, a pixel takes all bands from one tile."""
    h, w = grid["height"], grid["width"]
    refl = np.full((len(ASSETS), h, w), np.nan, dtype="float32")
    scl = np.zeros((h, w), dtype="uint8")
    filled = np.zeros((h, w), dtype=bool)
    weights = []
    for s in sorted(scenes, key=lambda s: grid["order"].index(s["tile"])):
        off, quant = boa_offsets(s["dir"] / "MTD_MSIL2A.xml")
        b2 = on_grid(s["files"]["B02"], grid)
        take = (b2 > 0) & ~filled
        for i, (asset, band) in enumerate(zip(ASSETS, tss_model.BANDS)):
            dn = b2 if asset == "B02" else on_grid(s["files"][asset], grid)
            v = (dn + off.get(asset, off.get(band, 0))) / quant
            v[dn == 0] = np.nan
            refl[i][take] = v[take]
        scl[take] = on_grid(s["files"]["SCL"], grid).astype("uint8")[take]
        filled |= take
        weights.append((float(s["cloud"]), int(take.sum())))
    cloud = sum(c * n for c, n in weights) / max(sum(n for _, n in weights), 1)   # area-weighted tile cloud cover
    return refl, scl, cloud


def cloud_score(date, tiles, grid, target):
    """Cloud Score+ (cs_cdf) of the date on the grid, from Earth Engine (in blocks of 2,000 pixels)."""
    import ee
    ee.Initialize(project=EE_PROJECT)
    d = ee.Date(date)
    col = (ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filterDate(d, d.advance(1, "day"))
           .filter(ee.Filter.Or(*[ee.Filter.stringContains("system:index", "T" + t) for t in tiles])))
    image = col.select("cs_cdf").mosaic().unmask(-1).toFloat()
    a, b, c, dd, e, f = list(grid["transform"])[:6]
    cs = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
    with tempfile.TemporaryDirectory() as tmp:
        for r0 in range(0, grid["height"], 2000):
            for c0 in range(0, grid["width"], 2000):
                cw, ch = min(2000, grid["width"] - c0), min(2000, grid["height"] - r0)
                url = image.getDownloadURL({"crs": grid["crs"], "crs_transform": [a, b, c + c0 * a, dd, e, f + r0 * e],
                                            "dimensions": f"{cw}x{ch}", "format": "GEO_TIFF"})
                part = Path(tmp) / "part.tif"
                curl(url, part)
                with rasterio.open(part) as src:
                    cs[r0:r0 + ch, c0:c0 + cw] = src.read(1)
    cs[cs < 0] = np.nan
    write(target, cs, grid["transform"], grid["crs"])
    return cs


def write(path, arr, transform, crs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
                       crs=crs, transform=transform, nodata=np.nan, compress="deflate") as dst:
        dst.write(arr.astype("float32"), 1)


def tss_map(bundle, refl, scl, cs, cloud, date, grid, location_type):
    """Model applied to every 40 m cell (in blocks of 1,024 rows of 10 m pixels)."""
    tr = grid["transform"]
    n_rows, n_cols = grid["height"] // 4, grid["width"] // 4
    to_ll = Transformer.from_crs(grid["crs"], "EPSG:4326", always_xy=True)
    tss = np.full((n_rows, n_cols), np.nan, dtype="float32")
    for a in range(0, n_rows, 256):
        b = min(a + 256, n_rows)
        rows = slice(a * 4, b * 4)
        bands = {n: refl[i][rows] for i, n in enumerate(tss_model.BANDS)}
        r, cc = np.mgrid[a:b, 0:n_cols]
        lon, lat = to_ll.transform(tr.c + (cc * 4 + 2) * tr.a, tr.f + (r * 4 + 2) * tr.e)
        cells = tss_model.cells_from_image(bands, scl[rows], cs[rows], lon, lat, date, location_type, cloud)
        tss[a:b] = tss_model.predict_tss(bundle, cells).reshape(b - a, n_cols)
    return tss


def water_mask(maps, min_cells):
    """Fixed water mask from the date with the most mapped cells."""
    best = max(maps, key=lambda d: np.isfinite(maps[d]).sum())
    lab, n = ndimage.label(np.isfinite(maps[best]))
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    if min_cells is None:
        return lab == int(np.argmax(sizes)) + 1
    return np.isin(lab, [i + 1 for i in range(n) if sizes[i] >= min_cells])


def main(case):
    c = CASES[case]
    data = ROOT / "data" / "storms" / case
    out = ROOT / "results" / "storms" / case
    data.mkdir(parents=True, exist_ok=True)
    if not (data / "items_used.csv").exists():
        find_scenes(case, data)
    download(case, data)

    item_dirs = sorted((data / "raw").glob("*/*/stac_item.json"))
    by_date = {}
    for f in item_dirs:
        by_date.setdefault(f.parent.parent.name, []).append(f.parent)
    all_scenes = [s for dirs in by_date.values() for s in scenes_of_date(dirs)]
    grid = make_grid(case, all_scenes, pd.read_csv(data / "scene_list.csv"))
    bundle = tss_model.load(ROOT / "models" / c["model"])

    maps = {}
    for date, dirs in sorted(by_date.items()):
        scenes = scenes_of_date(dirs)
        refl, scl, cloud = mosaic(scenes, grid)
        cs_file = data / "raw" / date / "CS_CDF_10m.tif"
        if cs_file.exists():
            with rasterio.open(cs_file) as src:
                cs = src.read(1)
        else:
            cs = cloud_score(date, [s["tile"] for s in scenes], grid, cs_file)
        maps[date] = tss_map(bundle, refl, scl, cs, cloud, date, grid, c["location_type"])
        write(out / f"TSS_{date}.tif", maps[date], grid["transform"] * grid["transform"].scale(4, 4), grid["crs"])
        print(date, f"median TSS {np.nanmedian(maps[date]):.1f} mg/L")

    mask = water_mask(maps, c["mask_min_cells"])
    rows = []
    for date, m in maps.items():
        v = m[mask & np.isfinite(m)]
        rows.append({"date": date, "days_from_landfall": (pd.Timestamp(date) - pd.Timestamp(c["landfall"])).days,
                     "seen_pct": round(100 * v.size / mask.sum(), 1), "median": np.median(v) if v.size else np.nan,
                     "p25": np.percentile(v, 25) if v.size else np.nan, "p75": np.percentile(v, 75) if v.size else np.nan})
    stats = pd.DataFrame(rows)
    stats.to_csv(out / "map_statistics.csv", index=False)
    write(out / "water_mask.tif", mask.astype("float32"), grid["transform"] * grid["transform"].scale(4, 4), grid["crs"])
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "okeechobee")

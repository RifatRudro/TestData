#!/usr/bin/env python
"""Download every dataset the paper needs into the project folder and build data/master.csv.

    python download_data.py                       # ISIC 2018 Task 3 (= HAM10000) + masks + metadata
    python download_data.py --with_bcn            # + BCN20000 external test (needs: pip install isic-cli)
    python download_data.py --with_test_meta      # + age/sex/site of ISIC 2018 test images (ISIC API)
    python download_data.py --skip_prepare        # download only

Layout created (inside the project):

    data/raw/isic2018/ISIC2018_Task3_Training_Input/      10,015 HAM10000 images
    data/raw/isic2018/ISIC2018_Task3_Training_GroundTruth/
    data/raw/isic2018/ISIC2018_Task3_Training_LesionGroupings.csv   (lesion_id per image)
    data/raw/isic2018/ISIC2018_Task3_Validation_Input/ + _GroundTruth/     193 images
    data/raw/isic2018/ISIC2018_Task3_Test_Input/ + _GroundTruth/          1,512 images
    data/raw/ham10000/HAM10000_metadata.csv               age / sex / localization (Dataverse or Kaggle)
    data/raw/ham10000/masks/                              lesion masks (Dataverse or Kaggle)
    data/raw/bcn20000/                                    optional
    data/master.csv, data/cache512/                       built by prepare_data.py

Sources: ISIC Challenge 2018 (CC BY-NC), HAM10000 Harvard Dataverse doi:10.7910/DVN/DBW86T
(CC BY-NC 4.0), Kaggle mirrors kmader/skin-cancer-mnist-ham10000 and
tschandl/ham10000-lesion-segmentations as fallback (needs ~/.kaggle/kaggle.json), ISIC Archive.
Downloads resume if interrupted; finished steps are skipped on re-run. ~3 GB in total.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ISIC_BASE = "https://isic-archive.s3.amazonaws.com/challenges/2018/"
ISIC_FILES = [
    "ISIC2018_Task3_Training_Input.zip",
    "ISIC2018_Task3_Training_GroundTruth.zip",
    "ISIC2018_Task3_Training_LesionGroupings.csv",
    "ISIC2018_Task3_Validation_Input.zip",
    "ISIC2018_Task3_Validation_GroundTruth.zip",
    "ISIC2018_Task3_Test_Input.zip",
    "ISIC2018_Task3_Test_GroundTruth.zip",
]
DATAVERSE = "https://dataverse.harvard.edu"
HAM_DOI = "doi:10.7910/DVN/DBW86T"
ISIC_API = "https://api.isic-archive.com/api/v2"
BCN_COLLECTION = 249
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
UA = {"User-Agent": "dermalign-download/1.0"}


# ----------------------------------------------------------------------------- helpers
def log(msg):
    print(f"[download] {msg}", flush=True)


def fetch(url, dest: Path, retries=4, chunk=1 << 20):
    """Resumable download (HTTP Range) with retries. Skips if dest exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"have {dest.name}")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={**UA, **({"Range": f"bytes={have}-"} if have else {})})
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored Range: restart
                    have = 0
                total = int(r.headers.get("Content-Length") or 0) + have
                mode = "ab" if have else "wb"
                done, last = have, time.time()
                with open(part, mode) as f:
                    while True:
                        buf = r.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        if time.time() - last > 5:
                            pct = f"{100 * done / total:5.1f}%" if total else ""
                            log(f"{dest.name}: {done / 1e6:8.1f} MB {pct}")
                            last = time.time()
            part.rename(dest)
            log(f"got {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            log(f"{dest.name}: attempt {attempt}/{retries} failed: {e}")
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"could not download {url}")


def get_json(url, timeout=60):
    req = urllib.request.Request(url, headers={**UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def extract(zip_path: Path, dest: Path):
    marker = dest / f".{zip_path.stem}.extracted"
    if marker.exists():
        return
    log(f"extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    marker.touch()


def img_dir(root: Path, name: str) -> Path:
    """Folder holding the images of one ISIC zip (zips normally contain a top-level folder)."""
    p = root / name
    return p if p.exists() else root


def find_one(root: Path, pattern: str):
    hits = sorted(root.rglob(pattern))
    return hits[0] if hits else None


# ----------------------------------------------------------------------------- sources
def get_isic2018(raw: Path, base: str):
    d = raw / "isic2018"
    for name in ISIC_FILES:
        f = fetch(base + name, d / "zips" / name if name.endswith(".zip") else d / name)
        if name.endswith(".zip"):
            extract(f, d)
    return d


def get_dataverse(raw: Path, base: str):
    """HAM10000 metadata (age/sex/site) and lesion masks from the Harvard Dataverse."""
    d = raw / "ham10000"
    meta_out, masks_dir = d / "HAM10000_metadata.csv", d / "masks"
    if meta_out.exists() and masks_dir.exists() and any(masks_dir.rglob("*.png")):
        log("have Dataverse metadata + masks")
        return meta_out, masks_dir
    try:
        listing = get_json(f"{base}/api/datasets/:persistentId/versions/:latest-published/files?persistentId={HAM_DOI}")
        files = listing.get("data", [])
    except Exception as e:  # noqa: BLE001
        log(f"Dataverse listing failed ({e})")
        return None, None
    by_name = {}
    for it in files:
        df = it.get("dataFile", {})
        for key in ("originalFileName", "filename"):
            if df.get(key):
                by_name[df[key]] = df["id"]
    log(f"Dataverse lists {len(files)} files: {sorted(by_name)[:12]}")
    meta_id = next((by_name[n] for n in by_name if n.startswith("HAM10000_metadata")), None)
    mask_id = next((by_name[n] for n in by_name if "segmentation" in n.lower() and n.endswith(".zip")), None)
    got_meta = got_mask = None
    if meta_id is not None:
        try:
            tmp = fetch(f"{base}/api/access/datafile/{meta_id}?format=original", d / "HAM10000_metadata.raw")
            m = pd.read_csv(tmp, sep=None, engine="python")
            m.to_csv(meta_out, index=False)
            got_meta = meta_out
        except Exception as e:  # noqa: BLE001
            log(f"metadata download failed ({e})")
    if mask_id is not None:
        try:
            z = fetch(f"{base}/api/access/datafile/{mask_id}", d / "zips" / "HAM10000_segmentations.zip")
            extract(z, masks_dir)
            got_mask = masks_dir
        except Exception as e:  # noqa: BLE001
            log(f"mask download failed ({e})")
    return got_meta, got_mask


def get_kaggle(raw: Path, need_meta: bool, need_masks: bool):
    """Fallback: Kaggle mirrors (requires the kaggle CLI and ~/.kaggle/kaggle.json)."""
    if not shutil.which("kaggle"):
        log("kaggle CLI not found - skipping Kaggle fallback (pip install kaggle + API token)")
        return None, None
    d = raw / "ham10000"
    meta = masks = None
    if need_meta:
        r = subprocess.run(["kaggle", "datasets", "download", "-d", "kmader/skin-cancer-mnist-ham10000",
                            "-f", "HAM10000_metadata.csv", "-p", str(d)], capture_output=True, text=True)
        z = d / "HAM10000_metadata.csv.zip"
        if z.exists():
            extract(z, d)
        meta = d / "HAM10000_metadata.csv" if (d / "HAM10000_metadata.csv").exists() else None
        if meta is None:
            log(f"Kaggle metadata failed: {r.stderr.strip()[:200]}")
    if need_masks:
        r = subprocess.run(["kaggle", "datasets", "download", "-d", "tschandl/ham10000-lesion-segmentations",
                            "-p", str(d / "zips"), "--unzip"], capture_output=True, text=True)
        src = d / "zips"
        pngs = list(src.rglob("*.png"))
        if pngs:
            (d / "masks").mkdir(parents=True, exist_ok=True)
            for p in pngs:
                shutil.move(str(p), d / "masks" / p.name)
            masks = d / "masks"
        else:
            log(f"Kaggle masks failed: {r.stderr.strip()[:200]}")
    return meta, masks


def metadata_from_isic(isic_dir: Path, out_csv: Path):
    """Fallback HAM10000 metadata built from the ISIC 2018 files (lesion_id + dx, no age/sex/site)."""
    gt = pd.read_csv(find_one(isic_dir, "ISIC2018_Task3_Training_GroundTruth.csv"))
    grp = pd.read_csv(find_one(isic_dir, "ISIC2018_Task3_Training_LesionGroupings.csv"))
    gt["dx"] = gt[CLASSES].values.argmax(1)
    gt["dx"] = gt["dx"].map(lambda k: CLASSES[k].lower())
    m = gt[["image", "dx"]].merge(grp, on="image", how="left")
    out = pd.DataFrame({"lesion_id": m["lesion_id"].fillna(m["image"]), "image_id": m["image"], "dx": m["dx"],
                        "dx_type": m.get("diagnosis_confirm_type"), "age": None, "sex": "unknown",
                        "localization": "unknown"})
    out.to_csv(out_csv, index=False)
    log(f"built {out_csv.name} from ISIC lesion groupings ({out.lesion_id.nunique()} lesions, no age/sex/site)")
    return out_csv


def get_test_meta(isic_dir: Path, api: str, workers=8):
    """age / sex / anatomical site of the ISIC 2018 test images from the ISIC API (optional)."""
    out = isic_dir / "ISIC2018_Task3_Test_Metadata.csv"
    if out.exists():
        return out
    ids = sorted(p.stem for p in img_dir(isic_dir, "ISIC2018_Task3_Test_Input").glob("ISIC_*.jpg"))

    def one(i):
        try:
            j = get_json(f"{api}/images/{i}/", timeout=30)
            c = (j.get("metadata") or {}).get("clinical") or {}
            return {"image_id": i, "age_approx": c.get("age_approx"), "sex": c.get("sex"),
                    "anatom_site_general": c.get("anatom_site_general")}
        except Exception:  # noqa: BLE001
            return {"image_id": i}

    log(f"querying ISIC API for {len(ids)} test images ...")
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, ids))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    log(f"test metadata: {df.get('sex', pd.Series(dtype=object)).notna().sum()} / {len(df)} images have values")
    return out


def get_bcn(raw: Path):
    """BCN20000 via the official ISIC command-line tool."""
    d = raw / "bcn20000"
    if (d / "metadata.csv").exists():
        return d / "metadata.csv", d
    if not shutil.which("isic"):
        log("isic-cli not found: pip install isic-cli, then re-run with --with_bcn")
        return None, None
    d.mkdir(parents=True, exist_ok=True)
    log("downloading BCN20000 with isic-cli (several GB, takes a while) ...")
    r = subprocess.run(["isic", "image", "download", "--collections", str(BCN_COLLECTION), str(d)])
    meta = d / "metadata.csv"
    if r.returncode != 0 or not meta.exists():
        log("isic-cli download failed - see its message above")
        return None, None
    return meta, d


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--isic_base", default=ISIC_BASE)
    ap.add_argument("--dataverse", default=DATAVERSE)
    ap.add_argument("--isic_api", default=ISIC_API)
    ap.add_argument("--no_kaggle", action="store_true")
    ap.add_argument("--with_bcn", action="store_true")
    ap.add_argument("--with_test_meta", action="store_true")
    ap.add_argument("--skip_prepare", action="store_true")
    ap.add_argument("--cache_short", type=int, default=512)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    data = Path(args.data_dir)
    raw = data / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    isic = get_isic2018(raw, args.isic_base)
    meta, masks = get_dataverse(raw, args.dataverse.rstrip("/"))
    if (meta is None or masks is None) and not args.no_kaggle:
        km, kk = get_kaggle(raw, meta is None, masks is None)
        meta, masks = meta or km, masks or kk
    if meta is None:
        (raw / "ham10000").mkdir(parents=True, exist_ok=True)
        meta = metadata_from_isic(isic, raw / "ham10000" / "HAM10000_metadata_from_isic.csv")
        log("NOTE: no age/sex/site metadata -> the --use_meta variant will see 'unknown' only")
    if masks is None:
        log("NOTE: no lesion masks found -> alignment loss and mask-based XAI metrics will be skipped. "
            "Get them from the HAM10000 Dataverse page or Kaggle 'tschandl/ham10000-lesion-segmentations' "
            f"and unzip into {raw / 'ham10000' / 'masks'}")
    test_meta = get_test_meta(isic, args.isic_api.rstrip("/")) if args.with_test_meta else None
    bcn_meta, bcn_dir = get_bcn(raw) if args.with_bcn else (None, None)

    # sanity checks
    train_dir = img_dir(isic, "ISIC2018_Task3_Training_Input")
    val_dir = img_dir(isic, "ISIC2018_Task3_Validation_Input")
    test_dir = img_dir(isic, "ISIC2018_Task3_Test_Input")
    n_train = len(list(train_dir.glob("ISIC_*.jpg")))
    n_test = len(list(test_dir.glob("ISIC_*.jpg")))
    n_masks = len(list(masks.rglob("*.png"))) if masks else 0
    log(f"training images: {n_train} (expected 10015), test images: {n_test} (expected 1512), masks: {n_masks}")

    if args.skip_prepare:
        return
    cmd = [sys.executable, str(ROOT / "prepare_data.py"),
           "--ham_dir", str(train_dir), "--ham_metadata", str(meta),
           "--isic18_test_dir", str(test_dir),
           "--isic18_test_gt", str(find_one(isic, "ISIC2018_Task3_Test_GroundTruth.csv")),
           "--isic18_val_dir", str(val_dir),
           "--isic18_val_gt", str(find_one(isic, "ISIC2018_Task3_Validation_GroundTruth.csv")),
           "--dhash", "--check_test_overlap",
           "--cache_dir", str(data / f"cache{args.cache_short}"), "--cache_short", str(args.cache_short),
           "--workers", str(args.workers), "--out", str(data / "master.csv")]
    if masks:
        cmd += ["--ham_masks_dir", str(masks)]
    if test_meta:
        cmd += ["--isic18_test_meta", str(test_meta)]
    if bcn_meta:
        cmd += ["--external", "bcn20000", str(bcn_meta), str(bcn_dir)]
    log("running prepare_data.py ...")
    subprocess.run(cmd, check=True, cwd=ROOT)
    log(f"done: {data / 'master.csv'}  ->  next: bash run_all.sh")


if __name__ == "__main__":
    main()

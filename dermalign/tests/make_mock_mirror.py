#!/usr/bin/env python
"""Build a local mock of the ISIC S3 bucket, the Harvard Dataverse API and the ISIC API from the
synthetic data, so download_data.py can be tested offline:

    python tests/make_synthetic.py --out /tmp/syn
    python tests/make_mock_mirror.py --syn /tmp/syn --www /tmp/www
    (cd /tmp/www && python -m http.server 8765) &
    python download_data.py --data_dir /tmp/dl --isic_base http://127.0.0.1:8765/challenges/2018/ \
        --dataverse http://127.0.0.1:8765 --isic_api http://127.0.0.1:8765/api/v2 --with_test_meta
"""
import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd


def zip_dir(files, zpath, arc_dir):
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w") as z:
        for f in files:
            z.write(f, f"{arc_dir}/{Path(f).name}")
        z.writestr(f"{arc_dir}/LICENSE.txt", "CC-BY-NC (mock)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn", required=True)
    ap.add_argument("--www", required=True)
    a = ap.parse_args()
    syn, www = Path(a.syn), Path(a.www)
    ch = www / "challenges" / "2018"
    ch.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(syn / "HAM10000" / "HAM10000_metadata.csv")
    imgs = sorted((syn / "HAM10000").rglob("ISIC_*.jpg"))
    zip_dir(imgs, ch / "ISIC2018_Task3_Training_Input.zip", "ISIC2018_Task3_Training_Input")
    cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    gt = pd.DataFrame({"image": meta.image_id, **{c: (meta.dx.str.upper() == c).astype(float) for c in cols}})
    tmp = www / "_tmp"
    tmp.mkdir(exist_ok=True)
    gt.to_csv(tmp / "ISIC2018_Task3_Training_GroundTruth.csv", index=False)
    zip_dir([tmp / "ISIC2018_Task3_Training_GroundTruth.csv"], ch / "ISIC2018_Task3_Training_GroundTruth.zip",
            "ISIC2018_Task3_Training_GroundTruth")
    pd.DataFrame({"image": meta.image_id, "lesion_id": meta.lesion_id,
                  "diagnosis_confirm_type": meta.dx_type}).to_csv(ch / "ISIC2018_Task3_Training_LesionGroupings.csv", index=False)
    timgs = sorted((syn / "ISIC2018_Task3_Test_Input").glob("ISIC_*.jpg"))
    tgt = pd.read_csv(syn / "ISIC2018_Task3_Test_GroundTruth.csv")
    zip_dir(timgs, ch / "ISIC2018_Task3_Test_Input.zip", "ISIC2018_Task3_Test_Input")
    tgt.to_csv(tmp / "ISIC2018_Task3_Test_GroundTruth.csv", index=False)
    zip_dir([tmp / "ISIC2018_Task3_Test_GroundTruth.csv"], ch / "ISIC2018_Task3_Test_GroundTruth.zip",
            "ISIC2018_Task3_Test_GroundTruth")
    vimgs, vgt = timgs[:20], tgt.iloc[:20]
    zip_dir(vimgs, ch / "ISIC2018_Task3_Validation_Input.zip", "ISIC2018_Task3_Validation_Input")
    vgt.to_csv(tmp / "ISIC2018_Task3_Validation_GroundTruth.csv", index=False)
    zip_dir([tmp / "ISIC2018_Task3_Validation_GroundTruth.csv"], ch / "ISIC2018_Task3_Validation_GroundTruth.zip",
            "ISIC2018_Task3_Validation_GroundTruth")

    # Dataverse API mock
    lst = www / "api" / "datasets" / ":persistentId" / "versions" / ":latest-published"
    lst.mkdir(parents=True, exist_ok=True)
    (lst / "files").write_text(json.dumps({"status": "OK", "data": [
        {"label": "HAM10000_metadata.tab", "dataFile": {"id": 1001, "filename": "HAM10000_metadata.tab",
                                                         "originalFileName": "HAM10000_metadata.csv"}},
        {"label": "HAM10000_segmentations_lesion_tschandl.zip",
         "dataFile": {"id": 1002, "filename": "HAM10000_segmentations_lesion_tschandl.zip"}},
        {"label": "HAM10000_images_part_1.zip", "dataFile": {"id": 1003, "filename": "HAM10000_images_part_1.zip"}},
    ]}))
    acc = www / "api" / "access" / "datafile"
    acc.mkdir(parents=True, exist_ok=True)
    meta.to_csv(acc / "1001", sep="\t", index=False)
    masks = sorted((syn / "HAM10000_segmentations").glob("*.png"))
    zip_dir(masks, acc / "1002", "HAM10000_segmentations_lesion_tschandl")

    # ISIC API mock: /api/v2/images/<id>/ -> JSON (served as index.html)
    for p in timgs:
        d = www / "api" / "v2" / "images" / p.stem
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(json.dumps({"isic_id": p.stem, "metadata": {"clinical": {
            "age_approx": 55, "sex": "female", "anatom_site_general": "posterior torso"}}}))
    print(f"mock mirror ready in {www}")


if __name__ == "__main__":
    main()

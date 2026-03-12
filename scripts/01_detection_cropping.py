"""
01_detect_and_crop.py
======================
Two-stage detect → crop pipeline for the DrivenData Conser-vision competition.

Stage 1 (detect):  Runs SpeciesNet's built-in MegaDetector via its detector-only
                    mode, producing a detections JSON per split.
Stage 2 (crop):    Reads the detections JSON and crops animal bounding boxes to
                    a structured directory with a metadata CSV.

Stages can be run together or independently (--skip_detect / --skip_crop).

Requirements:
    pip install speciesnet pillow pandas tqdm

Usage — full pipeline:
    python 01_detect_and_crop.py --data_dir ./data

Usage — detect only (e.g., on a GPU machine):
    python 01_detect_and_crop.py --data_dir ./data --skip_crop

Usage — crop only (rerun at different threshold without re-detecting):
    python 01_detect_and_crop.py --data_dir ./data --skip_detect --conf_threshold 0.15

Expected data layout (--data_dir):
    data/
      train_features/       <- training images
      train_features.csv    <- columns: id, filepath
      train_labels.csv      <- columns: id, <label cols>
      test_features/        <- test images
      test_features.csv     <- columns: id, filepath

Output:
    data/
      detections/
        train_detections.json   <- SpeciesNet detector-only output
        test_detections.json
      crops/
        train/                  <- cropped training images
        test/                   <- cropped test images
        crop_metadata.csv       <- mapping table for downstream classifier
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Stage 1: Detection via SpeciesNet
# ---------------------------------------------------------------------------

def run_detection(image_folder: str, output_json: str) -> None:
    """
    Run SpeciesNet in detector-only mode on a folder of images.
    Wraps the speciesnet CLI so it handles model download, batching, 
    checkpointing, and GPU detection automatically.
    """
    cmd = [
        sys.executable, "-m", "speciesnet.scripts.run_model",
        "--detector_only",
        "--folders", image_folder,
        "--predictions_json", output_json,
    ]
    print(f"\n{'='*60}")
    print(f"Running MegaDetector (via SpeciesNet) on:\n  {image_folder}")
    print(f"Output: {output_json}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\nWARNING: SpeciesNet exited with code {result.returncode}")
        print("Check output above for errors. Common fixes:")
        print("  - Ensure speciesnet is installed: pip install speciesnet")
        print("  - GPU issues: pip install torch torchvision --upgrade")
    else:
        print(f"\nDetection complete in {elapsed:.1f}s")


def detect_splits(data_dir: Path, det_dir: Path, splits: list[str]) -> None:
    """Run detection for each split."""
    det_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        image_folder = data_dir / f"{split}_features"
        output_json = det_dir / f"{split}_detections.json"

        if not image_folder.exists():
            print(f"WARNING: {image_folder} not found, skipping {split}")
            continue

        if output_json.exists():
            print(f"\nDetections file already exists: {output_json}")
            print("  SpeciesNet will resume from its last checkpoint.")

        run_detection(str(image_folder), str(output_json))


# ---------------------------------------------------------------------------
# Stage 2: Cropping from detection JSON
# ---------------------------------------------------------------------------

def load_detections(det_json_path: str) -> dict:
    """
    Load SpeciesNet detector-only JSON output.
    Returns a dict mapping filepath → list of detections.
    
    Each detection has: category, label, conf, bbox [x, y, w, h] (normalized)
    """
    with open(det_json_path, "r") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    det_map = {}
    for pred in predictions:
        fp = pred["filepath"]
        dets = pred.get("detections", [])
        if dets is None:
            dets = []
        det_map[fp] = dets

    return det_map


def crop_single_image(
    image_path: str,
    detections: list,
    conf_threshold: float,
    box_expansion: float,
    category_filter: str = "1",  # animals only
) -> list:
    """
    Crop animal detections from a single image.

    Returns list of dicts with keys: crop (PIL.Image), conf, bbox
    """
    img = Image.open(image_path)
    img_w, img_h = img.size
    crops = []

    for det in detections:
        if det["conf"] < conf_threshold:
            continue
        if category_filter and det["category"] != category_filter:
            continue

        bx, by, bw, bh = det["bbox"]

        # Expand bbox proportionally
        exp_x = bw * box_expansion
        exp_y = bh * box_expansion
        bx = max(0.0, bx - exp_x)
        by = max(0.0, by - exp_y)
        bw = min(1.0 - bx, bw + 2 * exp_x)
        bh = min(1.0 - by, bh + 2 * exp_y)

        # Convert to pixels
        left = int(bx * img_w)
        top = int(by * img_h)
        right = int((bx + bw) * img_w)
        bottom = int((by + bh) * img_h)

        if (right - left) < 8 or (bottom - top) < 8:
            continue

        crops.append({
            "crop": img.crop((left, top, right, bottom)),
            "conf": det["conf"],
            "bbox": [bx, by, bw, bh],  # expanded bbox
        })

    return crops


def crop_split(
    features_csv: str,
    data_dir: Path,
    det_json_path: str,
    crop_dir: str,
    split_name: str,
    conf_threshold: float,
    box_expansion: float,
) -> list:
    """
    Crop all images for one split. Returns metadata rows.
    """
    df = pd.read_csv(features_csv)
    det_map = load_detections(det_json_path)

    os.makedirs(crop_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Cropping {split_name}: {len(df)} images")
    print(f"  Detection JSON: {det_json_path} ({len(det_map)} entries)")
    print(f"  Conf threshold: {conf_threshold}")
    print(f"  Box expansion:  {box_expansion}")
    print(f"{'='*60}")

    metadata_rows = []
    n_no_det = 0
    n_multi_det = 0
    n_missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Cropping {split_name}"):
        img_id = row["id"]
        filepath = row["filepath"]  # e.g., "train_features/xxxx.jpg"
        img_path = str(data_dir / filepath)

        if not os.path.exists(img_path):
            n_missing += 1
            continue

        # Match to detections — SpeciesNet may store absolute or relative paths
        # Try matching on filename, or the filepath as given
        dets = None
        for key in [img_path, filepath, os.path.basename(filepath)]:
            if key in det_map:
                dets = det_map[key]
                break

        # Fallback: match on filename suffix
        if dets is None:
            basename = os.path.basename(filepath)
            for det_key in det_map:
                if det_key.endswith(basename):
                    dets = det_map[det_key]
                    break

        if dets is None:
            dets = []

        # Filter to animal detections above threshold
        animal_dets = [
            d for d in dets
            if d["category"] == "1" and d["conf"] >= conf_threshold
        ]
        n_detections = len(animal_dets)

        if n_detections == 0:
            # No animal detected → full-image fallback
            n_no_det += 1
            crop_fname = f"{img_id}_full.jpg"
            crop_path = os.path.join(crop_dir, crop_fname)

            try:
                img = Image.open(img_path).convert("RGB")
                img.save(crop_path, "JPEG", quality=95)
            except Exception as e:
                print(f"  WARNING: Could not open {img_path}: {e}")
                continue

            metadata_rows.append({
                "original_id": img_id,
                "split": split_name,
                "crop_filename": crop_fname,
                "det_conf": 0.0,
                "bbox_x": 0.0,
                "bbox_y": 0.0,
                "bbox_w": 1.0,
                "bbox_h": 1.0,
                "n_detections": 0,
                "is_full_image": True,
            })
            continue

        if n_detections > 1:
            n_multi_det += 1

        # Crop each detection
        crops = crop_single_image(
            img_path, animal_dets,
            conf_threshold=conf_threshold,
            box_expansion=box_expansion,
        )

        for i, crop_info in enumerate(crops):
            suffix = f"_crop{i}" if len(crops) > 1 else "_crop"
            crop_fname = f"{img_id}{suffix}.jpg"
            crop_path = os.path.join(crop_dir, crop_fname)

            try:
                crop_info["crop"].convert("RGB").save(
                    crop_path, "JPEG", quality=95
                )
            except Exception as e:
                print(f"  WARNING: Could not save crop for {img_id}: {e}")
                continue

            bx, by, bw, bh = crop_info["bbox"]
            metadata_rows.append({
                "original_id": img_id,
                "split": split_name,
                "crop_filename": crop_fname,
                "det_conf": round(crop_info["conf"], 4),
                "bbox_x": round(bx, 4),
                "bbox_y": round(by, 4),
                "bbox_w": round(bw, 4),
                "bbox_h": round(bh, 4),
                "n_detections": n_detections,
                "is_full_image": False,
            })

    print(f"\n  Summary for {split_name}:")
    print(f"    Total images:          {len(df)}")
    print(f"    Missing from disk:     {n_missing}")
    print(f"    No animal detected:    {n_no_det} (full-image fallback)")
    print(f"    Multiple detections:   {n_multi_det}")
    print(f"    Total crops saved:     {len(metadata_rows)}")

    return metadata_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect (MegaDetector via SpeciesNet) → Crop pipeline"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root data directory containing train/test features and CSVs",
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=0.2,
        help="Detection confidence threshold (default: 0.2, MDv5 typical)",
    )
    parser.add_argument(
        "--box_expansion", type=float, default=0.1,
        help="Fractional bbox expansion (default: 0.1 = 10%% padding per side)",
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["train", "test"],
        help="Which splits to process (default: train test)",
    )
    parser.add_argument(
        "--skip_detect", action="store_true",
        help="Skip detection stage (reuse existing JSONs)",
    )
    parser.add_argument(
        "--skip_crop", action="store_true",
        help="Skip cropping stage (detection only)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    det_dir = data_dir / f"detections_{args.conf_threshold}"
    crop_dir = data_dir / f"crops_{args.conf_threshold}"

    # -----------------------------------------------------------------------
    # Stage 1: Detection
    # -----------------------------------------------------------------------
    if not args.skip_detect:
        detect_splits(data_dir, det_dir, args.splits)
    else:
        print("Skipping detection stage (--skip_detect)")

    # -----------------------------------------------------------------------
    # Stage 2: Cropping
    # -----------------------------------------------------------------------
    if not args.skip_crop:
        all_metadata = []

        for split in args.splits:
            features_csv = data_dir / f"{split}_features.csv"
            det_json = det_dir / f"{split}_detections.json"
            split_crop_dir = crop_dir / split

            if not features_csv.exists():
                print(f"WARNING: {features_csv} not found, skipping {split}")
                continue
            if not det_json.exists():
                print(f"WARNING: {det_json} not found — run detection first")
                continue

            rows = crop_split(
                features_csv=str(features_csv),
                data_dir=data_dir,
                det_json_path=str(det_json),
                crop_dir=str(split_crop_dir),
                split_name=split,
                conf_threshold=args.conf_threshold,
                box_expansion=args.box_expansion,
            )
            all_metadata.extend(rows)

        # Save metadata
        meta_df = pd.DataFrame(all_metadata)
        crop_dir.mkdir(parents=True, exist_ok=True)
        meta_path = crop_dir / "crop_metadata.csv"
        meta_df.to_csv(meta_path, index=False)
        print(f"\nMetadata saved to: {meta_path}")
        print(f"Total rows: {len(meta_df)}")

        if len(meta_df) > 0:
            print("\n--- Quick Stats ---")
            summary = meta_df.groupby("split").agg(
                n_images=("original_id", "nunique"),
                n_crops=("crop_filename", "count"),
                pct_full_image=("is_full_image", "mean"),
                avg_det_conf=("det_conf", "mean"),
            ).round(3)
            print(summary.to_string())
    else:
        print("Skipping crop stage (--skip_crop)")

    print("\nDone!")
    if not args.skip_crop:
        print("Next step: train classifier on crops directory.")
        print(f"  Crops:    {crop_dir / '<split>'}/")
        print(f"  Metadata: {crop_dir / 'crop_metadata.csv'}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# Undistort comparison: original vs live intrinsics vs refined intrinsics

import json, os, cv2, numpy as np
from pathlib import Path

# === CONFIG ===
# Point these to your two JSONs:
LIVE_JSON     = r"intrinsics_NDI_1080.json"          # the “live capture” one
REFINED_JSON  = r"intrinsics_NDI_1080_refined.json"  # the rerun/refined one

# Where your captured frames are. If your JSON's "samples" are relative
# (e.g., "calib_shots_NDI\\calib_2025...png"), this base will be joined first.
BASE_DIR      = Path(".")  # change to project root if needed

# How many images to compare (picked from the JSON's "samples" list):
NUM_IMAGES    = 8

# Output folder for side-by-side comparison PNGs:
OUT_DIR       = Path("undistort_comparisons")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ==============

def load_intrinsics(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)
    w, h = map(int, data["image_size"])
    return (w, h), K, dist, data

def safe_read(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def undistort_with(K, dist, img):
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1.0, (w, h))
    und = cv2.undistort(img, K, dist, None, newK)
    return und

def path_from_sample(sample_path: str, base_dir: Path) -> Path:
    # Normalize slashes and make relative samples work
    sample_path = sample_path.replace("\\", os.sep).replace("/", os.sep)
    p = Path(sample_path)
    if not p.is_absolute():
        p = base_dir / p
    return p

def make_panel(orig, und_live, und_ref, label_live="Live", label_ref="Refined"):
    # Annotate images and stack horizontally
    def put_label(img, text):
        out = img.copy()
        cv2.putText(out, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)
        return out
    a = put_label(orig, "Original")
    b = put_label(und_live, label_live)
    c = put_label(und_ref, label_ref)
    # Resize to same height (just in case) and hstack
    h = min(a.shape[0], b.shape[0], c.shape[0])
    def rh(x): 
        return cv2.resize(x, (int(x.shape[1]*h/x.shape[0]), h), interpolation=cv2.INTER_AREA)
    panel = cv2.hconcat([rh(a), rh(b), rh(c)])
    return panel

def main():
    # Load both intrinsics
    (wL, hL), K_live, dist_live, meta_live = load_intrinsics(LIVE_JSON)
    (wR, hR), K_ref,  dist_ref,  meta_ref  = load_intrinsics(REFINED_JSON)

    # Sanity: warn if the declared image sizes in JSONs differ
    if (wL, hL) != (wR, hR):
        print(f"[WARN] Image sizes differ: live={wL}x{hL}, refined={wR}x{hR}. Proceeding anyway.")

    # Choose sample list (prefer the live JSON’s samples)
    samples = meta_live.get("samples") or []
    if not samples:
        # fallback: try refined
        samples = meta_ref.get("samples") or []
    if not samples:
        raise RuntimeError("No 'samples' found in either JSON. Re-run with captured frames.")

    take = samples[:NUM_IMAGES]
    print(f"[INFO] Comparing {len(take)} image(s). Saving panels to: {OUT_DIR.resolve()}")

    for i, s in enumerate(take, 1):
        img_path = path_from_sample(s, BASE_DIR)
        img = safe_read(img_path)

        # Optionally apply the same flip the JSON was captured with (for visual parity)
        # If you want to preview exactly as captured, uncomment these two blocks.
        # if bool(meta_live.get("flipped_during_capture", False)):
        #     img_live_view = cv2.flip(img, -1)
        # else:
        #     img_live_view = img
        # if bool(meta_ref.get("flipped_during_capture", False)):
        #     img_ref_view = cv2.flip(img, -1)
        # else:
        #     img_ref_view = img

        # In most cases, it's better to undistort the SAME original pixel array with each model:
        img_live_und = undistort_with(K_live, dist_live, img)
        img_ref_und  = undistort_with(K_ref,  dist_ref,  img)

        panel = make_panel(img, img_live_und, img_ref_und, "Live JSON", "Refined JSON")
        out_path = OUT_DIR / f"compare_{i:02d}.png"
        cv2.imwrite(str(out_path), panel)
        print(f"[OK] wrote {out_path}")

    print("\nDone. Review the panels and pick the model that produces straighter checkerboard lines "
          "(especially toward the corners).")

if __name__ == "__main__":
    main()

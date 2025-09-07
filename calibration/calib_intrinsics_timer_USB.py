#!/usr/bin/env python3
import argparse, os, time, json, math, glob
from pathlib import Path
import cv2
import numpy as np

def parse_pattern(s: str):
    a, b = s.lower().split('x')
    return (int(a), int(b))

def rms_corner_shift(a: np.ndarray, b: np.ndarray) -> float:
    da = a.reshape(-1, 2); db = b.reshape(-1, 2)
    d = np.linalg.norm(da - db, axis=1)
    return float(np.sqrt(np.mean(d**2)))

def find_corners(gray, pattern_size, use_sb=False):
    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            return True, corners
        # fall back to classic if SB fails on this frame
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not found:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def calibrate(objpoints, imgpoints, image_size, square_cm, pattern, samples_list, flipped, use_rational=False):
    flags = 0
    if use_rational:
        flags |= cv2.CALIB_RATIONAL_MODEL
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None, flags=flags)
    # per-image RMS
    per_img_err = []
    total_err = 0.0; total_pts = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(imgpoints[i])
        per_img_err.append(float(err))
        total_err += (err * len(imgpoints[i]))**2
        total_pts += len(imgpoints[i])
    rms = math.sqrt(total_err / max(1, total_pts))

    return {
        "image_size": [image_size[0], image_size[1]],
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "rms_reproj_error": rms,
        "square_size_cm": square_cm,
        "checkerboard": [pattern[0], pattern[1]],
        "samples": samples_list,
        "flipped_during_capture": bool(flipped),
        "per_image_error": per_img_err
    }, (K, dist, rvecs, tvecs)

def build_obj_grid(cols, rows, square_cm):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_cm
    return objp

def run_folder_mode(args):
    pat_cols, pat_rows = parse_pattern(args.pattern)
    pattern_size = (pat_cols, pat_rows)
    objp = build_obj_grid(pat_cols, pat_rows, args.square_cm)
    objpoints, imgpoints, samples = [], [], []

    image_paths = []
    for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"):
        image_paths.extend(glob.glob(str(Path(args.images_dir) / ext)))
    image_paths.sort()
    if not image_paths:
        print(f"[ERR] No images found in: {args.images_dir}")
        return

    # Determine image size from first image
    first = cv2.imread(image_paths[0])
    if first is None:
        print("[ERR] Unable to read first image.")
        return
    if args.flip: first = cv2.flip(first, -1)
    h, w = first.shape[:2]
    image_size = (w, h)

    taken = 0
    for p in image_paths:
        img = cv2.imread(p)
        if img is None: 
            print(f"[WARN] Skipping unreadable: {p}"); 
            continue
        if args.flip: img = cv2.flip(img, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray, pattern_size)
        if found:
            imgpoints.append(corners)
            objpoints.append(objp.copy())
            samples.append(p)
            taken += 1
            print(f"[OK] Corners: {p}")
        else:
            print(f"[..] No grid: {p}")

    if taken < 5:
        print(f"[WARN] Only {taken} valid samples—aborting calibration.")
        return

    result, calib = calibrate(objpoints, imgpoints, image_size, args.square_cm, (pat_cols, pat_rows), samples, args.flip, use_rational=args.rational)
    K, dist, rvecs, tvecs = calib
    # --- Optional: drop worst-N frames and recalibrate ---
    if args.drop_outliers > 0 and "per_image_error" in result:
        errs = list(enumerate(result["per_image_error"]))            # (index, per-image RMS)
        errs.sort(key=lambda x: x[1], reverse=True)                  # worst first
        to_drop = min(args.drop_outliers, len(errs))
        drop_idx = {i for i, _ in errs[:to_drop]}
        if drop_idx:
            print(f"[INFO] Dropping {len(drop_idx)} worst frames: {sorted(drop_idx)}")
            objpoints = [op for i, op in enumerate(objpoints) if i not in drop_idx]
            imgpoints = [ip for i, ip in enumerate(imgpoints) if i not in drop_idx]
            samples   = [sp for i, sp in enumerate(samples)   if i not in drop_idx]

            # Re-run calibration after pruning
            result, calib = calibrate(objpoints, imgpoints, image_size, args.square_cm,
                                      (pat_cols, pat_rows), samples, args.flip,
                                      use_rational=args.rational)
            K, dist, rvecs, tvecs = calib
            print(f"[INFO] RMS after outlier drop: {result['rms_reproj_error']:.4f}")

    # --- Save final intrinsics JSON ---
    with open(args.save_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] Saved intrinsics → {args.save_json}")
    print(f"[INFO] Final RMS reprojection error: {result['rms_reproj_error']:.4f}")
    print(f"[INFO] Samples used: {len(samples)}")

    # --- Optional: undistort preview (uses the first remaining sample) ---
    if getattr(args, "preview_undistort", False) and samples:
        preview_path = samples[0]
        img = cv2.imread(preview_path)
        if img is not None:
            if args.flip:
                img = cv2.flip(img, -1)
            h, w = img.shape[:2]
            newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1.0, (w, h))
            und = cv2.undistort(img, K, dist, None, newK)
            cv2.imshow("undistort preview", und)
            cv2.waitKey(0)
            cv2.destroyWindow("undistort preview")


def run_live_mode(args):
    pat_cols, pat_rows = parse_pattern(args.pattern)
    pattern_size = (pat_cols, pat_rows)
    res_w, res_h = map(int, args.res.lower().split('x'))
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
    if not cap.isOpened():
        print("[ERR] Could not open camera."); return

    objp = build_obj_grid(pat_cols, pat_rows, args.square_cm)
    objpoints, imgpoints, samples = [], [], []
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    last_capture_time = 0.0; last_corners = None; sample_count = 0; flipped = args.flip
    print("[INFO] Press 'c' to capture, 'u' flip toggle, 'q'/ESC to finish. Window must have focus.")
    cv2.namedWindow("calib_intrinsics", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok: break
        if flipped: frame = cv2.flip(frame, -1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray, pattern_size)

        overlay = frame.copy()
        status = "NO GRID"; color=(0,0,255)
        can_capture = False
        if found:
            status="GRID OK"; color=(0,200,0)
            now = time.time()
            if now - last_capture_time >= args.interval:
                if last_corners is None:
                    can_capture = True
                else:
                    can_capture = rms_corner_shift(corners, last_corners) >= args.min_shift
            # auto-capture
            if can_capture:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                out_img = out_dir / f"calib_{stamp}_{sample_count:03d}.png"
                cv2.imwrite(str(out_img), frame)
                imgpoints.append(corners)
                objpoints.append(objp.copy())
                samples.append(str(out_img))
                last_corners = corners.copy()
                last_capture_time = now
                sample_count += 1
                print(f"[CAPTURE] {out_img}  (total={sample_count})")

            cv2.drawChessboardCorners(overlay, pattern_size, corners, True)

        # HUD
        cv2.putText(overlay, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(overlay, f"samples: {sample_count}", (overlay.shape[1]-220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(overlay, f"flip: {'ON' if flipped else 'OFF'}  res: {res_w}x{res_h}",
                    (overlay.shape[1]-320, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.imshow("calib_intrinsics", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('u'):
            flipped = not flipped
        elif key == ord('c') and found:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_img = out_dir / f"calib_{stamp}_{sample_count:03d}.png"
            cv2.imwrite(str(out_img), frame)
            imgpoints.append(corners)
            objpoints.append(objp.copy())
            samples.append(str(out_img))
            last_corners = corners.copy()
            last_capture_time = time.time()
            sample_count += 1
            print(f"[MANUAL] {out_img}  (total={sample_count})")

        if args.max_samples > 0 and sample_count >= args.max_samples:
            print(f"[INFO] Reached max samples ({args.max_samples})."); break

    cap.release(); cv2.destroyAllWindows()

    if len(imgpoints) < 5:
        print("[WARN] Fewer than 5 samples—aborting calibration."); return

    image_size = (res_w, res_h)
    result = calibrate(objpoints, imgpoints, image_size, args.square_cm, (pat_cols, pat_rows), samples, flipped)
    with open(args.save_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] Saved intrinsics → {args.save_json}")
    print(f"[INFO] RMS reprojection error: {result['rms_reproj_error']:.4f}")
    print(f"[INFO] Samples used: {len(imgpoints)}")

def main():
    ap = argparse.ArgumentParser(description="Intrinsic calibration: live camera OR images folder.")
    ap.add_argument("--images-dir", help="If set, run in folder mode using all images in this directory.")
    ap.add_argument("--cam", type=int, default=0, help="Live: camera index")
    ap.add_argument("--res", default="640x480", help="Live: resolution WxH")
    ap.add_argument("--pattern", default="8x5", help="Checkerboard inner corners, e.g. 9x5")
    ap.add_argument("--square-cm", type=float, default=2.5, help="Checker square size in cm")
    ap.add_argument("--interval", type=float, default=1.0, help="Live: seconds between auto-samples")
    ap.add_argument("--min-shift", type=float, default=8.0, help="Live: minimum RMS corner shift (px)")
    ap.add_argument("--max-samples", type=int, default=50, help="Live: stop after N samples (0=no limit)")
    ap.add_argument("--flip", action="store_true", help="Flip frames/images 180°")
    ap.add_argument("--out-dir", default="calib_shots", help="Live: where to save captured frames")
    ap.add_argument("--save-json", default="intrinsics.json", help="Output intrinsics JSON path")
    ap.add_argument("--sb-corners", action="store_true", help="Use findChessboardCornersSB (more precise if available)")
    ap.add_argument("--rational", action="store_true", help="Enable CALIB_RATIONAL_MODEL (k4..k6)")
    ap.add_argument("--drop-outliers", type=int, default=0, help="Drop N worst per-image reprojection-error frames before final solve")
    ap.add_argument("--preview-undistort", action="store_true", help="Show undistort preview of a sample at the end")

    args = ap.parse_args()

    if args.images_dir:
        run_folder_mode(args)
    else:
        run_live_mode(args)

if __name__ == "__main__":
    main()

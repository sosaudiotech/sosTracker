#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, time, json, math, glob
from pathlib import Path
import cv2
import numpy as np

# --- Optional NDI support ---
try:
    import NDIlib as ndi
except Exception:
    ndi = None

# ---------- helpers ----------
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

def calibrate(objpoints, imgpoints, image_size, square_cm, pattern, samples_list,
              flipped, use_rational=False, use_fisheye=False, fix_principal=False):
    """
    Calibrate intrinsics from detected checkerboard corners.

    Parameters
    ----------
    objpoints : list[np.ndarray]
        List of (N,3) float32/float64 arrays of 3D board points per image (Z=0 plane).
    imgpoints : list[np.ndarray]
        List of (N,1,2) float32/float64 arrays of detected 2D image points per image.
    image_size : tuple[int,int]
        (width, height) of images.
    square_cm : float
        Checker square size in centimeters (for metadata only).
    pattern : tuple[int,int]
        (cols, rows) inner-corner counts used during detection (for metadata only).
    samples_list : list[str]
        Paths to images used (for metadata only).
    flipped : bool
        Whether frames were flipped 180° during capture (metadata; keep runtime consistent).
    use_rational : bool
        If True and not fisheye, enable CALIB_RATIONAL_MODEL (k4..k6).
    use_fisheye : bool
        If True, use cv2.fisheye.calibrate (D = [k1..k4], no tangential).
    fix_principal : bool
        If True (fisheye only), try a second pass fixing principal point *after* a stable solve.

    Returns
    -------
    result : dict
        JSON-serializable calibration result with keys:
        - model: "pinhole" or "fisheye"
        - image_size, K, dist, rms_reproj_error, square_size_cm, checkerboard,
          samples, flipped_during_capture, per_image_error
    calib : tuple
        (K, dist_or_D, rvecs, tvecs) for downstream use.
    """
    import math
    import numpy as np
    import cv2

    W, H = int(image_size[0]), int(image_size[1])

    # ---------- FISHEYE MODEL ----------
    if use_fisheye:
        # Ensure fisheye-expected shapes/dtypes
        objp_f = [np.ascontiguousarray(op.reshape(-1, 1, 3), dtype=np.float64) for op in objpoints]
        imgp_f = [np.ascontiguousarray(ip.reshape(-1, 1, 2), dtype=np.float64) for ip in imgpoints]

        K = np.zeros((3, 3), dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)

        # Stage 1: simple flags (no CHECK_COND), often more stable for the first solve
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rvecs, tvecs = [], []

        try:
            rms0, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objectPoints=objp_f,
                imagePoints=imgp_f,
                image_size=(W, H),
                K=K, D=D, rvecs=rvecs, tvecs=tvecs,
                flags=flags,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
            )
        except cv2.error:
            # Stage 2: seed fisheye from a pinhole solve, then use USE_INTRINSIC_GUESS
            _ret, K0, _dist0, _r0, _t0 = cv2.calibrateCamera(
                objpoints, imgpoints, (W, H), None, None, flags=0
            )
            K = K0.astype(np.float64)
            D = np.zeros((4, 1), dtype=np.float64)
            flags2 = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                      cv2.fisheye.CALIB_FIX_SKEW |
                      cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
            rvecs, tvecs = [], []
            rms0, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objectPoints=objp_f,
                imagePoints=imgp_f,
                image_size=(W, H),
                K=K, D=D, rvecs=rvecs, tvecs=tvecs,
                flags=flags2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
            )

        # Optional: refine with fixed principal point if requested (and if it doesn't hurt RMS)
        if fix_principal:
            flags3 = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                      cv2.fisheye.CALIB_FIX_SKEW |
                      cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT |
                      cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
            rvecs2, tvecs2 = [], []
            rms1, K2, D2, rvecs2, tvecs2 = cv2.fisheye.calibrate(
                objectPoints=objp_f,
                imagePoints=imgp_f,
                image_size=(W, H),
                K=K.copy(), D=D.copy(), rvecs=rvecs2, tvecs=tvecs2,
                flags=flags3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
            )
            # Accept only if RMS doesn't significantly degrade
            if rms1 <= rms0 * 1.05:
                rms0, K, D, rvecs, tvecs = rms1, K2, D2, rvecs2, tvecs2

        # Per-image reprojection error (average per-corner L2 in pixels)
        per_img_err = []
        total_ss = 0.0
        total_pts = 0
        for i, (op, ip) in enumerate(zip(objp_f, imgp_f)):
            proj, _ = cv2.fisheye.projectPoints(op, rvecs[i], tvecs[i], K, D)
            # ip and proj are (N,1,2)
            err = cv2.norm(ip, proj, cv2.NORM_L2) / len(ip)
            per_img_err.append(float(err))
            total_ss += (err * len(ip))**2
            total_pts += len(ip)
        rms_pix = math.sqrt(total_ss / max(1, total_pts))

        result = {
            "model": "fisheye",
            "image_size": [W, H],
            "K": K.tolist(),
            "dist": D.reshape(-1).tolist(),  # k1..k4
            "rms_reproj_error": float(rms_pix),
            "square_size_cm": float(square_cm),
            "checkerboard": [int(pattern[0]), int(pattern[1])],
            "samples": list(samples_list),
            "flipped_during_capture": bool(flipped),
            "per_image_error": per_img_err
        }
        return result, (K, D, rvecs, tvecs)

    # ---------- PINHOLE MODEL ----------
    flags = 0
    if use_rational:
        flags |= cv2.CALIB_RATIONAL_MODEL

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (W, H), None, None, flags=flags
    )

    # Per-image reprojection error (average per-corner L2 in pixels)
    per_img_err = []
    total_ss = 0.0
    total_pts = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(imgpoints[i])
        per_img_err.append(float(err))
        total_ss += (err * len(imgpoints[i]))**2
        total_pts += len(imgpoints[i])
    rms_pix = math.sqrt(total_ss / max(1, total_pts))

    result = {
        "model": "pinhole",
        "image_size": [W, H],
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),  # classic (up to k3) or rational (k1..k6)
        "rms_reproj_error": float(rms_pix),
        "square_size_cm": float(square_cm),
        "checkerboard": [int(pattern[0]), int(pattern[1])],
        "samples": list(samples_list),
        "flipped_during_capture": bool(flipped),
        "per_image_error": per_img_err
    }
    return result, (K, dist, rvecs, tvecs)



def build_obj_grid(cols, rows, square_cm):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_cm
    return objp

def undistort_with_stable(K, dist, img, prefer_tight=True):
    """
    Stable undistortion: clamps extreme fx/fy, recenters principal point,
    and uses remap for predictable scaling.
    """
    h, w = img.shape[:2]
    K = np.array(K, dtype=np.float64).copy()
    dist = np.array(dist, dtype=np.float64).reshape(-1, 1)

    # Center principal point
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0

    # Clamp ridiculous focal lengths
    fx, fy = K[0, 0], K[1, 1]
    max_fx = 2.0 * w
    max_fy = 2.0 * h
    if fx > max_fx or fy > max_fy:
        s = min(max_fx / fx, max_fy / fy)
        K[0, 0] *= s
        K[1, 1] *= s

    # Get new camera matrix with alpha=0 (tight crop)
    alpha = 0.0 if prefer_tight else 0.5
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))

    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)
    und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Crop if tight
    x, y, rw, rh = roi
    if prefer_tight and rw > 0 and rh > 0:
        und = und[y:y+rh, x:x+rw].copy()

    return und, newK, roi

# ---------- NDI helpers ----------
def list_ndi_sources(wait_ms: int):
    """Minimal discovery (matches your working test script)."""
    if ndi is None:
        raise RuntimeError("NDIlib not available. Install ndi-python + NDI runtime.")
    if not ndi.initialize():
        raise RuntimeError("Failed to initialize NDI")

    finder = ndi.find_create_v2()
    if finder is None:
        ndi.destroy()
        raise RuntimeError("Failed to create NDI finder")

    time.sleep(max(0, wait_ms) / 1000.0)
    sources = ndi.find_get_current_sources(finder) or []
    names = []
    for s in sources:
        name = s.ndi_name.decode() if isinstance(s.ndi_name, (bytes, bytearray)) else s.ndi_name
        url  = getattr(s, "url_address", None)
        if isinstance(url, (bytes, bytearray)): url = url.decode()
        names.append((name, url))
    ndi.find_destroy(finder)
    return sources, names

def open_ndi_source(source_substr: str = None, index: int = None, url: str = None, wait_ms: int = 5000):
    """Open an NDI receiver via URL, index, or name/substring."""
    sources, names = list_ndi_sources(wait_ms=wait_ms)

    print("[NDI] Discovered sources:")
    if not names:
        print("  (none)")
    else:
        for i, (n, u) in enumerate(names):
            print(f"  [{i}] {n}  <{u or ''}>")

    picked = None
    if url:
        # Direct URL connect
        src = ndi.Source()
        src.url_address = url.encode() if isinstance(url, str) else url
        picked = src
    elif index is not None:
        if 0 <= index < len(sources):
            picked = sources[index]
        else:
            raise RuntimeError(f"NDI index {index} out of range (0..{len(sources)-1}).")
    elif source_substr:
        lower = source_substr.strip().lower()
        exact, partial = None, None
        for s in sources:
            name = s.ndi_name.decode() if isinstance(s.ndi_name, (bytes, bytearray)) else s.ndi_name
            if name.strip().lower() == lower:
                exact = s; break
            if lower in name.lower():
                partial = s
        picked = exact or partial
        if picked is None:
            raise RuntimeError(f"NDI source not found: '{source_substr}'")
    else:
        raise RuntimeError("Provide --ndi-url OR --ndi-index OR --ndi-source.")

    # Receiver settings – don't throttle; let sender choose fastest format
    rc = ndi.RecvCreateV3()
    rc.color_format = getattr(ndi, "RECV_COLOR_FORMAT_FASTEST", 0)
    rc.bandwidth    = getattr(ndi, "RECV_BANDWIDTH_HIGHEST", 2)
    rc.allow_video_fields = False

    recv = ndi.recv_create_v3(rc)
    if recv is None:
        raise RuntimeError("Failed to create NDI receiver")

    ndi.recv_connect(recv, picked)
    return recv

def ndi_to_cv2_frame(recv, target_size=None, timeout_ms=1200):
    """
    Robust NDI frame grab: supports UYVY (your AIDA) and BGRA.
    Auto-detects by buffer size; copies before freeing.
    """
    t, v, a, m = ndi.recv_capture_v2(recv, timeout_ms)
    if t != getattr(ndi, "FRAME_TYPE_VIDEO", 2):  # 2 in most wrappers
        return None
    try:
        h, w = v.yres, v.xres
        buf = np.frombuffer(v.data, dtype=np.uint8)
        # Detect format by buffer size (most reliable across wrappers)
        if buf.size == h * w * 2:
            # UYVY (YUV422) — matches your working NDI server
            arr = buf.reshape(h, w, 2).copy()
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_UYVY)
        elif buf.size == h * w * 4:
            # BGRA
            arr = buf.reshape(h, w, 4).copy()
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        else:
            if not hasattr(ndi_to_cv2_frame, "_warned_fmt"):
                print(f"[NDI] Unrecognized buffer size ({buf.size}) for {w}x{h}; skipping.")
                ndi_to_cv2_frame._warned_fmt = True
            return None

        if target_size and (frame_bgr.shape[1], frame_bgr.shape[0]) != target_size:
            frame_bgr = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_AREA)
        return frame_bgr
    finally:
        ndi.recv_free_video_v2(recv, v)

# ---------- modes ----------
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
            print(f"[WARN] Skipping unreadable: {p}")
            continue
        if args.flip: img = cv2.flip(img, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray, pattern_size, use_sb=args.sb_corners)
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

    result, calib = calibrate(
        objpoints, imgpoints, image_size,
        args.square_cm, (pat_cols, pat_rows),
        samples, args.flip,
        use_rational=args.rational,
        use_fisheye=args.fisheye,
        fix_principal=args.fix_principal
    )
    K, dist, rvecs, tvecs = calib


    # Optional outlier drop & re-solve
    if args.drop_outliers > 0 and "per_image_error" in result:
        errs = list(enumerate(result["per_image_error"]))
        errs.sort(key=lambda x: x[1], reverse=True)
        to_drop = min(args.drop_outliers, len(errs))
        drop_idx = {i for i, _ in errs[:to_drop]}
        if drop_idx:
            print(f"[INFO] Dropping {len(drop_idx)} worst frames: {sorted(drop_idx)}")
            objpoints = [op for i, op in enumerate(objpoints) if i not in drop_idx]
            imgpoints = [ip for i, ip in enumerate(imgpoints) if i not in drop_idx]
            samples   = [sp for i, sp in enumerate(samples)   if i not in drop_idx]
            result, calib = calibrate(objpoints, imgpoints, image_size, args.square_cm, (pat_cols, pat_rows), samples, args.flip, use_rational=args.rational)
            K, dist, rvecs, tvecs = calib
            print(f"[INFO] RMS after outlier drop: {result['rms_reproj_error']:.4f}")

    with open(args.save_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] Saved intrinsics → {args.save_json}")
    print(f"[INFO] Final RMS reprojection error: {result['rms_reproj_error']:.4f}")
    print(f"[INFO] Samples used: {len(samples)}")

    # Optional undistort preview
    if getattr(args, "preview_undistort", False) and samples:
        preview_path = samples[0]
        img = cv2.imread(preview_path)
        if img is not None:
            if args.flip:
                img = cv2.flip(img, -1)
            h, w = img.shape[:2]
            newK, _ = cv2.getOptimalNewCameraMatrix(np.array(result["K"]), np.array(result["dist"]), (w, h), 1.0, (w, h))
            und, newK, roi = undistort_with_stable(result["K"], result["dist"], img, prefer_tight=True)
            cv2.imshow("undistort preview", und)
            cv2.waitKey(0)
            cv2.destroyWindow("undistort preview")

def run_live_mode(args):
    pat_cols, pat_rows = parse_pattern(args.pattern)
    pattern_size = (pat_cols, pat_rows)
    res_w, res_h = map(int, args.res.lower().split('x'))

    # Input: NDI or USB
    recv = None
    cap = None
    if (args.ndi_source is not None) or (args.ndi_index is not None) or (args.ndi_url is not None):
        if ndi is None:
            print("[ERR] NDI requested but NDIlib not available.")
            return
        try:
            recv = open_ndi_source(
                source_substr=args.ndi_source,
                index=args.ndi_index,
                url=args.ndi_url,
                wait_ms=args.ndi_wait_ms
            )
            get_frame = lambda: ndi_to_cv2_frame(recv, (res_w, res_h), timeout_ms=args.ndi_timeout_ms)
            print(f"[INFO] NDI mode ON at {res_w}x{res_h}")
        except Exception as e:
            print(f"[ERR] {e}")
            return
    else:
        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
        if not cap.isOpened():
            print("[ERR] Could not open camera.")
            return
        get_frame = lambda: (cap.read()[1] if cap.isOpened() else None)
        print(f"[INFO] USB mode ON (cam={args.cam}) at {res_w}x{res_h}")

    objp = build_obj_grid(pat_cols, pat_rows, args.square_cm)
    objpoints, imgpoints, samples = [], [], []
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    last_capture_time = 0.0; last_corners = None; sample_count = 0; flipped = args.flip
    print("[INFO] Press 'c' to capture, 'u' flip toggle, 'q'/ESC to finish. Window must have focus.")

    if not args.headless:
        cv2.namedWindow("calib_intrinsics", cv2.WINDOW_NORMAL)

    no_frame_start = None
    def reconnect():
        nonlocal recv, get_frame, no_frame_start
        if recv is None:  # USB path — nothing to do
            return
        print("[NDI] Reconnecting…")
        try:
            ndi.recv_destroy(recv)
        except Exception:
            pass
        recv = open_ndi_source(
            source_substr=args.ndi_source,
            index=args.ndi_index,
            url=args.ndi_url,
            wait_ms=args.ndi_wait_ms
        )
        get_frame = lambda: ndi_to_cv2_frame(recv, (res_w, res_h), timeout_ms=args.ndi_timeout_ms)
        no_frame_start = None
        print("[NDI] Reconnected.")

    while True:
        frame = get_frame()
        if frame is None:
            # heartbeat + optional reconnect (NDI path)
            if no_frame_start is None:
                no_frame_start = time.time()
            elif (time.time() - no_frame_start) * 1000 >= args.ndi_reconnect_ms:
                reconnect()
            if int(time.time()) % 2 == 0:
                print("[NDI] …waiting for frames" if recv is not None else "[USB] …no frame")
            # avoid tight spin
            time.sleep(0.02)
            # When headless, keep looping; in GUI mode, allow key handling below to continue
            key = 255
            if not args.headless:
                try:
                    key = cv2.waitKey(1) & 0xFF
                except Exception:
                    key = 255
            if key in (27, ord('q')):
                break
            elif key == ord('u'):
                flipped = not flipped
            continue
        else:
            no_frame_start = None

        if flipped:
            frame = cv2.flip(frame, -1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray, pattern_size, use_sb=args.sb_corners)

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
                    (overlay.shape[1]-360, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        if not args.headless:
            try:
                cv2.imshow("calib_intrinsics", overlay)
                key = cv2.waitKey(1) & 0xFF
            except Exception as e:
                print(f"[WARN] imshow failed: {e}. Forcing headless mode.")
                args.headless = True
                key = 255
        else:
            key = 255

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

    # Cleanup
    if cap is not None:
        cap.release()
    if recv is not None:
        try:
            ndi.recv_destroy(recv)
        except Exception:
            pass
        ndi.destroy()
    if not args.headless:
        cv2.destroyAllWindows()

    if len(imgpoints) < 5:
        print("[WARN] Fewer than 5 samples—aborting calibration."); return

    image_size = (res_w, res_h)
    result, _calib = calibrate(objpoints, imgpoints, image_size, args.square_cm, (pat_cols, pat_rows), samples, flipped, use_rational=args.rational)
    with open(args.save_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] Saved intrinsics → {args.save_json}")
    print(f"[INFO] RMS reprojection error: {result['rms_reproj_error']:.4f}")
    print(f"[INFO] Samples used: {len(imgpoints)}")

    # --- Optional: undistort preview ---
    if getattr(args, "preview_undistort", False) and samples:
        print(f"[INFO] Showing undistort preview (press any key to close)...")

    # Pick a sample image (first kept sample)
    sample_path = samples[0]
    img = cv2.imread(sample_path)
    if img is None:
        print(f"[WARN] Could not read preview image: {sample_path}")
    else:
        # Keep runtime orientation consistent with how we calibrated
        if args.flip:
            img = cv2.flip(img, -1)

        h, w = img.shape[:2]

        # Pull intrinsics/dist from the calibration result you just computed
        K_np = np.array(result["K"], dtype=np.float64)

        if args.fisheye or result.get("model") == "fisheye":
            # --- FISHEYE PATH ---
            D_np = np.array(result["dist"], dtype=np.float64).reshape(-1, 1)

            # Choose rectification/undistort params:
            # balance (0..1): 0 = crop more (straighter), 1 = preserve FOV
            # fov_scale: expand/contract FOV of the new camera matrix
            R = np.eye(3)
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K_np, D_np, (w, h), R, balance=args.balance, fov_scale=args.fov_scale
            )

            # Rectify map + remap
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K_np, D_np, R, newK, (w, h), cv2.CV_16SC2
            )
            und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

        else:
            # --- PINHOLE PATH ---
            dist_np = np.array(result["dist"], dtype=np.float64).reshape(-1, 1)
            newK, _ = cv2.getOptimalNewCameraMatrix(K_np, dist_np, (w, h), alpha=1.0, newImgSize=(w, h))
            und = cv2.undistort(img, K_np, dist_np, None, newK)

        cv2.imshow("undistort preview", und)
        cv2.waitKey(0)
        cv2.destroyWindow("undistort preview")


def main():
    ap = argparse.ArgumentParser(description="Intrinsic calibration: live (USB/NDI) or images folder.")
    # folder vs live
    ap.add_argument("--images-dir", help="If set, run in folder mode using all images in this directory.")
    # USB live
    ap.add_argument("--cam", type=int, default=0, help="Live (USB): camera index")
    # NDI live
    ap.add_argument("--ndi-source", help="Live (NDI): source name substring (case-insensitive).")
    ap.add_argument("--ndi-index", type=int, help="Live (NDI): pick source by discovered index.")
    ap.add_argument("--ndi-url", help="Live (NDI): direct URL to connect.")
    ap.add_argument("--ndi-wait-ms", type=int, default=5000, help="NDI discovery wait in ms (default 5000).")
    ap.add_argument("--ndi-timeout-ms", type=int, default=1200, help="NDI per-capture timeout in ms.")
    ap.add_argument("--ndi-reconnect-ms", type=int, default=8000, help="Reconnect if no frames for this long (ms).")

    ap.add_argument("--fisheye", action="store_true",help="Use fisheye model (cv2.fisheye.calibrate: k1..k4; no tangential terms)")
    ap.add_argument("--fix-principal", action="store_true",help="Fisheye: fix principal point (recommended for some webcams)")
    ap.add_argument("--balance", type=float, default=0.0,help="Fisheye undistort balance [0..1], 0 = max crop, 1 = keep FOV")
    ap.add_argument("--fov-scale", type=float, default=1.0,help="Fisheye estimateNewCameraMatrixForUndistortRectify fov_scale")

    # common live
    ap.add_argument("--res", default="640x480", help="Live: resolution WxH")
    ap.add_argument("--pattern", default="10x6", help="Checkerboard inner corners, e.g. 9x5")
    ap.add_argument("--square-cm", type=float, default=3.5, help="Checker square size in cm")
    ap.add_argument("--interval", type=float, default=1.0, help="Live: seconds between auto-samples")
    ap.add_argument("--min-shift", type=float, default=8.0, help="Live: minimum RMS corner shift (px)")
    ap.add_argument("--max-samples", type=int, default=50, help="Live: stop after N samples (0=no limit)")
    ap.add_argument("--flip", action="store_true", help="Flip frames/images 180°")
    ap.add_argument("--out-dir", default="calib_shots", help="Live: where to save captured frames")
    ap.add_argument("--save-json", default="intrinsics.json", help="Output intrinsics JSON path")
    ap.add_argument("--sb-corners", action="store_true", help="Use findChessboardCornersSB (more precise if available)")
    ap.add_argument("--rational", action="store_true", help="Enable CALIB_RATIONAL_MODEL (k4..k6)")
    ap.add_argument("--drop-outliers", type=int, default=0, help="Folder: drop N worst frames before final solve")
    ap.add_argument("--preview-undistort", action="store_true", help="Folder: show undistort preview at the end")
    ap.add_argument("--headless", action="store_true", help="Run without GUI windows (no imshow/waitKey).")

    args = ap.parse_args()

    if args.images_dir:
        run_folder_mode(args)
    else:
        run_live_mode(args)

if __name__ == "__main__":
    main()

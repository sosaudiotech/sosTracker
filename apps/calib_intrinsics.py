# calib_intrinsics.py
import cv2, numpy as np, json, time

CHECKERBOARD = (9,5)              # inner corners (cols, rows)
SQUARE_SIZE_CM = 2.5               # cm per square
MIN_SAMPLES = 15                    # 20–30 is better

def main(cam_index=2, out_path="intrinsics_cam2.json"):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    assert cap.isOpened(), "Camera not opened"

    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_CM

    objpoints, imgpoints = [], []
    print("[i] Press SPACE to add sample, ENTER to calibrate, ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, -1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        vis = frame.copy()
        if ret:
            cv2.drawChessboardCorners(vis, CHECKERBOARD, corners, ret)
        cv2.putText(vis, f"samples: {len(objpoints)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Calib - press SPACE to capture", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        elif k == 32 and ret:  # SPACE
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp.copy())
            imgpoints.append(corners2)
            print(f"[+] Added sample {len(objpoints)}")
        elif k == 13:  # ENTER
            if len(objpoints) < MIN_SAMPLES:
                print(f"[!] Need at least {MIN_SAMPLES} samples")
                continue
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
            h, w = gray.shape[:2]
            # Optional: improve principal point after undistort
            newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 0)
            print(f"[✓] RMS reprojection error: {ret:.3f}")
            payload = {
                "image_size": [int(w), int(h)],
                "K": K.tolist(),
                "dist": dist.reshape(-1).tolist(),
                "rms_reproj_error": float(ret),
                "square_size_cm": SQUARE_SIZE_CM,
                "checkerboard": list(CHECKERBOARD)
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[✓] Saved to {out_path}")
            time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(cam_index=2, out_path="intrinsics_cam2.json")

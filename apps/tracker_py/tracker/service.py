# -*- coding: utf-8 -*-
from __future__ import annotations




import os, math, json, asyncio, time, threading
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse, HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO

try:
    from torchreid.utils import FeatureExtractor
except Exception:
    FeatureExtractor = None

# -------- Types --------
Vector3 = Tuple[float, float, float]

class TrackVector(BaseModel):
    cameraId: str
    origin_cm: Vector3
    direction_unit: Vector3
    pos_estimate_cm: Vector3
    confidence: Optional[float] = None
    track_id: Optional[str] = None

class TickResponse(BaseModel):
    timestamp: str
    vectors: List[TrackVector]

class CameraInfo(BaseModel):
    name: str
    index: int
    position_cm: Vector3
    pan_deg: float
    tilt_deg: float
    fov_deg: float
    opened: bool
    flip: Optional[str] = "none"

class CameraSpec(BaseModel):
    name: str
    position_cm: Vector3
    pan_deg: float
    tilt_deg: float
    fov_deg: float
    device: Optional[str] = None   # <— allow device strings like "video=@device_pnp_..."
    index: Optional[int] = None  # optional in config
    flip: Optional[str] = "none"   # "none" | "h" | "v" | "hv"

def _parse_flip_code(val: Optional[object]) -> Optional[int]:
    """
    Map config to OpenCV flip codes:
      1 = horizontal, 0 = vertical, -1 = both, None = none.
    Accepts: "h", "horizontal", "mirror", "horiz", "hor",
             "v", "vertical",
             "hv", "vh", "both",
             1, 0, -1, True/False, "true"/"false"/"yes"/"no".
    """
    if val is None:
        return None

    # Accept numeric directly
    if isinstance(val, (int, float)):
        v = int(val)
        return v if v in (-1, 0, 1) else None

    # Accept booleans as horizontal mirror
    if isinstance(val, bool):
        return 1 if val else None

    s = str(val).strip().lower()
    if s in {"1", "+1"}:
        return 1
    if s in {"0"}:
        return 0
    if s in {"-1"}:
        return -1

    # common synonyms
    mapping = {
        "none": None,
        "h": 1, "hor": 1, "horiz": 1, "horizontal": 1, "mirror": 1, "mirrored": 1, "x": 1,
        "v": 0, "vert": 0, "vertical": 0, "y": 0,
        "hv": -1, "vh": -1, "both": -1, "invert": -1, "udlr": -1
    }
    if s in {"true", "yes", "on"}:
        return 1
    if s in {"false", "no", "off"}:
        return None

    return mapping.get(s, None)


class ApplyConfig(BaseModel):
    usb_cameras: List[CameraSpec]

# -------- Live preview storage (thread-safe per camera) --------
_preview_lock = defaultdict(threading.Lock)        # one lock per cam_id
_preview_jpeg = defaultdict(lambda: None)          # latest JPEG bytes per cam_id
_preview_meta = defaultdict(lambda: {"fps": 0.0, "ts": 0.0})
_preview_cfg = {
    "max_width": 960,    # downscale for the web
    "jpeg_quality": 70,  # 60–80 is a sweet spot
    "draw_boxes": True,  # turn off for max speed
    "enabled": True,
}



def _draw_overlays(frame_bgr, dets=None, fps=None, note=None):
    """Draw detection boxes + a tiny HUD. dets: list of dicts with keys xyxy,label,conf"""
    if dets and _preview_cfg["draw_boxes"]:
        for d in dets:
            x1, y1, x2, y2 = map(int, d['xyxy'])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = d.get('label', 'obj')
            conf = d.get('conf', None)
            txt = f"{label} {conf:.2f}" if conf is not None else label
            cv2.putText(frame_bgr, txt, (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    hud = []
    if fps is not None:
        hud.append(f"{fps:.1f} FPS")
    if note:
        hud.append(note)
    if hud:
        text = " | ".join(hud)
        cv2.putText(frame_bgr, text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    return frame_bgr


def update_preview(cam_id: str, frame_bgr, dets=None, fps=None, note=None):
    """Downscale, draw, encode JPEG, stash bytes for /frame and /stream endpoints."""
    if not _preview_cfg["enabled"] or frame_bgr is None:
        return
    h, w = frame_bgr.shape[:2]
    maxw = _preview_cfg["max_width"]
    if w > maxw:
        scale = maxw / float(w)
        frame_bgr = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    frame_bgr = _draw_overlays(frame_bgr, dets=dets, fps=fps, note=note)

    q = int(_preview_cfg["jpeg_quality"])  # JPEG encode
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return

    with _preview_lock[cam_id]:
        _preview_jpeg[cam_id] = buf.tobytes()
        _preview_meta[cam_id]["fps"] = fps or 0.0
        _preview_meta[cam_id]["ts"]  = time.time()


# -------- Camera tracker --------
class CameraTracker:
    def __init__(self, cam_source: "int|str", cam_name: str,
                 cam_position_cm: Vector3, pan_deg: float, tilt_deg: float, fov_deg: float,
                 flip: Optional[str] = "none"):
        """
        cam_source:
          - int: numeric index (0,1,2,...)
          - str: DirectShow/FFmpeg device name; 'video=<FriendlyName>' preferred.
                 If not prefixed, we'll add 'video=' automatically.
        """
        self.cap = None
        self.backend = None  # keep track of which backend succeeded
        self.index = -1
        self.source = str(cam_source)
        self.name = cam_name
        self.flip_mode = flip or "none"
        self.flip_code = _parse_flip_code(self.flip_mode)
        def _try_open(src, backend):
            cap = cv2.VideoCapture(src, backend)
            
            if cap.isOpened():
                return cap, backend
            cap.release()
            return None, None

        # Open by index (fast path)
        if isinstance(cam_source, int):
            self.index = cam_source
            self.source = f"index:{cam_source}"
            # Try DSHOW → MSMF → ANY
            self.cap, self.backend = _try_open(cam_source, cv2.CAP_DSHOW)
            if not self.cap:
                self.cap, self.backend = _try_open(cam_source, cv2.CAP_MSMF)
            if not self.cap:
                self.cap, self.backend = _try_open(cam_source, cv2.CAP_ANY)

        else:
            # Open by name; ensure 'video=' prefix for DirectShow via FFmpeg
            name = str(cam_source).strip()
            if not name.lower().startswith("video="):
                name = f"video={name}"
            self.source = name

            # Try FFMPEG (dshow) → DSHOW → MSMF → ANY
            self.cap, self.backend = _try_open(name, cv2.CAP_FFMPEG)
            if not self.cap:
                self.cap, self.backend = _try_open(name, cv2.CAP_DSHOW)
            if not self.cap:
                self.cap, self.backend = _try_open(name, cv2.CAP_MSMF)
            if not self.cap:
                self.cap, self.backend = _try_open(name, cv2.CAP_ANY)

        if not self.cap:
            raise RuntimeError(f"Unable to open camera source: {cam_source!r}")

        # Apply desired capture size after a successful open
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Pose & optics
        self.position = np.array(cam_position_cm, dtype=float)
        self.pan = float(pan_deg)
        self.tilt = float(tilt_deg)
        self.fov = float(fov_deg)

        # Telemetry
        self._last_ts = 0.0  # for FPS estimation
    


    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def _angle_offsets(self, cx: float, cy: float, w: int, h: int):
        offset_x = (cx - w / 2) / (w / 2)
        offset_y = (cy - h / 2) / (h / 2)
        angle_x = offset_x * (self.fov / 2.0)
        angle_y = offset_y * ((self.fov * 9.0 / 16.0) / 2.0)
        return angle_x, angle_y

    def _dir_from_angles(self, ax: float, ay: float):
        yaw = math.radians(self.pan + ax)
        pitch = math.radians(self.tilt - ay)
        x = math.sin(yaw) * math.cos(pitch)
        y = math.sin(pitch)
        z = math.cos(yaw) * math.cos(pitch)
        v = np.array([x, y, z], dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _estimate_fps(self) -> float:
        now = time.time()
        fps = 0.0
        if self._last_ts:
            dt = now - self._last_ts
            if dt > 0:
                fps = 1.0 / dt
        self._last_ts = now
        return fps

    def tick_once(self, model, extractor, use_reid):
        ok, frame = self.cap.read()
        if not ok:
            return []

        # EARLIEST stage: flip here
        flip_note = "none"
        if getattr(self, "flip_code", None) is not None:
            frame = cv2.flip(frame, int(self.flip_code))  # cast to int for safety
            flip_note = {1: "h", 0: "v", -1: "hv"}.get(int(self.flip_code), "none")

        h, w = frame.shape[:2]
        out = []
        overlay_dets = []
        try:
            results = model(frame)[0]
            names = results.names if hasattr(results, "names") else getattr(model, "names", {})
            if hasattr(results, "boxes") and len(results.boxes) > 0:
                for det in results.boxes:
                    cls_idx = int(det.cls)
                    if (names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)) != "person":
                        continue
                    x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().tolist()
                    cx, cy = (x1 + x2)/2.0, (y1 + y2)/2.0
                    ax, ay = self._angle_offsets(cx, cy, w, h)
                    direction = self._dir_from_angles(ax, ay)
                    pos_estimate = self.position + direction * 100.0
                    out.append(TrackVector(
                        cameraId=self.name,
                        origin_cm=tuple(self.position.tolist()),
                        direction_unit=tuple(direction.tolist()),
                        pos_estimate_cm=tuple(pos_estimate.tolist()),
                        confidence=float(det.conf[0].cpu().item()) if hasattr(det, "conf") else None
                    ))
                    overlay_dets.append({"xyxy": (x1, y1, x2, y2), "label": "person", "conf": float(det.conf[0]) if hasattr(det, "conf") else 0.0})
        except Exception:
            pass

        fps_est = self._estimate_fps()
        try:
            update_preview(self.name, frame, dets=overlay_dets, fps=fps_est, note=f"flip={flip_note}")
        except Exception:
            pass

        return out


# -------- Helpers (TOP LEVEL, not inside class!) --------

def resolve_indices(usb_specs: Dict[str, Dict]) -> Dict[str, "int|str"]:
    mapping: Dict[str, "int|str"] = {}
    # 1) device string in config
    for name, spec in usb_specs.items():
        dev = spec.get("device")
        if isinstance(dev, str) and dev.strip():
            mapping[name] = dev.strip()

    # 2) env override
    env_map = os.getenv("DEFAULT_INDEX_MAP")
    if env_map:
        for token in env_map.split(","):
            if ":" in token:
                n, idx = token.split(":", 1)
                n = n.strip()
                if n in usb_specs and n not in mapping:
                    try:
                        mapping[n] = int(idx.strip())
                    except ValueError:
                        pass

    names = list(usb_specs.keys())
    missing = [n for n in names if n not in mapping]

    taken = {v for v in mapping.values() if isinstance(v, int)}

    def probe(idx: int) -> bool:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        cap.release()
        return ok

    for name in missing:
        cfg_idx = usb_specs[name].get("index", None)
        if isinstance(cfg_idx, int):
            mapping[name] = cfg_idx
            taken.add(cfg_idx)
            continue
        # choose first probed available index (0..9)
        for i in range(10):
            if i in taken:
                continue
            if probe(i):
                mapping[name] = i
                taken.add(i)
                break

    return mapping


def load_cameras_from_config(config_path: Optional[str]) -> List[CameraTracker]:
    trackers: List[CameraTracker] = []
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        usb: Dict = cfg.get("usb_cameras") or {}
        if usb:
            src_map = resolve_indices(usb)
            for name, spec in usb.items():
                src = src_map.get(name, spec.get("index", 0))

                cam = CameraTracker(
                    cam_source=src,
                    cam_name=str(name),
                    cam_position_cm=tuple(spec.get("position_cm", [0, 0, 0])),
                    pan_deg=float(spec.get("pan_deg", 0)),
                    tilt_deg=float(spec.get("tilt_deg", 0)),
                    fov_deg=float(spec.get("fov_deg", 78)),
                    flip=spec.get("flip", "none"),   # <-- from JSON
                )

                # 👇 Add this line
                print(f"[startup] {cam.name} flip={cam.flip_mode!r} code={cam.flip_code} src={cam.source}")

                trackers.append(cam)
            return trackers
    # fallback when no config / no usb cameras
    return []   # <— important

    #    CameraTracker(0, "usb_cam_1", (250, 152, 300), 0, 0, 78),
    #    CameraTracker(2, "usb_cam_2", (-250, 152, 300), 10, 0, 78),
    

# -------- Env & bootstrap --------
PORT        = int(os.getenv("PORT", "7001"))
MODEL_YOLO  = os.getenv("MODEL_YOLO", "yolov8n.pt")
USE_REID    = os.getenv("USE_REID", "0") == "1"
REID_MODEL  = os.getenv("REID_MODEL", "osnet_x1_0")
DEVICE      = os.getenv("DEVICE", "cpu")
STREAM_HZ   = float(os.getenv("STREAM_HZ", "5"))
CONFIG_JSON = os.getenv("CONFIG_JSON") or os.getenv("CONFIG_PATH")

yolo_model = YOLO(MODEL_YOLO)
reid_extractor = None
if USE_REID:
    if FeatureExtractor is None:
        raise RuntimeError("torchreid not available but USE_REID=1.")
    reid_extractor = FeatureExtractor(model_name=REID_MODEL, model_path="", device=DEVICE)

cameras: List[CameraTracker] = load_cameras_from_config(CONFIG_JSON) or []

# -------- FastAPI app --------
app = FastAPI(title="Tracker Service", version="1.2.0")

@app.get("/health")
def health():
    opened = sum(1 for c in cameras if c.is_opened())
    return {"ok": True, "cameras_total": len(cameras), "cameras_opened": opened, "reid": USE_REID}

@app.get("/cameras", response_model=List[CameraInfo])
def list_cameras():
    return [
        CameraInfo(
            name=c.name, index=c.index, position_cm=tuple(c.position.tolist()),
            pan_deg=c.pan, tilt_deg=c.tilt, fov_deg=c.fov, opened=c.is_opened(),
            flip=getattr(c, "flip_mode", "none")
        )
        for c in cameras
    ]

@app.get("/cameras/scan")
def scan_dshow_devices():
    """Return a list of DirectShow 'video=...' names if ffmpeg is available; otherwise empty list."""
    import shutil, subprocess
    if not shutil.which("ffmpeg"):
        return {"dshow": []}
    # Ask ffmpeg to list DirectShow devices
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
        capture_output=True, text=True
    )
    names = []
    for line in (proc.stderr or "").splitlines():
        line = line.strip()
        # lines look like: "[dshow @ 000...]  "USB Video Device""
        if "dshow" in line and line.endswith('"') and '"' in line:
            try:
                quoted = line.split('"')
                if len(quoted) >= 2:
                    nm = quoted[1]
                    if nm and nm not in names:
                        names.append(nm)
            except Exception:
                pass
    return {"dshow": [f"video={n}" for n in names]}

@app.post("/config")
def apply_config(cfg: ApplyConfig):
    global cameras
    # release old cameras
    for c in cameras:
        try:
            if c.cap:
                c.cap.release()
        except Exception:
            pass

    # Build specs dict from the posted config
    specs = {
        spec.name: {
            "device": spec.device,
            "index": spec.index,
            "position_cm": spec.position_cm,
            "pan_deg": spec.pan_deg,
            "tilt_deg": spec.tilt_deg,
            "fov_deg": spec.fov_deg,
            "flip": spec.flip,
        }
        for spec in cfg.usb_cameras
    }

    index_map = resolve_indices(specs)

    new_cams = []
    errors = {}
    for spec in cfg.usb_cameras:
        try:
            src = index_map.get(spec.name, spec.index or 0)
            cam = CameraTracker(
                cam_source=src,
                cam_name=spec.name,
                cam_position_cm=spec.position_cm,
                pan_deg=spec.pan_deg,
                tilt_deg=spec.tilt_deg,
                fov_deg=spec.fov_deg,
                flip=spec.flip,
            )
            print(f"[/config] {cam.name} flip={cam.flip_mode!r} code={cam.flip_code} src={cam.source}")
            new_cams.append(cam)
        except Exception as e:
            errors[spec.name] = str(e)

    cameras = new_cams
    opened = sum(1 for c in cameras if c.is_opened())
    return {
        "ok": True,
        "count": len(cameras),
        "opened": opened,
        "errors": errors,
        "source_map": index_map,
    }


@app.get("/tick", response_model=TickResponse)
def tick():
    vectors: List[TrackVector] = []
    for cam in cameras:
        vectors.extend(cam.tick_once(yolo_model, reid_extractor, USE_REID))
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "vectors": vectors}

@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    try:
        interval = 1.0 / max(1.0, STREAM_HZ)
        while True:
            vectors: List[TrackVector] = []
            for cam in cameras:
                vectors.extend(cam.tick_once(yolo_model, reid_extractor, USE_REID))
            payload = {"timestamp": datetime.now(timezone.utc).isoformat(),
                       "vectors": [v.dict() for v in vectors]}
            await ws.send_json(payload)
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        pass



# -------- Live preview endpoints --------

@app.get("/frame/{cam_id}.jpg")
def get_frame_jpg(cam_id: str):
    # Return 204 if no frame yet, so the browser can retry gracefully
    with _preview_lock[cam_id]:
        jpg = _preview_jpeg[cam_id]
    if jpg is None:
        return Response(status_code=204)
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }
    return Response(content=jpg, media_type="image/jpeg", headers=headers)


def _mjpeg_generator(cam_id: str):
    boundary = b"--frame"
    while True:
        with _preview_lock[cam_id]:
            jpg = _preview_jpeg[cam_id]
        if jpg is not None:
            chunk = (boundary +
                     b"\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                     str(len(jpg)).encode() + b"\r\n\r\n" +
                     jpg + b"\r\n")
            yield chunk
        time.sleep(0.03)  # ~33 fps ceiling; adjust as desired


@app.get("/stream/{cam_id}")
def stream_mjpeg(cam_id: str):
    return StreamingResponse(_mjpeg_generator(cam_id),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/preview")
def preview_page():
    # List known camera IDs; fall back to any that have published frames
    cam_ids = [c.name for c in cameras]
    for k in _preview_jpeg.keys():
        if k not in cam_ids:
            cam_ids.append(k)
    cam_ids = sorted(cam_ids)

    html = f"""
<!doctype html><html><head>
<meta charset='utf-8'><title>Live Preview</title>
<style>
  body{{background:#111;color:#eee;font-family:system-ui,Segoe UI,Roboto,Arial}}
  .grid{{display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));padding:16px}}
  .card{{background:#1d1f22;border-radius:12px;overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,.35)}}
  .hdr{{display:flex;justify-content:space-between;align-items:center;padding:10px 12px;background:#2a2d31}}
  .hdr .cam{{font-weight:600}}
  .imgwrap{{display:block;line-height:0}}
  img, video{{width:100%;height:auto;display:block}}
  .meta{{font-size:12px;opacity:.8;padding:8px 12px}}
  a{{color:#8ec7ff}}
</style>
</head><body>
<div class='grid'>
  {''.join([f"""
  <div class='card'>
    <div class='hdr'><div class='cam'>{cam}</div><div><a href='/stream/{cam}'>MJPEG</a></div></div>
    <a class='imgwrap' href='/stream/{cam}'>
      <img id='img_{cam}' src='/frame/{cam}.jpg'>
    </a>
    <div class='meta'>Tip: Click image to switch to MJPEG stream.</div>
  </div>
  """ for cam in cam_ids])}
</div>
<script>
  const cams = {json.dumps(cam_ids)};
  setInterval(()=>{{
    const ts = Date.now();
    cams.forEach(c=>{{
      const el = document.getElementById('img_'+c);
      if (el) el.src = '/frame/'+c+'.jpg?ts='+ts;
    }});
  }}, 400); // ~2.5 fps polling
</script>
</body></html>
"""
    return HTMLResponse(content=html)


# -------- Entrypoint --------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=PORT, reload=False)

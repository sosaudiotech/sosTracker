# -*- coding: utf-8 -*-
from __future__ import annotations

"""
FastAPI tracker service
- YOLO (+ optional OSNet ReID)
- /health, /cameras, /config, /tick, /ws
Env:
  PORT=7001
  MODEL_YOLO=yolov8n.pt
  USE_REID=0|1 (default 0)
  REID_MODEL=osnet_x1_0
  DEVICE=cpu|cuda
  STREAM_HZ=5
  CONFIG_JSON=../../config/room_config.json
  DEFAULT_INDEX_MAP=usb_cam_1:0,usb_cam_2:2  (optional explicit mapping)
"""

import os, math, json, asyncio
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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

class CameraSpec(BaseModel):
    name: str
    position_cm: Vector3
    pan_deg: float
    tilt_deg: float
    fov_deg: float
    index: Optional[int] = None  # optional in config

class ApplyConfig(BaseModel):
    usb_cameras: List[CameraSpec]

# -------- Camera tracker --------
class CameraTracker:
    def __init__(self, cam_index: int, cam_name: str,
                 cam_position_cm: Vector3, pan_deg: float, tilt_deg: float, fov_deg: float):
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.index = cam_index
        self.name = cam_name
        self.position = np.array(cam_position_cm, dtype=float)
        self.pan = float(pan_deg)
        self.tilt = float(tilt_deg)
        self.fov = float(fov_deg)

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

    def tick_once(self, model: YOLO, extractor: Optional["FeatureExtractor"], use_reid: bool):
        ok, frame = self.cap.read()
        if not ok:
            return []

        h, w = frame.shape[:2]
        out: List[TrackVector] = []

        results = model(frame)[0]
        names = results.names if hasattr(results, "names") else getattr(model, "names", {})
        for det in results.boxes:
            cls_idx = int(det.cls)
            cls_name = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
            if cls_name != "person":
                continue

            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().tolist()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            conf = None
            try:
                conf = float(det.conf[0].cpu().item())
            except Exception:
                pass

            ax, ay = self._angle_offsets(cx, cy, w, h)
            direction = self._dir_from_angles(ax, ay)
            pos_estimate = self.position + direction * 100.0

            track_id = None
            if use_reid and extractor is not None:
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size > 0:
                    try:
                        crop_resized = cv2.resize(crop, (256, 128))
                        feat = extractor([crop_resized])[0]
                        track_id = f"id_{int(np.argmax(feat))}"
                    except Exception:
                        pass

            out.append(TrackVector(
                cameraId=self.name,
                origin_cm=tuple(self.position.tolist()),
                direction_unit=tuple(direction.tolist()),
                pos_estimate_cm=tuple(pos_estimate.tolist()),
                confidence=conf,
                track_id=track_id
            ))
        return out

# -------- Helpers (TOP LEVEL, not inside class!) --------
def resolve_indices(usb_specs: Dict[str, Dict]) -> Dict[str, int]:
    """
    Returns { name: index } using:
      1) DEFAULT_INDEX_MAP env (e.g., 'usb_cam_1:0,usb_cam_2:2'), then
      2) scan 0..9 for available indices and assign in name order.
    """
    mapping: Dict[str, int] = {}
    env_map = os.getenv("DEFAULT_INDEX_MAP")
    if env_map:
        for token in env_map.split(","):
            if ":" in token:
                n, idx = token.split(":", 1)
                try:
                    mapping[n.strip()] = int(idx.strip())
                except ValueError:
                    pass

    names = list(usb_specs.keys())
    missing = [n for n in names if n not in mapping]

    taken = set(mapping.values())

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
            index_map = resolve_indices(usb)
            for name, spec in usb.items():
                idx = int(index_map.get(name, spec.get("index", 0)))
                trackers.append(CameraTracker(
                    cam_index=idx,
                    cam_name=str(name),
                    cam_position_cm=tuple(spec.get("position_cm", [0, 0, 0])),
                    pan_deg=float(spec.get("pan_deg", 0)),
                    tilt_deg=float(spec.get("tilt_deg", 0)),
                    fov_deg=float(spec.get("fov_deg", 78)),
                ))
            return trackers
    # fallback
    return [
        CameraTracker(0, "usb_cam_1", (250, 152, 300), 0, 0, 78),
        CameraTracker(2, "usb_cam_2", (-250, 152, 300), 10, 0, 78),
    ]

# -------- Env & bootstrap --------
PORT        = int(os.getenv("PORT", "7001"))
MODEL_YOLO  = os.getenv("MODEL_YOLO", "yolov8n.pt")
USE_REID    = os.getenv("USE_REID", "0") == "1"
REID_MODEL  = os.getenv("REID_MODEL", "osnet_x1_0")
DEVICE      = os.getenv("DEVICE", "cpu")
STREAM_HZ   = float(os.getenv("STREAM_HZ", "5"))
CONFIG_JSON = os.getenv("CONFIG_JSON")

yolo_model = YOLO(MODEL_YOLO)
reid_extractor = None
if USE_REID:
    if FeatureExtractor is None:
        raise RuntimeError("torchreid not available but USE_REID=1.")
    reid_extractor = FeatureExtractor(model_name=REID_MODEL, model_path="", device=DEVICE)

cameras: List[CameraTracker] = load_cameras_from_config(CONFIG_JSON)

# -------- FastAPI app --------
app = FastAPI(title="Tracker Service", version="1.0.0")

@app.get("/health")
def health():
    opened = sum(1 for c in cameras if c.is_opened())
    return {"ok": True, "cameras_total": len(cameras), "cameras_opened": opened, "reid": USE_REID}

@app.get("/cameras", response_model=List[CameraInfo])
def list_cameras():
    return [
        CameraInfo(
            name=c.name, index=c.index, position_cm=tuple(c.position.tolist()),
            pan_deg=c.pan, tilt_deg=c.tilt, fov_deg=c.fov, opened=c.is_opened()
        )
        for c in cameras
    ]

@app.post("/config")
def apply_config(cfg: ApplyConfig):
    global cameras
    for c in cameras:
        try:
            if c.cap: c.cap.release()
        except Exception:
            pass

    specs = {
        spec.name: {
            "index": spec.index,
            "position_cm": spec.position_cm,
            "pan_deg": spec.pan_deg,
            "tilt_deg": spec.tilt_deg,
            "fov_deg": spec.fov_deg,
        } for spec in cfg.usb_cameras
    }
    index_map = resolve_indices(specs)

    cameras = [
        CameraTracker(
            cam_index=int(index_map.get(spec.name, spec.index or 0)),
            cam_name=spec.name,
            cam_position_cm=spec.position_cm,
            pan_deg=spec.pan_deg,
            tilt_deg=spec.tilt_deg,
            fov_deg=spec.fov_deg,
        )
        for spec in cfg.usb_cameras
    ]
    opened = sum(1 for c in cameras if c.is_opened())
    return {"ok": True, "count": len(cameras), "opened": opened, "index_map": index_map}

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


# -------- Entrypoint --------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="127.0.0.1", port=PORT, reload=False)

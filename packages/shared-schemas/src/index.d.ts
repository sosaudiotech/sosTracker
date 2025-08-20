export type Vector3 = [number, number, number];

export interface TrackVector {
  cameraId: string;
  origin_cm: Vector3;
  direction_unit: Vector3;
  pos_estimate_cm: Vector3;
  confidence?: number;
  track_id?: string;
}

export interface TickResponse {
  timestamp: string;
  vectors: TrackVector[];
}

export interface PTZCommand {
  cameraId: string;
  pan: number;   // normalize to -1..+1 or degrees consistently
  tilt: number;
  zoom: number;  // 0..1 per your latest standard
  uid?: string;
}

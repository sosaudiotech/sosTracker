// apps/orchestrator/src/server.js
import fs from "fs";
import path from "path";
import fetch from "node-fetch";
import express from "express";
import http from "http";
import { WebSocket, WebSocketServer } from "ws";
import dotenv from "dotenv";
import { fileURLToPath } from "url";

dotenv.config();

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3000;
const PY_BASE = process.env.PY_TRACKER_BASE || "http://127.0.0.1:7001";
const CONFIG_PATH = process.env.CONFIG_PATH || path.resolve(process.cwd(), "../../config/room_config.json");

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: "/ws" });

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const UI_DIR = path.join(__dirname, "ui");
app.use("/static", express.static(UI_DIR));
app.get("/", (_req, res) => res.sendFile(path.join(UI_DIR, "index.html")));

// --- Simulation state ---
let SIM_MODE = process.env.SIM_MODE === "1";
let simPose = { x: 0, y: 120, z: 200 };
let simHz = Number(process.env.SIM_HZ || 5);

function broadcastTelemetry(payload) {
    const msg = JSON.stringify({ type: "telemetry", payload });
    for (const c of wss.clients) {
        if (c.readyState === 1) c.send(msg);
    }
}

let pyWS = null;
let lastConfig = null;

function makeVectorFromPose(camera) {
    const origin = camera.position_cm;
    const dir = [simPose.x - origin[0], simPose.y - origin[1], simPose.z - origin[2]];
    const len = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    const unit = [dir[0] / len, dir[1] / len, dir[2] / len];
    return {
        cameraId: camera.name,
        origin_cm: origin,
        direction_unit: unit,
        pos_estimate_cm: [origin[0] + unit[0] * 100, origin[1] + unit[1] * 100, origin[2] + unit[2] * 100],
        confidence: 0.99
    };
}

async function loadConfig() {
    lastConfig = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf8"));
    return lastConfig;
}

async function applyConfigToPython() {
    if (!lastConfig) await loadConfig();
    const usb = lastConfig.usb_cameras || {};
    const usb_cameras = Object.entries(usb).map(([name, spec]) => ({
        name,
        // index optional; service.py will resolve if missing
        index: spec.index ?? undefined,
        position_cm: spec.position_cm ?? [0, 0, 0],
        pan_deg: spec.pan_deg ?? 0,
        tilt_deg: spec.tilt_deg ?? 0,
        fov_deg: spec.fov_deg ?? 78
    }));

    try {
        const r = await fetch(`${PY_BASE}/config`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ usb_cameras })
        });
        const j = await r.json().catch(() => ({}));
        console.log(`Tracker config applied: ${usb_cameras.length} camera(s).`, j.index_map ? `Index map: ${JSON.stringify(j.index_map)}` : "");
    } catch (e) {
        console.error("Failed to apply tracker config:", e.message);
    }
}

function connectTrackerWS() {
    const PY_WS = PY_BASE.replace("http", "ws") + "/ws";
    try {
        pyWS = new WebSocket(PY_WS);
        pyWS.on("open", () => console.log("Connected to tracker WS"));
        pyWS.on("message", data => {
            if (SIM_MODE) return; // ignore real data when sim is on
            try {
                const payload = JSON.parse(data.toString());
                broadcastTelemetry({ ...payload, source: "python" });
            } catch { /* ignore */ }
        });
        pyWS.on("close", () => {
            console.warn("Tracker WS closed; reconnecting in 2s");
            setTimeout(connectTrackerWS, 2000);
        });
        pyWS.on("error", () => { /* swallow */ });
    } catch {
        setTimeout(connectTrackerWS, 2000);
    }
}

async function mainLoop() {
    await loadConfig();
    await applyConfigToPython();
    connectTrackerWS();

    setInterval(async () => {
        if (SIM_MODE && lastConfig?.usb_cameras) {
            const cams = Object.entries(lastConfig.usb_cameras).map(([name, spec]) => ({
                name,
                position_cm: spec.position_cm,
                pan_deg: spec.pan_deg,
                tilt_deg: spec.tilt_deg,
                fov_deg: spec.fov_deg
            }));
            const vectors = cams.map(makeVectorFromPose);
            broadcastTelemetry({ timestamp: new Date().toISOString(), vectors, source: "sim" });
            return;
        }

        // Fallback polling only if WS not open and not in SIM
        const wsOpen = pyWS && pyWS.readyState === 1;
        if (!wsOpen) {
            try {
                const r = await fetch(`${PY_BASE}/tick`);
                const payload = await r.json();
                broadcastTelemetry({ ...payload, source: "python-poll" });
            } catch { /* ignore transient errors */ }
        }
    }, Math.max(50, 1000 / (SIM_MODE ? simHz : 5)));
}
mainLoop();

// --- REST passthroughs & helpers ---
app.get("/health", async (_req, res) => {
    try {
        const r = await fetch(`${PY_BASE}/health`);
        res.json(await r.json());
    } catch {
        res.status(502).json({ ok: false });
    }
});

app.post("/api/ptz", async (req, res) => {
    // TODO: route to your PTZ/Q-SYS adapter
    console.log("PTZ:", req.body);
    res.json({ ok: true });
});

app.get("/api/config", (_req, res) => {
    try {
        const cfg = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf8"));
        res.json(cfg);
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

// Simulation controls
app.post("/api/sim/start", (_req, res) => { SIM_MODE = true; res.json({ ok: true, sim: true }); });
app.post("/api/sim/stop", (_req, res) => { SIM_MODE = false; res.json({ ok: true, sim: false }); });
app.post("/api/sim/pose", (req, res) => {
    const { x, y, z } = req.body || {};
    if ([x, y, z].some(v => typeof v !== "number")) return res.status(400).json({ error: "x,y,z required" });
    simPose = { x, y, z };
    res.json({ ok: true, simPose });
});
app.get("/api/sim/status", (_req, res) => res.json({ sim: SIM_MODE, simPose, simHz }));

// Debug viewer
app.get("/tap", (_req, res) => {
    res.setHeader("content-type", "text/html; charset=utf-8");
    res.end(`<!doctype html><html><head><meta charset="utf-8"><title>Telemetry Tap</title>
  <style>body{font:14px system-ui;margin:20px} pre{white-space:pre-wrap;border:1px solid #444;padding:10px;border-radius:8px;max-height:60vh;overflow:auto}</style>
  </head><body><h1>Telemetry Tap</h1><div>Status: <span id="s">connecting…</span></div><pre id="out"></pre>
  <script>
    const s=document.getElementById('s'),out=document.getElementById('out');
    function log(x){ out.textContent+=x+"\\n"; out.scrollTop=out.scrollHeight; }
    const ws=new WebSocket('ws://'+location.host+'/ws');
    ws.addEventListener('open',()=>s.textContent='connected ✅');
    ws.addEventListener('close',()=>s.textContent='disconnected ❌');
    ws.addEventListener('error',e=>{s.textContent='error ⚠️';log(String(e))});
    ws.addEventListener('message',ev=>{
      try{
        const msg=JSON.parse(ev.data);
        if(msg.type==='telemetry'){
          const {timestamp,vectors}=msg.payload;
          log(new Date(timestamp).toLocaleTimeString()+' — '+vectors.length+' vectors');
          if(vectors[0]) log(JSON.stringify(vectors[0],null,2));
        } else { log(ev.data); }
      }catch(e){ log(ev.data); }
    });
  </script></body></html>`);
});

server.listen(PORT, () => {
    console.log(`Orchestrator listening on http://localhost:${PORT}`);
});

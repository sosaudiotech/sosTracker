// apps/orchestrator/ecosystem.config.cjs

const path = require("path");

module.exports = {
    apps: [
        {
            name: "orchestrator",
            cwd: __dirname,
            script: "src/server.js",
            interpreter: "node",
            env: {
                NODE_ENV: "production",
                PY_TRACKER_BASE: "http://127.0.0.1:7001",
                CONFIG_PATH: path.join(__dirname, "../../config/room_config.json")
            }
        },
        {
            name: "tracker_py",
            cwd: path.join(__dirname, "../tracker_py/tracker"),
            script: "service.py",
            interpreter: "python", // or "python3" or full path if needed
            env: {
                PORT: 7001,
                MODEL_YOLO: "yolov8n.pt",
                USE_REID: "0",
                DEVICE: "cpu",
                STREAM_HZ: "5",
                PYTHONUNBUFFERED: "1"
            }
        }
    ]
};


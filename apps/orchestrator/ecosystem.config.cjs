// apps/orchestrator/ecosystem.config.cjs
module.exports = {
  apps: [
    {
      name: "orchestrator",
      cwd: __dirname,
      script: "node",
      args: "src/server.js",
      env: {
        NODE_ENV: "production",
        PY_TRACKER_BASE: "http://127.0.0.1:7001",
        CONFIG_PATH: path.join(__dirname, "../../config/room_config.json")
      }
    },
    {
      name: "tracker_py",
      cwd: path.join(__dirname, "../tracker_py/tracker"),
      script: "python",
      args: "service.py",
      interpreter: "none",
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

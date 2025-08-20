// visualizer_logic.js
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.module.js';
//import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/loaders/GLTFLoader.js';
import { createOrbitControls } from '/static/OrbitControls.js';


let scene, camera, renderer, controls;
let speakerSphere, raycaster, mouse, isDragging = false;
const cornerLabels = [];
let directionLines = [];
let micDirectionLines = [];

// PTZ WebSocket broadcasting and visualization will now happen live in updateTrackingVectors()

let ws;
init();
loadRoomConfig();

function init() {

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const container = document.getElementById('canvas-container');
    const width = window.innerWidth;
    const height = window.innerHeight;

    camera = new THREE.PerspectiveCamera(45, width / height, 1, 5000);
    camera.position.set(1000, 600, 1000);
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    controls = createOrbitControls(THREE, camera, renderer.domElement, { panSpeed: 0.002 });
    controls.target.set(0, 150, 0);
    controls.rotateSpeed = 0.25;
    controls.update();
    controls.onUpdate = updateCornerLabels;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const gridHelper = new THREE.GridHelper(732, 24, 0x444444, 0x222222);
    gridHelper.material.opacity = 0.5;
    gridHelper.material.transparent = true;
    scene.add(gridHelper);

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    window.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    window.addEventListener('keydown', onKeyDown);

    addAxesHelper();
    addLegendOverlay();

    animate();
}

async function setSimPose(x, y, z) {
    await fetch('/api/sim/pose', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ x, y, z })
    });
}


function add3DModelIcon(url, position, scale = 10) {
    const loader = new GLTFLoader();
    loader.load(url, (gltf) => {
        const model = gltf.scene;
        model.position.set(...position);
        model.scale.set(scale, scale, scale);
        scene.add(model);
    }, undefined, (error) => {
        console.error(`Error loading model ${url}:`, error);
    });
}


function onKeyDown(event) {
    if (event.key === 'r') {
        controls.target.set(0, 0, 0);
        camera.position.set(500, 300, 500);
        controls.update();
    }
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function createDeviceLabel(text) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    context.font = '16px Arial';
    context.fillStyle = 'white';
    context.fillText(text, 0, 20);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(60, 30, 1);
    return sprite;
}

function addDeviceLabels(devices) {
    for (const [id, device] of Object.entries(devices)) {
        const [x, y, z] = device.position_cm;
        const label = createDeviceLabel(device.name || id);
        label.position.set(x, y + 20, z);
        scene.add(label);
    }
}

function createCameraFOVCone(origin, panDeg, fovDeg, tiltDeg = 0) {
    const length = 200;
    const aspectWidth = 16;
    const aspectHeight = 9;
    const diagonalFactor = Math.sqrt(aspectWidth ** 2 + aspectHeight ** 2);

    const halfDiagRad = THREE.MathUtils.degToRad(fovDeg / 2);
    const halfH = Math.tan(halfDiagRad) * (aspectWidth / diagonalFactor);
    const halfV = Math.tan(halfDiagRad) * (aspectHeight / diagonalFactor);
    //const halfH = halfV * aspectRatio;

    const geometry = new THREE.BufferGeometry();

    const vertices = new Float32Array([
        0, 0, 0,
        -length * halfH, length * halfV, length,
        length * halfH, length * halfV, length,
        length * halfH, -length * halfV, length,
        -length * halfH, -length * halfV, length
    ]);

    const indices = [
        0, 1, 2,
        0, 2, 3,
        0, 3, 4,
        0, 4, 1
    ];

    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({ color: 0x0000ff, transparent: true, opacity: 0.1, side: THREE.DoubleSide });
    const frustum = new THREE.Mesh(geometry, material);

    frustum.position.set(...origin);

    const panRad = THREE.MathUtils.degToRad(panDeg);
    const tiltRad = THREE.MathUtils.degToRad(tiltDeg);
    const euler = new THREE.Euler(tiltRad, panRad, 0, 'YXZ');
    frustum.setRotationFromEuler(euler);


    return frustum;
}

function addFOVRay(cam) {
    const [x, y, z] = cam.position_cm;
    const origin = new THREE.Vector3(x, y, z);
    const panRad = THREE.MathUtils.degToRad(cam.pan_deg || 0);
    const tiltRad = THREE.MathUtils.degToRad(cam.tilt_deg || 0);

    const dir = new THREE.Vector3(
        Math.sin(panRad) * Math.cos(-tiltRad),
        Math.sin(-tiltRad),
        Math.cos(panRad) * Math.cos(-tiltRad)
    ).normalize();

    const target = origin.clone().add(dir.multiplyScalar(500));

    const geometry = new THREE.BufferGeometry().setFromPoints([origin, target]);
    const material = new THREE.LineDashedMaterial({ color: 0x666666, dashSize: 20, gapSize: 10 });
    const line = new THREE.Line(geometry, material);
    line.computeLineDistances();

    scene.add(line);
}
  
function updateTrackingVectors() {
    const ptzEstimateColor = 0x00ffcc;
    const speaker = new THREE.Vector3(speakerSphere.position.x, speakerSphere.position.y, speakerSphere.position.z);
    const infoPanel = document.getElementById('info-panel');
    if (infoPanel) {
        infoPanel.innerHTML = '';

        const { x, y, z } = speakerSphere.position;
        const speakerLabel = document.createElement('div');
        speakerLabel.style.fontWeight = 'bold';
        speakerLabel.style.marginBottom = '4px';
        speakerLabel.textContent = `🎤 Speaker Position: x=${x.toFixed(1)}, y=${y.toFixed(1)}, z=${z.toFixed(1)}`;
        infoPanel.appendChild(speakerLabel);
    }

    for (const line of directionLines) scene.remove(line);
    directionLines = [];
    for (const line of micDirectionLines) scene.remove(line);
    micDirectionLines = [];

    const speakerPos = speakerSphere.position;

    for (const [id, ptz] of Object.entries(window.ptzCameras || {})) {
        const origin = new THREE.Vector3(...ptz.position_cm);
        const delta = speaker.clone().sub(origin);
        const distance = delta.length();

        const panDeg = -THREE.MathUtils.radToDeg(Math.atan2(delta.x, delta.z));
        const tiltDeg = THREE.MathUtils.radToDeg(Math.atan2(delta.y, Math.sqrt(delta.x ** 2 + delta.z ** 2)));

        const normPan = (panDeg / 170).toFixed(4);
        const normTilt = ((tiltDeg + 30) / 120 * (2.9808 + 0.9936) - 0.9936).toFixed(4);
        const normZoom = Math.min(1.0, (distance / 1000)).toFixed(4);



        fetch('/api/ptz', {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({
                cameraId: id,
                pan: Number(normPan),
                tilt: Number(normTilt),
                zoom: Number(normZoom),
                uid: `viz-${Date.now()}`
            })
        }).catch(() => { });



        const geometry = new THREE.BufferGeometry().setFromPoints([origin, speaker]);
        const material = new THREE.LineDashedMaterial({ color: 0xffff00, dashSize: 5, gapSize: 5 });
        const line = new THREE.Line(geometry, material);
        line.computeLineDistances();
        scene.add(line);
        directionLines.push(line);
    }
    for (const [id, cam] of Object.entries(window.usbCameras || {})) {
        const camOrigin = new THREE.Vector3(...cam.position_cm);
        const speaker = new THREE.Vector3(speakerSphere.position.x, speakerSphere.position.y, speakerSphere.position.z);
        const camDelta = speaker.clone().sub(camOrigin);

        // Convert to camera local coordinates
        const panRad = THREE.MathUtils.degToRad(cam.pan_deg || 0);
        const tiltRad = THREE.MathUtils.degToRad(cam.tilt_deg || 0);
        const camQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(tiltRad, panRad, 0, 'YXZ'));
        const inverseQuat = camQuat.clone().invert();
        const localDelta = camDelta.clone().applyQuaternion(inverseQuat);

        const xNorm = localDelta.x / (Math.abs(localDelta.z) * Math.tan(THREE.MathUtils.degToRad(cam.fov_deg / 2)));
        const yNorm = localDelta.y / (Math.abs(localDelta.z) * Math.tan(THREE.MathUtils.degToRad(cam.fov_deg / 2)) * (9 / 16));

        const label = document.createElement('div');
        label.style.fontSize = '12px';
        label.style.color = '#aaa';
        label.textContent = `${id}: x=${xNorm.toFixed(2)}, y=${yNorm.toFixed(2)}`;
        if (infoPanel) infoPanel.appendChild(label);
        const origin = new THREE.Vector3(...cam.position_cm);
        const target = new THREE.Vector3(speakerPos.x, speakerPos.y, speakerPos.z);
        const geometry = new THREE.BufferGeometry().setFromPoints([origin, target]);
        const material = new THREE.LineBasicMaterial({ color: 0xff00ff });
        const line = new THREE.Line(geometry, material);
        scene.add(line);
        directionLines.push(line);
    }

    for (const [id, mic] of Object.entries(window.microphones || {})) {
        const origin = new THREE.Vector3(...mic.position_cm);
        const speaker = new THREE.Vector3(speakerSphere.position.x, speakerSphere.position.y, speakerSphere.position.z);
        const delta = speaker.clone().sub(origin);

        const distance = delta.length();
        if (distance === 0) continue;

        const azimuthRad = Math.atan2(delta.x, -delta.z);
        const elevationRad = Math.asin(-delta.y / distance);

        const azimuthDeg = THREE.MathUtils.radToDeg(azimuthRad).toFixed(1);
        const elevationDeg = THREE.MathUtils.radToDeg(elevationRad).toFixed(1);

        const micLabel = document.createElement('div');
        micLabel.style.fontSize = '12px';
        micLabel.style.color = '#ffcc66';
        micLabel.textContent = `${id}: az=${azimuthDeg}°, el=${elevationDeg}°`;
        if (infoPanel) infoPanel.appendChild(micLabel);

        const dir = new THREE.Vector3(
            Math.sin(azimuthRad) * Math.cos(elevationRad),
            -Math.sin(elevationRad),
            -Math.cos(azimuthRad) * Math.cos(elevationRad)
        ).normalize();

        const target = origin.clone().add(dir.multiplyScalar(1000));
        const geometry = new THREE.BufferGeometry().setFromPoints([origin, target]);
        const material = new THREE.LineBasicMaterial({ color: 0xffaa00 });
        const line = new THREE.Line(geometry, material);
        scene.add(line);
        micDirectionLines.push(line);
        micDirectionLines.push(line);
    }
}

function drawRoom(config) {
    const { dimensions_cm, microphones, usb_cameras, ptz_cameras, speaker_zone_cm } = config;

    const room = new THREE.BoxHelper(
        new THREE.Mesh(new THREE.BoxGeometry(dimensions_cm.x, dimensions_cm.y, dimensions_cm.z)),
        0x888888
    );
    room.position.set(dimensions_cm.x / 2, dimensions_cm.y / 2, dimensions_cm.z / 2);
    room.geometry.translate(0, dimensions_cm.y / 2, 0);
    scene.add(room);

    window.microphones = microphones;
    for (const [id, mic] of Object.entries(microphones)) {
        const sphere = new THREE.Mesh(
            new THREE.SphereGeometry(10, 16, 16),
            new THREE.MeshStandardMaterial({ color: 0xff0000 })
        );
        const [x, y, z] = mic.position_cm;
        sphere.position.set(x, y, z);
        scene.add(sphere);
    }
   

    window.usbCameras = usb_cameras;
    window.ptzCameras = ptz_cameras;
    for (const [id, cam] of Object.entries(usb_cameras)) {
        const box = new THREE.Mesh(
            new THREE.BoxGeometry(20, 20, 20),
            new THREE.MeshStandardMaterial({ color: 0x0000ff })
        );
        const [x, y, z] = cam.position_cm;
        box.position.set(x, y, z);
        scene.add(box);

        const cone = createCameraFOVCone([x, y, z], cam.pan_deg, cam.fov_deg, cam.tilt_deg || 0);
        scene.add(cone);

        addFOVRay(cam);
    }

    for (const [id, cam] of Object.entries(ptz_cameras)) {
        const cyl = new THREE.Mesh(
            new THREE.CylinderGeometry(10, 10, 30, 16),
            new THREE.MeshStandardMaterial({ color: 0xffff00 })
        );
        const [x, y, z] = cam.position_cm;
        cyl.position.set(x, y, z);
        scene.add(cyl);
    }

    const zoneSize = [
        speaker_zone_cm.max[0] - speaker_zone_cm.min[0],
        speaker_zone_cm.max[1] - speaker_zone_cm.min[1],
        speaker_zone_cm.max[2] - speaker_zone_cm.min[2]
    ];
    const zoneCenter = [
        (speaker_zone_cm.max[0] + speaker_zone_cm.min[0]) / 2,
        (speaker_zone_cm.max[1] + speaker_zone_cm.min[1]) / 2,
        (speaker_zone_cm.max[2] + speaker_zone_cm.min[2]) / 2
    ];
    const zone = new THREE.Mesh(
        new THREE.BoxGeometry(...zoneSize),
        new THREE.MeshStandardMaterial({ color: 0x00ff00, transparent: true, opacity: 0.2 })
    );
    zone.position.set(...zoneCenter);
    scene.add(zone);
}

async function loadRoomConfig() {
    const res = await fetch('api/config');
    const config = await res.json();
    drawRoom(config);
    addDraggableSpeaker();
    addCornerLabels(config);
    addDeviceLabels(config.microphones);
    addDeviceLabels(config.usb_cameras);
    addDeviceLabels(config.ptz_cameras);
    addCalibrationMarkers(config.calibration_markers || {});
    add3DModelIcon('/models/mic.glb', [0, 150, 0], 12);

}

function addDraggableSpeaker() {
    const geometry = new THREE.SphereGeometry(12, 20, 20);
    const material = new THREE.MeshStandardMaterial({ color: 0xaa00ff });
    speakerSphere = new THREE.Mesh(geometry, material);
    speakerSphere.position.set(0, 120, 400);
    speakerSphere.name = 'speaker';
    scene.add(speakerSphere);
}

function addCornerLabels(config) {
    const { dimensions_cm } = config;
    const corners = [
        [0, 0, 0],
        [dimensions_cm.x, 0, 0],
        [0, dimensions_cm.y, 0],
        [0, 0, dimensions_cm.z],
        [dimensions_cm.x, dimensions_cm.y, 0],
        [dimensions_cm.x, 0, dimensions_cm.z],
        [0, dimensions_cm.y, dimensions_cm.z],
        [dimensions_cm.x, dimensions_cm.y, dimensions_cm.z],
    ];

    //for (const pos of corners) {
    //    const label = createCornerLabel(`${pos[0]},${pos[1]},${pos[2]}`);
    //    label.position.set(...pos);
    //    cornerLabels.push(label);
    //    scene.add(label);
    //}
}

function addCalibrationMarkers(markers) {
    for (const [id, marker] of Object.entries(markers)) {
        const [x, y, z] = marker.position_cm;
        const color = parseInt(marker.color || '0xffff00');

        const mesh = new THREE.Mesh(
            new THREE.SphereGeometry(8, 16, 16),
            new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.7 })
        );
        mesh.position.set(x, y, z);
        scene.add(mesh);

        const label = createDeviceLabel(id);
        label.position.set(x, y + 15, z);
        scene.add(label);
    }
}

function createCornerLabel(text) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    context.font = '40px Arial';
    context.fillStyle = 'white';
    context.fillText(text, 0, 20);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(80, 40, 1);
    return sprite;
}

function updateCornerLabels() {
    for (const label of cornerLabels) {
        label.lookAt(camera.position);
    }
}

function addAxesHelper() {
    const axesHelper = new THREE.AxesHelper(100);
    axesHelper.position.set(0, 0, 0);
    scene.add(axesHelper);
}

function addLegendOverlay() {
    const controlPanel = document.createElement('div');
    controlPanel.style.position = 'absolute';
    controlPanel.style.bottom = '10px';
    controlPanel.style.right = '10px';
    controlPanel.style.padding = '10px';
    controlPanel.style.background = 'rgba(0,0,0,0.6)';
    controlPanel.style.color = '#fff';
    controlPanel.style.fontSize = '12px';
    controlPanel.style.display = 'grid';
    controlPanel.style.gridTemplateColumns = 'repeat(4, auto)';
    controlPanel.style.gap = '4px';

    const axes = ['x', 'z', 'y'];
    const steps = [10, 100];
    const simOn = document.createElement('button');
    simOn.textContent = 'SIM ON';
    simOn.onclick = () => fetch('/api/sim/start', { method: 'POST' });

    const simOff = document.createElement('button');
    simOff.textContent = 'SIM OFF';
    simOff.onclick = () => fetch('/api/sim/stop', { method: 'POST' });

    controlPanel.appendChild(simOn);
    controlPanel.appendChild(simOff);

    for (const axis of axes) {
        const deltas = [
            { sign: -1, step: 100 },
            { sign: -1, step: 10 },
            { sign: 1, step: 10 },
            { sign: 1, step: 100 }
        ];
        for (const { sign, step } of deltas) {
            const btn = document.createElement('button');
            const dir = sign > 0 ? '+' : '–';
            btn.textContent = `${axis}${dir}${step}`;
            btn.style.padding = '2px 6px';
            btn.style.fontSize = '12px';
            btn.onclick = () => {
                speakerSphere.position[axis] += sign * step;
                updateTrackingVectors();
            };
            controlPanel.appendChild(btn);
        }
    }


    document.body.appendChild(controlPanel);
    const panel = document.createElement('div');
    panel.id = 'info-panel';
    panel.style.position = 'absolute';
    panel.style.bottom = '10px';
    panel.style.left = '10px';
    panel.style.padding = '10px';
    panel.style.background = 'rgba(0,0,0,0.6)';
    panel.style.color = '#fff';
    panel.style.fontSize = '12px';
    panel.style.maxHeight = '25vh';
    panel.style.overflowY = 'auto';
    panel.innerHTML = '<strong>Live Data</strong><br>';
    document.body.appendChild(panel);
    const legend = document.createElement('div');
    legend.style.position = 'absolute';
    legend.style.top = '10px';
    legend.style.left = '10px';
    legend.style.padding = '10px';
    legend.style.background = 'rgba(0,0,0,0.6)';
    legend.style.color = '#fff';
    legend.style.fontSize = '14px';
    legend.innerHTML = `
    <strong>Legend</strong><br>
    🔴 Microphone<br>
    🔵 USB Camera<br>
    🔷 Camera FOV<br>
    🟡 PTZ Camera<br>
    🟢 Calibration Points<br>
    🟣 Speaker (Draggable)
  `;
    document.body.appendChild(legend);
}

function addIconSprite(svgString, position) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const size = 64;
    canvas.width = size;
    canvas.height = size;

    const img = new Image();
    const svgBlob = new Blob([svgString], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(svgBlob);

    img.onload = () => {
        ctx.clearRect(0, 0, size, size);
        ctx.drawImage(img, 0, 0, size, size);
        URL.revokeObjectURL(url);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
        const sprite = new THREE.Sprite(material);
        sprite.scale.set(40, 40, 1); // adjust size in scene
        sprite.position.set(...position);
        scene.add(sprite);
    };

    img.src = url;
}

// --- NEW: render tracker vectors coming from Python (via Orchestrator) ---
let trackerLines = [];
function drawTrackerVectors(payload) {
    // clear old
    for (const line of trackerLines) scene.remove(line);
    trackerLines.length = 0;

    const { vectors } = payload || {};
    if (!Array.isArray(vectors)) return;

    for (const v of vectors) {
        const a = new THREE.Vector3(...v.origin_cm);
        const b = new THREE.Vector3(...v.pos_estimate_cm);
        const geom = new THREE.BufferGeometry().setFromPoints([a, b]);
        const mat = new THREE.LineBasicMaterial({ color: 0x00ffcc }); // cyan = tracker
        const line = new THREE.Line(geom, mat);
        scene.add(line);
        trackerLines.push(line);
    }

    // optional: update info panel
    const infoPanel = document.getElementById('info-panel');
    if (infoPanel) {
        infoPanel.innerHTML = `<strong>Live Data</strong><br/>Vectors: ${vectors.length}`;
    }
}

function onMouseDown(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(speakerSphere);
    if (intersects.length > 0) {
        isDragging = true;
        controls.enabled = false;
    } else {
        controls.enabled = true;
    }
}

function onMouseMove(event) {
    if (!isDragging) return;
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);

    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -speakerSphere.position.y);
    const intersectPoint = new THREE.Vector3();
    raycaster.ray.intersectPlane(plane, intersectPoint);

    const minX = 0, maxX = 1800;
    const minZ = -1000, maxZ = 1000;

    if (intersectPoint) {
        const clampedX = Math.max(minX, Math.min(maxX, intersectPoint.x));
        const clampedZ = Math.max(minZ, Math.min(maxZ, intersectPoint.z));
        speakerSphere.position.x = clampedX;
        speakerSphere.position.z = clampedZ;
        updateTrackingVectors();
    }

  
}

function onMouseUp(event) {
    if (isDragging) {
        isDragging = false;
    }
    controls.enabled = true;
    // push new sim pose to backend (so the cyan tracker overlay follows your draggable dot)
    setSimPose(speakerSphere.position.x, speakerSphere.position.y, speakerSphere.position.z);

}

function panDegToVector(deg) {
    const rad = THREE.MathUtils.degToRad(deg);
    return [Math.sin(rad), 0, Math.cos(rad)];
}

// --- NEW: subscribe to orchestrator telemetry ---
function connectTelemetry() {
    const ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onmessage = (ev) => {
        try {
            const msg = JSON.parse(ev.data);
            if (msg.type === 'telemetry') {
                drawTrackerVectors(msg.payload); // { timestamp, vectors }
            }
        } catch { }
    };
}
connectTelemetry();

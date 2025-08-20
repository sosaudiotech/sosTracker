// static/OrbitControls.js
export function createOrbitControls(THREE, camera, domElement, options = {}) {
    const {
        EventDispatcher, MOUSE, TOUCH, Quaternion, Spherical, Vector2, Vector3, MathUtils
    } = THREE;

    class OrbitControls extends EventDispatcher {
        constructor(camera, domElement) {
            super();
            this.camera = camera;
            this.domElement = domElement;

            this.enabled = true;
            this.target = new Vector3();
            this.minDistance = 0;
            this.maxDistance = Infinity;
            this.enableDamping = true;
            this.dampingFactor = 0.05;
            this.rotateSpeed = 0.5;
            this.zoomSpeed = 1.2;
            this.panSpeed = options.panSpeed || 0.25; // Exposed pan sensitivity

            this.spherical = new Spherical();
            this.sphericalDelta = new Spherical();
            this.scale = 1;
            this.panOffset = new Vector3();
            this.zoomChanged = false;
            this.rotateStart = new Vector2();
            this.rotateEnd = new Vector2();
            this.rotateDelta = new Vector2();
            this.panStart = new Vector2();
            this.panEnd = new Vector2();
            this.panDelta = new Vector2();
            this.state = 'none';

            this._onMouseDown = this.onMouseDown.bind(this);
            this._onMouseMove = this.onMouseMove.bind(this);
            this._onMouseUp = this.onMouseUp.bind(this);
            this._onMouseWheel = this.onMouseWheel.bind(this);

            domElement.addEventListener('mousedown', this._onMouseDown);
            domElement.addEventListener('wheel', this._onMouseWheel, { passive: false });
            domElement.addEventListener('contextmenu', e => e.preventDefault());

            this.update();
        }

        update() {
            const offset = new Vector3();
            offset.copy(this.camera.position).sub(this.target);
            this.spherical.setFromVector3(offset);
            this.spherical.theta += this.sphericalDelta.theta;
            this.spherical.phi += this.sphericalDelta.phi;
            this.spherical.makeSafe();
            this.spherical.radius *= this.scale;
            this.spherical.radius = Math.max(this.minDistance, Math.min(this.maxDistance, this.spherical.radius));
            offset.setFromSpherical(this.spherical);
            this.target.add(this.panOffset);
            this.camera.position.copy(this.target).add(offset);
            this.camera.lookAt(this.target);

            // If using 3D sprites for coordinate labels, update them here based on camera and target.
            if (typeof this.onUpdate === 'function') {
                this.onUpdate();
            }

            if (this.enableDamping) {
                this.sphericalDelta.theta *= (1 - this.dampingFactor);
                this.sphericalDelta.phi *= (1 - this.dampingFactor);
                this.panOffset.multiplyScalar(1 - this.dampingFactor);
            } else {
                this.sphericalDelta.set(0, 0, 0);
                this.panOffset.set(0, 0, 0);
            }

            this.scale = 1;
        }

        onMouseDown(event) {
            if (!this.enabled) return;

            if (event.button === MOUSE.LEFT) {
                if (event.shiftKey) {
                    this.state = 'pan';
                    this.panStart.set(event.clientX, event.clientY);
                } else {
                    this.state = 'rotate';
                    this.rotateStart.set(event.clientX, event.clientY);
                }
                this.domElement.addEventListener('mousemove', this._onMouseMove);
                this.domElement.addEventListener('mouseup', this._onMouseUp);
            }
        }

        onMouseMove(event) {
            if (!this.enabled) return;

            if (this.state === 'rotate') {
                this.rotateEnd.set(event.clientX, event.clientY);
                this.rotateDelta.subVectors(this.rotateEnd, this.rotateStart).multiplyScalar(this.rotateSpeed);

                const element = this.domElement;
                this.sphericalDelta.theta -= 2 * Math.PI * this.rotateDelta.x / element.clientHeight;
                this.sphericalDelta.phi -= 2 * Math.PI * this.rotateDelta.y / element.clientHeight;

                this.rotateStart.copy(this.rotateEnd);

            } else if (this.state === 'pan') {
                this.panEnd.set(event.clientX, event.clientY);
                this.panDelta.subVectors(this.panEnd, this.panStart).multiplyScalar(this.panSpeed);

                const offset = new THREE.Vector3();
                const pan = new THREE.Vector3();

                pan.setFromMatrixColumn(this.camera.matrix, 0); // x axis
                pan.multiplyScalar(-this.panDelta.x);
                offset.add(pan);

                pan.setFromMatrixColumn(this.camera.matrix, 1); // y axis
                pan.multiplyScalar(this.panDelta.y);
                offset.add(pan);

                this.panOffset.add(offset);
                this.panStart.copy(this.panEnd);
            }
        }

        onMouseUp() {
            this.domElement.removeEventListener('mousemove', this._onMouseMove);
            this.domElement.removeEventListener('mouseup', this._onMouseUp);
            this.state = 'none';
        }

        onMouseWheel(event) {
            if (!this.enabled) return;

            event.preventDefault();
            event.stopPropagation();

            if (event.deltaY < 0) {
                this.scale *= this.zoomSpeed;
            } else if (event.deltaY > 0) {
                this.scale /= this.zoomSpeed;
            }
        }

        dispose() {
            this.domElement.removeEventListener('mousedown', this._onMouseDown);
            this.domElement.removeEventListener('wheel', this._onMouseWheel);
        }
    }

    return new OrbitControls(camera, domElement);
}

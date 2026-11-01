import * as THREE from 'three';
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls.js';

const EYE_HEIGHT = 1.5;
const WALK_SPEED = 1.6;
const SPRINT_SPEED = 3.2;
const BODY_RADIUS = 0.3;

/**
 * First-person roaming (WASD + mouse look) with wall collision against the invisible
 * collision mesh, a top-down plan view, a point/splat visibility toggle, a two-click
 * measuring tool and a shareable camera link.
 *
 * Touch screens have no pointer lock or keys: walking there keeps the drag-to-look of the
 * inside view and moves with an on-screen stick (#joystick), with the same collision.
 */
export class RoamController {
  constructor(viewer) {
    this.viewer = viewer;
    this.touch = window.matchMedia('(hover: none) and (pointer: coarse)').matches;
    this.fp = new PointerLockControls(viewer.camera, viewer.renderer.domElement);
    this.stick = { x: 0, y: 0 };  // joystick: x right, y forward, each -1..1
    this._bindStick();
    this.keys = new Set();
    this.walking = false;
    this.planView = false;
    this.measuring = false;
    this.measurePts = [];
    this.measureGroup = new THREE.Group();
    viewer.threeScene.add(this.measureGroup);
    this.savedPose = null;
    this.ray = new THREE.Raycaster();
    this.ray.firstHitOnly = true;
    this.onMeasure = () => {};

    this.fp.addEventListener('unlock', () => { this.walking = false; this._emitMode(); });
    window.addEventListener('keydown', (e) => this._key(e, true));
    window.addEventListener('keyup', (e) => this._key(e, false));
    viewer.renderer.domElement.addEventListener('click', (e) => this._click(e));
  }

  _emitMode() {
    window.dispatchEvent(new CustomEvent('roam-mode', {
      detail: { walking: this.walking, plan: this.planView, measuring: this.measuring },
    }));
  }

  _key(e, down) {
    if (e.target && ['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
    const k = e.key.toLowerCase();
    if (down) {
      this.keys.add(k);
      if (k === 'm') this.togglePlan();
      else if (k === 'p') this.toggleVisual();
      else if (k === 'u') this.toggleMeasure();
      else if (k === 'f') this.toggleWalk();
      else if (!this.walking && k === 'o') this.viewer.toggleLookAround();
      else if (!this.walking && k === '[') this.viewer.enterViewpoint((this.viewer.viewpointIndex ?? 0) - 1);
      else if (!this.walking && k === ']') this.viewer.enterViewpoint((this.viewer.viewpointIndex ?? 0) + 1);
    } else {
      this.keys.delete(k);
    }
  }

  floorY() { return 0; }

  _bindStick() {
    const pad = document.getElementById('joystick');
    const knob = document.getElementById('joy-knob');
    if (!pad || !knob) return;
    let id = null;
    const move = (e) => {
      const r = pad.getBoundingClientRect();
      const half = r.width / 2;
      let dx = (e.clientX - r.left - half) / half, dy = (e.clientY - r.top - half) / half;
      const len = Math.hypot(dx, dy);
      if (len > 1) { dx /= len; dy /= len; }
      this.stick.x = dx; this.stick.y = -dy;
      knob.style.transform = `translate(${dx * half * 0.6}px, ${dy * half * 0.6}px)`;
      this.viewer.invalidate();
    };
    const end = (e) => {
      if (e.pointerId !== id) return;
      id = null; this.stick.x = this.stick.y = 0;
      knob.style.transform = '';
    };
    pad.addEventListener('pointerdown', (e) => { id = e.pointerId; pad.setPointerCapture(id); move(e); e.preventDefault(); });
    pad.addEventListener('pointermove', (e) => { if (e.pointerId === id) move(e); });
    pad.addEventListener('pointerup', end);
    pad.addEventListener('pointercancel', end);
  }

  /** Leave walk mode, whichever way it was entered. */
  stopWalk() {
    if (!this.walking) return;
    if (!this.touch) { this.fp.unlock(); return; }
    this.walking = false;
    this.stick.x = this.stick.y = 0;
    document.body.classList.remove('touch-walk');
    this._emitMode();
  }

  toggleWalk() {
    if (this.walking) { this.stopWalk(); return; }
    if (this.planView) this.togglePlan();
    const v = this.viewer;
    if (this.touch) {
      v.controls.autoRotate = false;
      v.camera.position.y = this.floorY() + EYE_HEIGHT;
      v.setInside(true);  // drag keeps turning the view; the stick walks
      this.walking = true;
      document.body.classList.add('touch-walk');
      this._emitMode();
      return;
    }
    v.controls.enabled = false;
    v.controls.autoRotate = false;
    v.camera.position.y = this.floorY() + EYE_HEIGHT;
    this.fp.lock();
    this.walking = true;
    this._emitMode();
    this.fp.addEventListener('unlock', () => {
      v.controls.enabled = true;
      if (v.inside) {
        v.setInside(true); // keep turning in place where the walk ended
      } else {
        const f = new THREE.Vector3();
        v.camera.getWorldDirection(f);
        v.controls.target.copy(v.camera.position).addScaledVector(f, 2);
      }
    }, { once: true });
  }

  togglePlan() {
    const v = this.viewer;
    this.stopWalk();
    this.planView = !this.planView;
    if (this.planView) {
      this.savedPose = { pos: v.camera.position.clone(), target: v.controls.target.clone(), inside: !!v.inside };
      if (v.inside) v.setInside(false); // overview needs normal orbit + zoom
      const box = this._sceneBounds();
      const c = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      v.camera.position.set(c.x, box.max.y + Math.max(size.x, size.z) * 0.9, c.z + 0.001);
      v.controls.target.set(c.x, c.y, c.z);
    } else if (this.savedPose) {
      v.camera.position.copy(this.savedPose.pos);
      v.controls.target.copy(this.savedPose.target);
      if (this.savedPose.inside) v.setInside(true);
    }
    this._emitMode();
  }

  toggleVisual() {
    const v = this.viewer;
    if (v.pointCloud) v.pointCloud.visible = !v.pointCloud.visible;
    if (v.blueprint) v.blueprint.visible = !v.blueprint.visible;
    if (v.collisionMesh && !v.pointCloud && !v.splatViewer) v.collisionMesh.visible = !v.collisionMesh.visible;
  }

  toggleMeasure() {
    this.measuring = !this.measuring;
    if (!this.measuring) { this.clearMeasure(); }
    this._emitMode();
  }

  clearMeasure() {
    this.measurePts = [];
    while (this.measureGroup.children.length) {
      const c = this.measureGroup.children[0];
      this.measureGroup.remove(c);
      c.geometry?.dispose();
    }
  }

  _sceneBounds() {
    const box = new THREE.Box3();
    const v = this.viewer;
    if (v.pointCloud) { v.pointCloud.geometry.computeBoundingBox(); box.union(v.pointCloud.geometry.boundingBox); }
    if (v.collisionMesh) { v.collisionMesh.geometry.computeBoundingBox(); box.union(v.collisionMesh.geometry.boundingBox); }
    if (box.isEmpty()) box.set(new THREE.Vector3(-5, 0, -5), new THREE.Vector3(5, 3, 5));
    return box;
  }

  _pick(e) {
    const v = this.viewer;
    const rect = v.renderer.domElement.getBoundingClientRect();
    const m = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.ray.setFromCamera(m, v.camera);
    if (v.collisionMesh) {
      const h = this.ray.intersectObject(v.collisionMesh, true);
      if (h.length) return h[0].point;
    }
    // fall back to the floor plane Y=0
    const p = new THREE.Vector3();
    return this.ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), p) ? p : null;
  }

  _click(e) {
    if (!this.measuring || this.walking) return;
    const p = this._pick(e);
    if (!p) return;
    if (this.measurePts.length >= 2) this.clearMeasure();
    this.measurePts.push(p.clone());
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.05, 12, 12), new THREE.MeshBasicMaterial({ color: 0xf59e0b, depthTest: false }));
    dot.position.copy(p);
    dot.renderOrder = 10;
    this.measureGroup.add(dot);
    if (this.measurePts.length === 2) {
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(this.measurePts),
        new THREE.LineBasicMaterial({ color: 0xf59e0b, depthTest: false }));
      line.renderOrder = 10;
      this.measureGroup.add(line);
      this.onMeasure(this.measurePts[0].distanceTo(this.measurePts[1]));
    }
  }

  shareLink() {
    const c = this.viewer.camera.position;
    const u = new URL(window.location.href);
    u.searchParams.set('cam', [c.x, c.y, c.z].map((n) => n.toFixed(2)).join(','));
    return u.toString();
  }

  applyLinkPose() {
    const cam = new URLSearchParams(window.location.search).get('cam');
    if (!cam) return false;
    const [x, y, z] = cam.split(',').map(Number);
    if (![x, y, z].every(Number.isFinite)) return false;
    this.viewer.camera.position.set(x, y, z);
    return true;
  }

  _blocked(dir) {
    const mesh = this.viewer.collisionMesh;
    if (!mesh) return false;
    const origin = this.viewer.camera.position.clone();
    origin.y = this.floorY() + 0.5; // knee-to-waist height ray, ignores low floor patches
    this.ray.set(origin, dir);
    this.ray.far = BODY_RADIUS;
    const hit = this.ray.intersectObject(mesh, true);
    this.ray.far = Infinity;
    return hit.length > 0;
  }

  update(dt) {
    if (!this.walking) return;
    let speed = this.keys.has('shift') ? SPRINT_SPEED : WALK_SPEED;
    let fwd = (this.keys.has('w') || this.keys.has('arrowup') ? 1 : 0) - (this.keys.has('s') || this.keys.has('arrowdown') ? 1 : 0);
    let side = (this.keys.has('d') || this.keys.has('arrowright') ? 1 : 0) - (this.keys.has('a') || this.keys.has('arrowleft') ? 1 : 0);
    const push = Math.hypot(this.stick.x, this.stick.y);
    if (!fwd && !side && push > 0.15) {  // small dead zone so a resting thumb does not drift
      fwd = this.stick.y; side = this.stick.x;
      speed = WALK_SPEED * Math.min(1, push);
    }
    if (!fwd && !side) return;
    const cam = this.viewer.camera;
    const f = new THREE.Vector3();
    cam.getWorldDirection(f);
    f.y = 0;
    f.normalize();
    const r = new THREE.Vector3().crossVectors(f, new THREE.Vector3(0, 1, 0));
    const dir = f.multiplyScalar(fwd).add(r.multiplyScalar(side)).normalize();
    // slide: try the full move, then each axis separately
    const before = cam.position.clone();
    for (const d of [dir, new THREE.Vector3(dir.x, 0, 0).normalize(), new THREE.Vector3(0, 0, dir.z).normalize()]) {
      if (!Number.isFinite(d.x) || d.lengthSq() === 0) continue;
      if (!this._blocked(d)) { cam.position.addScaledVector(d, speed * dt); break; }
    }
    cam.position.y = this.floorY() + EYE_HEIGHT;
    // touch walking looks with the orbit controls: carry their look target along
    if (this.touch) this.viewer.controls.target.add(cam.position.clone().sub(before));
  }
}

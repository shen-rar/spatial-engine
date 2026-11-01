import * as THREE from 'three';
import { GridPlanner } from './GridPlanner.js';

/**
 * Field System HUD: live plan (minimap) built from the backend's per-cell map status,
 * the "real frame" reference panel, the enter-the-space gate and the status stamp.
 */
const COLORS = {
  0: [45, 58, 72],      // unseen
  1: [143, 211, 194],  // open
  2: [240, 113, 103],  // blocked
};
const CAPTURE = [240, 199, 94];

export class FieldHUD {
  constructor(viewer, apiBase, sceneId) {
    this.viewer = viewer;
    this.apiBase = apiBase;
    this.sceneId = sceneId;
    this.map = null;             // { nx, nz, origin, cell, status, capture }
    this.route = [];             // waypoints of the active route, [x, y, z]
    this.props = [];             // placed sandbox props, [{x, y, z}]
    this.onMapClick = () => {};

    this.canvas = document.getElementById('minimap');
    this.ctx = this.canvas.getContext('2d');
    this.base = document.createElement('canvas');
    this.canvas.addEventListener('click', (e) => this._mapClick(e));
    this.$ = (id) => document.getElementById(id);
  }

  async loadMap() {
    if (!this.viewer.navReady) return false;
    try {
      const r = await fetch(`${this.apiBase}/api/v1/scenes/${this.sceneId}/map`);
      if (!r.ok) return false;
      const j = await r.json();
      const bin = atob(j.status_b64);
      const status = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) status[i] = bin.charCodeAt(i);
      this.map = {
        nx: j.nx, nz: j.nz, origin: j.origin_xz, cell: j.cell_size_m, status,
        capture: j.capture_cells,
      };
      this.viewer.gridPlanner = new GridPlanner(this.map);
      this._paintBase();
      return true;
    } catch {
      return false;
    }
  }

  /** Map status at a world position: 0 unseen, 1 open, 2 blocked (1 if no map loaded). */
  statusAt(x, z) {
    if (!this.map) return 1;
    const m = this.map;
    const ix = Math.floor((x - m.origin[0]) / m.cell);
    const iz = Math.floor((z - m.origin[1]) / m.cell);
    if (ix < 0 || iz < 0 || ix >= m.nx || iz >= m.nz) return 0;
    return m.status[ix * m.nz + iz];
  }

  /** Precompute the static layer (status + capture path) once; overlays are drawn per frame. */
  _paintBase() {
    const m = this.map;
    this.base.width = m.nx;
    this.base.height = m.nz;
    const bctx = this.base.getContext('2d');
    const img = bctx.createImageData(m.nx, m.nz);
    for (let ix = 0; ix < m.nx; ix++) {
      for (let iz = 0; iz < m.nz; iz++) {
        const c = COLORS[m.status[ix * m.nz + iz]] || COLORS[0];
        const p = (iz * m.nx + ix) * 4;     // image rows are z, columns are x
        img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 255;
      }
    }
    bctx.putImageData(img, 0, 0);
    bctx.fillStyle = `rgb(${CAPTURE.join(',')})`;
    for (const [ix, iz] of m.capture) bctx.fillRect(ix, iz, 1, 1);
  }

  _worldToCanvas(x, z) {
    const m = this.map;
    const ix = (x - m.origin[0]) / m.cell;
    const iz = (z - m.origin[1]) / m.cell;
    const s = this.canvas.width / Math.max(m.nx, m.nz);
    const ox = (this.canvas.width - m.nx * s) / 2;
    const oy = (this.canvas.height - m.nz * s) / 2;
    return [ox + ix * s, oy + iz * s, s];
  }

  _canvasToWorld(px, py) {
    const m = this.map;
    const s = this.canvas.width / Math.max(m.nx, m.nz);
    const ox = (this.canvas.width - m.nx * s) / 2;
    const oy = (this.canvas.height - m.nz * s) / 2;
    const ix = (px - ox) / s;
    const iz = (py - oy) / s;
    return [m.origin[0] + ix * m.cell, m.origin[1] + iz * m.cell];
  }

  _mapClick(e) {
    if (!this.map) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * this.canvas.width;
    const py = ((e.clientY - rect.top) / rect.height) * this.canvas.height;
    const [x, z] = this._canvasToWorld(px, py);
    this.onMapClick(x, z);
  }

  /** Redraw the live plan: base map, route, placed props and camera. */
  draw() {
    const ctx = this.ctx;
    const W = this.canvas.width;
    ctx.fillStyle = '#0a0f14';
    ctx.fillRect(0, 0, W, W);
    if (!this.map) {
      ctx.fillStyle = '#8b9bab';
      ctx.font = '600 16px monospace';
      ctx.fillText('no overview yet', 20, 40);
      return;
    }
    const m = this.map;
    const [x0, y0, s] = this._worldToCanvas(m.origin[0], m.origin[1]);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.base, x0, y0, m.nx * s, m.nz * s);

    if (this.route.length > 1) {
      ctx.strokeStyle = '#ffffff';
      ctx.setLineDash([8, 5]);
      ctx.lineWidth = 3;
      ctx.beginPath();
      this.route.forEach(([x, , z], i) => {
        const [px, py] = this._worldToCanvas(x, z);
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const p of this.props) {
      const [px, py] = this._worldToCanvas(p.x, p.z);
      ctx.fillStyle = '#b79cff';
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fill();
    }
    this._drawYou(ctx, W);
  }

  /**
   * "You are here": the camera as a blue dot ringed in white and dark (readable on every map
   * colour) with a cone as wide as the view, pointing where the camera looks. Sizes are in
   * canvas pixels scaled to how large the map is shown, so the marker reads the same at any size.
   */
  _drawYou(ctx, W) {
    const cam = this.viewer.camera;
    const [cx, cy] = this._worldToCanvas(cam.position.x, cam.position.z);
    const k = W / Math.max(1, this.canvas.clientWidth || W);  // canvas px per CSS px
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    const heading = Math.atan2(dir.z, dir.x);
    const flat = Math.hypot(dir.x, dir.z) > 0.15;  // looking straight down (overview): no cone
    if (flat) {
      // horizontal field of view from the vertical one and the aspect ratio
      const half = Math.atan(Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2) * cam.aspect);
      const len = 34 * k;
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, len);
      g.addColorStop(0, 'rgba(37,99,235,0.85)');
      g.addColorStop(1, 'rgba(37,99,235,0.15)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, len, heading - half, heading + half);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';  // edges keep the cone readable on the coral cells
      ctx.lineWidth = 1 * k;
      ctx.stroke();
    }
    const r = 5 * k;
    ctx.beginPath();
    ctx.arc(cx, cy, r + 2.5 * k, 0, Math.PI * 2);
    ctx.fillStyle = '#0a0f14';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx, cy, r + 1.2 * k, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = '#2563eb';
    ctx.fill();
    if (flat) {  // a small white tick on the dot shows the heading even when the cone is faint
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.4 * k;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(heading) * r * 0.9, cy + Math.sin(heading) * r * 0.9);
      ctx.stroke();
    }
  }

  startLoop() {
    // redraw only when the 3D view drew a frame (it does whenever the camera, route or props change)
    let seen = -1;
    const tick = () => {
      const f = this.viewer.framesDrawn ?? 0;
      if (f !== seen) { seen = f; this.draw(); }
      requestAnimationFrame(tick);
    };
    tick();
  }

  setStamp(live, label) {
    this.$('status-stamp').innerHTML = live ? 'Live' : `Offline<span>${label}</span>`;
  }

  setSubtitle(text) { this.$('subtitle').textContent = text; }

  showReference() {
    const panel = this.$('ref-panel');
    const img = this.$('ref-img');
    img.onload = () => panel.classList.remove('empty');
    img.onerror = () => panel.classList.add('empty');
    img.src = `${this.apiBase}/scenes/${this.sceneId}/artifacts/reference.jpg`;
  }

  setupGate(roam) {
    const gate = this.$('center-hint');
    gate.onclick = () => roam.toggleWalk();
    window.addEventListener('roam-mode', (e) => gate.classList.toggle('hidden', e.detail.walking));
  }
}

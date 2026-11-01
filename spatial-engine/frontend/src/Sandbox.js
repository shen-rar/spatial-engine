import * as THREE from 'three';

/** Prop kinds offered in the sandbox palette. Dimensions are metres (scene units). */
const KINDS = {
  cone:    { build: () => coneMesh(), height: 0.7 },
  barrier: { build: () => boxMesh(1.2, 0.9, 0.25, 0xff4d32), height: 0.9 },
  crate:   { build: () => boxMesh(0.6, 0.6, 0.6, 0x8b6b4a), height: 0.6 },
  sign:    { build: () => signMesh(), height: 1.2 },
};
const KIND_BY_KEY = { 1: 'cone', 2: 'barrier', 3: 'crate', 4: 'sign' };
const STORE_KEY = (scene) => `spatial.sandbox.v1.${scene}`;

function boxMesh(w, h, d, color) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshStandardMaterial({ color, roughness: 0.7 }));
  m.position.y = h / 2;
  return m;
}
function coneMesh() {
  const m = new THREE.Mesh(new THREE.ConeGeometry(0.22, 0.7, 16), new THREE.MeshStandardMaterial({ color: 0xff6a1a, roughness: 0.5 }));
  m.position.y = 0.35;
  return m;
}
function signMesh() {
  const g = new THREE.Group();
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 1.0, 8), new THREE.MeshStandardMaterial({ color: 0x333333 }));
  post.position.y = 0.5;
  const face = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.4, 0.03), new THREE.MeshStandardMaterial({ color: 0xd7ff3f }));
  face.position.y = 1.0;
  g.add(post, face);
  return g;
}

/**
 * Sandbox: place labelled props on the floor. Placement is checked against the backend map
 * (blocked and unseen cells refuse), props persist per scene and can be shared as a link.
 */
export class Sandbox {
  constructor(viewer, hud, roam) {
    this.viewer = viewer;
    this.hud = hud;
    this.roam = roam;
    this.active = false;
    this.carry = null;          // { kind, rot, mesh }
    this.group = new THREE.Group();
    viewer.threeScene.add(this.group);
    this.props = [];            // { id, kind, x, y, z, rot, mesh }
    this.ray = new THREE.Raycaster();
    this.ray.firstHitOnly = true;
    this.nextId = 1;
    this.onRoute = () => {};

    const $ = (id) => document.getElementById(id);
    this.$ = $;
    this.panel = $('sandbox');
    document.querySelectorAll('#prop-palette button').forEach((b) =>
      b.addEventListener('click', () => this.pick(b.dataset.prop)));
    $('sandbox-close').onclick = () => this.toggle(false);
    $('sandbox-place').onclick = () => this.place();
    $('sandbox-rotate').onclick = () => this.rotate();
    $('sandbox-cancel').onclick = () => this.cancel();
    $('sandbox-route').onclick = () => this.routeToEnd();
    $('sandbox-clear').onclick = () => this.clear();
    $('sandbox-share').onclick = () => this.copyLink();
    $('btn-sandbox').onclick = () => this.toggle();

    viewer.renderer.domElement.addEventListener('click', (e) => this._click(e));
    window.addEventListener('keydown', (e) => this._key(e));
  }

  /* ---------- lifecycle ---------- */

  toggle(force) {
    this.active = force ?? !this.active;
    this.panel.classList.toggle('hidden', !this.active);
    this.$('btn-sandbox').classList.toggle('on', this.active);
    this.$('toolbar').classList.toggle('hidden', this.active); // sandbox replaces the toolbar
    this.$('reticle').classList.toggle('hidden', !this.active);
    if (!this.active) this.cancel();
    if (this.active && this.roam.walking) this.roam.toggleWalk();
    this._refreshCount();
  }

  load(layoutJson) {
    let items = [];
    try {
      items = JSON.parse(layoutJson) || [];
    } catch { items = []; }
    for (const it of items) if (KINDS[it.kind]) this._spawn(it.kind, it.x, it.z, it.rot || 0);
    this._refreshCount();
  }

  /** Load from the URL (shared link) first, then this browser's saved layout. */
  restore() {
    const q = new URLSearchParams(window.location.search).get('layout');
    if (q) {
      try { this.load(atob(q.replace(/-/g, '+').replace(/_/g, '/'))); } catch { /* bad link */ }
    } else {
      try { this.load(localStorage.getItem(STORE_KEY(this.hud.sceneId)) || '[]'); } catch { /* storage off */ }
    }
    this.viewer.invalidate();
  }

  _save() {
    const data = JSON.stringify(this.props.map(({ kind, x, z, rot }) => ({ kind, x, z, rot })));
    try { localStorage.setItem(STORE_KEY(this.hud.sceneId), data); } catch { /* storage off */ }
  }

  /* ---------- props ---------- */

  _spawn(kind, x, z, rot) {
    const mesh = KINDS[kind].build();
    const holder = new THREE.Group();
    holder.add(mesh);
    holder.position.set(x, 0, z);
    holder.rotation.y = rot;
    this.group.add(holder);
    const prop = { id: this.nextId++, kind, x, y: 0, z, rot, mesh: holder };
    this.props.push(prop);
    this.hud.props = this.props;
    return prop;
  }

  _refreshCount() {
    this.$('sandbox-count').textContent = `${this.props.length} placed`;
    this.$('legend-prop').style.display = this.props.length ? 'inline-flex' : 'none';
  }

  pick(kind) {
    if (!this.active) this.toggle(true);
    this.carry?.mesh && this.group.remove(this.carry.mesh);
    const holder = new THREE.Group();
    holder.add(KINDS[kind].build());
    holder.traverse((o) => { if (o.material) { o.material = o.material.clone(); o.material.transparent = true; o.material.opacity = 0.55; } });
    this.group.add(holder);
    this.carry = { kind, rot: 0, mesh: holder };
    document.querySelectorAll('#prop-palette button').forEach((b) => b.classList.toggle('on', b.dataset.prop === kind));
    this.$('sandbox-carry-actions').classList.remove('hidden');
    this._hint(`Placing ${kind}: click the floor, R rotates, ESC cancels`);
  }

  rotate() {
    if (!this.carry) return;
    this.carry.rot = (this.carry.rot + Math.PI / 2) % (Math.PI * 2);
  }

  cancel() {
    if (this.carry) this.group.remove(this.carry.mesh);
    this.carry = null;
    document.querySelectorAll('#prop-palette button').forEach((b) => b.classList.remove('on'));
    this.$('sandbox-carry-actions').classList.add('hidden');
    this._hint('');
  }

  /** The point under the screen centre (reticle), on the collision mesh or the floor. */
  _aimPoint() {
    const v = this.viewer;
    this.ray.setFromCamera(new THREE.Vector2(0, 0), v.camera);
    if (v.collisionMesh) {
      const h = this.ray.intersectObject(v.collisionMesh, true);
      if (h.length) return h[0].point;
    }
    const p = new THREE.Vector3();
    return this.ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), p) ? p : null;
  }

  _place(x, z) {
    const status = this.hud.statusAt(x, z);
    if (status !== 1) {
      const reason = status === 2 ? 'blocked by the map' : 'unseen: no capture coverage here';
      this._toast(`Can't place: ${reason}`);
      return false;
    }
    this._spawn(this.carry.kind, x, z, this.carry.rot);
    this._save();
    this._refreshCount();
    this._toast(`${this.carry.kind} placed`);
    return true;
  }

  place() {
    if (!this.carry) return;
    const p = this._aimPoint();
    if (!p) return this._toast("Can't place: aim at the floor");
    if (this._place(p.x, p.z)) this.cancel();
  }

  /** Remove the placed prop the reticle is on (within 1.2 m of the aim point). */
  removeAimed() {
    const p = this._aimPoint();
    if (!p) return;
    let best = null, bestD = 1.2;
    for (const prop of this.props) {
      const d = Math.hypot(prop.x - p.x, prop.z - p.z);
      if (d < bestD) { bestD = d; best = prop; }
    }
    if (!best) return this._toast('Aim at a placed prop to remove it');
    this.group.remove(best.mesh);
    this.props = this.props.filter((q) => q !== best);
    this.hud.props = this.props;
    this._save();
    this._refreshCount();
  }

  clear() {
    for (const p of this.props) this.group.remove(p.mesh);
    this.props = [];
    this.hud.props = [];
    this.hud.route = [];
    this.viewer.clearPath();
    this._save();
    this._refreshCount();
    this._toast('Sandbox cleared');
  }

  async routeToEnd() {
    const end = this.props.length ? this.props[this.props.length - 1] : null;
    if (!end) return this._toast('Place a prop first: its position is the route end');
    await this.routeTo(end.x, 0, end.z);
  }

  async routeTo(x, y, z) {
    const path = await this.viewer.navigateToPoint(new THREE.Vector3(x, y, z));
    if (!path) return this._toast('route: no path');
    this.hud.route = path.waypoints_3d;
    this.$('legend-route').style.display = 'inline-flex';
    this.onRoute(path);
    return path;
  }

  /* ---------- sharing ---------- */

  layoutLink() {
    const data = JSON.stringify(this.props.map(({ kind, x, z, rot }) => ({ kind, x: +x.toFixed(2), z: +z.toFixed(2), rot: +rot.toFixed(3) })));
    const b64 = btoa(data).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    const u = new URL(window.location.href);
    u.searchParams.set('layout', b64);
    return u.toString();
  }

  async copyLink() {
    const url = this.layoutLink();
    try {
      await navigator.clipboard.writeText(url);
      this._toast('Layout link copied');
    } catch {
      window.prompt('Copy this layout link', url);
    }
  }

  /* ---------- input ---------- */

  _click(e) {
    if (!this.active || !this.carry) return;
    this.place();
  }

  _key(e) {
    if (e.target && ['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
    const k = e.key.toLowerCase();
    if (k === 'g') return this.toggle();
    if (!this.active) return;
    if (KIND_BY_KEY[e.key]) return this.pick(KIND_BY_KEY[e.key]);
    if (k === 'r') return this.rotate();
    if (k === 't') return this.routeToEnd();
    if (k === 'escape') return this.cancel();
    if (k === 'delete' || k === 'backspace') return this.removeAimed();
  }

  /* ---------- feedback ---------- */

  _hint(text) {
    const el = this.$('carry-hint');
    el.textContent = text;
    el.classList.toggle('hidden', !text);
  }

  _toast(text) {
    const el = this.$('toast');
    el.textContent = text;
    el.style.display = 'block';
    clearTimeout(this._t);
    this._t = setTimeout(() => { el.style.display = 'none'; }, 1800);
  }

  /** Per-frame: move the translucent ghost to the reticle while carrying a prop. */
  update() {
    if (!this.carry) return;
    const p = this._aimPoint();
    if (!p) return;
    const ok = this.hud.statusAt(p.x, p.z) === 1;
    this.carry.mesh.position.set(p.x, 0, p.z);
    this.carry.mesh.rotation.y = this.carry.rot;
    this.carry.mesh.traverse((o) => { if (o.material) o.material.color.set(ok ? 0xffffff : 0xff4d32); });
  }
}

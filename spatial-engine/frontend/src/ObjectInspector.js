import * as THREE from 'three';

const PICK_COLOUR = 0x7ff3ff;   // selected piece
const HOVER_COLOUR = 0xffe7b0;  // piece under the cursor
const BASE_COLOUR = 0xffd27a;   // furniture amber (as drawn by the viewer)

// how "Find in scene" groups a blueprint's pieces
const GROUPS = [
  ['Furniture', ['sofa', 'sectional', 'armchair', 'coffee_table', 'side_table', 'dining_set', 'round_dining', 'pool_table',
    'stool', 'outdoor_sofa', 'outdoor_armchair', 'ottoman', 'bench', 'piano', 'staircase']],
  ['Lighting', ['chandelier', 'pendant_light', 'floor_lamp', 'table_lamp']],
  ['Décor', ['vase', 'decor', 'sculpture', 'cushion', 'plant', 'rug']],
  ['Walls and windows', ['wall_art', 'mirror', 'curtain', 'fireplace', 'tv']],
  ['Kitchen, storage and appliances', ['kitchen_island', 'counter', 'bar_counter', 'sideboard', 'shelf', 'refrigerator']],
];

/**
 * Blueprint object inspector: click a furniture piece to highlight it and open a card with
 * its photo from the capture, its name and its dimensions. The orbit centres on the piece, so
 * dragging (or "Orbit 360°") turns around it.
 */
export class ObjectInspector {
  constructor(viewer, apiBase) {
    this.viewer = viewer;
    this.apiBase = apiBase;
    this.ray = new THREE.Raycaster();
    this.selected = null;
    this.hovered = null;
    this.$ = (id) => document.getElementById(id);
    const canvas = viewer.renderer.domElement;
    let down = null;
    canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
    canvas.addEventListener('pointerup', (e) => {
      // a click, not the end of a drag that turned the view
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5 || performance.now() - down.t > 500) return;
      if (this._busy()) return;
      this.select(this._pick(e));
    });
    canvas.addEventListener('pointermove', (e) => {
      if (e.buttons || this._busy()) return;
      this._hover(this._pick(e));
      const tip = this.$('obj-tip');
      tip.style.left = `${e.clientX + 14}px`;
      tip.style.top = `${e.clientY + 14}px`;
    });
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.selected) this.select(null);
    });
    this.$('obj-close').onclick = () => this.select(null);
    this.$('obj-orbit').onclick = () => this.toggleOrbit();
    this._buildList();
  }

  /**
   * "Find in scene" for a blueprint: every piece, grouped and alphabetical, filtered as you
   * type. Picking a row does what clicking the piece does; hovering a row highlights it.
   */
  _buildList() {
    const list = this.$('obj-list');
    const input = this.$('search-input');
    if (!list || !input) return;
    const groupOf = (t) => (GROUPS.find(([, types]) => types.includes(t)) ?? ['Other'])[0];
    const rows = this.objects.map((o) => {
      const row = document.createElement('button');
      row.className = 'obj-row';
      row.type = 'button';
      const w = o.info.dims?.width;
      row.innerHTML = `<span class="n"></span><span class="m"></span>`;
      row.querySelector('.n').textContent = o.info.name;
      const kind = (o.info.type ?? 'object').replace(/_/g, ' ');
      row.querySelector('.m').textContent = `${kind[0].toUpperCase()}${kind.slice(1)}${w ? ` · ${w.toFixed(2)} m` : ''}`;
      row.onclick = () => this.select(o);
      row.onmouseenter = () => this._hover(o, false);
      row.onmouseleave = () => this._hover(null, false);
      o.row = row;
      return { o, row, group: groupOf(o.info.type), key: `${o.info.name} ${o.info.type} ${o.info.id}`.toLowerCase().replace(/_/g, ' ') };
    });
    const sections = [...GROUPS.map(([g]) => g), 'Other'].map((g) => {
      const items = rows.filter((r) => r.group === g).sort((a, b) => a.o.info.name.localeCompare(b.o.info.name));
      if (!items.length) return null;
      const head = document.createElement('div');
      head.className = 'obj-group';
      head.textContent = g;
      return { head, items };
    }).filter(Boolean);
    const body = this.$('obj-list-items');
    body.replaceChildren(...sections.flatMap((s) => [s.head, ...s.items.map((r) => r.row)]));
    const count = this.$('obj-count');
    const filter = () => {
      const words = input.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
      let shown = 0;
      for (const s of sections) {
        let any = false;
        for (const r of s.items) {
          const ok = words.every((w) => r.key.includes(w));
          r.row.hidden = !ok;
          any ||= ok;
          shown += ok;
        }
        s.head.hidden = !any;
      }
      count.textContent = words.length ? `${shown} of ${rows.length} objects` : `${rows.length} objects`;
      this._first = sections.flatMap((s) => s.items).find((r) => !r.row.hidden)?.o ?? null;
    };
    input.placeholder = 'Filter objects, e.g. lamp, sofa, vase';
    input.addEventListener('input', filter);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && this._first) this.select(this._first);
      if (e.key === 'Escape') { input.value = ''; filter(); input.blur(); }
    });
    this.$('search-btn').onclick = () => { if (this._first) this.select(this._first); };
    list.hidden = false;
    filter();
  }

  _markList(o) {
    for (const x of this.objects) x.row?.classList.toggle('on', x === o);
    o?.row?.scrollIntoView({ block: 'nearest' });
  }

  get objects() { return this.viewer.blueprintObjects ?? []; }

  /** Walking, measuring and the prop tray use clicks themselves. */
  _busy() {
    const v = this.viewer;
    return v.roam.walking || v.roam.measuring || v.sandbox?.active;
  }

  _pick(e) {
    if (!this.objects.length) return null;
    const rect = this.viewer.renderer.domElement.getBoundingClientRect();
    const m = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.ray.setFromCamera(m, this.viewer.camera);
    const hits = this.ray.intersectObjects(this.objects.map((o) => o.box), false);
    return hits.length ? this.objects.find((o) => o.box === hits[0].object) : null;
  }

  _paint(o) {
    if (!o) return;
    o.lines.material.color.setHex(o === this.selected ? PICK_COLOUR : o === this.hovered ? HOVER_COLOUR : BASE_COLOUR);
    o.box.visible = o === this.selected;
  }

  _hover(o, tip = true) {
    if (o === this.hovered) return;
    const prev = this.hovered;
    this.hovered = o;
    this._paint(prev);
    this._paint(o);
    this.viewer.renderer.domElement.style.cursor = o ? 'pointer' : '';
    const t = this.$('obj-tip');
    t.textContent = o ? o.info.name : '';
    t.classList.toggle('hidden', !tip || !o || o === this.selected);
    this.viewer.invalidate(2);
  }

  select(o) {
    const prev = this.selected;
    this.selected = o;
    this._paint(prev);
    this._paint(o);
    this._markList(o);
    this.$('obj-tip').classList.add('hidden');
    document.body.classList.toggle('inspecting', !!o);
    this.$('obj-card').classList.toggle('hidden', !o);
    if (!o) {
      this.viewer.controls.autoRotate = false;
      this.$('obj-orbit').classList.remove('on');
      this.viewer.invalidate();
      return;
    }
    // phones: the list sheet makes way for the card
    document.body.classList.remove('find-open');
    this.$('btn-find')?.classList.remove('on');
    this._fillCard(o);
    this._focus(o);
  }

  _fillCard(o) {
    const { info } = o;
    const img = this.$('obj-img');
    const art = `${this.apiBase}/scenes/${this.viewer.sceneId}/artifacts`;
    this.$('obj-photo').classList.toggle('empty', !info.photo);
    if (info.photo) img.src = `${art}/${info.photo}`;
    img.alt = info.name;
    this.$('obj-name').textContent = info.name;
    this.$('obj-type').textContent = (info.type ?? 'object').replace(/_/g, ' ');
    // calibrated scenes convert scene units to metres; the build is metric already (factor 1)
    const k = this.viewer.calibrated ? this.viewer.scaleFactor : 1;
    const rows = [['width', 'Width'], ['depth', 'Depth'], ['height', 'Height'], ['mounted_at', 'Bottom edge at'],
      ['above_floor', 'Above floor']];
    this.$('obj-dims').innerHTML = rows.filter(([key]) => info.dims?.[key] != null).map(([key, label]) => {
      const m = info.dims[key] * k;
      const typical = !info.measured?.includes(key);
      const ft = m / 0.3048, inch = Math.round((ft % 1) * 12);
      return `<div class="dim${typical ? ' typical' : ''}"><span>${label}</span><b>${m.toFixed(2)} m</b>`
        + `<em>${Math.floor(ft)}′ ${inch}″${typical ? ' · typical' : ''}</em></div>`;
    }).join('');
  }

  /** Glide the orbit centre onto the piece and frame it, keeping the current viewing direction. */
  _focus(o) {
    const v = this.viewer;
    if (v.roam.planView) v.roam.togglePlan();
    if (v.inside) v.setInside(false);
    const target = o.center.clone();
    const dir = v.camera.position.clone().sub(v.controls.target);
    if (dir.lengthSq() < 1e-6) dir.set(1, 1, 1);
    dir.normalize();
    if (dir.y < 0.35) { dir.y = 0.35; dir.normalize(); }  // look down onto it a little
    const dist = Math.max(2.2, o.radius * 2.6);
    const toPos = target.clone().addScaledVector(dir, dist);
    const fromPos = v.camera.position.clone(), fromTarget = v.controls.target.clone();
    const t0 = performance.now(), dur = 650;
    const step = () => {
      const t = Math.min(1, (performance.now() - t0) / dur), e = t * t * (3 - 2 * t);
      v.camera.position.lerpVectors(fromPos, toPos, e);
      v.controls.target.lerpVectors(fromTarget, target, e);
      v.invalidate(2);
      if (t < 1 && this.selected === o) requestAnimationFrame(step);
    };
    step();
  }

  /** Slow turn around the selected piece (one turn every ~15 s). */
  toggleOrbit() {
    const c = this.viewer.controls;
    c.autoRotate = !c.autoRotate;
    c.autoRotateSpeed = 4;
    this.$('obj-orbit').classList.toggle('on', c.autoRotate);
    this.viewer.invalidate();
  }
}

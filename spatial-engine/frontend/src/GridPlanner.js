/**
 * In-browser route planning on the floor-plan grid, for static hosting where there is no
 * backend to run the Theta* planner. Same grid as the minimap (status 1 = walkable with the
 * agent radius already applied), A* over 8 neighbours, then line-of-sight shortcuts so the
 * route is a few straight legs instead of a staircase.
 *
 * Routes keep to the filmed part of the house: the capture walk itself (someone walked it, so
 * it is passable even where splat clutter marks the grid blocked) and open cells near it. The
 * grid's floor reaches past the walls, so unrestricted routes would detour around the outside.
 */
const EYE_HEIGHT_M = 1.5;  // the backend planner's camera_eye_height_m
const CORRIDOR_M = 0.35;   // half-width of the always-passable lane along the capture walk
const NEAR_M = 1.5;        // open cells this close to the walk are usable too
const MAX_STEP_M = 6.0;    // a longer jump between capture spots is a bad merge, not a walk

export class GridPlanner {
  constructor(map) {
    this.m = map;  // { nx, nz, origin: [x, z], cell, status: Uint8Array (ix outer, iz inner), capture }
    this.ok = this._usable();
  }

  /** 1 where routes may go: the capture lane, plus open cells within NEAR_M of it. */
  _usable() {
    const m = this.m, n = m.nx * m.nz;
    const pts = m.capture ?? [];
    if (pts.length < 2) return Uint8Array.from(m.status, (s) => (s === 1 ? 1 : 0));
    const dist = new Int16Array(n).fill(-1);
    const queue = new Int32Array(n);
    let tail = 0;
    const r = Math.ceil(CORRIDOR_M / m.cell);
    const seed = (ix, iz) => {
      for (let dx = -r; dx <= r; dx++) {
        for (let dz = -r; dz <= r; dz++) {
          const jx = ix + dx, jz = iz + dz;
          if (jx < 0 || jz < 0 || jx >= m.nx || jz >= m.nz || dx * dx + dz * dz > r * r) continue;
          const j = jx * m.nz + jz;
          if (dist[j] < 0) { dist[j] = 0; queue[tail++] = j; }
        }
      }
    };
    for (let k = 1; k < pts.length; k++) {
      const [ax, az] = pts[k - 1], [bx, bz] = pts[k];
      const len = Math.hypot(bx - ax, bz - az);
      if (len * m.cell > MAX_STEP_M) continue;
      const steps = Math.max(1, Math.ceil(len));
      for (let t = 0; t <= steps; t++) seed(Math.round(ax + (bx - ax) * t / steps), Math.round(az + (bz - az) * t / steps));
    }
    const ok = new Uint8Array(n);
    const maxD = Math.ceil(NEAR_M / m.cell);
    for (let head = 0; head < tail; head++) {  // breadth-first outwards from the lane
      const i = queue[head];
      ok[i] = dist[i] === 0 || m.status[i] === 1 ? 1 : 0;
      if (dist[i] >= maxD) continue;
      const ix = Math.floor(i / m.nz), iz = i % m.nz;
      for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const jx = ix + dx, jz = iz + dz;
        if (jx < 0 || jz < 0 || jx >= m.nx || jz >= m.nz) continue;
        const j = jx * m.nz + jz;
        if (dist[j] < 0 && m.status[j] === 1) { dist[j] = dist[i] + 1; queue[tail++] = j; }
      }
    }
    return ok;
  }

  _walkable(ix, iz) {
    const m = this.m;
    return ix >= 0 && iz >= 0 && ix < m.nx && iz < m.nz && this.ok[ix * m.nz + iz] === 1;
  }

  _cell(x, z) {
    const m = this.m;
    return [Math.floor((x - m.origin[0]) / m.cell), Math.floor((z - m.origin[1]) / m.cell)];
  }

  /** Nearest walkable cell within ~2 m (ring search), or null. */
  _snap(ix, iz) {
    if (this._walkable(ix, iz)) return [ix, iz];
    const maxR = Math.ceil(2.0 / this.m.cell);
    for (let r = 1; r <= maxR; r++) {
      let best = null, bestD = Infinity;
      for (let dx = -r; dx <= r; dx++) {
        for (let dz = -r; dz <= r; dz++) {
          if (Math.max(Math.abs(dx), Math.abs(dz)) !== r || !this._walkable(ix + dx, iz + dz)) continue;
          const d = dx * dx + dz * dz;
          if (d < bestD) { bestD = d; best = [ix + dx, iz + dz]; }
        }
      }
      if (best) return best;
    }
    return null;
  }

  _lineOfSight(a, b) {
    let [x0, z0] = a;
    const [x1, z1] = b;
    const dx = Math.abs(x1 - x0), dz = Math.abs(z1 - z0);
    const sx = x0 < x1 ? 1 : -1, sz = z0 < z1 ? 1 : -1;
    let err = dx - dz;
    for (;;) {
      if (!this._walkable(x0, z0)) return false;
      if (x0 === x1 && z0 === z1) return true;
      const e2 = 2 * err;
      if (e2 > -dz) { err -= dz; x0 += sx; }
      if (e2 < dx) { err += dx; z0 += sz; }
    }
  }

  /** Same shape as the backend's plan-path response, or null when no route exists. */
  plan(startXYZ, goalXYZ) {
    const m = this.m;
    const s = this._snap(...this._cell(startXYZ[0], startXYZ[2]));
    const g = this._snap(...this._cell(goalXYZ[0], goalXYZ[2]));
    if (!s || !g) return null;
    const n = m.nx * m.nz, idx = (ix, iz) => ix * m.nz + iz;
    const gScore = new Float32Array(n).fill(Infinity);
    const came = new Int32Array(n).fill(-1);
    const closed = new Uint8Array(n);
    const heap = [];  // binary min-heap of [f, index]
    const push = (f, i) => {
      heap.push([f, i]);
      for (let k = heap.length - 1; k > 0;) {
        const p = (k - 1) >> 1;
        if (heap[p][0] <= heap[k][0]) break;
        [heap[p], heap[k]] = [heap[k], heap[p]]; k = p;
      }
    };
    const pop = () => {
      const top = heap[0], last = heap.pop();
      if (heap.length) {
        heap[0] = last;
        for (let k = 0; ;) {
          const l = 2 * k + 1, r = l + 1;
          let c = k;
          if (l < heap.length && heap[l][0] < heap[c][0]) c = l;
          if (r < heap.length && heap[r][0] < heap[c][0]) c = r;
          if (c === k) break;
          [heap[c], heap[k]] = [heap[k], heap[c]]; k = c;
        }
      }
      return top;
    };
    const h = (ix, iz) => {  // octile distance
      const dx = Math.abs(ix - g[0]), dz = Math.abs(iz - g[1]);
      return Math.max(dx, dz) + (Math.SQRT2 - 1) * Math.min(dx, dz);
    };
    const si = idx(...s), gi = idx(...g);
    gScore[si] = 0;
    push(h(...s), si);
    while (heap.length) {
      const [, i] = pop();
      if (closed[i]) continue;
      if (i === gi) break;
      closed[i] = 1;
      const ix = Math.floor(i / m.nz), iz = i % m.nz;
      for (let dx = -1; dx <= 1; dx++) {
        for (let dz = -1; dz <= 1; dz++) {
          if (!dx && !dz) continue;
          const jx = ix + dx, jz = iz + dz;
          if (!this._walkable(jx, jz)) continue;
          if (dx && dz && !(this._walkable(ix + dx, iz) && this._walkable(ix, iz + dz))) continue;  // no corner cutting
          const j = idx(jx, jz), t = gScore[i] + (dx && dz ? Math.SQRT2 : 1);
          if (t < gScore[j]) { gScore[j] = t; came[j] = i; push(t + h(jx, jz), j); }
        }
      }
    }
    if (si !== gi && came[gi] < 0) return null;

    const cells = [];
    for (let i = gi; i >= 0; i = i === si ? -1 : came[i]) cells.push([Math.floor(i / m.nz), i % m.nz]);
    cells.reverse();
    const legs = [cells[0]];  // keep a corner only when the straight line past it is blocked
    for (let k = 1; k < cells.length - 1; k++) {
      if (!this._lineOfSight(legs[legs.length - 1], cells[k + 1])) legs.push(cells[k]);
    }
    if (cells.length > 1) legs.push(cells[cells.length - 1]);

    const toWorld = ([ix, iz]) => [m.origin[0] + (ix + 0.5) * m.cell, EYE_HEIGHT_M, m.origin[1] + (iz + 0.5) * m.cell];
    const waypoints = legs.map(toWorld);
    waypoints[0] = [startXYZ[0], EYE_HEIGHT_M, startXYZ[2]];
    let total = 0;
    for (let k = 1; k < waypoints.length; k++) {
      total += Math.hypot(waypoints[k][0] - waypoints[k - 1][0], waypoints[k][2] - waypoints[k - 1][2]);
    }
    return { waypoints_3d: waypoints, total_length_m: total };
  }
}

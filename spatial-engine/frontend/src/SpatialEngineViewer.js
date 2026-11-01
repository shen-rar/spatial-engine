import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { RoamController } from './RoamController.js';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

// Patch Three.js BufferGeometry and Mesh prototypes for O(log N) BVH raycasting
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

export class SpatialEngineViewer {
  /**
   * sceneInfo is this scene's entry from /api/v1/scenes (style and which layer files exist);
   * without it (static hosting) the viewer probes for the files itself.
   */
  constructor(containerDOM, apiBaseUrl, sceneId, sceneInfo = null) {
    this.container = containerDOM;
    this.apiBaseUrl = apiBaseUrl;
    this.sceneId = sceneId;
    this.sceneInfo = sceneInfo;

    this.threeScene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(
      65, window.innerWidth / window.innerHeight, 0.05, 150.0
    );
    this.camera.position.set(0, 1.5, 0);

    // No MSAA: splats are fill-rate bound and gain nothing from it; resolution adapts instead.
    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.maxPixelRatio = Math.min(window.devicePixelRatio, 2);
    this.minPixelRatio = 0.5;
    this.pixelRatio = this.maxPixelRatio;
    this.renderer.setPixelRatio(this.pixelRatio);
    this.container.appendChild(this.renderer.domElement);
    this.threeScene.background = new THREE.Color(0x0a0f1a);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.pointCloud = null;
    this.roam = new RoamController(this);
    this.sandbox = null;

    this.raycaster = new THREE.Raycaster();
    this.raycaster.firstHitOnly = true;

    this.collisionMesh = null;
    this.splatViewer = null;
    this.activePathMesh = null;
    this.poiMarkersGroup = new THREE.Group();
    this.threeScene.add(this.poiMarkersGroup);

    this.flightCurve = null;
    this.flightProgress = 0.0;
    this.flightSpeedMetersPerSec = 1.65;
    this.flightTotalLength = 1.0;
    this.isFlying = false;
    this.clock = new THREE.Clock();

    // On-demand rendering: draw while something changes, then idle the GPU
    this._redrawFrames = 0;
    this._lastView = new THREE.Matrix4();
    this._frameTimes = [];

    this._bindEvents();
  }

  /** The GPU WebGL runs on, as the browser reports it ('' if it hides it). */
  gpuName() {
    const gl = this.renderer.getContext();
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return String((ext && gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) || gl.getParameter(gl.RENDERER) || '');
  }

  /** Keep drawing for the next ~1.5 s (splat sorting finishes in the background meanwhile). */
  invalidate(frames = 90) {
    this._redrawFrames = Math.max(this._redrawFrames, frames);
  }

  _setPixelRatio(pr) {
    if (Math.abs(pr - this.pixelRatio) < 0.01) return;
    this.pixelRatio = pr;
    this.renderer.setPixelRatio(pr);
    // the splat viewer reads the ratio once at start; keep its viewport and focal in step
    const sv = this.splatViewer?.viewer;
    if (sv) {
      sv.devicePixelRatio = pr;
      if (sv.splatMesh) sv.splatMesh.devicePixelRatio = pr;
    }
  }

  /**
   * Dynamic resolution: while the view moves, trade sharpness for a steady ~50+ fps (integrated
   * GPUs cannot fill 200k splats at full resolution); once it stops, redraw at full resolution.
   */
  _adaptResolution(dt, moving) {
    // splats beyond 1.5x device pixels cost a lot of fill for detail the eye barely sees
    const top = this.splatViewer ? Math.min(this.maxPixelRatio, 1.5) : this.maxPixelRatio;
    if (!moving) {
      this._frameTimes.length = 0;
      if (this._idleFor > 0.35 && this.pixelRatio !== top) {
        this._setPixelRatio(top);
        this._redrawFrames = Math.min(this._redrawFrames, 4);  // a few sharp frames, then rest
      }
      return;
    }
    // learned across movements, so each one starts at a smooth size; phone GPUs start low and climb
    this._motionRatio ??= this.roam.touch ? Math.min(top, 1) : top;
    this._setPixelRatio(this._motionRatio);
    if (dt > 0.5) return;  // tab switch or load hitch, not a rendering cost
    const target = 1 / 50;
    this._frameTimes.push(dt);
    // one very slow frame (walking up to a wall fills the screen with splats) acts at once;
    // otherwise average a few frames so vsync jitter does not make the resolution hunt
    if (dt < target * 2.5 && this._frameTimes.length < 6) return;
    const avg = this._frameTimes.reduce((a, b) => a + b, 0) / this._frameTimes.length;
    this._frameTimes.length = 0;
    if (avg > target * 1.15) {
      // fill cost scales with pixel count, i.e. with the ratio squared
      this._motionRatio = Math.max(this.minPixelRatio, this._motionRatio * Math.max(0.6, Math.sqrt(target / avg)));
    } else if (avg < 1 / 55) {  // vsync caps most screens at 60 fps, so "fast" means near 16.7 ms
      this._motionRatio = Math.min(top, this._motionRatio * 1.06);
    }
  }

  async loadDigitalTwin() {
    const art = `${this.apiBaseUrl}/scenes/${this.sceneId}/artifacts`;
    const info = this.sceneInfo;
    const has = (key, file) => (info?.files ? Promise.resolve(!!info.files[key]) : this._exists(`${art}/${file}`));
    const meshUrl = `${art}/collision_mesh.glb`;
    const pointsUrl = `${art}/points.ply`;
    this._startRenderLoop();  // draw from the first byte: layers appear as they arrive

    // Everything starts at once. The navigation grid is optional and slowest, so it is awaited last.
    const navP = fetch(`${this.apiBaseUrl}/api/v1/scenes/${this.sceneId}/initialize`, { method: 'POST' })
      .then((r) => r.ok).catch(() => false)
      // static hosting (no backend): an exported map file still gives the plan and routes
      .then((ok) => ok || fetch(`${this.apiBaseUrl}/api/v1/scenes/${this.sceneId}/map`, { method: 'HEAD' })
        .then((r) => r.ok).catch(() => false));
    const meshP = has('mesh', 'collision_mesh.glb').then((ok) => (ok ? new GLTFLoader().loadAsync(meshUrl) : null))
      .catch((err) => { console.warn('collision mesh unavailable:', err.message); return null; });
    const camsP = fetch(`${art}/cameras.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
    const styleP = info?.style ? Promise.resolve(info.style)
      : fetch(`${this.apiBaseUrl}/scenes/${this.sceneId}/scene.json`)
        .then((r) => (r.ok ? r.json() : {})).then((j) => j.style ?? null).catch(() => null);
    const [style, hasKsplat, hasPly, hasPoints] = await Promise.all([
      styleP, has('ksplat', 'visual_splats.ksplat'), has('ply', 'visual_splats.ply'), has('points', 'points.ply')]);
    this.style = style;

    this.threeScene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dl = new THREE.DirectionalLight(0xffffff, 0.8);
    dl.position.set(5, 10, 5);
    this.threeScene.add(dl);

    // Scene style: a blueprint scene draws only lines, never the photoreal layers
    let blueprintLoaded = false;
    if (style === 'blueprint') {
      try {
        const r = await fetch(`${art}/blueprint.json`);
        if (r.ok) { this._addBlueprint(await r.json()); blueprintLoaded = true; }
      } catch (err) { console.warn('blueprint failed:', err.message); }
    }

    // Visual layer: 3DGS if present (compact .ksplat first, it streams in), else coloured points.
    // The splat download starts now and runs alongside the collision mesh below.
    let splatP = Promise.resolve(false);
    if (!blueprintLoaded && (hasKsplat || hasPly)) {
      const splatUrl = `${art}/visual_splats.${hasKsplat ? 'ksplat' : 'ply'}`;
      this.splatViewer = new GaussianSplats3D.DropInViewer({
        gpuAcceleratedSort: false,
        sharedMemoryForWorkers: window.crossOriginIsolated === true,  // needs the COOP/COEP headers
        integerBasedSort: true,
        halfPrecisionCovariancesOnGPU: true,
        sphericalHarmonicsDegree: 1,  // the trainer exports SH degree 1 (glossy floors, glass)
      });
      this.threeScene.add(this.splatViewer);
      splatP = this.splatViewer.addSplatScene(splatUrl, {
        splatAlphaRemovalThreshold: 5, showLoadingUI: false, progressiveLoad: hasKsplat,
      }).then(() => { this._setPixelRatio(this.pixelRatio); this.invalidate(); return true; })
        .catch((err) => {
          console.warn('splats failed:', err.message);
          this.threeScene.remove(this.splatViewer);
          this.splatViewer = null;
          return false;
        });
    }

    // Collision mesh (invisible, used for BVH raycasts)
    const gltf = await meshP;
    const meshLoaded = !!gltf;
    if (gltf) {
      gltf.scene.traverse((child) => {
        if (child.isMesh) {
          child.geometry.computeBoundsTree();
          child.material = new THREE.MeshBasicMaterial({ visible: false });
          this.collisionMesh = child;
        }
      });
      this.threeScene.add(gltf.scene);
    }

    const splatLoaded = await splatP;
    let pointsLoaded = false;
    if (!splatLoaded && !blueprintLoaded && hasPoints) {
      try {
        const geo = await new PLYLoader().loadAsync(pointsUrl);
        const hasColor = !!geo.getAttribute('color');
        this.pointCloud = new THREE.Points(geo, new THREE.PointsMaterial({
          size: 0.06, vertexColors: hasColor, color: hasColor ? 0xffffff : 0x64b4ff, sizeAttenuation: true,
        }));
        this.threeScene.add(this.pointCloud);
        pointsLoaded = true;
        geo.computeBoundingBox();
        const c = geo.boundingBox.getCenter(new THREE.Vector3());
        const size = geo.boundingBox.getSize(new THREE.Vector3()).length();
        this.camera.position.set(c.x, c.y + size * 0.5, c.z + size * 0.8);
        this.controls.target.copy(c);
      } catch (err) { console.warn('points failed:', err.message); }
    }
    if (splatLoaded && this.collisionMesh) {
      // splats have no cheap bounds: frame the camera from the collision mesh instead
      this.collisionMesh.geometry.computeBoundingBox();
      const bb = this.collisionMesh.geometry.boundingBox;
      const c = bb.getCenter(new THREE.Vector3());
      const d = bb.getSize(new THREE.Vector3()).length();
      this.camera.position.set(c.x, c.y + d * 0.15, c.z + d * 0.3);
      this.controls.target.copy(c);
    }
    if (!splatLoaded && !pointsLoaded && !blueprintLoaded) {
      if (this.collisionMesh) {
        // No photoreal layer: show the real collision mesh instead of fake geometry
        this.collisionMesh.material = new THREE.MeshStandardMaterial({ color: 0x5b7a99, roughness: 0.8, side: THREE.DoubleSide });
        this.collisionMesh.geometry.computeBoundingBox();
        const bb = this.collisionMesh.geometry.boundingBox;
        const c = bb.getCenter(new THREE.Vector3());
        const d = bb.getSize(new THREE.Vector3()).length();
        this.camera.position.set(c.x, c.y + d * 0.6, c.z + d * 0.7);
        this.controls.target.copy(c);
      } else {
        this._addVisibleRoomGeometry();
      }
    }

    // Capture viewpoints: start inside the room at eye height, where the splats were trained;
    // scene.json "start_view" picks the spot, else the middle of the walk
    this.viewpoints = [];
    const cams = (splatLoaded || pointsLoaded || blueprintLoaded) ? await camsP : null;
    if (cams?.centers) {
      this.viewpoints = cams.centers.map((c, i) => ({
        pos: new THREE.Vector3(...c),
        fwd: cams.forwards ? new THREE.Vector3(...cams.forwards[i]) : null,
      }));
    }
    // a blueprint reads best from above; [ and ] still step inside to the capture spots.
    // ?view=<i> opens at that capture spot (the blueprint links a piece to where it was filmed)
    const linkView = new URLSearchParams(window.location.search).get('view');
    const startView = linkView !== null && Number.isInteger(+linkView) ? +linkView
      : this.sceneInfo?.start_view ?? Math.floor(this.viewpoints.length / 2);
    if (this.viewpoints.length && !blueprintLoaded) this.enterViewpoint(startView);

    this.navReady = await navP;
    this.scaleFactor = 1.0;
    this.calibrated = false;
    if (this.navReady) {
      try {
        const c = await (await fetch(`${this.apiBaseUrl}/api/v1/scenes/${this.sceneId}/calibration`)).json();
        this.scaleFactor = c.scale_factor; this.calibrated = c.calibrated;
      } catch { /* keep defaults */ }
    }
    if (this.roam.applyLinkPose() && this.inside) this.setInside(true);
    this.invalidate();
    return { splatLoaded, pointsLoaded, blueprintLoaded, meshLoaded, navReady: this.navReady, viewpoints: this.viewpoints.length };
  }

  /**
   * Blueprint style: the space as lines only, like a 2D plan extruded into 3D. Outline and
   * corners are bright, interior walls softer, furniture line drawings amber; a drafting grid
   * lies on the floor. Frames the camera above the whole plan.
   */
  _addBlueprint(bp) {
    const colours = { outline: 0xeaf6ff, corner: 0xeaf6ff, wall: 0x9fd4ff, object: 0xffd27a };  // furniture in amber
    const group = new THREE.Group();
    const lines = (segs, color) => {
      const pos = new Float32Array(segs.length * 6);
      segs.forEach((s, i) => pos.set(s.slice(0, 6), i * 6));
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color }));
    };
    for (const kind of ['outline', 'corner', 'wall']) {
      const segs = bp.segments.filter((s) => s[6] === kind);
      if (segs.length) group.add(lines(segs, colours[kind]));
    }
    // each furniture piece is its own line set (segment index 7 names the piece), with an
    // invisible box around it in the plan frame, so a click can pick and highlight it
    const yaw = bp.yaw_rad ?? 0, cy = Math.cos(yaw), sy = Math.sin(yaw);
    const byObj = new Map();
    for (const s of bp.segments) if (s[6] === 'object') byObj.set(s[7] ?? -1, [...(byObj.get(s[7] ?? -1) ?? []), s]);
    this.blueprintObjects = [];
    for (const [i, segs] of byObj) {
      const mesh = lines(segs, colours.object);
      group.add(mesh);
      const info = bp.objects?.[i];
      if (i < 0 || !info) continue;
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const s of segs) {
        for (const [x, y, z] of [s.slice(0, 3), s.slice(3, 6)]) {
          const p = [cy * x - sy * z, y, sy * x + cy * z];  // world -> plan frame
          for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
        }
      }
      const pad = 0.05, size = hi.map((h, k) => Math.max(h - lo[k], 0.1) + 2 * pad);
      const c = hi.map((h, k) => (h + lo[k]) / 2);
      const box = new THREE.Mesh(new THREE.BoxGeometry(...size),
        new THREE.MeshBasicMaterial({ color: 0x7ff3ff, transparent: true, opacity: 0.1, depthWrite: false }));
      box.position.set(cy * c[0] + sy * c[2], c[1], -sy * c[0] + cy * c[2]);  // plan -> world
      box.rotation.y = yaw;
      box.visible = false;  // shown as a soft fill while the piece is selected; raycasts ignore visibility
      group.add(box);
      this.blueprintObjects.push({ info, lines: mesh, box, center: box.position.clone(), radius: Math.hypot(...size) / 2 });
    }
    const lo = new THREE.Vector3(...bp.bounds.min), hi = new THREE.Vector3(...bp.bounds.max);
    const c = lo.clone().add(hi).multiplyScalar(0.5);
    const span = Math.ceil(Math.max(hi.x - lo.x, hi.z - lo.z) + 4);
    const grid = new THREE.GridHelper(span, span * 2, 0x2d5f8f, 0x1c4370);
    grid.position.set(c.x, (bp.floor_y ?? 0) - 0.005, c.z);
    grid.rotation.y = -(bp.yaw_rad ?? 0);  // grid lines run along the walls
    group.add(grid);
    this.threeScene.add(group);
    this.threeScene.background = new THREE.Color(0x0b2340);
    this.blueprint = group;
    const d = hi.clone().sub(lo).length();
    this.camera.position.set(c.x + d * 0.35, d * 0.55, c.z + d * 0.45);
    this.controls.target.set(c.x, 0, c.z);
  }

  /**
   * Draw another scene's blueprint lines over this one, always on top (furniture amber, walls
   * blue): shows how the blueprint twin lines up with the photoreal splats. Call again to remove.
   */
  async toggleBlueprintOverlay(sceneId) {
    if (this.overlay) {
      this.threeScene.remove(this.overlay);
      this.overlay = null;
      this.invalidate();
      return false;
    }
    const bp = await (await fetch(`${this.apiBaseUrl}/scenes/${sceneId}/artifacts/blueprint.json`)).json();
    this.overlay = new THREE.Group();
    for (const [kinds, color] of [[['object'], 0xffb020], [['outline', 'corner', 'wall'], 0x5fb6ff]]) {
      const segs = bp.segments.filter((s) => kinds.includes(s[6]));
      const pos = new Float32Array(segs.length * 6);
      segs.forEach((s, i) => pos.set(s.slice(0, 6), i * 6));
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      const lines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true }));
      lines.renderOrder = 20;
      this.overlay.add(lines);
    }
    this.threeScene.add(this.overlay);
    this.invalidate();
    return true;
  }

  /** Jump to capture viewpoint i (wraps), level the horizon and look the way the camera did. */
  enterViewpoint(i) {
    const n = this.viewpoints.length;
    if (!n) return;
    this.viewpointIndex = ((i % n) + n) % n;
    const vp = this.viewpoints[this.viewpointIndex];
    this.camera.position.copy(vp.pos);
    const fwd = vp.fwd ? vp.fwd.clone().setY(0).normalize() : null;
    this.setInside(true, fwd);
  }

  /**
   * Inside mode turns the orbit controls into a look-around: the target sits 5 cm in front
   * of the eye, so dragging rotates the view in place (full 360 yaw) instead of orbiting the
   * building. Outside mode restores normal orbit, pan and zoom.
   */
  setInside(on, fwd = null) {
    const c = this.controls;
    this.inside = on;
    if (on) {
      if (!fwd) {
        fwd = new THREE.Vector3();
        this.camera.getWorldDirection(fwd);
        fwd.y = 0;
        if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1);
        fwd.normalize();
      }
      c.target.copy(this.camera.position).addScaledVector(fwd, 0.05);
      c.enableZoom = false;
      c.enablePan = false;
      c.rotateSpeed = -0.4; // drag moves the scene with the cursor, like a panorama viewer
    } else {
      c.enableZoom = true;
      c.enablePan = true;
      c.rotateSpeed = 1.0;
      c.autoRotate = false;
    }
    window.dispatchEvent(new CustomEvent('view-mode', { detail: { inside: on, lookAround: c.autoRotate } }));
  }

  /** Slow, continuous 360 turn in place (one turn every ~24 s). */
  toggleLookAround(force) {
    if (!this.inside) this.setInside(true);
    const c = this.controls;
    c.autoRotate = force ?? !c.autoRotate;
    c.autoRotateSpeed = 2.5;
    window.dispatchEvent(new CustomEvent('view-mode', { detail: { inside: true, lookAround: c.autoRotate } }));
  }

  _addVisibleRoomGeometry() {
    // Floor
    const floorGeo = new THREE.PlaneGeometry(10, 8);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x2a3a4a,
      roughness: 0.9,
      side: THREE.DoubleSide
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0.01;
    this.threeScene.add(floor);

    // Walls
    const wallMat = new THREE.MeshStandardMaterial({
      color: 0x3a4a5a,
      roughness: 0.8,
      side: THREE.DoubleSide
    });

    // Back wall
    const backWall = new THREE.Mesh(new THREE.PlaneGeometry(10, 3), wallMat);
    backWall.position.set(5, 1.5, 8);
    this.threeScene.add(backWall);

    // Front wall
    const frontWall = new THREE.Mesh(new THREE.PlaneGeometry(10, 3), wallMat);
    frontWall.position.set(5, 1.5, 0);
    frontWall.rotation.y = Math.PI;
    this.threeScene.add(frontWall);

    // Left wall
    const leftWall = new THREE.Mesh(new THREE.PlaneGeometry(8, 3), wallMat);
    leftWall.position.set(0, 1.5, 4);
    leftWall.rotation.y = Math.PI / 2;
    this.threeScene.add(leftWall);

    // Right wall
    const rightWall = new THREE.Mesh(new THREE.PlaneGeometry(8, 3), wallMat);
    rightWall.position.set(10, 1.5, 4);
    rightWall.rotation.y = -Math.PI / 2;
    this.threeScene.add(rightWall);

    // Add a grid for reference
    const gridHelper = new THREE.GridHelper(10, 10, 0x64b4ff, 0x2a3a4a);
    gridHelper.position.y = 0.02;
    this.threeScene.add(gridHelper);
  }

  async navigateToPoint(targetWorldPoint) {
    const startXYZ = [this.camera.position.x, this.camera.position.y, this.camera.position.z];
    const goalXYZ = [targetWorldPoint.x, targetWorldPoint.y, targetWorldPoint.z];

    let pathData = null;
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/v1/navigation/plan-path`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scene_id: this.sceneId, start_xyz: startXYZ, goal_xyz: goalXYZ }),
      });
      if (response.ok) pathData = await response.json();
      else if (response.status === 422) return null;  // the backend found no route
    } catch { /* no backend */ }
    // static hosting: plan on the minimap's grid in the browser (set by the HUD once loaded)
    if (!pathData) pathData = this.gridPlanner?.plan(startXYZ, goalXYZ) ?? null;
    if (!pathData) {
      console.error('Pathplanning failed: no route');
      return null;
    }

    this._renderNavigationRibbon(pathData.waypoints_3d);
    this._beginCameraFlight(pathData.waypoints_3d, pathData.total_length_m);
    return pathData;
  }

  _renderNavigationRibbon(waypoints3D) {
    if (this.activePathMesh) {
      this.threeScene.remove(this.activePathMesh);
      this.activePathMesh.geometry.dispose();
      this.activePathMesh.material.dispose();
    }
    if (waypoints3D.length < 2) return;

    const floorPoints = waypoints3D.map((pt) => new THREE.Vector3(pt[0], 0.12, pt[2]));
    const curve = new THREE.CatmullRomCurve3(floorPoints);
    const tubeGeo = new THREE.TubeGeometry(curve, Math.max(32, floorPoints.length * 2), 0.045, 8, false);
    const tubeMat = new THREE.MeshBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.85 });
    this.activePathMesh = new THREE.Mesh(tubeGeo, tubeMat);
    this.threeScene.add(this.activePathMesh);
    this.invalidate();
  }

  clearPath() {
    if (this.activePathMesh) {
      this.threeScene.remove(this.activePathMesh);
      this.activePathMesh.geometry.dispose();
      this.activePathMesh.material.dispose();
      this.activePathMesh = null;
    }
    this.isFlying = false;
    this.flightCurve = null;
    this.invalidate();
  }

  _beginCameraFlight(waypoints3D, totalLengthMeters) {
    if (waypoints3D.length < 2 || totalLengthMeters < 0.1) return;
    // stop ~0.9 m short of the goal so the camera does not end inside the target object
    const goal = waypoints3D[waypoints3D.length - 1];
    let trimmed = waypoints3D.filter((pt) => Math.hypot(pt[0] - goal[0], pt[2] - goal[2]) > 0.9);
    if (trimmed.length >= 2) {
      waypoints3D = trimmed;
      totalLengthMeters = Math.max(0.1, totalLengthMeters - 0.9);
    }
    const curvePts = waypoints3D.map((pt) => new THREE.Vector3(pt[0], pt[1], pt[2]));
    this.flightCurve = new THREE.CatmullRomCurve3(curvePts);
    this.flightTotalLength = totalLengthMeters;
    this.flightProgress = 0.0;
    this.isFlying = true;
  }

  async searchAndNavigateToPOI(naturalLanguageQuery) {
    const response = await fetch(`${this.apiBaseUrl}/api/v1/semantics/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scene_id: this.sceneId, query_text: naturalLanguageQuery, top_k: 3 }),
    });
    if (!response.ok) return [];
    const pois = await response.json();

    while (this.poiMarkersGroup.children.length > 0) {
      this.poiMarkersGroup.remove(this.poiMarkersGroup.children[0]);
    }

    pois.forEach((poi, idx) => {
      const minVec = new THREE.Vector3(...poi.bbox_min_xyz);
      const maxVec = new THREE.Vector3(...poi.bbox_max_xyz);
      const box3 = new THREE.Box3(minVec, maxVec);
      const helper = new THREE.Box3Helper(box3, idx === 0 ? 0xf59e0b : 0x3b82f6);
      this.poiMarkersGroup.add(helper);
    });
    this.invalidate();

    if (pois.length > 0) {
      const bestPoi = pois[0];
      const targetVec = new THREE.Vector3(...bestPoi.centroid_xyz);
      await this.navigateToPoint(targetVec);
    }
    return pois;
  }

  _bindEvents() {
    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.invalidate();
    });
    // any input may change the scene (props, measuring, toggles): draw for a moment after it
    for (const ev of ['pointerdown', 'pointermove', 'wheel', 'keydown', 'keyup']) {
      window.addEventListener(ev, () => this.invalidate(), { passive: true });
    }

    const travelTo = (event) => {
      if (!this.collisionMesh) return;
      const mouse = new THREE.Vector2(
        (event.clientX / window.innerWidth) * 2 - 1,
        -(event.clientY / window.innerHeight) * 2 + 1
      );
      this.raycaster.setFromCamera(mouse, this.camera);
      const hits = this.raycaster.intersectObject(this.collisionMesh, true);
      if (hits.length > 0) {
        this.navigateToPoint(hits[0].point);
      }
    };
    const canvas = this.renderer.domElement;
    canvas.addEventListener('dblclick', (event) => { if (this._lastPointer !== 'touch') travelTo(event); });
    // phones: a double tap travels there (browsers do not all turn a double tap into dblclick)
    let down = null, lastTap = null;
    canvas.addEventListener('pointerdown', (e) => {
      this._lastPointer = e.pointerType;
      down = { x: e.clientX, y: e.clientY, t: performance.now() };
    });
    canvas.addEventListener('pointerup', (e) => {
      if (e.pointerType !== 'touch' || !down) return;
      const now = performance.now();
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 10 || now - down.t > 300) { lastTap = null; return; }
      if (lastTap && now - lastTap.t < 350 && Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < 40) {
        lastTap = null;
        travelTo(e);
      } else {
        lastTap = { x: e.clientX, y: e.clientY, t: now };
      }
    });
  }

  _startRenderLoop() {
    if (this._looping) return;
    this._looping = true;
    this._idleFor = 0;
    const animate = () => {
      requestAnimationFrame(animate);
      const dt = this.clock.getDelta();

      if (this.isFlying && this.flightCurve) {
        this.flightProgress += (this.flightSpeedMetersPerSec * dt) / this.flightTotalLength;
        if (this.flightProgress >= 1.0) { this.flightProgress = 1.0; this.isFlying = false; }
        this.camera.position.copy(this.flightCurve.getPointAt(this.flightProgress));
        this.controls.target.lerp(this.flightCurve.getPointAt(Math.min(1.0, this.flightProgress + 0.04)), 0.18);
      }
      if (this.roam.walking) this.roam.update(dt);
      if (!this.roam.walking || this.roam.touch) {  // touch walking still looks by dragging
        this.controls.update(dt); // dt keeps auto look-around at the same speed on any refresh rate
        if (this.sandbox && !this.roam.walking) this.sandbox.update();
      }

      // Only draw while the view changes (plus a short tail for splat sorting and damping);
      // a still view costs no GPU time, which keeps the page and the rest of the laptop responsive.
      this.camera.updateMatrixWorld();
      // damping keeps nudging the camera by sub-millimetre amounts for seconds: that is not motion
      const m = this.camera.matrixWorld.elements, last = this._lastView.elements;
      let diff = 0;
      for (let i = 0; i < 16; i++) diff = Math.max(diff, Math.abs(m[i] - last[i]));
      const moving = diff > 2e-4;
      this._lastView.copy(this.camera.matrixWorld);
      if (moving) this.invalidate();  // the damping tail below this is invisible
      this._idleFor = moving ? 0 : this._idleFor + dt;
      // the splat sorter works in a worker: keep drawing until its newest order is on screen
      const sv = this.splatViewer?.viewer;
      if (sv && (sv.sortRunning || !sv.splatRenderReady || sv.renderNextFrame || sv.isLoadingOrUnloading())) this.invalidate(2);
      if (this._redrawFrames <= 0) return;
      this._redrawFrames--;
      this._adaptResolution(dt, moving);
      this.renderer.render(this.threeScene, this.camera);
      if (sv) sv.renderNextFrame = false;  // drop-in mode never clears it; we just drew that frame
      this.framesDrawn = (this.framesDrawn ?? 0) + 1;
    };
    animate();
  }
}

import { SpatialEngineViewer } from './SpatialEngineViewer.js';
import { PoiSearchUI } from './PoiSearchUI.js';
import { FieldHUD } from './FieldHUD.js';
import { Sandbox } from './Sandbox.js';
import { ObjectInspector } from './ObjectInspector.js';

// Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';
const REQUESTED = new URLSearchParams(window.location.search).get('scene');

async function main() {
  const $ = (id) => document.getElementById(id);

  // Scene list first (a few ms): it tells the viewer which layer files exist, so loading
  // starts straight away without probing requests, and drives the switcher below.
  let scenes = [];
  try {
    const r = await fetch(`${API_BASE_URL}/api/v1/scenes`);
    if (r.ok) scenes = (await r.json()).scenes ?? [];
  } catch { /* static hosting: only the current scene */ }
  // no ?scene=: open the first house in the list
  const SCENE_ID = REQUESTED || scenes.find((sc) => sc.place === sc.id)?.id || 'house';
  const sceneInfo = scenes.find((sc) => sc.id === SCENE_ID) ?? null;

  const viewer = new SpatialEngineViewer($('canvas-container'), API_BASE_URL, SCENE_ID, sceneInfo);
  window.__viewer = viewer; // handy for debugging in the console
  const roam = viewer.roam;
  const hud = new FieldHUD(viewer, API_BASE_URL, SCENE_ID);
  const sandbox = new Sandbox(viewer, hud, roam);
  viewer.sandbox = sandbox;

  // Measurement readout and calibration (first measurement asks for a real length once)
  const readout = $('readout');
  const showDistance = (units) => {
    readout.style.display = 'block';
    const m = units * viewer.scaleFactor;
    readout.textContent = viewer.calibrated
      ? `Distance: ${m.toFixed(2)} m (calibrated)`
      : `Distance: ${m.toFixed(2)} m (estimated: calibrate with a known length)`;
  };
  roam.onMeasure = async (units) => {
    showDistance(units);
    if (viewer.calibrated || !viewer.navReady) return;
    const real = parseFloat(window.prompt('Calibrate: real length of this segment in metres (blank to skip)', ''));
    if (!(real > 0)) return;
    let factor = real / units;  // static hosting cannot save it: this visit only
    try {
      const r = await fetch(`${API_BASE_URL}/api/v1/scenes/${SCENE_ID}/calibration`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ measured_units: units, real_metres: real }),
      });
      if (r.ok) factor = (await r.json()).scale_factor;
    } catch { /* no backend */ }
    viewer.scaleFactor = factor; viewer.calibrated = true;
    showDistance(units);
  };

  // Model switcher: the list holds one entry per house; the toggle flips that house between its
  // photoreal splats and its line blueprint (two scenes linked as twins by the backend).
  const go = (id) => {
    const url = new URL(window.location.href);
    url.search = '';  // a shared pose belongs to the old scene
    url.searchParams.set('scene', id);
    window.location.href = url.toString();
  };
  const sceneSelect = $('scene-select');
  const place = sceneInfo?.place ?? SCENE_ID;
  const places = scenes.filter((sc) => sc.place === sc.id);
  if (!places.some((sc) => sc.id === place)) places.unshift({ id: place, name: SCENE_ID });
  $('title').textContent = (places.find((sc) => sc.id === place)?.name ?? SCENE_ID).split(' · ')[0];
  sceneSelect.replaceChildren(...places.map((sc) => {
    const o = document.createElement('option');
    o.value = sc.id;
    o.textContent = sc.name.split(' · ')[0];  // "House · photoreal" -> "House"
    o.selected = sc.id === place;
    return o;
  }));
  sceneSelect.onchange = () => {
    // keep the current view style when the new house has it
    const target = scenes.find((sc) => sc.id === sceneSelect.value);
    go(sceneInfo?.style === 'blueprint' && target?.twins?.blueprint ? target.twins.blueprint : sceneSelect.value);
  };
  const toggle = $('style-toggle');
  if (sceneInfo?.twins) {
    const current = sceneInfo.style === 'blueprint' ? 'blueprint' : 'photoreal';
    toggle.hidden = false;
    for (const b of toggle.querySelectorAll('button[data-style]')) {
      b.classList.toggle('on', b.dataset.style === current);
      b.onclick = () => { if (b.dataset.style !== current) go(sceneInfo.twins[b.dataset.style]); };
    }
    // photoreal: the blueprint's lines drawn over the splats, to see that the two line up
    const overlay = $('btn-overlay');
    if (current === 'photoreal' && sceneInfo.twins.blueprint) {
      overlay.hidden = false;
      overlay.onclick = async () => overlay.classList.toggle('on', await viewer.toggleBlueprintOverlay(sceneInfo.twins.blueprint));
    }
  }

  // Toolbar
  $('btn-walk').onclick = () => roam.toggleWalk();
  $('btn-plan').onclick = () => roam.togglePlan();
  $('btn-measure').onclick = () => {
    roam.toggleMeasure();
    readout.style.display = roam.measuring ? 'block' : 'none';
    if (roam.measuring) readout.textContent = `Measure: ${roam.touch ? 'tap' : 'click'} two points`;
  };
  // phones: the object list and the floor plan open as sheets from the bottom bar
  $('btn-map').onclick = () => {
    const open = document.body.classList.toggle('map-open');
    $('btn-map').classList.toggle('on', open);
  };
  // look around (O), capture spots ([ ]) and points (P) stay on the keyboard
  window.addEventListener('roam-mode', (e) => {
    $('btn-walk').classList.toggle('on', e.detail.walking);
    $('btn-plan').classList.toggle('on', e.detail.plan);
    $('btn-measure').classList.toggle('on', e.detail.measuring);
    if (!e.detail.measuring) readout.style.display = 'none';
  });

  // Connection / layer status (status bar)
  const setDot = (id, textId, text, warn = false) => {
    $(id).classList.toggle('warning', warn);
    $(textId).textContent = text;
  };
  setDot('conn-status', 'conn-text', 'Viewer ready');
  setDot('mesh-status', 'mesh-text', 'Mesh: loading…', true);
  setDot('splat-status', 'splat-text', 'Layer: loading…', true);
  hud.setSubtitle('Loading scene…');

  try {
    const result = await viewer.loadDigitalTwin();

    setDot('mesh-status', 'mesh-text', result.meshLoaded ? 'Mesh: BVH ready' : 'Mesh: not available', !result.meshLoaded);
    const layer = result.blueprintLoaded ? 'Blueprint: rendering' : result.splatLoaded ? 'Splats: rendering'
      : result.pointsLoaded ? 'Point cloud: rendering' : 'Layer: none';
    const live = result.splatLoaded || result.pointsLoaded || result.blueprintLoaded;
    setDot('splat-status', 'splat-text', layer, !live);

    // splats are fill bound: an integrated GPU is several times slower than the laptop's
    // discrete one, which the browser only gets when Windows is told to give it
    const gpu = viewer.gpuName();
    if (result.splatLoaded && /intel|uhd|iris|microsoft basic|swiftshader|llvmpipe/i.test(gpu)
        && !/nvidia|geforce|radeon rx|arc a/i.test(gpu)) {
      $('gpu-item').hidden = false;
      $('gpu-text').textContent = 'GPU: integrated (slower)';
      $('gpu-item').title = `Rendering on ${gpu}. For smooth walking, set this browser to "High performance" in `
        + 'Windows Settings > System > Display > Graphics, then restart the browser.';
    }

    hud.setStamp(live, 'mesh only');
    const shown = result.blueprintLoaded ? 'Blueprint' : result.splatLoaded ? 'Photoreal' : result.pointsLoaded ? 'Point cloud' : 'Collision mesh';
    hud.setSubtitle(`${shown} · ${result.viewpoints || 0} capture spots · navigation ${result.navReady ? 'on' : 'off'}`);
    hud.showReference();
    hud.setupGate(roam);
    if (await hud.loadMap()) { hud.startLoop(); $('btn-map').hidden = false; }

    // Plan clicks set the route end; sandbox restores props from link or this browser
    hud.onMapClick = (x, z) => sandbox.routeTo(x, 0, z);
    sandbox.restore();
    sandbox.onRoute = (path) => {
      $('nav-info').classList.add('active');
      $('nav-distance').textContent = `${path.total_length_m.toFixed(2)} m`;
      $('nav-clearance').textContent = `${path.min_obstacle_clearance_m.toFixed(2)} m`;
      $('nav-waypoints').textContent = `${path.waypoints_3d.length}`;
    };
    $('clear-path-btn').addEventListener('click', () => { hud.route = []; $('legend-route').style.display = 'none'; });
  } catch (err) {
    console.error('Failed to load digital twin:', err);
    setDot('conn-status', 'conn-text', 'Error');
    hud.setStamp(false, 'load failed');
    hud.setSubtitle(`Failed to load "${SCENE_ID}". Check the backend at ${API_BASE_URL || 'this origin'}.`);
  }

  // Poi search keeps its own panel logic
  // blueprint scenes: every piece is listed under Find in scene and opens its card when picked
  // (or clicked in 3D); photoreal scenes search the scene's labelled points of interest
  if (viewer.blueprintObjects?.length) {
    window.__inspector = new ObjectInspector(viewer, API_BASE_URL);
    $('btn-find').hidden = false;
    $('btn-find').onclick = () => $('btn-find').classList.toggle('on', document.body.classList.toggle('find-open'));
    // the menu sits under the (now taller) search panel
    const place = () => { $('toolbar').style.top = `${$('search-panel').getBoundingClientRect().bottom + 10}px`; };
    place();
    window.addEventListener('resize', place);
  } else {
    new PoiSearchUI(viewer);
  }
}

main();

import * as THREE from 'three';

/**
 * Renders 3D navigation ribbons and animated camera flight controllers
 * for the Spatial Engine walkthrough experience.
 */
export class PathVisualizer {
  constructor(threeScene) {
    this.scene = threeScene;
    this.ribbonMeshes = [];
    this.markerMeshes = [];
    this.animationMixers = [];
  }

  /**
   * Creates an animated ribbon path with start/end markers
   * @param {number[][]} waypoints3D - Array of [x, y, z] waypoints
   * @param {object} options - Visual configuration
   */
  createAnimatedRibbon(waypoints3D, options = {}) {
    const {
      color = 0x10b981,
      ribbonRadius = 0.045,
      opacity = 0.85,
      showMarkers = true,
      markerColor = 0xf59e0b,
    } = options;

    if (waypoints3D.length < 2) return null;

    // Floor-level ribbon
    const floorPoints = waypoints3D.map(
      (pt) => new THREE.Vector3(pt[0], 0.12, pt[2])
    );
    const curve = new THREE.CatmullRomCurve3(floorPoints);
    const segments = Math.max(32, floorPoints.length * 2);
    const tubeGeo = new THREE.TubeGeometry(curve, segments, ribbonRadius, 8, false);

    const tubeMat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity,
    });

    const ribbonMesh = new THREE.Mesh(tubeGeo, tubeMat);
    this.scene.add(ribbonMesh);
    this.ribbonMeshes.push(ribbonMesh);

    // Start and end markers
    if (showMarkers && floorPoints.length >= 2) {
      const startMarker = this._createMarker(floorPoints[0], markerColor, 0.08);
      const endMarker = this._createMarker(
        floorPoints[floorPoints.length - 1],
        0xef4444,
        0.08
      );
      this.scene.add(startMarker);
      this.scene.add(endMarker);
      this.markerMeshes.push(startMarker, endMarker);
    }

    return { curve, ribbonMesh, totalLength: curve.getLength() };
  }

  _createMarker(position, color, radius) {
    const geo = new THREE.SphereGeometry(radius, 16, 16);
    const mat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.9,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(position);
    return mesh;
  }

  /**
   * Creates a camera flight controller that smoothly follows a CatmullRom curve
   * @param {THREE.CatmullRomCurve3} curve
   * @param {number} totalLengthMeters
   * @param {number} speedMps
   */
  createFlightController(curve, totalLengthMeters, speedMps = 1.65) {
    return {
      curve,
      totalLength: totalLengthMeters,
      speed: speedMps,
      progress: 0.0,
      active: true,

      update(dt, camera, controls) {
        if (!this.active) return false;

        this.progress += (this.speed * dt) / this.totalLength;
        if (this.progress >= 1.0) {
          this.progress = 1.0;
          this.active = false;
        }

        const pos = this.curve.getPointAt(this.progress);
        const lookT = Math.min(1.0, this.progress + 0.04);
        const lookTarget = this.curve.getPointAt(lookT);

        camera.position.copy(pos);
        if (controls) {
          controls.target.lerp(lookTarget, 0.18);
        }

        return this.active;
      },
    };
  }

  clearAll() {
    for (const mesh of this.ribbonMeshes) {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    for (const mesh of this.markerMeshes) {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    this.ribbonMeshes = [];
    this.markerMeshes = [];
  }
}
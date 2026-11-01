/**
 * Natural language spatial query UI controller.
 * Connects the search input to the SpatialEngineViewer POI search API
 * and renders results in the DOM overlay.
 */
export class PoiSearchUI {
  constructor(viewer) {
    this.viewer = viewer;
    this.searchInput = document.getElementById('search-input');
    this.searchBtn = document.getElementById('search-btn');
    this.resultsContainer = document.getElementById('poi-results');
    this.navInfo = document.getElementById('nav-info');
    this.navDistance = document.getElementById('nav-distance');
    this.navClearance = document.getElementById('nav-clearance');
    this.navWaypoints = document.getElementById('nav-waypoints');
    this.clearPathBtn = document.getElementById('clear-path-btn');

    this._bind();
  }

  _bind() {
    this.searchBtn.addEventListener('click', () => this._executeSearch());
    this.searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this._executeSearch();
    });
    this.clearPathBtn.addEventListener('click', () => this._clearPath());
  }

  async _executeSearch() {
    const query = this.searchInput.value.trim();
    if (!query) return;

    this.searchBtn.textContent = '...';
    this.searchBtn.disabled = true;

    try {
      const pois = await this.viewer.searchAndNavigateToPOI(query);
      this._renderResults(pois);
    } catch (err) {
      console.error('POI search failed:', err);
      this.resultsContainer.innerHTML = '<div class="poi-message error">Search failed. Is the backend running?</div>';
      this.resultsContainer.style.display = 'block';
    } finally {
      this.searchBtn.textContent = 'Search';
      this.searchBtn.disabled = false;
    }
  }

  _renderResults(pois) {
    this.resultsContainer.innerHTML = '';

    if (!pois || pois.length === 0) {
      this.resultsContainer.innerHTML = '<div class="poi-message">No matching objects.</div>';
      this.resultsContainer.style.display = 'block';
      return;
    }

    pois.forEach((poi, idx) => {
      const item = document.createElement('div');
      item.className = 'poi-item';
      item.innerHTML = `
        <div class="label">${poi.label}${idx === 0 ? ' <span style="color:var(--muted);font-weight:500">· best match</span>' : ''}</div>
        <div class="meta">
          <span>Conf: ${(poi.confidence * 100).toFixed(1)}%</span>
          <span>Pos: (${poi.centroid_xyz.map(v => v.toFixed(1)).join(', ')})</span>
        </div>
      `;
      item.addEventListener('click', () => {
        const target = { x: poi.centroid_xyz[0], y: poi.centroid_xyz[1], z: poi.centroid_xyz[2] };
        this.viewer.navigateToPoint(target);
      });
      this.resultsContainer.appendChild(item);
    });

    this.resultsContainer.style.display = 'block';
  }

  showNavigationInfo(pathData) {
    if (!pathData) return;
    this.navDistance.textContent = `${pathData.total_length_m.toFixed(1)} m`;
    this.navClearance.textContent = `${pathData.min_obstacle_clearance_m.toFixed(2)} m`;
    this.navWaypoints.textContent = `${pathData.waypoints_3d.length}`;
    this.navInfo.classList.add('active');
  }

  _clearPath() {
    this.viewer.clearPath();
    this.navInfo.classList.remove('active');
    this.resultsContainer.style.display = 'none';
    this.resultsContainer.innerHTML = '';
  }
}
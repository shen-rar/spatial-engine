import { defineConfig } from 'vite';

const api = process.env.API_PROXY_TARGET || 'http://localhost:8000';

// Cross-origin isolation lets the splat sorter share memory with its worker (faster sorting);
// 'credentialless' keeps the Google Fonts stylesheet loading. The backend sends the same headers.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

export default defineConfig({
  root: '.',
  publicDir: 'public',
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      output: {
        // libraries change rarely: separate chunks stay cached across app updates
        manualChunks: { three: ['three', 'three-mesh-bvh'], splats: ['@mkkellogg/gaussian-splats-3d'] },
      },
    },
  },
  // pre-bundle on server start, not on the first page load
  optimizeDeps: {
    include: ['three', 'three-mesh-bvh', '@mkkellogg/gaussian-splats-3d',
      'three/examples/jsm/loaders/GLTFLoader.js', 'three/examples/jsm/loaders/PLYLoader.js',
      'three/examples/jsm/controls/OrbitControls.js', 'three/examples/jsm/controls/PointerLockControls.js'],
  },
  server: {
    port: 5173,
    headers: isolation,
    warmup: { clientFiles: ['./src/main.js'] },
    proxy: {
      '/api': { target: api, changeOrigin: true },
      '/scenes': { target: api, changeOrigin: true },
    },
  },
  preview: { headers: isolation },
});

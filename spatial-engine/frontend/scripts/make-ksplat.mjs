#!/usr/bin/env node
/**
 * Convert trained splats (visual_splats.ply) to the viewer's compact .ksplat format.
 *
 *   node scripts/make-ksplat.mjs                     # every scene under ../data/scenes, when stale
 *   node scripts/make-ksplat.mjs path/to/visual_splats.ply [--force]
 *
 * A .ksplat stores 16-bit positions, scales, rotations and SH1 colour, about 40% of the PLY's
 * size, and the browser maps it directly instead of parsing a PLY, so scenes open much faster.
 * Splats below alpha 5/255 are dropped here, the same threshold the viewer uses.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

globalThis.window ??= globalThis;  // the library schedules work through window.setTimeout
const G = await import('@mkkellogg/gaussian-splats-3d');

const MIN_ALPHA = 5;
const COMPRESSION = 1;   // 16-bit attributes; level 2 visibly bands SH colour
const SH_DEGREE = 1;     // the trainer exports SH degree 1

async function convert(ply, force) {
  const out = ply.replace(/\.ply$/i, '.ksplat');
  if (!force && fs.existsSync(out) && fs.statSync(out).mtimeMs >= fs.statSync(ply).mtimeMs) {
    console.log(`up to date  ${out}`);
    return;
  }
  const t = Date.now();
  const buf = fs.readFileSync(ply);
  const data = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const sb = await G.PlyLoader.loadFromFileData(data, MIN_ALPHA, COMPRESSION, true, SH_DEGREE);
  fs.writeFileSync(out, Buffer.from(sb.bufferData));
  const mb = (n) => (n / 1e6).toFixed(1);
  console.log(`wrote ${out}: ${sb.getSplatCount()} splats, ${mb(buf.length)} MB -> ${mb(sb.bufferData.byteLength)} MB in ${Date.now() - t} ms`);
}

const args = process.argv.slice(2);
const force = args.includes('--force');
let plys = args.filter((a) => !a.startsWith('--'));
if (!plys.length) {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../data/scenes');
  plys = fs.readdirSync(root)
    .map((d) => path.join(root, d, 'artifacts', 'visual_splats.ply'))
    .filter((p) => fs.existsSync(p));
}
for (const p of plys) await convert(p, force);

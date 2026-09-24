import type { SceneParams } from '../../utils/sceneMapping';

/**
 * Procedural hillslope around the assessed 30 m cell.
 *
 * Local frame, in metres: the cell sits at the origin, u runs downslope
 * (towards the viewer, scene +z) and v runs across the slope (scene +x). The
 * surface is built so that its gradient at the cell equals the input slope
 * exactly — decorative relief is faded out near the cell for that reason.
 *
 * Everything here is pure and deterministic, so the 3D block and the 2D
 * fallback cross-section draw the same terrain.
 */

export const M_PER_UNIT = 20;
export const HALF_M = 600;
export const HALF = HALF_M / M_PER_UNIT;
export const CELL_M = 30.4625;
export const ROAD_HALF_WIDTH_M = 6;
const PROFILE_LENGTH_M = 380;
const DEG = Math.PI / 180;

const gauss = (x: number, sigma: number) => Math.exp(-(x * x) / (2 * sigma * sigma));

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function hash(x: number, y: number): number {
  let h = Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}

function valueNoise(x: number, y: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const sx = xf * xf * (3 - 2 * xf);
  const sy = yf * yf * (3 - 2 * yf);
  const a = hash(xi, yi);
  const b = hash(xi + 1, yi);
  const c = hash(xi, yi + 1);
  const d = hash(xi + 1, yi + 1);
  return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
}

function fbm(x: number, y: number): number {
  return valueNoise(x, y) * 0.55 + valueNoise(x * 2.1 + 7.7, y * 2.1 - 3.3) * 0.3 + valueNoise(x * 4.3 - 1.9, y * 4.3 + 5.1) * 0.15;
}

/** Deterministic pseudo-random number in [0, 1) for instance placement. */
export const random01 = (i: number, salt: number) => hash(i * 7919 + salt, salt * 104729 + i);

export const roadInBlock = (p: SceneParams) => p.roadDistance <= HALF_M - 24;
export const streamInBlock = (p: SceneParams) => p.streamOffset <= HALF_M - 16;
export const faultInBlock = (p: SceneParams) => faultSpan(p) !== null;

/** Cross-slope position of the stream centreline at downslope distance u. */
export function streamCentre(u: number, p: SceneParams): number {
  return p.streamOffset + 16 * Math.sin(u / 95 + 0.7);
}

/** Unit normal of the lineament trace in (v, u); the trace lies at faultDistance along it. */
const FAULT_NORMAL: [number, number] = (() => {
  const v = -0.55;
  const u = -0.83;
  const length = Math.hypot(v, u);
  return [v / length, u / length];
})();

export function faultOffset(u: number, v: number, p: SceneParams): number {
  return v * FAULT_NORMAL[0] + u * FAULT_NORMAL[1] - p.faultDistance;
}

/** Point on the lineament closest to the cell, and the trace direction, in metres (v, u). */
export function faultGeometry(p: SceneParams) {
  const foot: [number, number] = [FAULT_NORMAL[0] * p.faultDistance, FAULT_NORMAL[1] * p.faultDistance];
  const direction: [number, number] = [-FAULT_NORMAL[1], FAULT_NORMAL[0]];
  return { foot, direction };
}

/** Parameter range [s0, s1] along the trace that lies inside the block, or null. */
export function faultSpan(p: SceneParams, margin = 10): [number, number] | null {
  const { foot, direction } = faultGeometry(p);
  let s0 = -Infinity;
  let s1 = Infinity;
  const limit = HALF_M - margin;
  for (let axis = 0; axis < 2; axis += 1) {
    const d = direction[axis];
    const f = foot[axis];
    if (Math.abs(d) < 1e-9) {
      if (Math.abs(f) > limit) return null;
      continue;
    }
    const a = (-limit - f) / d;
    const b = (limit - f) / d;
    s0 = Math.max(s0, Math.min(a, b));
    s1 = Math.min(s1, Math.max(a, b));
  }
  return s1 - s0 > 40 ? [s0, s1] : null;
}

function hillslope(u: number, v: number, p: SceneParams): number {
  const gradient = Math.tan(p.slopeDeg * DEG);
  let h = -gradient * PROFILE_LENGTH_M * Math.tanh(u / PROFILE_LENGTH_M);

  // Plan curvature: a converging hollow (negative) or diverging spur (positive).
  h += p.planShape * 16 * gauss(v, 120) * gauss(u, 260);

  // Profile curvature: concave (negative) or convex (positive) along the fall line.
  const ue = 260 * Math.tanh(u / 260);
  h -= p.profileShape * 20 * ((ue * ue) / (260 * 260)) * gauss(v, 320);

  // Decorative relief, faded out near the cell so the local gradient stays exact.
  const r = Math.hypot(u, v);
  h += (fbm(u * 0.0045 + 11.3, v * 0.0045 - 3.1) - 0.5) * 14 * smoothstep(90, 320, r);
  return h;
}

/** Terrain height in metres relative to a flat reference through the cell. */
export function heightAt(u: number, v: number, p: SceneParams): number {
  let h = hillslope(u, v, p);

  if (streamInBlock(p)) {
    const dv = v - streamCentre(u, p);
    h -= 8 * gauss(dv, 11) + 5 * gauss(dv, 55);
  }

  if (p.gully > 0.01) {
    const gv = v - 5 * Math.sin(u / 42);
    const along = smoothstep(-30, 25, u) * (1 - smoothstep(240, 420, u));
    h -= p.gully * 7 * gauss(gv, 4 + p.gully * 5) * along;
  }

  if (faultInBlock(p)) {
    h -= 3 * gauss(faultOffset(u, v, p), 9);
  }

  if (roadInBlock(p)) {
    const roadU = -p.roadDistance;
    const weight = 1 - smoothstep(ROAD_HALF_WIDTH_M, ROAD_HALF_WIDTH_M + 16, Math.abs(u - roadU));
    if (weight > 0) {
      const bench = hillslope(roadU, v, p) - 1.5;
      h += (bench - h) * weight;
    }
  }
  return h;
}

export interface SurfaceMasks {
  wet: number;
  road: number;
  channel: number;
  fault: number;
}

/** Surface material masks at a point, in [0, 1]. */
export function surfaceMasks(u: number, v: number, p: SceneParams): SurfaceMasks {
  const r = Math.hypot(u, v);
  let wet = p.wetness * gauss(r, 170);
  let channel = 0;
  if (streamInBlock(p)) {
    const dv = v - streamCentre(u, p);
    channel = gauss(dv, 9);
    wet += 0.7 * gauss(dv, 38);
  }
  if (p.gully > 0.01) {
    const gv = v - 5 * Math.sin(u / 42);
    const along = smoothstep(-30, 25, u) * (1 - smoothstep(240, 420, u));
    channel = Math.max(channel, p.gully * 0.8 * gauss(gv, 3.5 + p.gully * 3) * along);
    wet += p.gully * 0.4 * gauss(gv, 14) * along;
  }
  const road = roadInBlock(p)
    ? 1 - smoothstep(ROAD_HALF_WIDTH_M - 1, ROAD_HALF_WIDTH_M + 1, Math.abs(u + p.roadDistance))
    : 0;
  const fault = faultInBlock(p) ? gauss(faultOffset(u, v, p), 6) : 0;
  return { wet: Math.min(1, wet), road, channel, fault };
}

export const siteHeight = (p: SceneParams) => heightAt(0, 0, p);

/** Standing water level for water-dominated land cover, in metres relative to the reference. */
export function waterLevel(p: SceneParams): number | null {
  const site = siteHeight(p);
  if (p.lulc === 80) return site + 1.5;
  if (p.lulc === 95) return site + 0.8;
  if (p.lulc === 90) return site + 0.2;
  return null;
}

/** Sea level relative to the reference, when the block reaches below 0 m a.s.l. */
export function seaLevel(p: SceneParams): number | null {
  // World elevation of a point is elevation + (h - h_site); the sea sits where that is 0.
  const site = siteHeight(p);
  const sea = site - p.elevation;
  let lowest = Infinity;
  for (let v = -HALF_M; v <= HALF_M; v += 100) {
    lowest = Math.min(lowest, heightAt(HALF_M, v, p), heightAt(-HALF_M, v, p));
  }
  return lowest < sea ? sea : null;
}

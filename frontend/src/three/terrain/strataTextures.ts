import { CanvasTexture, ClampToEdgeWrapping, RepeatWrapping, SRGBColorSpace } from 'three';

/**
 * Procedural rock textures for the cutaway faces, one per GLiM lithology
 * class. The top of each canvas is a thin soil horizon; below it the pattern
 * identifies the rock type. Patterns are schematic, in the manner of a
 * geological block diagram.
 */

const W = 256;
const H = 512;
const SOIL_PX = 14;

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function soil(ctx: CanvasRenderingContext2D) {
  const gradient = ctx.createLinearGradient(0, 0, 0, SOIL_PX + 6);
  gradient.addColorStop(0, '#3b2f22');
  gradient.addColorStop(0.7, '#57432e');
  gradient.addColorStop(1, 'rgba(87,67,46,0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, W, SOIL_PX + 6);
}

function metamorphic(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = '#6f6a6a';
  ctx.fillRect(0, 0, W, H);
  const random = rng(11);
  const bands = ['#8a8080', '#5c585b', '#9b8f86', '#6b6568', '#a39a90'];
  for (let y = SOIL_PX; y < H; y += 5 + random() * 9) {
    ctx.strokeStyle = bands[Math.floor(random() * bands.length)];
    ctx.lineWidth = 1.5 + random() * 4;
    ctx.beginPath();
    for (let x = 0; x <= W; x += 8) {
      const yy = y + Math.sin((x / W) * Math.PI * 4 + y * 0.05) * 6;
      if (x === 0) ctx.moveTo(x, yy);
      else ctx.lineTo(x, yy);
    }
    ctx.stroke();
  }
}

function acidPlutonic(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = '#8f7f78';
  ctx.fillRect(0, 0, W, H);
  const random = rng(23);
  const specks = ['#b7a39a', '#5a5352', '#c9b8ad', '#3f3a3a', '#a58c80'];
  for (let i = 0; i < 5200; i += 1) {
    ctx.fillStyle = specks[Math.floor(random() * specks.length)];
    const size = 1 + random() * 2.5;
    ctx.fillRect(random() * W, SOIL_PX + random() * (H - SOIL_PX), size, size);
  }
}

function sedimentary(ctx: CanvasRenderingContext2D) {
  const random = rng(37);
  const layers = ['#9c8563', '#b39b74', '#86704f', '#a8916a', '#7c6848', '#c0a97f'];
  let y = SOIL_PX;
  while (y < H) {
    const thickness = 6 + random() * 22;
    ctx.fillStyle = layers[Math.floor(random() * layers.length)];
    ctx.fillRect(0, y, W, thickness);
    ctx.fillStyle = 'rgba(40,30,20,0.18)';
    ctx.fillRect(0, y + thickness - 1, W, 1);
    y += thickness;
  }
}

function basicVolcanic(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = '#3d3e42';
  ctx.fillRect(0, 0, W, H);
  const random = rng(53);
  ctx.strokeStyle = 'rgba(15,15,18,0.55)';
  ctx.lineWidth = 1.5;
  for (let x = 0; x < W; x += 14 + random() * 10) {
    ctx.beginPath();
    ctx.moveTo(x, SOIL_PX);
    ctx.lineTo(x + (random() - 0.5) * 6, H);
    ctx.stroke();
  }
  for (let y = SOIL_PX + 20; y < H; y += 26 + random() * 30) {
    ctx.strokeStyle = 'rgba(90,92,98,0.35)';
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y + (random() - 0.5) * 4);
    ctx.stroke();
  }
}

function waterBodies(ctx: CanvasRenderingContext2D) {
  const random = rng(71);
  let y = SOIL_PX;
  while (y < H) {
    const thickness = 3 + random() * 9;
    const shade = 90 + random() * 40;
    ctx.fillStyle = `rgb(${shade * 0.72},${shade * 0.82},${shade * 0.9})`;
    ctx.fillRect(0, y, W, thickness);
    y += thickness;
  }
}

function unmapped(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = '#5a5e62';
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(200,205,210,0.22)';
  ctx.lineWidth = 2;
  for (let i = -H; i < W + H; i += 18) {
    ctx.beginPath();
    ctx.moveTo(i, SOIL_PX);
    ctx.lineTo(i + H, H + SOIL_PX);
    ctx.stroke();
  }
}

const PAINTERS: Record<number, (ctx: CanvasRenderingContext2D) => void> = {
  [-1]: unmapped,
  1: metamorphic,
  2: acidPlutonic,
  3: sedimentary,
  4: basicVolcanic,
  5: waterBodies,
};

const cache = new Map<number, CanvasTexture>();

export function strataTexture(lithology: number): CanvasTexture {
  const code = PAINTERS[lithology] ? lithology : -1;
  const cached = cache.get(code);
  if (cached) return cached;

  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    PAINTERS[code](ctx);
    soil(ctx);
  }
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  texture.wrapS = RepeatWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.anisotropy = 4;
  cache.set(code, texture);
  return texture;
}

/** Metres of depth that one texture height spans on the cutaway. */
export const STRATA_DEPTH_M = 700;

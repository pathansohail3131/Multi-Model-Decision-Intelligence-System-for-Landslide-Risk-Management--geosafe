import { Color, MeshStandardMaterial, type IUniform } from 'three';

/**
 * Terrain surface material: standard PBR lighting and shadows, with the
 * surface colour computed from land cover, rock exposure on steep faces,
 * soil wetness, road and channel masks, contour lines, and the result
 * overlays (susceptibility halo and illustrative failure zone).
 */

export interface TerrainUniforms {
  uGroundA: IUniform<Color>;
  uGroundB: IUniform<Color>;
  uRock: IUniform<Color>;
  uStripes: IUniform<number>;
  uContour: IUniform<number>;
  uContourStep: IUniform<number>;
  uDim: IUniform<number>;
  uResultColor: IUniform<Color>;
  uReveal: IUniform<number>;
  uFailure: IUniform<number>;
  uScanZ: IUniform<number>;
  uScanAmount: IUniform<number>;
  uPulse: IUniform<number>;
  uTime: IUniform<number>;
}

const VERTEX_HEAD = /* glsl */ `
attribute float aWet;
attribute float aRoad;
attribute float aChannel;
attribute float aFault;
varying float vWet;
varying float vRoad;
varying float vChannel;
varying float vFault;
varying vec3 vWorldPos;
varying vec3 vWorldNormal;
`;

const VERTEX_BODY = /* glsl */ `
vWet = aWet;
vRoad = aRoad;
vChannel = aChannel;
vFault = aFault;
vWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
vWorldNormal = normalize(mat3(modelMatrix) * objectNormal);
`;

const FRAGMENT_HEAD = /* glsl */ `
uniform vec3 uGroundA;
uniform vec3 uGroundB;
uniform vec3 uRock;
uniform float uStripes;
uniform float uContour;
uniform float uContourStep;
uniform float uDim;
uniform vec3 uResultColor;
uniform float uReveal;
uniform float uFailure;
uniform float uScanZ;
uniform float uScanAmount;
uniform float uPulse;
uniform float uTime;
varying float vWet;
varying float vRoad;
varying float vChannel;
varying float vFault;
varying vec3 vWorldPos;
varying vec3 vWorldNormal;

float tHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float tNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(tHash(i), tHash(i + vec2(1.0, 0.0)), u.x), mix(tHash(i + vec2(0.0, 1.0)), tHash(i + vec2(1.0, 1.0)), u.x), u.y);
}
`;

const FRAGMENT_COLOR = /* glsl */ `
{
  vec2 xz = vWorldPos.xz;
  float n = tNoise(xz * 0.45) * 0.6 + tNoise(xz * 1.7) * 0.4;
  vec3 ground = mix(uGroundA, uGroundB, n);

  // Cropland terraces follow the contours.
  float terrace = step(0.5, fract(vWorldPos.y * 1.6 + n * 0.3));
  ground = mix(ground, ground * mix(0.82, 1.12, terrace), uStripes);

  // Rock shows through on steep faces.
  float steep = 1.0 - clamp(vWorldNormal.y, 0.0, 1.0);
  float rock = smoothstep(0.42, 0.78, steep + (n - 0.5) * 0.12);
  ground = mix(ground, uRock * (0.85 + n * 0.3), rock);

  // Wet soil darkens; channels and roads use their own surfaces. Colours here are linear.
  ground = mix(ground, ground * 0.5 + vec3(0.0, 0.004, 0.012), clamp(vWet, 0.0, 1.0) * 0.8);
  ground = mix(ground, vec3(0.018, 0.03, 0.036), clamp(vChannel, 0.0, 1.0) * 0.9);
  ground = mix(ground, vec3(0.03, 0.03, 0.032), clamp(vRoad, 0.0, 1.0));
  ground = mix(ground, vec3(0.07, 0.026, 0.016), clamp(vFault, 0.0, 1.0) * 0.8);

  // Contour lines every uContourStep scene units.
  float cy = vWorldPos.y / uContourStep;
  float fw = max(fwidth(cy), 1e-4);
  float line = 1.0 - smoothstep(0.0, fw * 1.25, abs(fract(cy - 0.5) - 0.5));
  ground = mix(ground, ground * 0.6, line * uContour);

  // Model lens: de-emphasise the surface when another theme is selected.
  float lum = dot(ground, vec3(0.299, 0.587, 0.114));
  ground = mix(ground, vec3(lum) * 0.6, uDim);

  // Scan sweep while the scenario is being read.
  float scan = (1.0 - smoothstep(0.0, 0.7, abs(vWorldPos.z - uScanZ))) * uScanAmount;
  ground += vec3(0.1, 0.24, 0.32) * scan;

  // Result halo around the assessed cell.
  float d = length(xz);
  float halo = (1.0 - smoothstep(2.2, 7.5, d)) * uReveal;
  float ring = (1.0 - smoothstep(0.0, 0.09, abs(d - 7.2))) * uReveal;
  ground = mix(ground, uResultColor, halo * 0.38);
  ground = mix(ground, uResultColor * 1.15, ring * 0.85);

  // Waiting pulse: an expanding ring from the cell.
  float pr = fract(uTime * 0.7) * 9.0;
  float pulse = (1.0 - smoothstep(0.0, 0.18, abs(d - pr))) * (1.0 - pr / 9.0) * uPulse;
  ground += vec3(0.12, 0.26, 0.34) * pulse;

  // Illustrative failure zone downslope of the cell (not a predicted extent).
  vec2 q = vec2(xz.x / 3.1, (xz.y - 3.6) / 5.4);
  float e = length(q);
  float inside = 1.0 - smoothstep(0.94, 1.0, e);
  float hatch = step(0.55, fract((xz.x + xz.y) * 1.4));
  float edge = 1.0 - smoothstep(0.0, 0.045, abs(e - 1.0));
  ground = mix(ground, uResultColor * 0.8, inside * hatch * 0.32 * uFailure);
  ground = mix(ground, uResultColor * 1.2, edge * 0.9 * uFailure);

  diffuseColor.rgb = ground;
}
`;

export function createTerrainMaterial(): { material: MeshStandardMaterial; uniforms: TerrainUniforms } {
  const uniforms: TerrainUniforms = {
    uGroundA: { value: new Color('#2d4a2a') },
    uGroundB: { value: new Color('#3c5c30') },
    uRock: { value: new Color('#77716e') },
    uStripes: { value: 0 },
    uContour: { value: 0.3 },
    uContourStep: { value: 1 },
    uDim: { value: 0 },
    uResultColor: { value: new Color('#c9d1d9') },
    uReveal: { value: 0 },
    uFailure: { value: 0 },
    uScanZ: { value: -100 },
    uScanAmount: { value: 0 },
    uPulse: { value: 0 },
    uTime: { value: 0 },
  };

  const material = new MeshStandardMaterial({ roughness: 0.93, metalness: 0 });
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${VERTEX_HEAD}`)
      .replace('#include <fog_vertex>', `#include <fog_vertex>\n${VERTEX_BODY}`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${FRAGMENT_HEAD}`)
      .replace('#include <color_fragment>', `#include <color_fragment>\n${FRAGMENT_COLOR}`);
  };
  material.customProgramCacheKey = () => 'landslide-terrain-v1';
  return { material, uniforms };
}

const GROUND: Record<number, [string, string]> = {
  10: ['#2b4727', '#3d5d31'],
  20: ['#5b6838', '#6f7548'],
  30: ['#788747', '#8f955a'],
  40: ['#76853f', '#978a58'],
  50: ['#6a665f', '#7e786e'],
  60: ['#86704f', '#9a8263'],
  70: ['#cfd5da', '#e2e6e9'],
  80: ['#566450', '#69705a'],
  90: ['#4a5a38', '#5b6b45'],
  95: ['#2c4632', '#3a573d'],
  100: ['#6c7757', '#7c8663'],
};

const ROCK: Record<number, string> = {
  [-1]: '#5e6266',
  1: '#77716e',
  2: '#8e7f78',
  3: '#9c8563',
  4: '#45464b',
  5: '#6b7a86',
};

const DRY = new Color('#7d7249');

export function groundColours(lulc: number, vegetation: number): [Color, Color] {
  const [a, b] = GROUND[lulc] ?? GROUND[10];
  const vegetated = [10, 20, 30, 40, 90, 95, 100].includes(lulc);
  const dryness = vegetated ? (1 - vegetation) * 0.55 : 0;
  return [new Color(a).lerp(DRY, dryness), new Color(b).lerp(DRY, dryness)];
}

export const rockColour = (lithology: number) => new Color(ROCK[lithology] ?? ROCK[-1]);

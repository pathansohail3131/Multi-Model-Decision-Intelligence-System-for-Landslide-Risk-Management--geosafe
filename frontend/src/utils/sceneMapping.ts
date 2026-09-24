import type { FactorName, InputFeatures, ModelMetadata, NumericFactorSpec, ThemeName } from '../types/api';
import type { SceneElement } from '../types/simulation';

/**
 * Scene parameters derived from the 14 conditioning factors.
 *
 * The scene is a schematic of the assessed 30 m cell and its hillslope. Some
 * quantities are drawn to scale (slope angle, distances, elevation); others
 * have no physical size at this scale and are drawn qualitatively, normalised
 * against the real training distribution returned by the API. SCENE_MAPPING
 * documents which is which, and the viewport shows it to the user.
 */
export interface SceneParams {
  slopeDeg: number;
  aspectDeg: number;
  elevation: number;
  planShape: number;
  profileShape: number;
  wetness: number;
  gully: number;
  streamOffset: number;
  rain: number;
  faultDistance: number;
  roadDistance: number;
  vegetation: number;
  isFlat: boolean;
  lithology: number;
  lulc: number;
}

export type NumericSceneKey = Exclude<keyof SceneParams, 'isFlat' | 'lithology' | 'lulc'>;

export const NUMERIC_SCENE_KEYS: NumericSceneKey[] = [
  'slopeDeg',
  'aspectDeg',
  'elevation',
  'planShape',
  'profileShape',
  'wetness',
  'gully',
  'streamOffset',
  'rain',
  'faultDistance',
  'roadDistance',
  'vegetation',
];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const rank = (v: number, lo: number, hi: number) => (hi > lo ? clamp((v - lo) / (hi - lo), 0, 1) : 0);

function stats(meta: ModelMetadata, name: FactorName) {
  return (meta.factors.find((f) => f.name === name) as NumericFactorSpec).training;
}

function symmetricShape(meta: ModelMetadata, name: FactorName, value: number) {
  const s = stats(meta, name);
  const scale = Math.max(Math.abs(s.p01), Math.abs(s.p99), 1e-9);
  return clamp(value / scale, -1, 1);
}

export function toSceneParams(inputs: InputFeatures, meta: ModelMetadata): SceneParams {
  const twi = stats(meta, 'TWI');
  const spi = stats(meta, 'SPI');
  const rain = stats(meta, 'rainfall');
  return {
    slopeDeg: clamp(inputs.slope, 0, 75),
    aspectDeg: inputs.aspect,
    elevation: Math.max(0, inputs.elevation),
    planShape: symmetricShape(meta, 'plan_curv', inputs.plan_curv),
    profileShape: symmetricShape(meta, 'profile_curv', inputs.profile_curv),
    wetness: rank(inputs.TWI, twi.p05, twi.p99),
    gully: rank(Math.log1p(Math.max(inputs.SPI, 0)), Math.log1p(spi.median), Math.log1p(spi.p99)),
    streamOffset: Math.max(0, inputs.dist_drainage),
    rain: rank(inputs.rainfall, rain.min, rain.max),
    faultDistance: Math.max(0, inputs.dist_fault),
    roadDistance: Math.max(0, inputs.dist_road),
    vegetation: rank(inputs.NDVI, 0.05, 0.88),
    isFlat: inputs.slope <= 0,
    lithology: Math.round(inputs.lithology),
    lulc: Math.round(inputs.LULC),
  };
}

export interface MappingRow {
  factor: FactorName;
  element: string;
  rule: string;
  scaled: 'to scale' | 'qualitative' | 'position schematic';
}

export const SCENE_MAPPING: MappingRow[] = [
  { factor: 'slope', element: 'Hillslope gradient', rule: 'True angle at the assessed cell, no vertical exaggeration', scaled: 'to scale' },
  { factor: 'aspect', element: 'Compass rose', rule: 'Downslope (towards the viewer) faces the aspect bearing', scaled: 'to scale' },
  { factor: 'elevation', element: 'Altitude label and sea level', rule: 'Terrain below 0 m a.s.l. floods to sea level', scaled: 'to scale' },
  { factor: 'plan_curv', element: 'Hollow or spur across the slope', rule: 'Normalised to the training range; negative drawn as a converging hollow', scaled: 'qualitative' },
  { factor: 'profile_curv', element: 'Concave or convex along the slope', rule: 'Normalised to the training range; negative drawn as concave', scaled: 'qualitative' },
  { factor: 'TWI', element: 'Soil wetness around the cell', rule: 'Darkening ranked between the 5th and 99th training percentile', scaled: 'qualitative' },
  { factor: 'SPI', element: 'Gully incision below the cell', rule: 'Depth ranked on log(SPI) above the training median', scaled: 'qualitative' },
  { factor: 'dist_drainage', element: 'Stream channel', rule: 'Offset across the slope by the distance; direction is not known', scaled: 'position schematic' },
  { factor: 'rainfall', element: 'Rain density', rule: 'Ranked across the training range; animation is illustrative', scaled: 'qualitative' },
  { factor: 'lithology', element: 'Rock strata on the cutaway', rule: 'Pattern per GLiM class', scaled: 'qualitative' },
  { factor: 'dist_fault', element: 'Lineament trace', rule: 'Placed at the distance; orientation is not known', scaled: 'position schematic' },
  { factor: 'dist_road', element: 'Road bench', rule: 'Upslope at the distance; side is not known', scaled: 'position schematic' },
  { factor: 'LULC', element: 'Land cover', rule: 'Trees, crops, buildings, bare ground or water per class', scaled: 'qualitative' },
  { factor: 'NDVI', element: 'Vegetation density', rule: 'Ranked between NDVI 0.05 and 0.88', scaled: 'qualitative' },
];

/** Which scene elements belong to which thematic model, for the model lens. */
export const THEME_ELEMENTS: Record<ThemeName, SceneElement[]> = {
  topographic: ['terrain', 'site'],
  hydrological: ['stream', 'rain', 'site'],
  geological: ['strata', 'fault', 'site'],
  anthropogenic: ['road', 'vegetation', 'site'],
};

/** Scene element a parameter control should bring into view. */
export const FACTOR_ELEMENT: Record<FactorName, SceneElement> = {
  elevation: 'terrain',
  slope: 'terrain',
  aspect: 'terrain',
  plan_curv: 'site',
  profile_curv: 'site',
  TWI: 'site',
  SPI: 'site',
  dist_drainage: 'stream',
  rainfall: 'rain',
  lithology: 'strata',
  dist_fault: 'fault',
  dist_road: 'road',
  LULC: 'vegetation',
  NDVI: 'vegetation',
};

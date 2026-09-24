import { createContext, useContext, type RefObject } from 'react';
import { Color } from 'three';
import type { SceneParams } from '../utils/sceneMapping';

/**
 * Mutable per-frame state shared by the scene's components.
 *
 * React state changes only when the user or the API changes something; the
 * values that animate between those changes live here and are read inside
 * useFrame, so a slider drag or a result reveal never re-renders the tree
 * sixty times a second.
 */
export interface SceneRuntime {
  params: SceneParams;
  target: SceneParams;
  animating: boolean;
  /** Incremented whenever params changed, so geometry knows to rebuild. */
  version: number;
  /** Result reveal, 0 = none, 1 = fully shown. */
  reveal: number;
  /** Illustrative failure-zone overlay, 0..1. */
  failure: number;
  resultColor: Color;
  markerColor: Color;
  /** Scan sweep amount and position (scene units along z). */
  scanAmount: number;
  scanZ: number;
  /** Pulse amount while waiting for the model. */
  pulse: number;
  time: number;
  /** Scene-unit heights maintained by the terrain: bottom of the block and top of the cell. */
  baseY: number;
  siteY: number;
}

export function createRuntime(params: SceneParams): SceneRuntime {
  return {
    params: { ...params },
    target: { ...params },
    animating: false,
    version: 1,
    reveal: 0,
    failure: 0,
    resultColor: new Color('#c9d1d9'),
    markerColor: new Color('#c9d1d9'),
    scanAmount: 0,
    scanZ: 0,
    pulse: 0,
    time: 0,
    baseY: -10,
    siteY: 0,
  };
}

export const RuntimeContext = createContext<RefObject<SceneRuntime> | null>(null);

export function useRuntime(): RefObject<SceneRuntime> {
  const runtime = useContext(RuntimeContext);
  if (!runtime) throw new Error('useRuntime must be used inside the simulation scene');
  return runtime;
}

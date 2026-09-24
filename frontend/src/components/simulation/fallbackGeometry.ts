import { HALF_M, heightAt, siteHeight } from '../../three/terrain/terrainModel';
import type { SceneParams } from '../../utils/sceneMapping';

/** Terrain sampling along the fall line for the 2D fallback. The terrain model imports no three.js. */
export const crossSection = {
  profile(params: SceneParams, count: number) {
    return Array.from({ length: count }, (_, i) => {
      const u = -HALF_M + (i / (count - 1)) * HALF_M * 2;
      return { u, h: heightAt(u, 0, params) };
    });
  },
  height: (u: number, params: SceneParams) => heightAt(u, 0, params),
  site: (params: SceneParams) => siteHeight(params),
};

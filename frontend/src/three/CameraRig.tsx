import { CameraControls, CameraControlsImpl } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import { useSimulationStore } from '../store/simulationStore';
import type { CameraMode, SceneElement } from '../types/simulation';
import { toSceneParams, type SceneParams } from '../utils/sceneMapping';
import {
  HALF,
  HALF_M,
  M_PER_UNIT,
  faultGeometry,
  faultSpan,
  heightAt,
  roadInBlock,
  streamCentre,
  streamInBlock,
} from './terrain/terrainModel';

type Pose = [number, number, number, number, number, number];

const y = (u: number, v: number, p: SceneParams) => heightAt(u, v, p) / M_PER_UNIT;

function overview(p: SceneParams, compact: boolean): Pose {
  const site = y(0, 0, p);
  const scale = compact ? 1.3 : 1;
  return [60 * scale, site + 46 * scale, 80 * scale, -3, site * 0.5 - 6, 1];
}

function focusPose(element: SceneElement | null, p: SceneParams, compact: boolean): Pose {
  const site = y(0, 0, p);
  switch (element) {
    case 'terrain':
      // Side-on, so the slope angle reads directly.
      return [66, site + 12, 6, 0, site - 4, 2];
    case 'site':
      return [11, site + 9, 17, 0, site, 1.5];
    case 'stream': {
      if (!streamInBlock(p)) return overview(p, compact);
      const u = 120;
      const v = streamCentre(u, p);
      const x = v / M_PER_UNIT;
      const h = y(u, v, p);
      return [x + 14, h + 13, u / M_PER_UNIT + 20, x, h, u / M_PER_UNIT];
    }
    case 'road': {
      if (!roadInBlock(p)) return [8, site + 40, 56, 0, site, -HALF * 0.7];
      const z = -p.roadDistance / M_PER_UNIT;
      const h = y(-p.roadDistance, 0, p);
      return [18, h + 13, z + 22, 0, h, z];
    }
    case 'fault': {
      const span = faultSpan(p);
      if (!span) return [-40, site + 38, 50, -HALF * 0.6, site, -HALF * 0.6];
      const { foot, direction } = faultGeometry(p);
      const s = Math.max(span[0], Math.min(span[1], 0));
      const v = foot[0] + direction[0] * s;
      const u = foot[1] + direction[1] * s;
      const h = y(u, v, p);
      return [v / M_PER_UNIT + 14, h + 15, u / M_PER_UNIT + 22, v / M_PER_UNIT, h, u / M_PER_UNIT];
    }
    case 'strata': {
      // Front cutaway face, far enough back to keep the terrain edge in view.
      const front = y(HALF_M, 0, p);
      return [26, front + 8, HALF + 62, 0, front - 8, HALF - 4];
    }
    case 'vegetation':
      return [24, site + 15, 32, 0, site, 2];
    case 'rain':
      return [44, site + 50, 66, 0, site + 6, 0];
    default:
      return overview(p, compact);
  }
}

function poseFor(mode: CameraMode, element: SceneElement | null, p: SceneParams, compact: boolean): Pose | null {
  const site = y(0, 0, p);
  switch (mode) {
    case 'overview':
      return overview(p, compact);
    case 'configure':
      return focusPose(element, p, compact);
    case 'processing': {
      const [px, py, pz, tx, ty, tz] = overview(p, compact);
      return [px * 1.08, py + 6, pz * 1.08, tx, ty, tz];
    }
    case 'result':
      return compact ? [16, site + 15, 26, 0, site + 1, 3] : [14, site + 11, 22, 0, site + 1, 3];
    default:
      return null;
  }
}

export function CameraRig({ reducedMotion, compact, touch }: { reducedMotion: boolean; compact: boolean; touch: boolean }) {
  const controls = useRef<CameraControlsImpl>(null);
  const mode = useSimulationStore((s) => s.cameraMode);
  const focusElement = useSimulationStore((s) => s.focusElement);
  const resetToken = useSimulationStore((s) => s.cameraResetToken);
  const inputs = useSimulationStore((s) => s.inputs);
  const metadata = useSimulationStore((s) => s.metadata);
  const resultId = useSimulationStore((s) => s.result?.id);
  const setCameraMode = useSimulationStore((s) => s.setCameraMode);
  // Derived from the store rather than the runtime, which parents update after this effect runs.
  const target = useMemo(() => (inputs && metadata ? toSceneParams(inputs, metadata) : null), [inputs, metadata]);

  useEffect(() => {
    const instance = controls.current;
    if (!instance) return;
    const onUserStart = () => {
      if (useSimulationStore.getState().cameraMode !== 'processing') setCameraMode('inspect');
    };
    instance.addEventListener('controlstart', onUserStart);
    return () => instance.removeEventListener('controlstart', onUserStart);
  }, [setCameraMode]);

  // Move whenever the mode or focused element changes, or the terrain changes while configuring.
  const poseTarget = mode === 'configure' || mode === 'result' ? target : null;
  const initial = useRef(true);
  useEffect(() => {
    const instance = controls.current;
    if (!instance || !target) return;
    const params = poseTarget ?? target;
    const timer = window.setTimeout(() => {
      const pose = poseFor(mode, focusElement, params, compact);
      if (pose) void instance.setLookAt(...pose, !reducedMotion && !initial.current);
      initial.current = false;
    }, mode === 'configure' ? 120 : 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, focusElement, resetToken, resultId, poseTarget, compact, reducedMotion, target === null]);

  useFrame((_, delta) => {
    if (mode === 'processing' && !reducedMotion && controls.current) {
      controls.current.rotate(delta * 0.1, 0, false);
    }
  });

  return (
    <CameraControls
      ref={controls}
      makeDefault
      smoothTime={reducedMotion ? 0.01 : 0.7}
      minDistance={6}
      maxDistance={170}
      maxPolarAngle={Math.PI * 0.49}
      dollyToCursor
      touches={
        touch
          ? {
              one: CameraControlsImpl.ACTION.TOUCH_ROTATE,
              two: CameraControlsImpl.ACTION.TOUCH_DOLLY_TRUCK,
              three: CameraControlsImpl.ACTION.NONE,
            }
          : undefined
      }
    />
  );
}

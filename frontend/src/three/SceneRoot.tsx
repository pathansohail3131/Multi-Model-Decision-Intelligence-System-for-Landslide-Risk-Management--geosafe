import { useFrame } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import { Color } from 'three';
import { useResultIsStale, useSimulationStore } from '../store/simulationStore';
import type { InputFeatures, ModelMetadata } from '../types/api';
import { CLASS_COLOURS, DECISION_COLOURS } from '../utils/colors';
import { NUMERIC_SCENE_KEYS, toSceneParams, type NumericSceneKey } from '../utils/sceneMapping';
import { CameraRig } from './CameraRig';
import { Compass, ScaleBar } from './Compass';
import { FeatureOverlays } from './FeatureOverlays';
import { LandCover } from './LandCover';
import { Lighting } from './Lighting';
import { Rain } from './Rain';
import { RuntimeContext, createRuntime } from './runtime';
import { ScanSweep, SiteMarker } from './SiteMarker';
import { TerrainBlock } from './terrain/TerrainBlock';
import { HALF } from './terrain/terrainModel';

const EPSILON: Record<NumericSceneKey, number> = {
  slopeDeg: 0.01,
  aspectDeg: 0.05,
  elevation: 0.05,
  planShape: 0.0005,
  profileShape: 0.0005,
  wetness: 0.001,
  gully: 0.001,
  streamOffset: 0.05,
  rain: 0.001,
  faultDistance: 0.05,
  roadDistance: 0.05,
  vegetation: 0.001,
};

const NEUTRAL = new Color('#d0d7de');

export interface SceneQuality {
  segments: number;
  vegetationDensity: number;
  rainCapacity: number;
  shadows: boolean;
}

interface SceneRootProps {
  inputs: InputFeatures;
  metadata: ModelMetadata;
  quality: SceneQuality;
  reducedMotion: boolean;
  compact: boolean;
  touch: boolean;
}

export function SceneRoot({ inputs, metadata, quality, reducedMotion, compact, touch }: SceneRootProps) {
  const target = useMemo(() => toSceneParams(inputs, metadata), [inputs, metadata]);
  const runtime = useRef(createRuntime(target));
  const phase = useSimulationStore((s) => s.phase);
  const result = useSimulationStore((s) => s.result);
  const stale = useResultIsStale();

  useEffect(() => {
    const rt = runtime.current;
    rt.target = target;
    // Categorical properties switch at once; numeric ones glide in useFrame.
    rt.params.isFlat = target.isFlat;
    rt.params.lithology = target.lithology;
    rt.params.lulc = target.lulc;
    rt.animating = true;
    rt.version += 1;
  }, [target]);

  const resultColor = useMemo(() => {
    const prediction = result?.response.prediction;
    return prediction ? new Color(CLASS_COLOURS[prediction.class.key]) : NEUTRAL.clone();
  }, [result]);
  const markerColor = useMemo(() => {
    const prediction = result?.response.prediction;
    if (!prediction) return NEUTRAL.clone();
    return new Color(prediction.decision ? DECISION_COLOURS[prediction.decision.key] : CLASS_COLOURS[prediction.class.key]);
  }, [result]);

  useFrame((state, delta) => {
    const rt = runtime.current;
    const dt = Math.min(delta, 0.1);
    rt.time = state.clock.elapsedTime;

    if (rt.animating) {
      const k = reducedMotion ? 1 : 1 - Math.exp(-dt * 6.5);
      let settled = true;
      for (const key of NUMERIC_SCENE_KEYS) {
        const goal = rt.target[key];
        let current = rt.params[key];
        if (key === 'aspectDeg') {
          const diff = ((((goal - current) % 360) + 540) % 360) - 180;
          current += diff * k;
          if (Math.abs(diff) < EPSILON[key]) current = goal;
          else settled = false;
        } else {
          current += (goal - current) * k;
          if (Math.abs(goal - current) < EPSILON[key]) current = goal;
          else settled = false;
        }
        rt.params[key] = current;
      }
      rt.version += 1;
      rt.animating = !settled;
    }

    const running = phase === 'simulating' || phase === 'predicting' || phase === 'validating';
    const ease = reducedMotion ? 1 : 1 - Math.exp(-dt * 3.2);
    const hasResult = Boolean(result);
    const revealGoal = hasResult ? (stale ? 0.3 : 1) : 0;
    rt.reveal += (revealGoal - rt.reveal) * ease;
    const failureGoal = hasResult && result?.response.prediction.above_threshold && !stale ? 1 : 0;
    rt.failure += (failureGoal - rt.failure) * (reducedMotion ? 1 : 1 - Math.exp(-dt * 1.6));
    rt.resultColor.lerp(resultColor, ease);
    rt.markerColor.lerp(markerColor, ease);

    const scanning = phase === 'simulating' || phase === 'predicting';
    rt.scanAmount += ((scanning && !reducedMotion ? 1 : 0) - rt.scanAmount) * (1 - Math.exp(-dt * 5));
    if (rt.scanAmount > 0.001) {
      rt.scanZ = -HALF + ((state.clock.elapsedTime * 38) % (HALF * 2));
    }
    rt.pulse += ((running ? 1 : 0) - rt.pulse) * (1 - Math.exp(-dt * 5));
  });

  return (
    <RuntimeContext.Provider value={runtime}>
      <Lighting shadows={quality.shadows} />
      <CameraRig reducedMotion={reducedMotion} compact={compact} touch={touch} />
      <TerrainBlock segments={quality.segments} />
      <FeatureOverlays />
      <LandCover density={quality.vegetationDensity} />
      <Rain capacity={quality.rainCapacity} animate={!reducedMotion} />
      <SiteMarker />
      <ScanSweep />
      <Compass />
      <ScaleBar />
      <mesh rotation-x={-Math.PI / 2} position={[0, -60, 0]} receiveShadow>
        <circleGeometry args={[160, 48]} />
        <meshStandardMaterial color="#12161a" roughness={1} />
      </mesh>
    </RuntimeContext.Provider>
  );
}

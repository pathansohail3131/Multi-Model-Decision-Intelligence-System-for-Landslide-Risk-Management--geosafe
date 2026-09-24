import { AdaptiveDpr, PerformanceMonitor } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { useState } from 'react';
import { useCoarsePointer, useCompactLayout, useReducedMotion } from '../hooks/useMediaQuery';
import { useSimulationStore } from '../store/simulationStore';
import { SceneRoot, type SceneQuality } from './SceneRoot';

const FULL: SceneQuality = { segments: 140, vegetationDensity: 1, rainCapacity: 1500, shadows: true };
const REDUCED: SceneQuality = { segments: 96, vegetationDensity: 0.5, rainCapacity: 600, shadows: false };

export default function SimulationCanvas() {
  const inputs = useSimulationStore((s) => s.inputs);
  const metadata = useSimulationStore((s) => s.metadata);
  const setWebglFailed = useSimulationStore((s) => s.setWebglFailed);
  const setHovered = useSimulationStore((s) => s.setHovered);
  const compact = useCompactLayout();
  const touch = useCoarsePointer();
  const reducedMotion = useReducedMotion();
  const [degraded, setDegraded] = useState(false);
  const quality = compact || touch || degraded ? REDUCED : FULL;

  if (!inputs || !metadata) return null;

  return (
    <Canvas
      className="simulation-canvas"
      shadows={quality.shadows}
      dpr={compact ? [1, 1.5] : [1, 1.75]}
      camera={{ position: [60, 44, 76], fov: 36, near: 1, far: 600 }}
      gl={{ antialias: true, powerPreference: 'high-performance' }}
      onPointerMissed={() => setHovered(null)}
      onCreated={({ gl }) => {
        const canvas = gl.domElement;
        canvas.addEventListener('webglcontextlost', (event) => {
          // preventDefault lets the browser hand the context back later.
          event.preventDefault();
          console.error('[scene] WebGL context lost — falling back to 2D until it is restored');
          setWebglFailed();
        });
        canvas.addEventListener('webglcontextrestored', () => {
          console.info('[scene] WebGL context restored — rebuilding the scene');
          useSimulationStore.getState().retryWebgl();
        });
      }}
    >
      <color attach="background" args={['#0f1317']} />
      <fog attach="fog" args={['#0f1317', 120, 300]} />
      <PerformanceMonitor onDecline={() => setDegraded(true)} flipflops={2} />
      <AdaptiveDpr pixelated={false} />
      <SceneRoot
        inputs={inputs}
        metadata={metadata}
        quality={quality}
        reducedMotion={reducedMotion}
        compact={compact}
        touch={touch}
      />
    </Canvas>
  );
}

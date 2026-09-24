import { Html, Line } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import { Color, type Group, type Mesh, type MeshStandardMaterial, DoubleSide } from 'three';
import type { Line2 } from 'three-stdlib';
import { useResultIsStale, useSimulationStore } from '../store/simulationStore';
import { formatProbability } from '../utils/format';
import { useRuntime } from './runtime';
import { CELL_M, HALF, M_PER_UNIT, heightAt } from './terrain/terrainModel';

const OUTLINE_STEPS = 6;
const HALF_CELL = CELL_M / 2;

function outlinePoints(): [number, number][] {
  const corners: [number, number][] = [
    [-HALF_CELL, -HALF_CELL],
    [HALF_CELL, -HALF_CELL],
    [HALF_CELL, HALF_CELL],
    [-HALF_CELL, HALF_CELL],
  ];
  const points: [number, number][] = [];
  for (let c = 0; c < 4; c += 1) {
    const [v0, u0] = corners[c];
    const [v1, u1] = corners[(c + 1) % 4];
    for (let s = 0; s < OUTLINE_STEPS; s += 1) {
      const t = s / OUTLINE_STEPS;
      points.push([v0 + (v1 - v0) * t, u0 + (u1 - u0) * t]);
    }
  }
  points.push(corners[0]);
  return points;
}

/** The assessed 30 m cell: its true-scale outline, a survey pole and the result flag. */
export function SiteMarker() {
  const runtime = useRuntime();
  const result = useSimulationStore((s) => s.result);
  const phase = useSimulationStore((s) => s.phase);
  const setHovered = useSimulationStore((s) => s.setHovered);
  const setActiveFactor = useSimulationStore((s) => s.setActiveFactor);
  const stale = useResultIsStale();
  const outline = useMemo(outlinePoints, []);
  const line = useRef<Line2>(null);
  const pole = useRef<Group>(null);
  const flag = useRef<Mesh>(null);
  const built = useRef(-1);
  const neutral = useMemo(() => new Color('#d0d7de'), []);

  useFrame((state) => {
    const rt = runtime.current;
    if (built.current !== rt.version) {
      built.current = rt.version;
      const positions: number[] = [];
      for (const [v, u] of outline) {
        positions.push(v / M_PER_UNIT, heightAt(u, v, rt.params) / M_PER_UNIT + 0.08, u / M_PER_UNIT);
      }
      line.current?.geometry.setPositions(positions);
      pole.current?.position.set(0, rt.siteY, 0);
    }
    if (flag.current) {
      const material = flag.current.material as MeshStandardMaterial;
      material.color.copy(neutral).lerp(rt.markerColor, Math.min(1, rt.reveal * 1.4));
      material.emissive.copy(material.color).multiplyScalar(0.18 + rt.pulse * 0.25 * (0.5 + 0.5 * Math.sin(state.clock.elapsedTime * 6)));
      flag.current.rotation.y = Math.sin(state.clock.elapsedTime * 1.3) * 0.12;
    }
    if (line.current) {
      const material = line.current.material as unknown as { color: Color; linewidth: number };
      material.color.copy(neutral).lerp(rt.resultColor, rt.reveal);
    }
  });

  const prediction = result?.response.prediction;
  let caption = 'Assessed cell · 30 m';
  if (phase === 'predicting' || phase === 'simulating') caption = 'Running model…';
  else if (prediction && !stale) caption = `${prediction.class.label} · ${formatProbability(prediction.susceptibility)}`;
  else if (prediction && stale) caption = 'Parameters changed · re-run';

  return (
    <group>
      <Line ref={line} points={[[0, 0, 0], [1, 0, 0]]} lineWidth={2.2} color="#d0d7de" />
      <group
        ref={pole}
        onPointerOver={(event) => {
          event.stopPropagation();
          setHovered('site');
        }}
        onPointerOut={() => setHovered(null)}
        onClick={(event) => {
          event.stopPropagation();
          setActiveFactor('slope');
        }}
      >
        <mesh position={[0, 1.7, 0]} castShadow>
          <cylinderGeometry args={[0.05, 0.06, 3.4, 8]} />
          <meshStandardMaterial color="#e7ebef" roughness={0.5} />
        </mesh>
        <mesh ref={flag} position={[0.55, 3.05, 0]} castShadow>
          <planeGeometry args={[1.05, 0.62]} />
          <meshStandardMaterial color="#d0d7de" roughness={0.6} side={DoubleSide} />
        </mesh>
        <Html position={[0, 4.4, 0]} center zIndexRange={[30, 0]} pointerEvents="none">
          <span className={`scene-label scene-label--site${prediction && !stale ? ' is-result' : ''}`}>{caption}</span>
        </Html>
      </group>
    </group>
  );
}

/** Translucent plane sweeping down the block while the scenario is sampled. */
export function ScanSweep() {
  const runtime = useRuntime();
  const mesh = useRef<Mesh>(null);
  useFrame(() => {
    const rt = runtime.current;
    if (!mesh.current) return;
    mesh.current.visible = rt.scanAmount > 0.01;
    mesh.current.position.z = rt.scanZ;
    mesh.current.position.y = (rt.baseY + 30) / 2;
    (mesh.current.material as MeshStandardMaterial).opacity = 0.1 * rt.scanAmount;
  });
  return (
    <mesh ref={mesh} visible={false}>
      <planeGeometry args={[HALF * 2, 60]} />
      <meshStandardMaterial color="#8fd3e8" transparent opacity={0} depthWrite={false} side={DoubleSide} emissive="#8fd3e8" emissiveIntensity={0.6} />
    </mesh>
  );
}

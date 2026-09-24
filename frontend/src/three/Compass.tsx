import { Html } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import { BufferGeometry, Float32BufferAttribute, type Group } from 'three';
import { useSimulationStore } from '../store/simulationStore';
import { compassPoint, formatNumber } from '../utils/format';
import { useRuntime } from './runtime';
import { HALF } from './terrain/terrainModel';

const DEG = Math.PI / 180;

/**
 * Compass rose on the plinth. The block is always drawn with downslope towards
 * the viewer, so aspect is shown by turning the compass: the rose's bearing
 * that points at the viewer is the slope's aspect.
 */
export function Compass() {
  const runtime = useRuntime();
  const aspect = useSimulationStore((s) => s.inputs?.aspect ?? 0);
  const flat = useSimulationStore((s) => (s.inputs?.slope ?? 1) <= 0);
  const plinthGroup = useRef<Group>(null);
  const rose = useRef<Group>(null);

  const needle = useMemo(() => {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new Float32BufferAttribute([0, 0, -2.3, -0.45, 0, 0, 0.45, 0, 0], 3));
    geometry.computeVertexNormals();
    return geometry;
  }, []);

  useFrame((_, delta) => {
    const rt = runtime.current;
    // Right-hand plinth edge, near the front: the side the default cameras look from.
    plinthGroup.current?.position.set(HALF + 3.8, rt.baseY + 0.02, HALF - 4);
    if (rose.current) {
      // Canonical north is -z; rotate so the aspect bearing points at +z (the viewer).
      const target = (rt.params.aspectDeg - 180) * DEG;
      let diff = target - rose.current.rotation.y;
      diff = Math.atan2(Math.sin(diff), Math.cos(diff));
      rose.current.rotation.y += diff * (1 - Math.exp(-delta * 6));
    }
  });

  return (
    <group ref={plinthGroup}>
      <group ref={rose} scale={0.72}>
        <mesh rotation-x={-Math.PI / 2}>
          <ringGeometry args={[2.35, 2.55, 48]} />
          <meshBasicMaterial color="#8a96a3" />
        </mesh>
        <mesh geometry={needle}>
          <meshBasicMaterial color="#d9793a" side={2} />
        </mesh>
        <mesh geometry={needle} rotation-y={Math.PI}>
          <meshBasicMaterial color="#5b6672" side={2} />
        </mesh>
        <group position={[0, 0.2, -3.2]}>
          <Html center zIndexRange={[10, 0]} pointerEvents="none">
            <span className="scene-compass-n">N</span>
          </Html>
        </group>
      </group>
      <group position={[0.6, 0.2, -5.2]}>
        <Html center zIndexRange={[10, 0]} pointerEvents="none">
          <span className="scene-label scene-label--muted">
            {flat ? 'Flat cell · aspect not defined' : `Downslope faces ${formatNumber(aspect)}° ${compassPoint(aspect)}`}
          </span>
        </Html>
      </group>
    </group>
  );
}

/** 200 m scale bar on the plinth's front edge. */
export function ScaleBar() {
  const runtime = useRuntime();
  const group = useRef<Group>(null);
  useFrame(() => {
    group.current?.position.set(HALF - 16, runtime.current.baseY + 0.04, HALF + 3.8);
  });
  return (
    <group ref={group}>
      <mesh rotation-x={-Math.PI / 2} position={[5, 0, 0]}>
        <planeGeometry args={[10, 0.22]} />
        <meshBasicMaterial color="#c9d1d9" />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position={[2.5, 0.01, 0]}>
        <planeGeometry args={[5, 0.22]} />
        <meshBasicMaterial color="#39424c" />
      </mesh>
      <group position={[5, 0.2, 1.3]}>
        <Html center zIndexRange={[10, 0]} pointerEvents="none">
          <span className="scene-label scene-label--muted">200 m · block 1.2 km</span>
        </Html>
      </group>
    </group>
  );
}

import { useFrame } from '@react-three/fiber';
import { useEffect, useMemo } from 'react';
import { BoxGeometry, InstancedMesh, Matrix4, MeshBasicMaterial } from 'three';
import { useSimulationStore } from '../store/simulationStore';
import { useRuntime } from './runtime';
import { HALF, random01 } from './terrain/terrainModel';

const TOP = 42;
const FALL = 60;

/**
 * Rain density follows the site's annual rainfall rank. Annual rainfall is a
 * climatological mean, so the animation is illustrative only — it does not
 * depict a storm the model knows about.
 */
export function Rain({ capacity, animate }: { capacity: number; animate: boolean }) {
  const runtime = useRuntime();
  const lens = useSimulationStore((s) => s.lens);

  const { mesh, offsets } = useMemo(() => {
    const geometry = new BoxGeometry(0.02, 0.9, 0.02);
    const material = new MeshBasicMaterial({ color: '#a9bfd0', transparent: true, opacity: 0.2, depthWrite: false });
    const instanced = new InstancedMesh(geometry, material, capacity);
    instanced.frustumCulled = false;
    const values = new Float32Array(capacity * 3);
    for (let i = 0; i < capacity; i += 1) {
      values[i * 3] = (random01(i, 21) * 2 - 1) * HALF;
      values[i * 3 + 1] = random01(i, 22) * FALL;
      values[i * 3 + 2] = (random01(i, 23) * 2 - 1) * HALF;
    }
    return { mesh: instanced, offsets: values };
  }, [capacity]);

  useEffect(
    () => () => {
      mesh.geometry.dispose();
      (mesh.material as MeshBasicMaterial).dispose();
      mesh.dispose();
    },
    [mesh],
  );

  const matrix = useMemo(() => new Matrix4(), []);

  useFrame((state, delta) => {
    const rt = runtime.current;
    const count = Math.round(capacity * (0.12 + 0.88 * rt.params.rain));
    mesh.count = count;
    const material = mesh.material as MeshBasicMaterial;
    // Close-up views put streaks right in front of the lens; fade rain out as the camera approaches.
    const distance = state.camera.position.length();
    const nearFade = Math.min(1, Math.max(0, (distance - 30) / 40));
    const target = (lens === 'all' ? 0.2 : lens === 'hydrological' ? 0.42 : 0.05) * nearFade;
    material.opacity += (target - material.opacity) * (1 - Math.exp(-delta * 6));
    mesh.visible = material.opacity > 0.01;

    const t = animate ? state.clock.elapsedTime : 0;
    const drift = 0.18;
    for (let i = 0; i < count; i += 1) {
      const fall = (offsets[i * 3 + 1] + t * 26) % FALL;
      const y = TOP - fall;
      const x = offsets[i * 3] + drift * fall * 0.1;
      matrix.makeRotationZ(-drift * 0.1);
      matrix.setPosition(x > HALF ? x - HALF * 2 : x, y, offsets[i * 3 + 2]);
      mesh.setMatrixAt(i, matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  });

  return <primitive object={mesh} />;
}

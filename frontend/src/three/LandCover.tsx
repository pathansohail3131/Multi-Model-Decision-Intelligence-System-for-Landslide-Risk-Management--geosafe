import { useFrame } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import {
  BoxGeometry,
  Color,
  ConeGeometry,
  CylinderGeometry,
  DodecahedronGeometry,
  IcosahedronGeometry,
  InstancedMesh,
  Matrix4,
  MeshStandardMaterial,
  Quaternion,
  Vector3,
} from 'three';
import { useSimulationStore } from '../store/simulationStore';
import type { SceneParams } from '../utils/sceneMapping';
import { useRuntime } from './runtime';
import {
  HALF,
  HALF_M,
  M_PER_UNIT,
  ROAD_HALF_WIDTH_M,
  heightAt,
  random01,
  roadInBlock,
  seaLevel,
  streamCentre,
  streamInBlock,
  waterLevel,
} from './terrain/terrainModel';

const GRID = 72;

/** Instances per kind, for one land-cover class, as a density per candidate point. */
interface CoverRecipe {
  trees: number;
  shrubs: number;
  tufts: number;
  buildings: number;
  rocks: number;
  reeds: number;
  mangroves: number;
}

function recipe(p: SceneParams): CoverRecipe {
  const v = p.vegetation;
  const none: CoverRecipe = { trees: 0, shrubs: 0, tufts: 0, buildings: 0, rocks: 0, reeds: 0, mangroves: 0 };
  switch (p.lulc) {
    case 10:
      return { ...none, trees: 0.04 + 0.2 * v, shrubs: 0.04 };
    case 20:
      return { ...none, shrubs: 0.08 + 0.18 * v, trees: 0.01 };
    case 30:
      return { ...none, tufts: 0.14 + 0.16 * v, trees: 0.008 };
    case 40:
      return { ...none, trees: 0.008, tufts: 0.015 };
    case 50:
      return { ...none, buildings: 0.16, trees: 0.02 + 0.03 * v };
    case 60:
      return { ...none, rocks: 0.08, shrubs: 0.02 };
    case 80:
      return { ...none, trees: 0.03 * v, shrubs: 0.02 };
    case 90:
      return { ...none, reeds: 0.25, shrubs: 0.03 };
    case 95:
      return { ...none, mangroves: 0.3 + 0.15 * v };
    default:
      return { ...none, tufts: 0.05 };
  }
}

const matrix = new Matrix4();
const quaternion = new Quaternion();
const scale = new Vector3();
const position = new Vector3();
const up = new Vector3(0, 1, 0);
const colour = new Color();

function makeMesh(geometry: THREE_Geometry, material: MeshStandardMaterial, capacity: number, castShadow: boolean) {
  const mesh = new InstancedMesh(geometry, material, capacity);
  mesh.castShadow = castShadow;
  mesh.receiveShadow = true;
  mesh.count = 0;
  mesh.frustumCulled = false;
  return mesh;
}
type THREE_Geometry = ConstructorParameters<typeof InstancedMesh>[0];

export function LandCover({ density }: { density: number }) {
  const runtime = useRuntime();
  const lens = useSimulationStore((s) => s.lens);
  const capacity = Math.round(GRID * GRID * Math.min(1, density));

  const meshes = useMemo(() => {
    const crown = new IcosahedronGeometry(1, 2);
    crown.translate(0, 0.4, 0);
    const trunk = new CylinderGeometry(0.035, 0.055, 1, 6);
    trunk.translate(0, 0.5, 0);
    const box = new BoxGeometry(1, 1, 1);
    box.translate(0, 0.5, 0);
    const rock = new DodecahedronGeometry(1, 0);
    const reed = new ConeGeometry(0.05, 1, 4);
    reed.translate(0, 0.5, 0);
    const tuft = new ConeGeometry(0.18, 0.35, 5);
    tuft.translate(0, 0.17, 0);

    const foliage = new MeshStandardMaterial({ roughness: 0.95 });
    return {
      crowns: makeMesh(crown, foliage, capacity * 2, true),
      trunks: makeMesh(trunk, new MeshStandardMaterial({ color: '#4a3a2c', roughness: 1 }), capacity, true),
      buildings: makeMesh(box, new MeshStandardMaterial({ roughness: 0.75 }), 260, true),
      rocks: makeMesh(rock, new MeshStandardMaterial({ roughness: 0.95, flatShading: true }), 400, true),
      reeds: makeMesh(reed, new MeshStandardMaterial({ color: '#8c8f55', roughness: 1 }), capacity, false),
      tufts: makeMesh(tuft, new MeshStandardMaterial({ color: '#9aa35e', roughness: 1, flatShading: true }), capacity, false),
    };
  }, [capacity]);

  useEffect(
    () => () => {
      for (const mesh of Object.values(meshes)) {
        mesh.geometry.dispose();
        (mesh.material as MeshStandardMaterial).dispose();
        mesh.dispose();
      }
    },
    [meshes],
  );

  const built = useRef(-1);
  useEffect(() => {
    built.current = -1;
  }, [meshes]);

  useFrame((_, delta) => {
    const rt = runtime.current;
    if (built.current !== rt.version) {
      built.current = rt.version;
      populate(rt.params);
    }
    const target = lens === 'all' || lens === 'anthropogenic' ? 1 : 0.38;
    const ease = 1 - Math.exp(-delta * 6);
    for (const mesh of Object.values(meshes)) {
      const material = mesh.material as MeshStandardMaterial;
      material.userData.base ??= material.color.clone();
      const current = (material.userData.scalar as number | undefined) ?? 1;
      const next = current + (target - current) * ease;
      material.userData.scalar = next;
      material.color.copy(material.userData.base as Color).multiplyScalar(next);
    }
  });

  function populate(p: SceneParams) {
    const r = recipe(p);
    const water = Math.max(seaLevel(p) ?? -Infinity, waterLevel(p) ?? -Infinity);
    const counts = { crowns: 0, trunks: 0, buildings: 0, rocks: 0, reeds: 0, tufts: 0 };
    const step = (HALF_M * 2) / GRID;
    const leafBase = new Color('#2f5a2c').lerp(new Color('#7c7a44'), (1 - p.vegetation) * 0.6);

    for (let gz = 0; gz < GRID; gz += 1) {
      for (let gx = 0; gx < GRID; gx += 1) {
        const i = gz * GRID + gx;
        if (random01(i, 3) > density) continue;
        const v = -HALF_M + (gx + random01(i, 1)) * step;
        const u = -HALF_M + (gz + random01(i, 2)) * step;
        if (Math.abs(v) > HALF_M - 8 || Math.abs(u) > HALF_M - 8) continue;
        if (Math.hypot(u, v) < 26) continue; // keep the assessed cell clear
        if (roadInBlock(p) && Math.abs(u + p.roadDistance) < ROAD_HALF_WIDTH_M + 10) continue;
        if (streamInBlock(p) && Math.abs(v - streamCentre(u, p)) < 16) continue;
        const h = heightAt(u, v, p);
        const pick = random01(i, 4);
        const size = 0.75 + random01(i, 5) * 0.5;
        const yaw = random01(i, 6) * Math.PI * 2;
        const x = v / M_PER_UNIT;
        const z = u / M_PER_UNIT;
        const y = h / M_PER_UNIT;
        const underwater = h < water;

        let threshold = 0;
        if (!underwater && pick < (threshold += r.trees)) {
          const height = 0.95 * size;
          const crownRadius = 0.3 * size;
          quaternion.setFromAxisAngle(up, yaw);
          matrix.compose(position.set(x, y, z), quaternion, scale.set(1, height, 1));
          meshes.trunks.setMatrixAt(counts.trunks++, matrix);
          matrix.compose(position.set(x, y + height * 0.95, z), quaternion, scale.set(crownRadius, crownRadius * 1.05, crownRadius));
          meshes.crowns.setMatrixAt(counts.crowns, matrix);
          meshes.crowns.setColorAt(counts.crowns++, colour.copy(leafBase).offsetHSL(0, 0, (random01(i, 7) - 0.5) * 0.08));
          continue;
        }
        if (!underwater && pick < (threshold += r.shrubs)) {
          const radius = 0.2 * size;
          matrix.compose(position.set(x, y - 0.02, z), quaternion.setFromAxisAngle(up, yaw), scale.set(radius, radius * 0.75, radius));
          meshes.crowns.setMatrixAt(counts.crowns, matrix);
          meshes.crowns.setColorAt(counts.crowns++, colour.copy(leafBase).offsetHSL(0.02, -0.05, 0.04));
          continue;
        }
        if (!underwater && pick < (threshold += r.tufts)) {
          matrix.compose(position.set(x, y, z), quaternion.setFromAxisAngle(up, yaw), scale.set(size, size, size));
          meshes.tufts.setMatrixAt(counts.tufts++, matrix);
          continue;
        }
        if (!underwater && pick < (threshold += r.buildings) && Math.hypot(u, v) < 360 && counts.buildings < 260) {
          const width = (0.32 + random01(i, 8) * 0.34) * size;
          const depth = (0.3 + random01(i, 9) * 0.28) * size;
          const height = 0.22 + random01(i, 10) * 0.3;
          matrix.compose(position.set(x, y - 0.05, z), quaternion.setFromAxisAngle(up, Math.round(yaw / (Math.PI / 2)) * (Math.PI / 2)), scale.set(width, height, depth));
          meshes.buildings.setMatrixAt(counts.buildings, matrix);
          meshes.buildings.setColorAt(counts.buildings++, colour.set(random01(i, 11) > 0.7 ? '#a7836b' : '#c9c4ba').offsetHSL(0, 0, (random01(i, 12) - 0.5) * 0.1));
          continue;
        }
        if (pick < (threshold += r.rocks) && counts.rocks < 400) {
          const radius = 0.12 + random01(i, 13) * 0.25;
          matrix.compose(position.set(x, y + radius * 0.3, z), quaternion.setFromAxisAngle(up, yaw), scale.set(radius, radius * 0.7, radius));
          meshes.rocks.setMatrixAt(counts.rocks, matrix);
          meshes.rocks.setColorAt(counts.rocks++, colour.set('#8a8178').offsetHSL(0, 0, (random01(i, 14) - 0.5) * 0.12));
          continue;
        }
        if (pick < (threshold += r.reeds)) {
          matrix.compose(position.set(x, Math.max(y, water / M_PER_UNIT) - 0.1, z), quaternion.setFromAxisAngle(up, yaw), scale.set(1, 0.6 * size, 1));
          meshes.reeds.setMatrixAt(counts.reeds++, matrix);
          continue;
        }
        if (pick < (threshold += r.mangroves)) {
          const radius = 0.34 * size;
          const base = Math.max(y, water / M_PER_UNIT);
          matrix.compose(position.set(x, base + 0.15, z), quaternion.setFromAxisAngle(up, yaw), scale.set(radius, radius * 0.7, radius));
          meshes.crowns.setMatrixAt(counts.crowns, matrix);
          meshes.crowns.setColorAt(counts.crowns++, colour.set('#2d5238').offsetHSL(0, 0, (random01(i, 15) - 0.5) * 0.06));
        }
      }
    }

    for (const [key, mesh] of Object.entries(meshes) as [keyof typeof counts, InstancedMesh][]) {
      mesh.count = Math.min(counts[key], mesh.instanceMatrix.count);
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
  }

  return (
    <group>
      {Object.entries(meshes).map(([key, mesh]) => (
        <primitive key={key} object={mesh} />
      ))}
    </group>
  );
}

export const LAND_COVER_EXTENT = HALF;

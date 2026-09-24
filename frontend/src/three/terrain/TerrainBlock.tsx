import { useFrame, type ThreeEvent } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  type Mesh,
  MeshStandardMaterial,
  PlaneGeometry,
} from 'three';
import { useSimulationStore } from '../../store/simulationStore';
import type { Lens, SceneElement } from '../../types/simulation';
import { useRuntime } from '../runtime';
import { groundColours, rockColour, createTerrainMaterial } from './terrainMaterial';
import { HALF, M_PER_UNIT, heightAt, seaLevel, surfaceMasks, waterLevel } from './terrainModel';
import { STRATA_DEPTH_M, strataTexture } from './strataTextures';

const DIM_BY_LENS: Record<Lens, number> = {
  all: 0,
  topographic: 0,
  anthropogenic: 0.22,
  hydrological: 0.55,
  geological: 0.55,
};

const CONTOUR_BY_LENS: Record<Lens, number> = {
  all: 0.26,
  topographic: 0.85,
  anthropogenic: 0.12,
  hydrological: 0.12,
  geological: 0.12,
};

function createWallGeometry(segments: number) {
  const perEdge = segments + 1;
  const vertexCount = perEdge * 2 * 4;
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(vertexCount * 3), 3));
  geometry.setAttribute('uv', new BufferAttribute(new Float32Array(vertexCount * 2), 2));
  const indices: number[] = [];
  for (let edge = 0; edge < 4; edge += 1) {
    const offset = edge * perEdge * 2;
    for (let i = 0; i < segments; i += 1) {
      const a = offset + i * 2;
      const b = a + 1;
      const c = a + 2;
      const d = a + 3;
      indices.push(a, b, c, b, d, c);
    }
  }
  geometry.setIndex(indices);
  return geometry;
}

export function TerrainBlock({ segments }: { segments: number }) {
  const runtime = useRuntime();
  const lens = useSimulationStore((s) => s.lens);
  const setHovered = useSimulationStore((s) => s.setHovered);
  const setActiveFactor = useSimulationStore((s) => s.setActiveFactor);

  const { material, uniforms } = useMemo(createTerrainMaterial, []);
  const wallMaterial = useMemo(() => new MeshStandardMaterial({ roughness: 0.96, metalness: 0, side: DoubleSide }), []);
  const waterMaterial = useMemo(
    () =>
      new MeshStandardMaterial({
        color: '#2c5a70',
        roughness: 0.12,
        metalness: 0.05,
        transparent: true,
        opacity: 0.86,
      }),
    [],
  );

  const terrain = useMemo(() => {
    const geometry = new PlaneGeometry(HALF * 2, HALF * 2, segments, segments);
    geometry.rotateX(-Math.PI / 2);
    const count = geometry.attributes.position.count;
    for (const name of ['aWet', 'aRoad', 'aChannel', 'aFault']) {
      geometry.setAttribute(name, new BufferAttribute(new Float32Array(count), 1));
    }
    return geometry;
  }, [segments]);
  const walls = useMemo(() => createWallGeometry(segments), [segments]);

  const plinth = useRef<Mesh>(null);
  const sea = useRef<Mesh>(null);
  const lake = useRef<Mesh>(null);
  const built = useRef({ version: -1, lithology: Number.NaN });

  useEffect(() => {
    built.current.version = -1;
  }, [terrain, walls]);

  useEffect(
    () => () => {
      terrain.dispose();
      walls.dispose();
      material.dispose();
      wallMaterial.dispose();
      waterMaterial.dispose();
    },
    [terrain, walls, material, wallMaterial, waterMaterial],
  );

  useFrame((state, delta) => {
    const rt = runtime.current;
    const p = rt.params;

    if (rt.version !== built.current.version) {
      built.current.version = rt.version;
      const position = terrain.attributes.position as BufferAttribute;
      const coords = position.array as Float32Array;
      const wet = terrain.attributes.aWet.array as Float32Array;
      const road = terrain.attributes.aRoad.array as Float32Array;
      const channel = terrain.attributes.aChannel.array as Float32Array;
      const fault = terrain.attributes.aFault.array as Float32Array;

      let minY = Infinity;
      for (let i = 0; i < position.count; i += 1) {
        const x = coords[i * 3];
        const z = coords[i * 3 + 2];
        const u = z * M_PER_UNIT;
        const v = x * M_PER_UNIT;
        const y = heightAt(u, v, p) / M_PER_UNIT;
        coords[i * 3 + 1] = y;
        if (y < minY) minY = y;
        const masks = surfaceMasks(u, v, p);
        wet[i] = masks.wet;
        road[i] = masks.road;
        channel[i] = masks.channel;
        fault[i] = masks.fault;
      }
      position.needsUpdate = true;
      for (const name of ['aWet', 'aRoad', 'aChannel', 'aFault']) {
        (terrain.attributes[name] as BufferAttribute).needsUpdate = true;
      }
      terrain.computeVertexNormals();
      terrain.computeBoundingSphere();
      terrain.computeBoundingBox();

      const baseY = Math.floor(minY) - 7;
      rt.baseY = baseY;
      rt.siteY = heightAt(0, 0, p) / M_PER_UNIT;

      // Cutaway faces follow the terrain edge down to the base.
      const perEdge = segments + 1;
      const wallPositions = walls.attributes.position.array as Float32Array;
      const wallUvs = walls.attributes.uv.array as Float32Array;
      const edgeIndex = (edge: number, i: number) => {
        if (edge === 0) return segments * perEdge + i; // front, z = +HALF
        if (edge === 1) return i; // back, z = -HALF
        if (edge === 2) return i * perEdge; // left, x = -HALF
        return i * perEdge + segments; // right, x = +HALF
      };
      for (let edge = 0; edge < 4; edge += 1) {
        for (let i = 0; i < perEdge; i += 1) {
          const source = edgeIndex(edge, i) * 3;
          const x = coords[source];
          const top = coords[source + 1];
          const z = coords[source + 2];
          const base = (edge * perEdge + i) * 2;
          wallPositions.set([x, top, z, x, baseY, z], base * 3);
          const along = (i / segments) * 6;
          const depth = ((top - baseY) * M_PER_UNIT) / STRATA_DEPTH_M;
          wallUvs.set([along, 1, along, 1 - depth], base * 2);
        }
      }
      walls.attributes.position.needsUpdate = true;
      walls.attributes.uv.needsUpdate = true;
      walls.computeVertexNormals();
      walls.computeBoundingSphere();

      if (plinth.current) plinth.current.position.y = baseY - 0.45;

      const seaY = seaLevel(p);
      if (sea.current) {
        sea.current.visible = seaY !== null;
        if (seaY !== null) sea.current.position.y = seaY / M_PER_UNIT;
      }
      const lakeY = waterLevel(p);
      if (lake.current) {
        lake.current.visible = lakeY !== null && (seaY === null || lakeY > seaY);
        if (lakeY !== null) lake.current.position.y = lakeY / M_PER_UNIT;
      }

      const [groundA, groundB] = groundColours(p.lulc, p.vegetation);
      uniforms.uGroundA.value.copy(groundA);
      uniforms.uGroundB.value.copy(groundB);
      uniforms.uRock.value.copy(rockColour(p.lithology));
      uniforms.uStripes.value = p.lulc === 40 ? 1 : 0;

      if (p.lithology !== built.current.lithology) {
        built.current.lithology = p.lithology;
        wallMaterial.map = strataTexture(p.lithology);
        wallMaterial.needsUpdate = true;
      }
    }

    const ease = 1 - Math.exp(-delta * 6);
    uniforms.uDim.value += (DIM_BY_LENS[lens] - uniforms.uDim.value) * ease;
    uniforms.uContour.value += (CONTOUR_BY_LENS[lens] - uniforms.uContour.value) * ease;
    const wallTarget = lens === 'all' || lens === 'geological' ? 1 : 0.42;
    const wallScalar = wallMaterial.color.r + (wallTarget - wallMaterial.color.r) * ease;
    wallMaterial.color.setScalar(wallScalar);

    uniforms.uTime.value = state.clock.elapsedTime;
    uniforms.uReveal.value = rt.reveal;
    uniforms.uFailure.value = rt.failure;
    uniforms.uResultColor.value.copy(rt.resultColor);
    uniforms.uScanZ.value = rt.scanZ;
    uniforms.uScanAmount.value = rt.scanAmount;
    uniforms.uPulse.value = rt.pulse;
  });

  const hover = (element: SceneElement) => (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    setHovered(element);
  };

  return (
    <group>
      <mesh
        geometry={terrain}
        material={material}
        castShadow
        receiveShadow
        onPointerOver={hover('terrain')}
        onPointerOut={() => setHovered(null)}
        onClick={(event) => {
          event.stopPropagation();
          setActiveFactor('slope');
        }}
      />
      <mesh
        geometry={walls}
        material={wallMaterial}
        receiveShadow
        onPointerOver={hover('strata')}
        onPointerOut={() => setHovered(null)}
        onClick={(event) => {
          event.stopPropagation();
          setActiveFactor('lithology');
        }}
      />
      <mesh ref={plinth} receiveShadow>
        <boxGeometry args={[HALF * 2 + 14, 0.9, HALF * 2 + 14]} />
        <meshStandardMaterial color="#1a1f24" roughness={0.8} metalness={0.1} />
      </mesh>
      <mesh ref={sea} rotation-x={-Math.PI / 2} material={waterMaterial} visible={false} receiveShadow>
        <planeGeometry args={[HALF * 2, HALF * 2]} />
      </mesh>
      <mesh ref={lake} rotation-x={-Math.PI / 2} material={waterMaterial} visible={false} receiveShadow>
        <planeGeometry args={[HALF * 2, HALF * 2]} />
      </mesh>
    </group>
  );
}

import { Html } from '@react-three/drei';
import { useFrame, type ThreeEvent } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import { BufferAttribute, BufferGeometry, type Group, type Mesh, MeshStandardMaterial } from 'three';
import { useSimulationStore } from '../store/simulationStore';
import type { FactorName } from '../types/api';
import type { Lens, SceneElement } from '../types/simulation';
import { formatDistance } from '../utils/format';
import { sourcePlace, townHint } from '../utils/place';
import { THEME_ELEMENTS, toSceneParams, type SceneParams } from '../utils/sceneMapping';
import { useRuntime } from './runtime';
import {
  HALF,
  HALF_M,
  M_PER_UNIT,
  ROAD_HALF_WIDTH_M,
  faultGeometry,
  faultInBlock,
  faultSpan,
  heightAt,
  roadInBlock,
  streamCentre,
  streamInBlock,
} from './terrain/terrainModel';

const SAMPLES = 96;

function ribbonGeometry() {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array((SAMPLES + 1) * 2 * 3), 3));
  const indices: number[] = [];
  for (let i = 0; i < SAMPLES; i += 1) {
    const a = i * 2;
    indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }
  geometry.setIndex(indices);
  return geometry;
}

type Sample = { u: number; v: number; nu: number; nv: number; y: number };

function writeRibbon(geometry: BufferGeometry, samples: Sample[], halfWidthM: number) {
  const array = geometry.attributes.position.array as Float32Array;
  samples.forEach((s, i) => {
    const y = s.y / M_PER_UNIT;
    array.set(
      [
        (s.v + s.nv * halfWidthM) / M_PER_UNIT,
        y,
        (s.u + s.nu * halfWidthM) / M_PER_UNIT,
        (s.v - s.nv * halfWidthM) / M_PER_UNIT,
        y,
        (s.u - s.nu * halfWidthM) / M_PER_UNIT,
      ],
      i * 6,
    );
  });
  geometry.attributes.position.needsUpdate = true;
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
}

const emphasis = (lens: Lens, element: SceneElement) =>
  lens === 'all' || THEME_ELEMENTS[lens].includes(element) ? 1 : 0.3;

function buildRoad(p: SceneParams): Sample[] {
  const u = -p.roadDistance;
  return Array.from({ length: SAMPLES + 1 }, (_, i) => {
    const v = -HALF_M + (i / SAMPLES) * HALF_M * 2;
    return { u, v, nu: 1, nv: 0, y: heightAt(u, v, p) + 0.35 };
  });
}

function buildStream(p: SceneParams): Sample[] {
  return Array.from({ length: SAMPLES + 1 }, (_, i) => {
    const u = -HALF_M + (i / SAMPLES) * HALF_M * 2;
    const v = streamCentre(u, p);
    const bed = heightAt(u, v, p);
    return { u, v, nu: 0, nv: 1, y: bed + 1.6 };
  });
}

function buildFault(p: SceneParams): Sample[] {
  const { foot, direction } = faultGeometry(p);
  const [s0, s1] = faultSpan(p) ?? [0, 0];
  return Array.from({ length: SAMPLES + 1 }, (_, i) => {
    const s = s0 + (i / SAMPLES) * (s1 - s0);
    const v = foot[0] + direction[0] * s;
    const u = foot[1] + direction[1] * s;
    return { u, v, nu: -direction[0], nv: direction[1], y: heightAt(u, v, p) + 0.4 };
  });
}

function faultLabelPoint(p: SceneParams): [number, number, number] {
  const { foot, direction } = faultGeometry(p);
  const [s0, s1] = faultSpan(p) ?? [0, 0];
  const s = Math.max(s0, Math.min(s1, 0)) * 0.8 + (s0 + s1) * 0.1;
  const v = foot[0] + direction[0] * s;
  const u = foot[1] + direction[1] * s;
  return [v / M_PER_UNIT, heightAt(u, v, p) / M_PER_UNIT + 2, u / M_PER_UNIT];
}

interface OverlayProps {
  element: SceneElement;
  factor: FactorName;
  colour: string;
  halfWidthM: number;
  inBlock: (p: SceneParams) => boolean;
  build: (p: SceneParams) => Sample[];
  labelAt: (p: SceneParams) => [number, number, number];
  label: string;
  /** Html ignores its group's visibility, so a label off the block is not rendered at all. */
  showLabel: boolean;
  transparent?: boolean;
}

function Overlay({ element, factor, colour, halfWidthM, inBlock, build, labelAt, label, showLabel, transparent }: OverlayProps) {
  const runtime = useRuntime();
  const lens = useSimulationStore((s) => s.lens);
  const hovered = useSimulationStore((s) => s.hovered === element);
  const setHovered = useSimulationStore((s) => s.setHovered);
  const setActiveFactor = useSimulationStore((s) => s.setActiveFactor);
  const geometry = useMemo(ribbonGeometry, []);
  const material = useMemo(
    () =>
      new MeshStandardMaterial({
        color: colour,
        roughness: transparent ? 0.15 : 0.85,
        metalness: 0,
        transparent: Boolean(transparent),
        opacity: transparent ? 0.9 : 1,
        polygonOffset: true,
        polygonOffsetFactor: -2,
      }),
    [colour, transparent],
  );
  const mesh = useRef<Mesh>(null);
  const labelGroup = useRef<Group>(null);
  const built = useRef(-1);

  useEffect(
    () => () => {
      geometry.dispose();
      material.dispose();
    },
    [geometry, material],
  );

  useFrame((_, delta) => {
    const rt = runtime.current;
    const visible = inBlock(rt.params);
    if (mesh.current) mesh.current.visible = visible;
    if (labelGroup.current) labelGroup.current.visible = visible;
    if (visible && built.current !== rt.version) {
      built.current = rt.version;
      writeRibbon(geometry, build(rt.params), halfWidthM);
      labelGroup.current?.position.set(...labelAt(rt.params));
    }
    const target = emphasis(lens, element) * (hovered ? 1.35 : 1);
    const ease = 1 - Math.exp(-delta * 8);
    const base = material.userData.scalar ?? 1;
    const next = base + (target - base) * ease;
    material.userData.scalar = next;
    material.emissive.set(colour).multiplyScalar(hovered ? 0.25 : 0);
    material.color.set(colour).multiplyScalar(next);
  });

  return (
    <>
      <mesh
        ref={mesh}
        geometry={geometry}
        material={material}
        receiveShadow
        onPointerOver={(event: ThreeEvent<PointerEvent>) => {
          event.stopPropagation();
          setHovered(element);
        }}
        onPointerOut={() => setHovered(null)}
        onClick={(event) => {
          event.stopPropagation();
          setActiveFactor(factor);
        }}
      />
      <group ref={labelGroup}>
        {showLabel && (
          <Html center zIndexRange={[20, 0]} className="scene-label-wrap" pointerEvents="none">
            <span className={`scene-label scene-label--${element}`}>{label}</span>
          </Html>
        )}
      </group>
    </>
  );
}

/** Chips on the block edge for features that lie beyond the 1.2 km block. */
function OffBlockChip({ position, text }: { position: [number, number, number]; text: string }) {
  return (
    <group position={position}>
      <Html center zIndexRange={[20, 0]} pointerEvents="none">
        <span className="scene-label scene-label--offblock">{text}</span>
      </Html>
    </group>
  );
}

/** Nameplate above the front edge of the block. The typical values are not a real place, so they have none. */
function PlacePlaque({ params }: { params: SceneParams }) {
  const place = useSimulationStore((s) => sourcePlace(s.source));
  if (!place) return null;
  const hint = townHint(place);
  return (
    <group position={[0, heightAt(HALF_M, 0, params) / M_PER_UNIT + 5.2, HALF + 1]}>
      <Html center zIndexRange={[25, 0]} pointerEvents="none">
        <div className="place-plaque">
          <span className="place-plaque__eyebrow">This block · 1.2 km around the cell</span>
          <strong>{place.label}</strong>
          <span className="place-plaque__sub">{hint ? `${place.sublabel} · ${hint}` : place.sublabel}</span>
        </div>
      </Html>
    </group>
  );
}

export function FeatureOverlays() {
  const inputs = useSimulationStore((s) => s.inputs);
  const metadata = useSimulationStore((s) => s.metadata);
  const target = useMemo(() => (inputs && metadata ? toSceneParams(inputs, metadata) : null), [inputs, metadata]);
  if (!inputs || !target) return null;

  const road = inputs.dist_road;
  const stream = inputs.dist_drainage;
  const fault = inputs.dist_fault;
  const edgeY = (u: number, v: number) => heightAt(u, v, target) / M_PER_UNIT + 2.2;

  return (
    <group>
      <PlacePlaque params={target} />
      <Overlay
        element="road"
        factor="dist_road"
        colour="#3a3c3f"
        halfWidthM={ROAD_HALF_WIDTH_M}
        inBlock={roadInBlock}
        build={buildRoad}
        labelAt={(p) => [HALF - 5, heightAt(-p.roadDistance, (HALF - 5) * M_PER_UNIT, p) / M_PER_UNIT + 1.6, -p.roadDistance / M_PER_UNIT]}
        label={`Road · ${formatDistance(road)}`}
        showLabel={roadInBlock(target)}
      />
      <Overlay
        element="stream"
        factor="dist_drainage"
        colour="#2f8fd8"
        halfWidthM={5}
        inBlock={streamInBlock}
        build={buildStream}
        labelAt={(p) => {
          // Far enough up the channel to clear the nameplate on the front edge.
          const u = HALF_M - 300;
          const v = streamCentre(u, p);
          return [v / M_PER_UNIT, heightAt(u, v, p) / M_PER_UNIT + 2, u / M_PER_UNIT];
        }}
        label={`Stream / river · ${formatDistance(stream)}`}
        showLabel={streamInBlock(target)}
        transparent
      />
      <Overlay
        element="fault"
        factor="dist_fault"
        colour="#6b3a2c"
        halfWidthM={4}
        inBlock={faultInBlock}
        build={buildFault}
        labelAt={faultLabelPoint}
        label={`Lineament · ${formatDistance(fault)}`}
        showLabel={faultInBlock(target)}
      />
      {!roadInBlock(target) && (
        <OffBlockChip position={[0, edgeY(-HALF_M, 0), -HALF - 1]} text={`↑ Road ${formatDistance(road)} away`} />
      )}
      {!streamInBlock(target) && (
        <OffBlockChip position={[HALF + 1, edgeY(0, HALF_M), 0]} text={`Stream / river ${formatDistance(stream)} away →`} />
      )}
      {!faultInBlock(target) && (
        <OffBlockChip position={[-HALF - 1, edgeY(-HALF_M * 0.6, -HALF_M), -HALF * 0.6]} text={`← Lineament ${formatDistance(fault)} away`} />
      )}
    </group>
  );
}

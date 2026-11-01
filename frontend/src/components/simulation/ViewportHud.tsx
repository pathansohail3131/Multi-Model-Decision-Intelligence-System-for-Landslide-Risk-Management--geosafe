import { useState } from 'react';
import { useResultIsStale, useSimulationStore } from '../../store/simulationStore';
import type { FactorName, ThemeName } from '../../types/api';
import type { Lens, SceneElement } from '../../types/simulation';
import { CLASS_COLOURS, THEME_COLOURS, THEME_LABELS } from '../../utils/colors';
import { compassPoint, formatDistance, formatNumber } from '../../utils/format';
import { SCENE_MAPPING } from '../../utils/sceneMapping';
import { levelLabel } from '../../utils/scenario';
import { CrossIcon, HelpIcon, OverviewIcon, TargetIcon } from '../common/Icons';

const LENSES: Lens[] = ['all', 'topographic', 'hydrological', 'geological', 'anthropogenic'];

const LENS_CAPTION: Record<ThemeName, string> = {
  topographic: 'The topographic model sees only elevation, slope, aspect and curvature.',
  hydrological: 'The hydrological model sees only wetness, stream power, drainage distance and rainfall.',
  geological: 'The geological model sees only lithology and distance to lineaments.',
  anthropogenic: 'The anthropogenic model sees only road distance, land cover and NDVI.',
};

const HOVER_FACTORS: Record<SceneElement, FactorName[]> = {
  site: ['slope', 'plan_curv', 'profile_curv', 'TWI', 'SPI'],
  terrain: ['slope', 'aspect', 'elevation'],
  stream: ['dist_drainage'],
  road: ['dist_road'],
  fault: ['dist_fault'],
  strata: ['lithology'],
  vegetation: ['LULC', 'NDVI'],
  rain: ['rainfall'],
};

const HOVER_TITLE: Record<SceneElement, string> = {
  site: 'Assessed 30 m cell',
  terrain: 'Hillslope',
  stream: 'Stream / river channel (blue)',
  road: 'Road bench',
  fault: 'Lineament trace',
  strata: 'Rock strata',
  vegetation: 'Land cover',
  rain: 'Rainfall',
};

function HoverCard() {
  const hovered = useSimulationStore((s) => s.hovered);
  const inputs = useSimulationStore((s) => s.inputs);
  const metadata = useSimulationStore((s) => s.metadata);
  if (!hovered || !inputs || !metadata) return null;

  const lines = HOVER_FACTORS[hovered].map((factor) => {
    const spec = metadata.factors.find((f) => f.name === factor);
    if (!spec) return null;
    const value = inputs[factor];
    let text: string;
    if (spec.kind === 'categorical') text = levelLabel(metadata, factor as 'lithology' | 'LULC', value);
    else if (factor.startsWith('dist_')) text = formatDistance(value);
    else if (factor === 'aspect') text = `${formatNumber(value)}° ${compassPoint(value)}`;
    else text = `${formatNumber(value, spec.decimals)}${spec.unit ? ` ${spec.unit}` : ''}`;
    return (
      <div key={factor}>
        {spec.label}: <span className="mono">{text}</span>
      </div>
    );
  });

  return (
    <div className="hud hud--top-center">
      <div className="hud-card hover-card" role="status">
        <strong>{HOVER_TITLE[hovered]}</strong>
        {lines}
        <div className="muted" style={{ fontSize: 11, marginTop: 2 }}>
          Click to edit
        </div>
      </div>
    </div>
  );
}

function PhaseCaption() {
  const phase = useSimulationStore((s) => s.phase);
  const runStep = useSimulationStore((s) => s.runStep);
  let text: string | null = null;
  let note = '';
  if (phase === 'validating') text = 'Validating parameters';
  else if (runStep === 'lock') text = 'Scenario locked';
  else if (runStep === 'prepare') {
    text = 'Reading the 14 conditioning factors';
    note = 'conceptual sweep';
  } else if (phase === 'predicting') {
    text = 'Running 5 CatBoost models';
    note = 'real inference';
  } else if (runStep === 'transition') text = 'Applying prediction to the site model';
  if (!text) return null;
  return (
    <div className="hud hud--top-center" aria-hidden="true">
      <div className="hud-card phase-caption">
        {runStep !== 'transition' && <span className="spinner" />}
        {text}
        {note && <small>· {note}</small>}
      </div>
    </div>
  );
}

function MappingPopover({ onClose }: { onClose: () => void }) {
  const metadata = useSimulationStore((s) => s.metadata);
  return (
    <div className="hud-card mapping" role="dialog" aria-label="How the scene maps to the model inputs">
      <div className="section__head" style={{ marginBottom: 6 }}>
        <h3 className="section__title">What the scene shows</h3>
        <button type="button" className="icon-btn" aria-label="Close" onClick={onClose}>
          <CrossIcon />
        </button>
      </div>
      <p className="small-note" style={{ marginBottom: 8 }}>
        A schematic 1.2 km block around the assessed 30 m cell, generated from the parameters. It is a visual aid, not terrain
        data: only the parameter values go to the model. The failure-zone outline after a flagged result is illustrative — the
        model predicts susceptibility, not failure extent.
      </p>
      <table>
        <thead>
          <tr>
            <th>Factor</th>
            <th>Scene</th>
            <th>Drawn</th>
          </tr>
        </thead>
        <tbody>
          {SCENE_MAPPING.map((row) => (
            <tr key={row.factor}>
              <td>{metadata?.factors.find((f) => f.name === row.factor)?.label ?? row.factor}</td>
              <td>
                {row.element}
                <div className="muted">{row.rule}</div>
              </td>
              <td>
                <span className="scale-tag" data-scale={row.scaled}>
                  {row.scaled}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const SCENE_KEY = [
  { key: 'stream', label: 'Stream / river', tip: 'Blue ribbon: the nearest stream or river. Its distance from the cell is drawn to scale; its direction is not known.' },
  { key: 'road', label: 'Road', tip: 'Grey band: the nearest road, placed at its distance from the cell.' },
  { key: 'fault', label: 'Lineament', tip: 'Rust line: a DEM-derived structural lineament, a proxy rather than a mapped fault.' },
  { key: 'cell', label: 'Assessed cell', tip: 'White diamond: the 30 m cell the models score.' },
  { key: 'vegetation', label: 'Vegetation', tip: 'Green dots: land cover and NDVI at the cell.' },
  { key: 'strata', label: 'Rock layers', tip: 'Stripes on the block sides: the lithology at the cell.' },
  { key: 'rain', label: 'Rain', tip: 'Diagonal streaks: mean annual rainfall at the cell, drawn qualitatively.' },
];

function SceneKey() {
  return (
    <ul className="scene-key" aria-label="Scene colour key">
      {SCENE_KEY.map((item) => (
        <li key={item.key} title={item.tip}>
          <span className={`scene-key__swatch scene-key__swatch--${item.key}`} aria-hidden="true" />
          {item.label}
        </li>
      ))}
    </ul>
  );
}

function Legend() {
  const result = useSimulationStore((s) => s.result);
  const metadata = useSimulationStore((s) => s.metadata);
  const stale = useResultIsStale();
  if (!metadata) return null;
  const current = result?.response.prediction.class.key;
  return (
    <div className="hud hud--bottom-left">
      <div className="hud-card legend">
        <div className="legend__title">
          <span className="eyebrow">Site model</span>
          <span className="muted">schematic · slope and distances to scale</span>
        </div>
        {result ? (
          <>
            <div>
              Halo and flag show run #{result.index}
              {stale ? ' (parameters changed since)' : ''}
              {result.response.prediction.above_threshold && !stale ? '; hatched outline is an illustrative failure zone' : ''}.
            </div>
            <div className="legend__classes" aria-hidden="true">
              {metadata.classes.map((band) => (
                <div key={band.key} className={`legend__class${band.key === current ? ' is-current' : ''}`}>
                  <div className="legend__swatch" style={{ background: CLASS_COLOURS[band.key] }} />
                  <span>{band.label}</span>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div>Drag to orbit, scroll to zoom. Hover an element to see the parameter behind it.</div>
        )}
        <SceneKey />
      </div>
    </div>
  );
}

export function ViewportHud() {
  const lens = useSimulationStore((s) => s.lens);
  const setLens = useSimulationStore((s) => s.setLens);
  const setCameraMode = useSimulationStore((s) => s.setCameraMode);
  const resetCamera = useSimulationStore((s) => s.resetCamera);
  const cameraMode = useSimulationStore((s) => s.cameraMode);
  const [mappingOpen, setMappingOpen] = useState(false);

  return (
    <>
      <div className="hud hud--top-left">
        <div className="hud-card lens-bar" role="group" aria-label="Model lens">
          <span className="lens-bar__label">Model lens</span>
          {LENSES.map((option) => (
            <button key={option} type="button" aria-pressed={lens === option} onClick={() => setLens(option)} title={option === 'all' ? 'Show everything' : LENS_CAPTION[option]}>
              {option !== 'all' && <span className="chip" style={{ background: THEME_COLOURS[option] }} />}
              <span className={option === 'all' ? undefined : 'lens-name'}>{option === 'all' ? 'All' : THEME_LABELS[option]}</span>
            </button>
          ))}
        </div>
        {lens !== 'all' && <div className="hud-card lens-caption">{LENS_CAPTION[lens]}</div>}
      </div>

      <div className="hud hud--top-right">
        <div className="hud-card hud-toolbar" role="toolbar" aria-label="Camera">
          <button type="button" className="icon-btn" aria-pressed={cameraMode === 'overview'} title="Overview" aria-label="Overview camera" onClick={resetCamera}>
            <OverviewIcon />
          </button>
          <button type="button" className="icon-btn" aria-pressed={cameraMode === 'result'} title="Focus the assessed cell" aria-label="Focus the assessed cell" onClick={() => setCameraMode('result')}>
            <TargetIcon />
          </button>
          <button type="button" className="icon-btn" aria-pressed={mappingOpen} title="What the scene shows" aria-label="What the scene shows" onClick={() => setMappingOpen((open) => !open)}>
            <HelpIcon />
          </button>
        </div>
      </div>

      {mappingOpen && <MappingPopover onClose={() => setMappingOpen(false)} />}
      <PhaseCaption />
      <HoverCard />
      <Legend />
    </>
  );
}

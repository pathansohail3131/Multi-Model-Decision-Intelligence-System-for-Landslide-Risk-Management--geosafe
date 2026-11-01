import { useState } from 'react';
import { ApiError, predictionApi } from '../../services/predictionApi';
import { isRunningPhase, useSimulationStore } from '../../store/simulationStore';
import type { PlaceHit } from '../../types/api';
import { formatDistance, formatLatLon } from '../../utils/format';
import { districtSource, sourcePlace, townHint } from '../../utils/place';
import { HoverCard } from '../common/HoverCard';
import { DiceIcon, FlagIcon, MapIcon, PinIcon } from '../common/Icons';
import { PlaceSearch } from '../common/PlaceSearch';

export function ScenarioSource() {
  const source = useSimulationStore((s) => s.source);
  const metadata = useSimulationStore((s) => s.metadata);
  const phase = useSimulationStore((s) => s.phase);
  const applyScenario = useSimulationStore((s) => s.applyScenario);
  const setDistrictOpen = useSimulationStore((s) => s.setDistrictOpen);
  const [loading, setLoading] = useState<0 | 1 | null>(null);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const disabled = isRunningPhase(phase) || !metadata;

  const loadSample = async (label: 0 | 1) => {
    setLoading(label);
    setError(null);
    try {
      const sample = await predictionApi.randomSample(label);
      applyScenario(sample.factors, {
        kind: 'training',
        label: sample.place ? `${sample.place.label} · sample #${sample.sample_id}` : `Training sample #${sample.sample_id}`,
        sample,
      });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not load a sample.');
    } finally {
      setLoading(null);
    }
  };

  const applyPlace = async (hit: PlaceHit) => {
    setApplying(true);
    setError(null);
    try {
      const site = await predictionApi.districtSample(hit.x, hit.y);
      if (site.factors && site.issues.length === 0) applyScenario(site.factors, districtSource(site));
      else setError(`${hit.name} cannot be scored here. ${site.issues[0] ?? ''}`.trim());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not read that place.');
    } finally {
      setApplying(false);
    }
  };

  let detail = '';
  if (source?.kind === 'median') detail = 'Middle value of each factor. Search a place, pick one on the map or load a recorded landslide.';
  if (source?.kind === 'training') {
    const s = source.sample;
    detail = `${s.label === 1 ? 'Recorded landslide' : 'No recorded landslide'}${!s.place && s.taluk ? ` · ${s.taluk} taluk` : ''}. ${s.note}`;
  }
  if (source?.kind === 'district') {
    const site = source.site;
    detail = `Real raster values at ${formatLatLon(site.lat, site.lon)}${
      site.nearest_landslide_m != null ? ` · nearest recorded landslide ${formatDistance(site.nearest_landslide_m)}` : ''
    }.`;
  }
  if (source?.kind === 'custom') detail = `Edited from: ${source.basedOn}.`;

  const place = sourcePlace(source);
  const hint = place ? townHint(place) : '';

  return (
    <section className="section" aria-labelledby="scenario-title">
      <div className="section__head">
        <h2 className="section__title" id="scenario-title">
          Scenario
        </h2>
      </div>
      <div className="source-card">
        <PlaceSearch onPick={(hit) => void applyPlace(hit)} disabled={disabled || applying} busy={applying} />
        <div>
          <p className="source-card__label">{source?.label ?? '—'}</p>
          {place && (
            <div className="place-line">
              <PinIcon />
              <div>
                <strong>{place.label}</strong>
                <span>{hint ? `${place.sublabel} · ${hint}` : place.sublabel}</span>
              </div>
            </div>
          )}
          <p className="source-card__detail">{detail}</p>
        </div>
        <div className="source-actions">
          <button type="button" className="btn btn--wide" disabled={disabled} onClick={() => setDistrictOpen(true)}>
            <MapIcon /> Pick a real site on the district map
          </button>
          <HoverCard title="Recorded landslide" text="A real landslide spot from GSI records (1 of 417)" colour="#d9793a">
            <button type="button" className="btn btn--sm" disabled={disabled || loading !== null} onClick={() => loadSample(1)}>
              {loading === 1 ? <span className="spinner" /> : <FlagIcon />} Recorded landslide (GSI)
            </button>
          </HoverCard>
          <HoverCard title="No recorded landslide" text="A spot with no landslide on record (1 of 4,519)" colour="#6fae8a">
            <button type="button" className="btn btn--sm" disabled={disabled || loading !== null} onClick={() => loadSample(0)}>
              {loading === 0 ? <span className="spinner" /> : <DiceIcon />} No recorded landslide
            </button>
          </HoverCard>
        </div>
        {error && (
          <p className="control__error" role="alert">
            {error}
          </p>
        )}
      </div>
    </section>
  );
}

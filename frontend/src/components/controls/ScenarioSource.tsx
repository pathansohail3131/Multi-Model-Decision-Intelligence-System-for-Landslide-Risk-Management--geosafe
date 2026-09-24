import { useState } from 'react';
import { ApiError, predictionApi } from '../../services/predictionApi';
import { isRunningPhase, useSimulationStore } from '../../store/simulationStore';
import { formatDistance, formatLatLon } from '../../utils/format';
import { medianScenario } from '../../utils/scenario';
import { DiceIcon, FlagIcon, MapIcon, TargetIcon } from '../common/Icons';

export function ScenarioSource() {
  const source = useSimulationStore((s) => s.source);
  const metadata = useSimulationStore((s) => s.metadata);
  const phase = useSimulationStore((s) => s.phase);
  const applyScenario = useSimulationStore((s) => s.applyScenario);
  const setDistrictOpen = useSimulationStore((s) => s.setDistrictOpen);
  const [loading, setLoading] = useState<0 | 1 | null>(null);
  const [error, setError] = useState<string | null>(null);
  const disabled = isRunningPhase(phase) || !metadata;

  const loadSample = async (label: 0 | 1) => {
    setLoading(label);
    setError(null);
    try {
      const sample = await predictionApi.randomSample(label);
      applyScenario(sample.factors, {
        kind: 'training',
        label: `Training sample #${sample.sample_id}`,
        sample,
      });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not load a sample.');
    } finally {
      setLoading(null);
    }
  };

  let detail = '';
  if (source?.kind === 'median') detail = 'Reference point built from training-data medians — not a real location.';
  if (source?.kind === 'training') {
    const s = source.sample;
    detail = `${s.label === 1 ? 'Recorded landslide' : 'Non-landslide sample'}${s.taluk ? ` · ${s.taluk} taluk` : ''}. ${s.note}`;
  }
  if (source?.kind === 'district') {
    const site = source.site;
    detail = `Real raster values at ${formatLatLon(site.lat, site.lon)}${
      site.nearest_landslide_m != null ? ` · nearest recorded landslide ${formatDistance(site.nearest_landslide_m)}` : ''
    }.`;
  }
  if (source?.kind === 'custom') detail = `Edited from: ${source.basedOn}.`;

  return (
    <section className="section" aria-labelledby="scenario-title">
      <div className="section__head">
        <h2 className="section__title" id="scenario-title">
          Scenario
        </h2>
      </div>
      <div className="source-card">
        <div>
          <p className="source-card__label">{source?.label ?? '—'}</p>
          <p className="source-card__detail">{detail}</p>
        </div>
        <div className="source-actions">
          <button type="button" className="btn btn--wide" disabled={disabled} onClick={() => setDistrictOpen(true)}>
            <MapIcon /> Pick a real site on the district map
          </button>
          <button type="button" className="btn btn--sm" disabled={disabled || loading !== null} onClick={() => loadSample(1)}>
            {loading === 1 ? <span className="spinner" /> : <FlagIcon />} Landslide site
          </button>
          <button type="button" className="btn btn--sm" disabled={disabled || loading !== null} onClick={() => loadSample(0)}>
            {loading === 0 ? <span className="spinner" /> : <DiceIcon />} Non-landslide
          </button>
          <button
            type="button"
            className="btn btn--sm btn--wide"
            disabled={disabled}
            onClick={() => metadata && applyScenario(medianScenario(metadata), { kind: 'median', label: 'Training-data median' })}
          >
            <TargetIcon /> Training-data median
          </button>
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

import { useState } from 'react';
import { ApiError, predictionApi } from '../../services/predictionApi';
import { isRunningPhase, useSimulationStore } from '../../store/simulationStore';
import { formatDistance, formatLatLon } from '../../utils/format';
import { sourcePlace, townHint } from '../../utils/place';
import { medianScenario } from '../../utils/scenario';
import { DiceIcon, FlagIcon, MapIcon, PinIcon, TargetIcon } from '../common/Icons';

const RECORDED_TOOLTIP =
  'Loads one of 417 places where the Geological Survey of India recorded a landslide. The models saw this point in training, so this is a demonstration, not a test.';
const NOT_RECORDED_TOOLTIP =
  'Loads one of 4,519 sampled points with no recorded landslide, at least 101 m from any recorded one. No record does not mean the place is safe.';

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
        label: sample.place ? `${sample.place.label} · sample #${sample.sample_id}` : `Training sample #${sample.sample_id}`,
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
  const placeDetail = place
    ? [
        place.sublabel,
        townHint(place),
        source?.kind === 'district' ? formatLatLon(source.site.lat, source.site.lon) : '',
      ]
        .filter(Boolean)
        .join(' · ')
    : '';

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
          {place && (
            <div className="place-line">
              <PinIcon />
              <div>
                <strong>{place.label}</strong>
                <span>{placeDetail}</span>
              </div>
            </div>
          )}
          <p className="source-card__detail">{detail}</p>
        </div>
        <div className="source-actions">
          <button type="button" className="btn btn--wide" disabled={disabled} onClick={() => setDistrictOpen(true)}>
            <MapIcon /> Pick a real site on the district map
          </button>
          <button type="button" className="btn btn--sm btn--wide" title={RECORDED_TOOLTIP} disabled={disabled || loading !== null} onClick={() => loadSample(1)}>
            {loading === 1 ? <span className="spinner" /> : <FlagIcon />} Recorded landslide (GSI)
          </button>
          <button type="button" className="btn btn--sm btn--wide" title={NOT_RECORDED_TOOLTIP} disabled={disabled || loading !== null} onClick={() => loadSample(0)}>
            {loading === 0 ? <span className="spinner" /> : <DiceIcon />} No recorded landslide
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

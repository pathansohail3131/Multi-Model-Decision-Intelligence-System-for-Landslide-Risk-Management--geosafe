import type { ModelMetadata } from '../../types/api';
import type { RunRecord } from '../../types/simulation';
import { CLASS_COLOURS } from '../../utils/colors';
import { formatPercent, formatProbability } from '../../utils/format';
import { CheckIcon, WarnIcon } from '../common/Icons';

/**
 * Class bands are drawn with equal widths because the operating threshold sits
 * at 0.137: on a linear 0–1 axis the four lower classes would share 40% of the
 * width. Positions inside a band are linear; the ticks give the real bounds.
 */
function scalePosition(value: number, bands: ModelMetadata['classes']): number {
  const width = 100 / bands.length;
  const index = bands.findIndex((band, i) => value < band.range[1] || i === bands.length - 1);
  const band = bands[Math.max(0, index)];
  const [low, high] = band.range;
  const within = Math.min(1, Math.max(0, (value - low) / (high - low || 1)));
  return (Math.max(0, index) + within) * width;
}

export function PredictionSummary({ run, metadata }: { run: RunRecord; metadata: ModelMetadata }) {
  const p = run.response.prediction;
  const bands = metadata.classes;
  const marker = scalePosition(p.susceptibility, bands);
  const thresholdIndex = bands.findIndex((band) => Math.abs(band.range[0] - p.threshold) < 1e-9);
  const thresholdPos = thresholdIndex >= 0 ? (thresholdIndex * 100) / bands.length : scalePosition(p.threshold, bands);
  const oof = metadata.threshold.oof_metrics;

  return (
    <section className="section" aria-labelledby="prediction-title">
      <div className="section__head">
        <h2 className="section__title" id="prediction-title">
          Prediction
        </h2>
        <span className="small-note">
          run #{run.index} · {run.sourceLabel}
        </span>
      </div>

      <div className="headline">
        <div>
          <p className="headline__class">
            <span className="chip" style={{ background: CLASS_COLOURS[p.class.key] }} />
            {p.class.label}
          </p>
          <p className="headline__caption">Landslide susceptibility of the assessed cell</p>
        </div>
        <div style={{ textAlign: 'right' }}>
          <p className="headline__value">{formatProbability(p.susceptibility)}</p>
          <p className="headline__caption">calibrated score, 0–1</p>
        </div>
      </div>

      <div className="class-scale" role="img" aria-label={`Score ${formatProbability(p.susceptibility)} falls in the ${p.class.label} class`}>
        <div className="class-scale__bands">
          {bands.map((band) => (
            <span
              key={band.key}
              className={`class-scale__band${band.key === p.class.key ? ' is-current' : ''}`}
              style={{ background: CLASS_COLOURS[band.key] }}
              title={`${band.label}: ${band.range[0].toFixed(3)}–${band.range[1].toFixed(3)}`}
            />
          ))}
          <span className="class-scale__threshold" style={{ left: `${thresholdPos}%` }} />
          <span className="class-scale__marker" style={{ left: `${marker}%` }} />
        </div>
        <div className="class-scale__ticks">
          {[0, ...bands.map((b) => b.range[1])].map((value, i) => (
            <span
              key={i}
              className={Math.abs(value - p.threshold) < 1e-9 ? 'is-threshold' : undefined}
              style={{ left: `${(i * 100) / bands.length}%` }}
            >
              {value === 0 || value === 1 ? value : value.toFixed(value < 0.1 ? 2 : 3).replace(/0+$/, '')}
            </span>
          ))}
        </div>
      </div>

      <p className="flag-line">
        {p.above_threshold ? <WarnIcon style={{ color: CLASS_COLOURS.high }} /> : <CheckIcon style={{ color: 'var(--ok)' }} />}
        <span>
          <strong>{p.above_threshold ? 'Flagged' : 'Not flagged'}</strong> at the operating threshold of{' '}
          <span className="mono">{p.threshold.toFixed(3)}</span>, chosen so a missed landslide costs{' '}
          {metadata.threshold.cost_false_negative ?? 10}× a false alarm.
          {oof && (
            <>
              {' '}
              On out-of-fold training predictions this threshold caught {formatPercent(oof.recall, 0)} of recorded landslides, with{' '}
              {formatPercent(oof.precision, 0)} of flagged cells being landslides.
            </>
          )}
        </span>
      </p>
    </section>
  );
}

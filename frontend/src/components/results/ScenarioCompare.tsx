import { useSimulationStore } from '../../store/simulationStore';
import type { ModelMetadata } from '../../types/api';
import type { RunRecord } from '../../types/simulation';
import { CLASS_COLOURS, THEME_LABELS } from '../../utils/colors';
import { formatNumber, formatProbability, formatSigned } from '../../utils/format';
import { changedFactors, levelLabel, specFor } from '../../utils/scenario';
import { PinIcon } from '../common/Icons';

function valueText(metadata: ModelMetadata, run: RunRecord, factor: Parameters<typeof specFor>[1]) {
  const spec = specFor(metadata, factor);
  const value = run.inputs[factor];
  return spec.kind === 'categorical'
    ? levelLabel(metadata, factor as 'lithology' | 'LULC', value)
    : `${formatNumber(value, spec.decimals)}${spec.unit ? ` ${spec.unit}` : ''}`;
}

export function ScenarioCompare({ current, metadata }: { current: RunRecord; metadata: ModelMetadata }) {
  const history = useSimulationStore((s) => s.history);
  const baselineId = useSimulationStore((s) => s.baselineId);
  const setBaseline = useSimulationStore((s) => s.setBaseline);
  const baseline = history.find((run) => run.id === baselineId) ?? null;
  const comparing = baseline && baseline.id !== current.id ? baseline : null;

  return (
    <section className="section" aria-labelledby="runs-title">
      <div className="section__head">
        <h2 className="section__title" id="runs-title">
          Scenario runs
        </h2>
        <span className="small-note">{history.length} this session</span>
      </div>

      {history.length < 2 && (
        <p className="small-note" style={{ marginBottom: 8 }}>
          Change a parameter and run again to build a what-if comparison. Pin a run as the baseline to compare against it.
        </p>
      )}

      <table className="runs">
        <thead>
          <tr>
            <th>#</th>
            <th>Scenario</th>
            <th style={{ textAlign: 'right' }}>Score</th>
            <th>Decision</th>
            <th>
              <span className="visually-hidden">Baseline</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {history.map((run) => {
            const p = run.response.prediction;
            const pinned = run.id === baselineId;
            return (
              <tr key={run.id} className={run.id === current.id ? 'is-current' : undefined}>
                <td className="mono">{run.index}</td>
                <td className="source" title={run.sourceLabel}>
                  {run.sourceLabel}
                </td>
                <td className="num">
                  <span className="chip" style={{ background: CLASS_COLOURS[p.class.key], marginRight: 5, width: 7, height: 7 }} />
                  {formatProbability(p.susceptibility)}
                </td>
                <td>{p.decision?.label ?? p.class.label}</td>
                <td>
                  <button
                    type="button"
                    className="icon-btn"
                    aria-pressed={pinned}
                    aria-label={pinned ? `Unpin run ${run.index} as baseline` : `Pin run ${run.index} as baseline`}
                    title={pinned ? 'Baseline — click to unpin' : 'Pin as baseline'}
                    onClick={() => setBaseline(pinned ? null : run.id)}
                  >
                    <PinIcon />
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {comparing && <Comparison baseline={comparing} current={current} metadata={metadata} />}
    </section>
  );
}

function Comparison({ baseline, current, metadata }: { baseline: RunRecord; current: RunRecord; metadata: ModelMetadata }) {
  const a = baseline.response.prediction;
  const b = current.response.prediction;
  const delta = b.susceptibility - a.susceptibility;
  const changes = changedFactors(metadata, baseline.inputs, current.inputs);

  return (
    <div className="compare" aria-label={`Run ${baseline.index} compared with run ${current.index}`}>
      <div className="compare__scores">
        <div className="compare__score">
          <b style={{ color: CLASS_COLOURS[a.class.key] }}>{formatProbability(a.susceptibility)}</b>
          <span>
            baseline #{baseline.index} · {a.class.label}
          </span>
        </div>
        <span className={`mono ${delta > 0 ? 'delta-up' : 'delta-down'}`}>{formatSigned(delta)}</span>
        <div className="compare__score">
          <b style={{ color: CLASS_COLOURS[b.class.key] }}>{formatProbability(b.susceptibility)}</b>
          <span>
            run #{current.index} · {b.class.label}
          </span>
        </div>
      </div>
      <table className="metrics-table">
        <tbody>
          {current.response.models.map((model) => {
            const before = baseline.response.models.find((m) => m.name === model.name)?.probability ?? 0;
            const d = model.probability - before;
            return (
              <tr key={model.name}>
                <td>{THEME_LABELS[model.name]}</td>
                <td className="num">{formatProbability(before)}</td>
                <td className="num">{formatProbability(model.probability)}</td>
                <td className={`num ${d > 0 ? 'delta-up' : d < 0 ? 'delta-down' : ''}`}>{formatSigned(d)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div>
        <p className="eyebrow" style={{ marginBottom: 4 }}>
          Changed parameters ({changes.length})
        </p>
        {changes.length === 0 ? (
          <p className="small-note">Identical inputs.</p>
        ) : (
          <ul>
            {changes.map((factor) => (
              <li key={factor}>
                {specFor(metadata, factor).label}: {valueText(metadata, baseline, factor)} → {valueText(metadata, current, factor)}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

import { useMemo, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { useSensitivity } from '../../hooks/useSensitivity';
import type { FactorName, ModelMetadata, SensitivityResponse } from '../../types/api';
import type { RunRecord } from '../../types/simulation';
import { CLASS_COLOURS, THEME_COLOURS, THEME_LABELS } from '../../utils/colors';
import { formatNumber, formatProbability } from '../../utils/format';
import { levelLabel, specFor } from '../../utils/scenario';

const W = 340;
const H = 170;
const PAD = { left: 34, right: 10, top: 10, bottom: 26 };

function niceMax(value: number) {
  const candidates = [0.1, 0.2, 0.3, 0.5, 0.75, 1];
  return candidates.find((c) => c >= value) ?? 1;
}

interface ChartProps {
  data: SensitivityResponse;
  metadata: ModelMetadata;
  run: RunRecord;
}

function LineChart({ data, metadata, run }: ChartProps) {
  const spec = specFor(metadata, data.factor);
  const [hover, setHover] = useState<number | null>(null);
  const isLog = spec.kind === 'numeric' && spec.scale === 'log';
  // Distances are heavily skewed towards zero, where the models change fastest.
  const isSqrt = !isLog && data.factor.startsWith('dist_');
  const xs = data.values;
  const xMin = isLog ? Math.max(xs[0], 1e-4) : xs[0];
  const xMax = xs[xs.length - 1];
  const theme = spec.theme;
  const thematicSeries = data.thematic[theme];
  const yMax = niceMax(Math.max(data.threshold * 1.6, ...data.susceptibility, ...data.full_model, ...thematicSeries) * 1.05);

  const tx = (x: number) => {
    let t: number;
    if (isLog) t = (Math.log(Math.max(x, xMin)) - Math.log(xMin)) / (Math.log(xMax) - Math.log(xMin) || 1);
    else if (isSqrt) t = (Math.sqrt(Math.max(x - xMin, 0))) / (Math.sqrt(xMax - xMin) || 1);
    else t = (x - xMin) / (xMax - xMin || 1);
    return PAD.left + t * (W - PAD.left - PAD.right);
  };
  const ty = (y: number) => PAD.top + (1 - y / yMax) * (H - PAD.top - PAD.bottom);
  const path = (series: number[]) => series.map((y, i) => `${i ? 'L' : 'M'}${tx(xs[i]).toFixed(1)},${ty(y).toFixed(1)}`).join('');
  const currentIndex = xs.findIndex((x) => Math.abs(x - data.current_value) < 1e-9);
  const decimals = spec.kind === 'numeric' ? spec.decimals : 0;
  const xTicks = isLog
    ? [xMin, Math.sqrt(xMin * xMax), xMax]
    : isSqrt
      ? [xMin, xMin + (xMax - xMin) / 4, xMax]
      : [xMin, (xMin + xMax) / 2, xMax];
  const colour = CLASS_COLOURS[run.response.prediction.class.key];

  const onMove = (event: ReactPointerEvent<SVGRectElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * (W - PAD.left - PAD.right) + PAD.left;
    let best = 0;
    xs.forEach((x, i) => {
      if (Math.abs(tx(x) - px) < Math.abs(tx(xs[best]) - px)) best = i;
    });
    setHover(best);
  };

  const readout = hover ?? currentIndex;
  return (
    <>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Susceptibility as ${spec.label} varies`}>
        {[0, yMax / 2, yMax].map((y) => (
          <g key={y}>
            <line className="grid-line" x1={PAD.left} x2={W - PAD.right} y1={ty(y)} y2={ty(y)} />
            <text x={PAD.left - 5} y={ty(y) + 3} textAnchor="end">
              {y.toFixed(2)}
            </text>
          </g>
        ))}
        {xTicks.map((x, i) => (
          <text key={i} x={tx(x)} y={H - 8} textAnchor={i === 0 ? 'start' : i === 2 ? 'end' : 'middle'}>
            {formatNumber(x, x < 10 ? Math.min(decimals, 2) : 0)}
          </text>
        ))}
        <line x1={PAD.left} x2={W - PAD.right} y1={ty(data.threshold)} y2={ty(data.threshold)} stroke="#e3e7eb" strokeDasharray="3 3" strokeWidth={1} opacity={0.6} />
        <text x={W - PAD.right} y={ty(data.threshold) - 4} textAnchor="end">
          threshold
        </text>
        <path d={path(thematicSeries)} fill="none" stroke={THEME_COLOURS[theme]} strokeWidth={1.4} opacity={0.9} />
        <path d={path(data.full_model)} fill="none" stroke={THEME_COLOURS.full} strokeWidth={1.2} strokeDasharray="4 3" opacity={0.7} />
        <path d={path(data.susceptibility)} fill="none" stroke={colour} strokeWidth={2.4} />
        {currentIndex >= 0 && (
          <>
            <line x1={tx(xs[currentIndex])} x2={tx(xs[currentIndex])} y1={PAD.top} y2={H - PAD.bottom} stroke="#e3e7eb" opacity={0.35} />
            <circle cx={tx(xs[currentIndex])} cy={ty(data.susceptibility[currentIndex])} r={4} fill={colour} stroke="#12161a" strokeWidth={2} />
          </>
        )}
        {hover !== null && (
          <line x1={tx(xs[hover])} x2={tx(xs[hover])} y1={PAD.top} y2={H - PAD.bottom} stroke="#82b4e6" opacity={0.6} />
        )}
        <rect
          x={PAD.left}
          y={PAD.top}
          width={W - PAD.left - PAD.right}
          height={H - PAD.top - PAD.bottom}
          fill="transparent"
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
        />
      </svg>
      <p className="chart-readout">
        {readout >= 0 &&
          `${spec.label} ${formatNumber(xs[readout], decimals)}${spec.unit ? ` ${spec.unit}` : ''} → score ${formatProbability(
            data.susceptibility[readout],
          )} · full ${formatProbability(data.full_model[readout])} · ${THEME_LABELS[theme].toLowerCase()} ${formatProbability(
            thematicSeries[readout],
          )}${readout === currentIndex ? ' (this scenario)' : ''}`}
      </p>
      <div className="chart-legend">
        {(isLog || isSqrt) && <span className="muted">x axis: {isLog ? 'log' : 'square-root'} scale</span>}
        <span>
          <i style={{ borderColor: colour }} /> Production score
        </span>
        <span>
          <i style={{ borderColor: THEME_COLOURS.full, borderTopStyle: 'dashed' }} /> Full model
        </span>
        <span>
          <i style={{ borderColor: THEME_COLOURS[theme] }} /> {THEME_LABELS[theme]} model
        </span>
      </div>
    </>
  );
}

function CategoryChart({ data, metadata, run }: ChartProps) {
  const name = data.factor as 'lithology' | 'LULC';
  const yMax = niceMax(Math.max(data.threshold * 1.6, ...data.susceptibility) * 1.05);
  const colour = CLASS_COLOURS[run.response.prediction.class.key];
  return (
    <div className="evidence">
      {data.values.map((code, i) => (
        <div key={code} className="evidence__row" style={{ cursor: 'default', gridTemplateColumns: '140px 1fr 50px' }}>
          <span className="evidence__name" style={{ fontWeight: code === data.current_value ? 650 : 400 }}>
            {levelLabel(metadata, name, code)}
            {code === data.current_value ? ' ●' : ''}
          </span>
          <span className="evidence__bar">
            <span className="evidence__fill" style={{ width: `${(data.susceptibility[i] / yMax) * 100}%`, background: code === data.current_value ? colour : 'var(--text-3)' }} />
            <span className="evidence__threshold" style={{ left: `${(data.threshold / yMax) * 100}%` }} />
          </span>
          <span className="evidence__value">{formatProbability(data.susceptibility[i])}</span>
        </div>
      ))}
    </div>
  );
}

export function SensitivityChart({ run, metadata }: { run: RunRecord; metadata: ModelMetadata }) {
  const defaultFactor = useMemo<FactorName>(
    () => run.response.explanations?.full?.by_factor[0]?.factor ?? 'slope',
    [run],
  );
  const [chosen, setChosen] = useState<FactorName | null>(null);
  const factor = chosen ?? defaultFactor;
  const state = useSensitivity(run, factor);

  return (
    <section className="section" aria-labelledby="sensitivity-title">
      <div className="section__head">
        <h2 className="section__title" id="sensitivity-title">
          What-if sweep
        </h2>
        <select
          className="field-input"
          style={{ width: 180, height: 26 }}
          value={factor}
          aria-label="Factor to sweep"
          onChange={(event) => setChosen(event.target.value as FactorName)}
        >
          {metadata.factors.map((f) => (
            <option key={f.name} value={f.name}>
              {f.label}
            </option>
          ))}
        </select>
      </div>

      {state.status === 'loading' && (
        <p className="small-note" style={{ display: 'flex', gap: 8, alignItems: 'center', minHeight: 170 }}>
          <span className="spinner" /> Re-running the models across the range…
        </p>
      )}
      {state.status === 'error' && <p className="notice notice--danger">{state.message}</p>}
      {state.status === 'ready' &&
        (specFor(metadata, factor).kind === 'categorical' ? (
          <CategoryChart data={state.data} metadata={metadata} run={run} />
        ) : (
          <LineChart data={state.data} metadata={metadata} run={run} />
        ))}
      {state.status === 'ready' && (
        <p className="small-note" style={{ marginTop: 6 }}>
          Every point is a real model evaluation of run #{run.index} with only this factor changed, swept across{' '}
          {state.data.range_basis}. It shows this scenario's response, not a global effect.
        </p>
      )}
    </section>
  );
}

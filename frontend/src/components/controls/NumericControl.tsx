import { useEffect, useId, useRef, useState, type CSSProperties } from 'react';
import type { NumericFactorSpec } from '../../types/api';
import { compassPoint, formatNumber } from '../../utils/format';
import { trainingRangeStatus } from '../../utils/scenario';
import { InfoIcon } from '../common/Icons';

const STEPS = 1000;

function sliderBounds(spec: NumericFactorSpec): [number, number] {
  const { min, max } = spec.training;
  return spec.scale === 'log' ? [Math.max(min, 1e-4), Math.max(max, 1e-3)] : [min, max];
}

function toPosition(spec: NumericFactorSpec, value: number): number {
  const [low, high] = sliderBounds(spec);
  if (spec.scale === 'log') {
    const v = Math.min(Math.max(value, low), high);
    return Math.round(((Math.log(v) - Math.log(low)) / (Math.log(high) - Math.log(low))) * STEPS);
  }
  return Math.round(((Math.min(Math.max(value, low), high) - low) / (high - low || 1)) * STEPS);
}

function fromPosition(spec: NumericFactorSpec, position: number): number {
  const [low, high] = sliderBounds(spec);
  const t = position / STEPS;
  const raw = spec.scale === 'log' ? Math.exp(Math.log(low) + t * (Math.log(high) - Math.log(low))) : low + t * (high - low);
  const factor = 10 ** spec.decimals;
  return Math.round(raw * factor) / factor;
}

function draftText(spec: NumericFactorSpec, value: number) {
  return Number.isFinite(value) ? String(Number(value.toFixed(spec.decimals))) : '';
}

interface Props {
  spec: NumericFactorSpec;
  value: number;
  error?: string;
  active: boolean;
  onChange: (value: number) => void;
  onFocus: () => void;
}

export function NumericControl({ spec, value, error, active, onChange, onFocus }: Props) {
  const id = useId();
  const [draft, setDraft] = useState(() => draftText(spec, value));
  const [draftError, setDraftError] = useState<string | null>(null);
  const [showInfo, setShowInfo] = useState(false);
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setDraft(draftText(spec, value));
    setDraftError(null);
  }, [spec, value]);

  useEffect(() => {
    if (active) container.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [active]);

  const commit = () => {
    const parsed = Number(draft.trim());
    if (draft.trim() === '' || !Number.isFinite(parsed)) {
      setDraftError('Enter a number.');
      return;
    }
    setDraftError(null);
    if (parsed !== value) onChange(parsed);
  };

  const position = toPosition(spec, value);
  const outside = trainingRangeStatus(spec, value);
  const [low, high] = spec.bounds;
  const physicalError = value < low || value > high ? `Physical limit is ${low} to ${high}.` : null;
  const message = draftError ?? error ?? physicalError;
  const suffix = spec.name === 'aspect' ? compassPoint(value) : spec.unit;
  const t = spec.training;
  const tick = (v: number) => `${(toPosition(spec, v) / STEPS) * 100}%`;

  return (
    <div ref={container} className={`control${active ? ' is-active' : ''}${message ? ' has-error' : ''}`}>
      <div className="control__row">
        <label className="control__label" htmlFor={`${id}-value`}>
          {spec.label}
        </label>
        <button
          type="button"
          className="icon-btn"
          aria-pressed={showInfo}
          aria-label={`About ${spec.label}`}
          onClick={() => setShowInfo((open) => !open)}
        >
          <InfoIcon />
        </button>
        <div className="control__value">
          <input
            id={`${id}-value`}
            className="field-input"
            inputMode="decimal"
            value={draft}
            aria-invalid={Boolean(message)}
            aria-describedby={`${id}-hint`}
            onFocus={onFocus}
            onChange={(event) => setDraft(event.target.value)}
            onBlur={commit}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                commit();
              }
              if (event.key === 'Escape') setDraft(draftText(spec, value));
            }}
          />
          <span className="control__suffix">{suffix}</span>
        </div>
      </div>

      <div className="control__track">
        <input
          type="range"
          className="slider"
          min={0}
          max={STEPS}
          step={1}
          value={position}
          style={{ '--fill': `${(position / STEPS) * 100}%` } as CSSProperties}
          aria-label={`${spec.label} slider`}
          aria-valuetext={`${formatNumber(value, spec.decimals)} ${spec.unit}`}
          onFocus={onFocus}
          onChange={(event) => onChange(fromPosition(spec, Number(event.target.value)))}
        />
        <span className="control__tick" style={{ left: tick(t.median) }} aria-hidden="true" />
        <span className="control__tick control__tick--landslide" style={{ left: tick(t.median_landslide) }} aria-hidden="true" />
      </div>

      <p className="control__hint" id={`${id}-hint`}>
        {spec.scale === 'log' ? 'Log scale · ' : ''}
        Training {formatNumber(t.min, spec.decimals)}–{formatNumber(t.max, spec.decimals)}
        <span className="tick-key" style={{ background: 'var(--text-3)', marginLeft: 6 }} />
        median {formatNumber(t.median, spec.decimals)}
        <span className="tick-key" style={{ background: '#d9793a', marginLeft: 6 }} />
        at landslides {formatNumber(t.median_landslide, spec.decimals)}
      </p>

      {message && (
        <p className="control__error" role="alert">
          {message}
        </p>
      )}
      {!message && outside && (
        <p className="control__warning">
          {outside === 'below' ? 'Below' : 'Above'} the training range — the models will be extrapolating.
        </p>
      )}
      {showInfo && <p className="control__description">{spec.description}</p>}
    </div>
  );
}

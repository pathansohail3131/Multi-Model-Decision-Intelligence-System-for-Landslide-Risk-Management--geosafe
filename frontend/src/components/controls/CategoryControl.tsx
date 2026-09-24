import { useEffect, useId, useRef, useState } from 'react';
import type { CategoricalFactorSpec } from '../../types/api';
import { formatPercent } from '../../utils/format';
import { InfoIcon } from '../common/Icons';

interface Props {
  spec: CategoricalFactorSpec;
  value: number;
  error?: string;
  active: boolean;
  onChange: (value: number) => void;
  onFocus: () => void;
}

export function CategoryControl({ spec, value, error, active, onChange, onFocus }: Props) {
  const id = useId();
  const [showInfo, setShowInfo] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const level = spec.levels.find((l) => l.code === value);

  useEffect(() => {
    if (active) container.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [active]);

  return (
    <div ref={container} className={`control${active ? ' is-active' : ''}${error ? ' has-error' : ''}`}>
      <div className="control__row">
        <label className="control__label" htmlFor={`${id}-select`}>
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
      </div>
      <select
        id={`${id}-select`}
        className="field-input"
        value={value}
        aria-invalid={Boolean(error)}
        aria-describedby={`${id}-hint`}
        onFocus={onFocus}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        {spec.levels.map((option) => (
          <option key={option.code} value={option.code}>
            {option.label} · {option.n_samples.toLocaleString('en-US')} samples
          </option>
        ))}
      </select>
      <p className="control__hint" id={`${id}-hint`}>
        {level
          ? `Recorded landslide rate in training data: ${formatPercent(level.landslide_rate)} (${level.n_landslides} of ${level.n_samples.toLocaleString('en-US')})`
          : 'Class not present in the training data.'}
      </p>
      {level && level.n_samples < 50 && (
        <p className="control__warning">Only {level.n_samples} training samples — this class's effect is poorly constrained.</p>
      )}
      {error && (
        <p className="control__error" role="alert">
          {error}
        </p>
      )}
      {showInfo && <p className="control__description">{spec.description}</p>}
    </div>
  );
}

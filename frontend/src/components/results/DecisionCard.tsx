import type { DecisionKey, ModelMetadata } from '../../types/api';
import type { RunRecord } from '../../types/simulation';
import { DECISION_COLOURS } from '../../utils/colors';
import { formatProbability } from '../../utils/format';
import { CheckIcon, EyeIcon, SearchIcon, WarnIcon } from '../common/Icons';

const ICONS: Record<DecisionKey, typeof WarnIcon> = {
  act_now: WarnIcon,
  survey_first: SearchIcon,
  monitor: EyeIcon,
  deprioritise: CheckIcon,
};

// Row 1 = high confidence, column 3 = susceptibility at or above the threshold.
const GRID: { key: DecisionKey; row: number; col: number }[] = [
  { key: 'deprioritise', row: 1, col: 2 },
  { key: 'act_now', row: 1, col: 3 },
  { key: 'monitor', row: 2, col: 2 },
  { key: 'survey_first', row: 2, col: 3 },
];

export function DecisionCard({ run, metadata }: { run: RunRecord; metadata: ModelMetadata }) {
  const p = run.response.prediction;
  const decision = p.decision;

  if (!decision || p.confidence === null) {
    return (
      <section className="section" aria-labelledby="decision-title">
        <div className="section__head">
          <h2 className="section__title" id="decision-title">
            Planner decision
          </h2>
        </div>
        <div className="notice notice--warn">
          <WarnIcon />
          <span>
            Confidence cannot be scaled because the Stage 5 reports are missing, so no planner decision is shown. The raw
            spread across the thematic models is <span className="mono">{formatProbability(p.disagreement)}</span>. Run{' '}
            <span className="mono">python run_pipeline.py stage 5</span> and restart the API.
          </span>
        </div>
      </section>
    );
  }

  const Icon = ICONS[decision.key];
  const colour = DECISION_COLOURS[decision.key];
  const split = p.confidence_split ?? 0;

  return (
    <section className="section" aria-labelledby="decision-title">
      <div className="section__head">
        <h2 className="section__title" id="decision-title">
          Planner decision
        </h2>
        <span className="small-note">susceptibility × confidence</span>
      </div>
      <div className="decision">
        <div className="decision__head">
          <span className="decision__icon" style={{ background: `${colour}26`, color: colour }}>
            <Icon />
          </span>
          <div>
            <p className="decision__label" style={{ color: colour }}>
              {decision.label}
            </p>
            <p className="decision__text">{decision.interpretation}</p>
          </div>
        </div>

        <div className="quadrant" role="img" aria-label={`Decision quadrant: ${decision.label}`}>
          <span className="quadrant__axis quadrant__axis--y">Confidence</span>
          {GRID.map((cell) => {
            const info = metadata.decisions.find((d) => d.key === cell.key);
            const current = cell.key === decision.key;
            return (
              <div
                key={cell.key}
                className={`quadrant__cell${current ? ' is-current' : ''}`}
                style={{
                  gridRow: cell.row,
                  gridColumn: cell.col,
                  borderColor: current ? DECISION_COLOURS[cell.key] : undefined,
                }}
              >
                <b style={{ color: current ? DECISION_COLOURS[cell.key] : undefined }}>{info?.label}</b>
                {cell.row === 1 ? 'high' : 'low'} conf. · {cell.col === 3 ? 'high' : 'low'} susc.
              </div>
            );
          })}
          <span className="quadrant__axis" style={{ gridRow: 3, gridColumn: '2 / 4' }}>
            Susceptibility vs threshold
          </span>
        </div>

        <div className="meter">
          <div className="meter__row">
            <span>Theme agreement (confidence)</span>
            <span className="mono">{formatProbability(p.confidence)}</span>
          </div>
          <div className="meter__track">
            <span className="meter__fill" style={{ width: `${p.confidence * 100}%`, background: colour }} />
            <span className="meter__split" style={{ left: `${split * 100}%` }} title={`Split at ${split.toFixed(3)}`} />
          </div>
          <p className="small-note">
            1 − spread of the four thematic models ({formatProbability(p.disagreement)}) ÷ the largest spread seen on{' '}
            {metadata.confidence.n_reference?.toLocaleString('en-US') ?? 'the'} out-of-fold training predictions. The dashed line is
            the median ({split.toFixed(3)}) that splits high from low. Relative to this dataset; not a probability of being right.
          </p>
        </div>
      </div>
    </section>
  );
}

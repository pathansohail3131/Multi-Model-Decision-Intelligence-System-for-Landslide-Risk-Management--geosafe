import { useState } from 'react';
import type { FactorName, ModelMetadata, ModelName } from '../../types/api';
import type { RunRecord } from '../../types/simulation';
import { THEME_COLOURS, THEME_LABELS } from '../../utils/colors';
import { formatNumber, formatSigned } from '../../utils/format';
import { levelLabel, specFor } from '../../utils/scenario';

const MODELS: ModelName[] = ['full', 'topographic', 'hydrological', 'geological', 'anthropogenic'];
const PUSH_UP = '#d9793a';
const PUSH_DOWN = '#6f9fc4';

function inputText(metadata: ModelMetadata, run: RunRecord, factor: FactorName) {
  const spec = specFor(metadata, factor);
  const value = run.inputs[factor];
  if (spec.kind === 'categorical') return levelLabel(metadata, factor as 'lithology' | 'LULC', value);
  return `${formatNumber(value, spec.decimals)}${spec.unit ? ` ${spec.unit}` : ''}`;
}

export function ExplanationChart({ run, metadata }: { run: RunRecord; metadata: ModelMetadata }) {
  const [model, setModel] = useState<ModelName>('full');
  const explanation = run.response.explanations?.[model] ?? null;

  if (!run.response.explanations) return null;

  const rows = explanation?.by_factor ?? [];
  const maxAbs = Math.max(0.25, ...rows.map((r) => Math.abs(r.shap)));
  const roadIsTop = model !== 'geological' && rows.slice(0, 3).some((r) => r.factor === 'dist_road');

  return (
    <section className="section" aria-labelledby="explain-title">
      <div className="section__head">
        <h2 className="section__title" id="explain-title">
          Why this score
        </h2>
        <span className="small-note">SHAP, per model</span>
      </div>
      <div className="segmented" role="tablist" aria-label="Model to explain" style={{ marginBottom: 10, flexWrap: 'wrap' }}>
        {MODELS.map((name) => (
          <button key={name} type="button" role="tab" aria-selected={model === name} onClick={() => setModel(name)}>
            <span className="chip" style={{ background: THEME_COLOURS[name], marginRight: 5, width: 7, height: 7 }} />
            {name === 'full' ? 'Full' : THEME_LABELS[name]}
          </button>
        ))}
      </div>

      {explanation ? (
        <>
          <div className="shap" role="list">
            {rows.map((row) => {
              const width = (Math.abs(row.shap) / maxAbs) * 50;
              const label = specFor(metadata, row.factor).label;
              return (
                <div
                  key={row.factor}
                  className="shap__row"
                  role="listitem"
                  aria-label={`${label} = ${inputText(metadata, run, row.factor)}: ${formatSigned(row.shap, 2)} log-odds`}
                >
                  <span className="shap__label" title={`${label} = ${inputText(metadata, run, row.factor)}`}>
                    {label}
                    <span className="muted"> · {inputText(metadata, run, row.factor)}</span>
                  </span>
                  <span className="shap__bar" aria-hidden="true">
                    <span className="shap__axis" />
                    <span
                      className="shap__fill"
                      style={{
                        left: row.shap >= 0 ? '50%' : `${50 - width}%`,
                        width: `${width}%`,
                        background: row.shap >= 0 ? PUSH_UP : PUSH_DOWN,
                      }}
                    />
                  </span>
                  <span className="shap__value" style={{ color: row.shap >= 0 ? PUSH_UP : PUSH_DOWN }}>
                    {formatSigned(row.shap, 2)}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="shap__legend">
            <span style={{ color: PUSH_DOWN }}>← lowers susceptibility</span>
            <span style={{ color: PUSH_UP }}>raises susceptibility →</span>
          </div>
          <p className="small-note" style={{ marginTop: 8 }}>
            Contributions in log-odds to the uncalibrated {model === 'full' ? 'full-feature' : THEME_LABELS[model].toLowerCase()} model
            (CatBoost native SHAP): base {formatSigned(explanation.base_log_odds, 2)} + contributions ={' '}
            {formatSigned(explanation.output_log_odds, 2)}. Calibration and blending happen afterwards, so these explain the model's
            ranking, not the final score directly. Encoded features are summed into their source factor.
          </p>
          {roadIsTop && (
            <p className="notice notice--warn" style={{ marginTop: 8 }}>
              Distance to road ranks highly. That is partly genuine road-cut instability and partly inventory bias: landslides far from
              roads are under-recorded. The data cannot separate the two.
            </p>
          )}
        </>
      ) : (
        <p className="small-note">No SHAP explanation is available for this model type.</p>
      )}
    </section>
  );
}

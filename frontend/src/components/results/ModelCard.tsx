import { useSimulationStore } from '../../store/simulationStore';
import { THEME_COLOURS } from '../../utils/colors';
import { formatPercent } from '../../utils/format';

/** What the model is, and the evaluation numbers the pipeline actually recorded. */
export function ModelCard({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const metadata = useSimulationStore((s) => s.metadata);
  if (!metadata) return null;
  const { threshold, production, confidence, dataset } = metadata;
  const baseRate = dataset.n_landslides / dataset.n_samples;
  const cvScheme = metadata.models[0]?.spatial_cv;

  return (
    <section className="section">
      <details className="model-card" open={defaultOpen}>
        <summary>
          <h2 className="section__title">About the model</h2>
        </summary>
        <div className="prose" style={{ marginTop: 10 }}>
          <p>
            <strong>Predicts:</strong> landslide susceptibility of one {Math.round(dataset.cell_size_m)} m terrain cell — how
            predisposed it is to failure given 14 static conditioning factors. It does not forecast when a landslide will happen.
          </p>
          <p>
            <strong>How:</strong> five CatBoost classifiers. Four see one physical theme each and nothing else; the fifth sees all
            factors. Each output is calibrated with isotonic regression, then blended: {production.rule}.
          </p>
          <p>
            <strong>Trained on:</strong> {dataset.n_samples.toLocaleString('en-US')} samples, {dataset.n_landslides} of them
            field-validated GSI landslides (base rate {formatPercent(baseRate)}).
          </p>

          <table className="metrics-table" aria-label="Spatial cross-validation results">
            <thead>
              <tr>
                <th>Model</th>
                <th style={{ textAlign: 'right' }}>AUC-PR</th>
                <th style={{ textAlign: 'right' }}>ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {metadata.models.map((model) => (
                <tr key={model.name} title={`Factors: ${model.factors.join(', ')}`}>
                  <td>
                    <span className="chip" style={{ background: THEME_COLOURS[model.name], marginRight: 6 }} />
                    {model.label}
                  </td>
                  <td className="num">
                    {model.spatial_cv?.average_precision_mean?.toFixed(3) ?? '—'}
                    {model.spatial_cv?.average_precision_std != null && (
                      <span className="muted"> ±{model.spatial_cv.average_precision_std.toFixed(3)}</span>
                    )}
                  </td>
                  <td className="num">{model.spatial_cv?.roc_auc_mean?.toFixed(3) ?? '—'}</td>
                </tr>
              ))}
              {production.oof_metrics && (
                <tr>
                  <td>
                    <strong>Production blend</strong>
                  </td>
                  <td className="num">{production.oof_metrics.average_precision.toFixed(3)}</td>
                  <td className="num">{production.oof_metrics.roc_auc.toFixed(3)}</td>
                </tr>
              )}
            </tbody>
          </table>
          <p className="small-note">
            Individual models: {cvScheme?.n_splits ?? 5}-fold spatial block cross-validation with{' '}
            {cvScheme?.block_size_m ? `${(cvScheme.block_size_m / 1000).toFixed(0)} km` : 'spatial'} blocks, stored in each model
            bundle. Production blend: calibrated out-of-fold predictions from Stage 5. AUC-PR leads because only{' '}
            {formatPercent(baseRate)} of samples are landslides.
          </p>

          <p>
            <strong>Operating threshold {threshold.value.toFixed(3)}</strong> ({threshold.method?.replace('_', '-')}, false negative
            cost {threshold.cost_false_negative}× false positive)
            {threshold.oof_metrics &&
              `: recall ${formatPercent(threshold.oof_metrics.recall)}, precision ${formatPercent(
                threshold.oof_metrics.precision,
              )}, flags ${formatPercent(threshold.oof_metrics.flagged_rate)} of samples.`}
          </p>
          <p>
            <strong>Confidence:</strong>{' '}
            {confidence.available
              ? `${confidence.method}. It ranks agreement; it is not a probability of being correct.`
              : 'unavailable — the Stage 5 reports were not found.'}
          </p>

          <p className="eyebrow" style={{ marginTop: 4 }}>
            Limitations
          </p>
          <ul>
            {metadata.caveats.map((caveat) => (
              <li key={caveat}>{caveat}</li>
            ))}
          </ul>
          {metadata.report_warnings.length > 0 && (
            <div className="notice notice--warn">
              <span>{metadata.report_warnings.join(' ')}</span>
            </div>
          )}
        </div>
      </details>
    </section>
  );
}

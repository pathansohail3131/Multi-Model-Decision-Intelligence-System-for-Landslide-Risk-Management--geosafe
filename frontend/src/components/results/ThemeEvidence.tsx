import { useSimulationStore } from '../../store/simulationStore';
import type { ModelMetadata, ThemeName } from '../../types/api';
import type { RunRecord } from '../../types/simulation';
import { THEME_COLOURS } from '../../utils/colors';
import { formatProbability } from '../../utils/format';

export function ThemeEvidence({ run, metadata }: { run: RunRecord; metadata: ModelMetadata }) {
  const lens = useSimulationStore((s) => s.lens);
  const setLens = useSimulationStore((s) => s.setLens);
  const p = run.response.prediction;
  const thematic = run.response.models.filter((m) => m.role === 'thematic');
  const full = run.response.models.find((m) => m.role === 'benchmark');
  const factorsOf = (name: string) => metadata.models.find((m) => m.name === name)?.factors.join(', ') ?? '';

  return (
    <section className="section" aria-labelledby="evidence-title">
      <div className="section__head">
        <h2 className="section__title" id="evidence-title">
          Evidence by model
        </h2>
        <span className="small-note">calibrated probability</span>
      </div>
      <div className="evidence">
        {thematic.map((model) => {
          const theme = model.name as ThemeName;
          return (
            <button
              key={model.name}
              type="button"
              className="evidence__row"
              aria-pressed={lens === theme}
              onClick={() => setLens(lens === theme ? 'all' : theme)}
              title={`Sees only: ${factorsOf(model.name)}. Uncalibrated score ${formatProbability(model.raw_probability)}. Click to apply the model lens.`}
            >
              <span className="evidence__name">
                <span className="chip" style={{ background: THEME_COLOURS[theme] }} />
                {model.label}
              </span>
              <span className="evidence__bar">
                <span className="evidence__fill" style={{ width: `${model.probability * 100}%`, background: THEME_COLOURS[theme] }} />
                <span className="evidence__threshold" style={{ left: `${p.threshold * 100}%` }} />
              </span>
              <span className="evidence__value">{formatProbability(model.probability)}</span>
            </button>
          );
        })}
        <span className="evidence__divider" />
        {full && (
          <div className="evidence__row" style={{ cursor: 'default' }} title={`Sees all 14 factors. Uncalibrated score ${formatProbability(full.raw_probability)}.`}>
            <span className="evidence__name">
              <span className="chip" style={{ background: THEME_COLOURS.full }} />
              {full.label}
            </span>
            <span className="evidence__bar">
              <span className="evidence__fill" style={{ width: `${full.probability * 100}%`, background: THEME_COLOURS.full }} />
              <span className="evidence__threshold" style={{ left: `${p.threshold * 100}%` }} />
            </span>
            <span className="evidence__value">{formatProbability(full.probability)}</span>
          </div>
        )}
      </div>
      <p className="formula" title="The production rule adopted in Stage 4">
        0.5 × (full {formatProbability(p.full_model)} + thematic mean {formatProbability(p.thematic_mean)}) ={' '}
        {formatProbability(p.susceptibility)}
      </p>
      <p className="small-note" style={{ marginTop: 6 }}>
        Each thematic model sees only its own factors, so their spread (std {formatProbability(p.disagreement)}) measures how much
        the physical evidence conflicts. Dashed line: operating threshold. Click a theme to see what it sees in the scene.
      </p>
    </section>
  );
}

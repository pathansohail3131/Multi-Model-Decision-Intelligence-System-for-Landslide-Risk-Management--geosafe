import { isRunningPhase, useSimulationStore } from '../../store/simulationStore';
import type { FactorSpec, ThemeName } from '../../types/api';
import { THEME_COLOURS, THEME_LABELS } from '../../utils/colors';
import { EyeIcon } from '../common/Icons';
import { CategoryControl } from './CategoryControl';
import { NumericControl } from './NumericControl';
import { ScenarioSource } from './ScenarioSource';

const THEMES: ThemeName[] = ['topographic', 'hydrological', 'geological', 'anthropogenic'];

export function ParameterPanel({ embedded = false }: { embedded?: boolean }) {
  const metadata = useSimulationStore((s) => s.metadata);
  const inputs = useSimulationStore((s) => s.inputs);
  const fieldErrors = useSimulationStore((s) => s.fieldErrors);
  const activeFactor = useSimulationStore((s) => s.activeFactor);
  const phase = useSimulationStore((s) => s.phase);
  const lens = useSimulationStore((s) => s.lens);
  const setInput = useSimulationStore((s) => s.setInput);
  const setActiveFactor = useSimulationStore((s) => s.setActiveFactor);
  const setLens = useSimulationStore((s) => s.setLens);

  if (!metadata || !inputs) return null;
  const locked = isRunningPhase(phase) || phase === 'resetting';

  const renderControl = (spec: FactorSpec) => {
    const common = {
      value: inputs[spec.name],
      error: fieldErrors[spec.name],
      active: activeFactor === spec.name,
      onChange: (value: number) => setInput(spec.name, value),
      onFocus: () => {
        if (activeFactor !== spec.name) setActiveFactor(spec.name);
      },
    };
    return spec.kind === 'numeric' ? (
      <NumericControl key={spec.name} spec={spec} {...common} />
    ) : (
      <CategoryControl key={spec.name} spec={spec} {...common} />
    );
  };

  const body = (
    <>
      <ScenarioSource />
      <fieldset className="controls-fieldset" disabled={locked} aria-label="Site parameters">
        {THEMES.map((theme) => {
          const model = metadata.models.find((m) => m.name === theme);
          const ap = model?.spatial_cv?.average_precision_mean;
          return (
            <section key={theme} className="theme-group" aria-labelledby={`theme-${theme}`}>
              <div className="theme-group__head">
                <span className="chip" style={{ background: THEME_COLOURS[theme] }} />
                <h3 className="theme-group__name" id={`theme-${theme}`}>
                  {THEME_LABELS[theme]}
                </h3>
                {ap != null && (
                  <span className="theme-group__metric" title="Spatial block cross-validation AUC-PR of this theme's model (base rate 0.084)">
                    model AUC-PR {ap.toFixed(3)}
                  </span>
                )}
                <button
                  type="button"
                  className="icon-btn"
                  aria-pressed={lens === theme}
                  aria-label={`Show only what the ${THEME_LABELS[theme]} model sees`}
                  title={`Model lens: ${THEME_LABELS[theme]}`}
                  onClick={() => setLens(lens === theme ? 'all' : theme)}
                >
                  <EyeIcon />
                </button>
              </div>
              <div className="theme-group__body">
                {metadata.factors.filter((f) => f.theme === theme).map(renderControl)}
              </div>
            </section>
          );
        })}
      </fieldset>
    </>
  );

  if (embedded) return body;

  return (
    <>
      <div className="panel__header">
        <div>
          <h2 className="panel__title">Simulation parameters</h2>
          <p className="muted" style={{ fontSize: 11.5 }}>
            14 conditioning factors of one 30 m cell, grouped by the model that sees them
          </p>
        </div>
      </div>
      <div className="panel__scroll">{body}</div>
    </>
  );
}

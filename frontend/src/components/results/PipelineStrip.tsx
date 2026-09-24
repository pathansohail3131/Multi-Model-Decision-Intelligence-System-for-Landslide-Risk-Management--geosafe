import { useSimulationStore } from '../../store/simulationStore';
import type { RunStep } from '../../types/simulation';

const STEPS = [
  { title: 'Inputs', detail: '14 factors', active: ['validate', 'lock'], hint: 'The 14 raw conditioning factors, validated' },
  { title: 'Encode', detail: 'aspect, SPI', active: ['prepare'], hint: 'Circular aspect (sin/cos), flat flag, log1p(SPI), categories' },
  { title: 'Models', detail: '5 CatBoost', active: ['infer'], hint: '4 thematic models + 1 full-feature benchmark' },
  { title: 'Calibrate', detail: 'isotonic', active: ['infer'], hint: 'Isotonic calibrators attached in Stage 5' },
  { title: 'Blend', detail: 'agreement', active: ['respond'], hint: '0.5 × (full + thematic mean); spread across themes' },
  { title: 'Decide', detail: 'class', active: ['transition'], hint: 'Susceptibility class, threshold and planner quadrant' },
] satisfies { title: string; detail: string; active: RunStep[]; hint: string }[];

const ORDER: RunStep[] = ['validate', 'lock', 'prepare', 'infer', 'respond', 'transition', 'analysis'];

/** Conceptual flow of one prediction. Each block names a real step the API performs. */
export function PipelineStrip() {
  const runStep = useSimulationStore((s) => s.runStep);
  const phase = useSimulationStore((s) => s.phase);
  const result = useSimulationStore((s) => s.result);
  const current = runStep ? ORDER.indexOf(runStep) : -1;

  const state = (active: RunStep[]) => {
    if (current < 0) return phase === 'result' || phase === 'modified' ? (result ? 'done' : 'idle') : 'idle';
    const indices = active.map((step) => ORDER.indexOf(step));
    if (indices.includes(current)) return 'active';
    return current > Math.max(...indices) ? 'done' : 'idle';
  };

  const timing = result?.response.timing_ms;
  return (
    <section className="section" aria-label="Prediction pipeline">
      <div className="section__head">
        <h2 className="section__title">Prediction pipeline</h2>
        {timing && (
          <span className="small-note mono">
            model {timing.inference.toFixed(0)} ms{timing.explanation != null ? ` · SHAP ${timing.explanation.toFixed(0)} ms` : ''}
          </span>
        )}
      </div>
      <ol className="pipeline" style={{ listStyle: 'none', margin: 0, padding: 0 }}>
        {STEPS.map((step) => (
          <li key={step.title} className="pipeline__step" data-state={state(step.active)} title={step.hint}>
            <b>{step.title}</b>
            <span>{step.detail}</span>
          </li>
        ))}
      </ol>
      <p className="pipeline__caption">
        Conceptual flow: every block is a real step in the prediction API, but the highlight timing is illustrative.
      </p>
    </section>
  );
}

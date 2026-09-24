import { useEffect, type ReactNode } from 'react';
import { useReducedMotion } from '../../hooks/useMediaQuery';
import { isRunningPhase, useResultIsStale, useSimulationStore } from '../../store/simulationStore';
import type { RunStep } from '../../types/simulation';
import { formatProbability } from '../../utils/format';
import { CameraIcon, PlayIcon, ResetIcon } from '../common/Icons';

const STEPS: { key: RunStep[]; label: string }[] = [
  { key: ['validate'], label: 'Validate' },
  { key: ['lock', 'prepare'], label: 'Prepare' },
  { key: ['infer'], label: 'Infer' },
  { key: ['respond', 'transition', 'analysis'], label: 'Result' },
];

const ORDER: RunStep[] = ['validate', 'lock', 'prepare', 'infer', 'respond', 'transition', 'analysis'];

export function ActionBar({ onRun }: { onRun: () => void }) {
  const phase = useSimulationStore((s) => s.phase);
  const runStep = useSimulationStore((s) => s.runStep);
  const backend = useSimulationStore((s) => s.backend);
  const result = useSimulationStore((s) => s.result);
  const source = useSimulationStore((s) => s.source);
  const error = useSimulationStore((s) => s.error);
  const dismissError = useSimulationStore((s) => s.dismissError);
  const beginReset = useSimulationStore((s) => s.beginReset);
  const completeReset = useSimulationStore((s) => s.completeReset);
  const resetCamera = useSimulationStore((s) => s.resetCamera);
  const stale = useResultIsStale();
  const reducedMotion = useReducedMotion();
  const running = isRunningPhase(phase);
  const canRun = backend.status === 'ready' && !running && phase !== 'resetting';

  useEffect(() => {
    if (phase !== 'resetting') return;
    const timer = window.setTimeout(completeReset, reducedMotion ? 0 : 450);
    return () => window.clearTimeout(timer);
  }, [phase, completeReset, reducedMotion]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        onRun();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onRun]);

  let message: ReactNode;
  if (backend.status !== 'ready') message = backend.stage;
  else if (phase === 'error' && error) {
    message = (
      <>
        <strong>{error.title}.</strong> {error.message}
      </>
    );
  } else if (phase === 'validating') message = 'Validating the 14 conditioning factors…';
  else if (runStep === 'lock') message = 'Scenario locked.';
  else if (runStep === 'prepare') message = 'Preparing the feature vector for the models…';
  else if (phase === 'predicting') message = 'Running the model: 4 thematic + 1 full-feature CatBoost, calibrated…';
  else if (phase === 'resetting') message = 'Resetting scenario…';
  else if (result && stale) {
    message = (
      <>
        Parameters changed since <strong>run #{result.index}</strong> — run again to compare.
      </>
    );
  } else if (result && phase === 'result') {
    const p = result.response.prediction;
    message = (
      <>
        <strong>Run #{result.index}</strong> · {p.class.label} susceptibility {formatProbability(p.susceptibility)} ·
        inference {result.response.timing_ms.inference.toFixed(0)} ms
      </>
    );
  } else if (phase === 'modified') message = `Scenario configured · ${source?.label ?? ''}`;
  else message = 'Configure the site parameters, then run the simulation.';

  const stepIndex = runStep ? ORDER.indexOf(runStep) : -1;
  const stepState = (keys: RunStep[]) => {
    if (phase === 'error' && error?.kind !== 'validation' && runStep === null && keys.includes('infer')) return 'failed';
    if (stepIndex < 0) return 'idle';
    const first = ORDER.indexOf(keys[0]);
    const last = ORDER.indexOf(keys[keys.length - 1]);
    if (stepIndex > last || (keys.includes('analysis') && stepIndex >= first)) return 'done';
    if (stepIndex >= first) return 'active';
    return 'idle';
  };

  return (
    <footer className="actionbar">
      <div className="actionbar__left">
        <button
          type="button"
          className="btn"
          onClick={beginReset}
          disabled={running || phase === 'resetting' || backend.status !== 'ready'}
          title="Restore the training-median scenario and clear the current result"
        >
          <ResetIcon />
          <span className="btn__label">Reset</span>
        </button>
        <button type="button" className="btn btn--ghost" onClick={resetCamera} title="Return the camera to the overview">
          <CameraIcon />
          <span className="btn__label">Camera</span>
        </button>
      </div>

      <div className="actionbar__status">
        {(running || runStep) && (
          <div className="run-steps" aria-hidden="true">
            {STEPS.map((step) => (
              <span key={step.label} className="run-steps__item" data-state={stepState(step.key)}>
                {step.label}
              </span>
            ))}
          </div>
        )}
        <p className={`actionbar__message${phase === 'error' ? ' is-error' : ''}`} title={typeof message === 'string' ? message : undefined}>
          {message}
        </p>
        {phase === 'error' && (
          <button type="button" className="btn btn--sm btn--ghost" onClick={dismissError}>
            Dismiss
          </button>
        )}
      </div>

      <div className="actionbar__right">
        <span className="kbd-hint">
          <span className="kbd">Ctrl</span> + <span className="kbd">Enter</span>
        </span>
        <button
          type="button"
          className={`btn btn--primary${running ? ' is-running' : ''}`}
          onClick={onRun}
          disabled={!canRun}
          aria-busy={running}
        >
          {running ? <span className="spinner" aria-hidden="true" /> : <PlayIcon />}
          {running ? 'Simulating' : result && !stale ? 'Run again' : 'Run simulation'}
        </button>
      </div>
    </footer>
  );
}

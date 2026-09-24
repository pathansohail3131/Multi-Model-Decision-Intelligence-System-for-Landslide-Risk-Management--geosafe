import { useEffect, useState } from 'react';
import { useSimulationStore } from '../../store/simulationStore';
import { CheckIcon, CrossIcon, MountainMark } from '../common/Icons';

type StepState = 'pending' | 'active' | 'done' | 'error';

function StepIcon({ state }: { state: StepState }) {
  if (state === 'done') return <CheckIcon style={{ width: 14, color: 'var(--ok)' }} />;
  if (state === 'error') return <CrossIcon style={{ width: 14, color: 'var(--danger)' }} />;
  if (state === 'active') return <span className="spinner" />;
  return <span className="dot" />;
}

/** Startup overlay driven entirely by the real health and metadata calls. */
export function BootScreen({ onRetry }: { onRetry: () => void }) {
  const backend = useSimulationStore((s) => s.backend);
  const metadata = useSimulationStore((s) => s.metadata);
  const ready = backend.status === 'ready' && Boolean(metadata);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (!ready) {
      setVisible(true);
      return;
    }
    const timer = window.setTimeout(() => setVisible(false), 320);
    return () => window.clearTimeout(timer);
  }, [ready]);

  if (!visible) return null;

  const failed = backend.status === 'error';
  const offline = backend.status === 'offline';
  const reached = backend.status === 'loading' || backend.status === 'ready' || failed;
  const modelsLoaded = backend.status === 'ready' || backend.stage === 'Reading model metadata';

  const steps: { label: string; state: StepState; detail?: string }[] = [
    { label: 'Loading interface', state: 'done' },
    {
      label: 'Connecting to prediction engine',
      state: reached ? 'done' : offline ? 'error' : 'active',
      detail: offline ? (backend.message ?? undefined) : undefined,
    },
    {
      label: 'Loading trained models',
      state: modelsLoaded ? 'done' : failed ? 'error' : reached ? 'active' : 'pending',
      detail: reached && !modelsLoaded ? `${backend.stage}${backend.elapsed != null ? ` · ${backend.elapsed.toFixed(1)} s` : ''}` : undefined,
    },
    {
      label: 'Reading model metadata',
      state: ready ? 'done' : backend.stage === 'Reading model metadata' ? 'active' : 'pending',
    },
  ];

  return (
    <div className={`boot${ready ? ' is-leaving' : ''}`} role="dialog" aria-modal="true" aria-label="Initializing simulation">
      <div className="boot__card">
        <MountainMark style={{ width: 34, height: 34 }} />
        <p className="boot__title">{ready ? 'READY' : 'INITIALIZING SIMULATION'}</p>
        <ul className="boot__steps" aria-live="polite">
          {steps.map((step) => (
            <li key={step.label} className="boot__step" data-state={step.state}>
              <span className="boot__icon">
                <StepIcon state={step.state} />
              </span>
              <span>
                {step.label}
                {step.detail && <span className="boot__detail">{step.detail}</span>}
              </span>
            </li>
          ))}
        </ul>

        {offline && (
          <div style={{ marginTop: 16 }}>
            <p className="small-note">The simulator needs the prediction API. From the project root, run:</p>
            <code className="boot__command">python -m uvicorn api.app.main:app --port 8000</code>
            <p className="small-note" style={{ marginTop: 8 }}>
              Retrying automatically…
            </p>
          </div>
        )}
        {failed && (
          <div style={{ marginTop: 16, display: 'grid', gap: 10 }}>
            <p className="notice notice--danger">{backend.message}</p>
            <button type="button" className="btn" onClick={onRetry}>
              Retry
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

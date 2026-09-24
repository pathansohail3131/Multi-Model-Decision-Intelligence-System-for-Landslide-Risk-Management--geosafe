import { useResultIsStale, useSimulationStore } from '../../store/simulationStore';
import type { SimulationPhase } from '../../types/simulation';

type Tone = 'neutral' | 'accent' | 'busy' | 'ok' | 'danger';

const PHASES: Record<SimulationPhase, { label: string; tone: Tone }> = {
  initializing: { label: 'Initializing', tone: 'busy' },
  ready: { label: 'Ready', tone: 'neutral' },
  modified: { label: 'Simulation configured', tone: 'accent' },
  validating: { label: 'Validating', tone: 'busy' },
  simulating: { label: 'Processing', tone: 'busy' },
  predicting: { label: 'Running model', tone: 'busy' },
  result: { label: 'Prediction complete', tone: 'ok' },
  error: { label: 'Error', tone: 'danger' },
  resetting: { label: 'Resetting', tone: 'busy' },
};

export function StatusBadge() {
  const phase = useSimulationStore((s) => s.phase);
  const stale = useResultIsStale();
  const entry = PHASES[phase];
  const label = phase === 'modified' && stale ? 'Parameters modified' : entry.label;
  return (
    <span className="status-badge" data-tone={entry.tone} role="status" aria-label={`Simulation state: ${label}`}>
      <span className="status-badge__dot" />
      {label.toUpperCase()}
    </span>
  );
}

export function EngineStatus() {
  const backend = useSimulationStore((s) => s.backend);
  const tone: Tone =
    backend.status === 'ready' ? 'ok' : backend.status === 'error' || backend.status === 'offline' ? 'danger' : 'busy';
  const text =
    backend.status === 'ready'
      ? 'Prediction engine online'
      : backend.status === 'offline'
        ? 'Engine offline'
        : backend.status === 'error'
          ? 'Engine error'
          : 'Loading models';
  return (
    <span className="engine-status" data-tone={tone} title={backend.message ?? backend.stage}>
      <span className="status-badge__dot" style={{ background: toneColour(tone) }} />
      <span className="engine-status__text">{text}</span>
    </span>
  );
}

function toneColour(tone: Tone) {
  switch (tone) {
    case 'ok':
      return 'var(--ok)';
    case 'danger':
      return 'var(--danger)';
    case 'busy':
      return 'var(--accent)';
    default:
      return 'var(--text-3)';
  }
}

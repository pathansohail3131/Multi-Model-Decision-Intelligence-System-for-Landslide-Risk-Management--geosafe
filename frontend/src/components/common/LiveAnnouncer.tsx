import { useEffect, useState } from 'react';
import { useSimulationStore } from '../../store/simulationStore';
import { formatProbability } from '../../utils/format';

/** Announces run progress and results to screen readers. */
export function LiveAnnouncer() {
  const phase = useSimulationStore((s) => s.phase);
  const result = useSimulationStore((s) => s.result);
  const error = useSimulationStore((s) => s.error);
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (phase === 'predicting') setMessage('Running the model.');
    if (phase === 'error' && error) setMessage(`${error.title}. ${error.message}`);
  }, [phase, error]);

  useEffect(() => {
    if (!result) return;
    const p = result.response.prediction;
    setMessage(
      `Prediction complete. ${p.class.label} susceptibility, score ${formatProbability(p.susceptibility)}${
        p.decision ? `. Planner decision: ${p.decision.label}` : ''
      }.`,
    );
  }, [result]);

  return (
    <div className="visually-hidden" aria-live="polite" aria-atomic="true">
      {message}
    </div>
  );
}

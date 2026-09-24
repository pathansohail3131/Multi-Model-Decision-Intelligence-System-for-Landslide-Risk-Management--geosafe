import { useCallback, useEffect, useRef } from 'react';
import { ApiError, predictionApi } from '../services/predictionApi';
import { useSimulationStore } from '../store/simulationStore';

const POLL_MS = 700;
const OFFLINE_RETRY_MS = 2500;

/**
 * Real startup sequence: poll the API's health endpoint until the model
 * bundles report loaded, then fetch the model metadata. Nothing here is timed
 * for effect — each step completes when the backend says it has.
 */
export function useBootSequence() {
  const setBackend = useSimulationStore((s) => s.setBackend);
  const initialise = useSimulationStore((s) => s.initialise);
  const attempt = useRef(0);

  const boot = useCallback(async () => {
    const run = ++attempt.current;
    const alive = () => run === attempt.current;
    setBackend({ status: 'connecting', stage: 'Connecting to prediction engine', message: null });

    for (;;) {
      if (!alive()) return;
      try {
        const health = await predictionApi.health();
        if (!alive()) return;
        if (health.status === 'error') {
          setBackend({ status: 'error', stage: 'Model loading failed', message: health.error });
          return;
        }
        if (health.status !== 'ready') {
          setBackend({ status: 'loading', stage: health.stage, message: null, elapsed: health.elapsed_s });
          await new Promise((resolve) => setTimeout(resolve, POLL_MS));
          continue;
        }
        setBackend({ status: 'loading', stage: 'Reading model metadata', elapsed: health.elapsed_s });
        const metadata = await predictionApi.metadata();
        if (!alive()) return;
        initialise(metadata);
        setBackend({ status: 'ready', stage: 'Ready', message: null });
        return;
      } catch (error) {
        if (!alive()) return;
        const message = error instanceof ApiError ? error.message : 'Unexpected error while connecting.';
        if (error instanceof ApiError && error.kind === 'http') {
          setBackend({ status: 'error', stage: 'Prediction engine error', message });
          return;
        }
        setBackend({ status: 'offline', stage: 'Waiting for prediction engine', message });
        await new Promise((resolve) => setTimeout(resolve, OFFLINE_RETRY_MS));
      }
    }
  }, [initialise, setBackend]);

  useEffect(() => {
    void boot();
    return () => {
      attempt.current += 1;
    };
  }, [boot]);

  return { retry: boot };
}

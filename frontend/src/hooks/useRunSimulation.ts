import { useCallback } from 'react';
import { ApiError, predictionApi } from '../services/predictionApi';
import { isRunningPhase, useSimulationStore } from '../store/simulationStore';
import type { FactorName } from '../types/api';
import type { FieldErrors, SimulationError } from '../types/simulation';
import { validateInputs } from '../utils/scenario';
import { useReducedMotion } from './useMediaQuery';

/**
 * The RUN SIMULATION sequence.
 *
 * Only two things here take real time: validation and the model call. The
 * short pauses around them give the scene time to show the scenario lock and
 * the sampling sweep; they are capped at a fraction of a second and collapse
 * almost to nothing under prefers-reduced-motion. The prediction itself always
 * comes from POST /api/predict.
 */
export function useRunSimulation() {
  const reducedMotion = useReducedMotion();

  return useCallback(async () => {
    const store = useSimulationStore.getState();
    const { metadata, inputs, phase, backend } = store;
    if (!metadata || !inputs || isRunningPhase(phase) || phase === 'resetting' || backend.status !== 'ready') return;

    const scenario = { ...inputs };
    const sourceLabel = store.source?.label ?? 'Scenario';
    const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, reducedMotion ? Math.min(ms, 40) : ms));

    store.beginRun();
    const errors = validateInputs(metadata, scenario);
    await pause(200);
    const invalid = Object.keys(errors) as FactorName[];
    if (invalid.length) {
      const names = invalid.map((name) => metadata.factors.find((f) => f.name === name)?.label ?? name);
      store.failRun(
        {
          kind: 'validation',
          title: 'Check the highlighted parameters',
          message: `${names.join(', ')} ${invalid.length === 1 ? 'is' : 'are'} outside physical limits.`,
        },
        errors,
      );
      return;
    }

    store.setRunStep('lock', 'simulating');
    await pause(380);
    store.setRunStep('prepare');
    await pause(520);
    store.setRunStep('infer', 'predicting');

    try {
      const response = await predictionApi.predict({ factors: scenario, explain: true });
      store.setRunStep('respond');
      store.completeRun({ at: response.generated_utc, sourceLabel, inputs: scenario, response });
    } catch (error) {
      const { simulationError, fieldErrors } = describeError(error);
      console.error('[run] prediction failed', error);
      store.failRun(simulationError, fieldErrors);
      return;
    }

    await pause(1100);
    const current = useSimulationStore.getState();
    if (current.phase === 'result' && current.runStep === 'transition') {
      current.setRunStep('analysis');
    }
  }, [reducedMotion]);
}

function describeError(error: unknown): { simulationError: SimulationError; fieldErrors: FieldErrors } {
  if (!(error instanceof ApiError)) {
    return {
      simulationError: { kind: 'http', title: 'Prediction failed', message: 'An unexpected error occurred.' },
      fieldErrors: {},
    };
  }
  if (error.kind === 'network') {
    return {
      simulationError: {
        kind: 'network',
        title: 'Prediction engine unreachable',
        message: error.message,
        detail: 'Start it from the project root: python -m uvicorn api.app.main:app --port 8000',
      },
      fieldErrors: {},
    };
  }
  if (error.kind === 'timeout') {
    return { simulationError: { kind: 'timeout', title: 'Prediction timed out', message: error.message }, fieldErrors: {} };
  }
  if (error.kind === 'malformed') {
    return { simulationError: { kind: 'malformed', title: 'Unexpected response', message: error.message }, fieldErrors: {} };
  }
  if (error.status === 422 && error.fields?.length) {
    const fieldErrors: FieldErrors = {};
    for (const field of error.fields) fieldErrors[field.field as FactorName] = field.message;
    return {
      simulationError: { kind: 'validation', title: 'The model rejected some values', message: error.message },
      fieldErrors,
    };
  }
  if (error.status === 503) {
    return { simulationError: { kind: 'models', title: 'Models unavailable', message: error.message }, fieldErrors: {} };
  }
  return { simulationError: { kind: 'http', title: 'Prediction failed', message: error.message }, fieldErrors: {} };
}

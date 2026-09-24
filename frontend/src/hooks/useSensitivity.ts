import { useEffect, useState } from 'react';
import { ApiError, predictionApi } from '../services/predictionApi';
import type { FactorName, SensitivityResponse } from '../types/api';
import type { RunRecord } from '../types/simulation';

type SensitivityState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ready'; data: SensitivityResponse }
  | { status: 'error'; message: string };

const cache = new Map<string, SensitivityResponse>();

/** Real model sweep of one factor for the scenario of a completed run. */
export function useSensitivity(run: RunRecord | null, factor: FactorName | null): SensitivityState {
  const key = run && factor ? `${run.id}:${factor}` : null;
  const [state, setState] = useState<SensitivityState>({ status: 'idle' });

  useEffect(() => {
    if (!run || !factor || !key) {
      setState({ status: 'idle' });
      return;
    }
    const cached = cache.get(key);
    if (cached) {
      setState({ status: 'ready', data: cached });
      return;
    }
    let cancelled = false;
    setState({ status: 'loading' });
    predictionApi
      .sensitivity({ factors: run.inputs, factor, points: 33 })
      .then((data) => {
        cache.set(key, data);
        if (!cancelled) setState({ status: 'ready', data });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({ status: 'error', message: error instanceof ApiError ? error.message : 'Sweep failed.' });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [key, run, factor]);

  return state;
}

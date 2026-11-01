import { create } from 'zustand';
import type { FactorName, InputFeatures, ModelMetadata } from '../types/api';
import type {
  BackendStatus,
  CameraMode,
  FieldErrors,
  Lens,
  RunRecord,
  RunStep,
  ScenarioSource,
  SceneElement,
  SimulationError,
  SimulationPhase,
} from '../types/simulation';
import { sourcePlace } from '../utils/place';
import { FACTOR_ELEMENT } from '../utils/sceneMapping';
import { medianScenario, sameScenario } from '../utils/scenario';

const MAX_HISTORY = 12;

export const RUNNING_PHASES: SimulationPhase[] = ['validating', 'simulating', 'predicting'];

export interface BackendState {
  status: BackendStatus;
  stage: string;
  message: string | null;
  elapsed: number | null;
}

interface SimulationState {
  phase: SimulationPhase;
  runStep: RunStep | null;
  backend: BackendState;
  metadata: ModelMetadata | null;
  inputs: InputFeatures | null;
  source: ScenarioSource | null;
  fieldErrors: FieldErrors;
  result: RunRecord | null;
  history: RunRecord[];
  baselineId: string | null;
  error: SimulationError | null;
  lens: Lens;
  cameraMode: CameraMode;
  focusElement: SceneElement | null;
  cameraResetToken: number;
  hovered: SceneElement | null;
  activeFactor: FactorName | null;
  districtOpen: boolean;
  webglFailed: boolean;
  /** Bumped to remount the canvas with a fresh WebGL context. */
  canvasKey: number;

  setBackend: (backend: Partial<BackendState>) => void;
  initialise: (metadata: ModelMetadata) => void;
  setInput: (name: FactorName, value: number) => void;
  applyScenario: (inputs: InputFeatures, source: ScenarioSource) => void;
  beginReset: () => void;
  completeReset: () => void;
  setLens: (lens: Lens) => void;
  setActiveFactor: (name: FactorName | null) => void;
  setCameraMode: (mode: CameraMode) => void;
  resetCamera: () => void;
  setHovered: (element: SceneElement | null) => void;
  setDistrictOpen: (open: boolean) => void;
  setWebglFailed: () => void;
  retryWebgl: () => void;

  beginRun: () => void;
  setRunStep: (step: RunStep, phase?: SimulationPhase) => void;
  failRun: (error: SimulationError, fieldErrors?: FieldErrors) => void;
  completeRun: (record: Omit<RunRecord, 'id' | 'index'>) => void;
  dismissError: () => void;
  setBaseline: (id: string | null) => void;
}

export const useSimulationStore = create<SimulationState>()((set, get) => ({
  phase: 'initializing',
  runStep: null,
  backend: { status: 'connecting', stage: 'Connecting', message: null, elapsed: null },
  metadata: null,
  inputs: null,
  source: null,
  fieldErrors: {},
  result: null,
  history: [],
  baselineId: null,
  error: null,
  lens: 'all',
  cameraMode: 'overview',
  focusElement: null,
  cameraResetToken: 0,
  hovered: null,
  activeFactor: null,
  districtOpen: false,
  webglFailed: false,
  canvasKey: 0,

  setBackend: (backend) => set((state) => ({ backend: { ...state.backend, ...backend } })),

  initialise: (metadata) =>
    set({
      metadata,
      inputs: medianScenario(metadata),
      source: { kind: 'median', label: 'Training-data median' },
      phase: 'ready',
    }),

  setInput: (name, value) => {
    const state = get();
    if (!state.inputs || RUNNING_PHASES.includes(state.phase)) return;
    const inputs = { ...state.inputs, [name]: value };
    const { [name]: _removed, ...fieldErrors } = state.fieldErrors;
    void _removed;
    const source: ScenarioSource =
      state.source?.kind === 'custom'
        ? state.source
        : { kind: 'custom', label: 'Custom scenario', basedOn: state.source?.label ?? 'median', place: sourcePlace(state.source) };
    const matchesResult = state.result && state.metadata && sameScenario(state.metadata, state.result.inputs, inputs);
    set({
      inputs,
      fieldErrors,
      source,
      phase: matchesResult ? 'result' : 'modified',
      error: state.error?.kind === 'validation' && Object.keys(fieldErrors).length === 0 ? null : state.error,
      activeFactor: name,
      focusElement: FACTOR_ELEMENT[name],
      cameraMode: 'configure',
    });
  },

  applyScenario: (inputs, source) => {
    if (RUNNING_PHASES.includes(get().phase)) return;
    set({
      inputs: { ...inputs },
      source,
      fieldErrors: {},
      error: null,
      phase: 'modified',
      cameraMode: 'overview',
      focusElement: null,
    });
  },

  beginReset: () => set({ phase: 'resetting', cameraMode: 'overview', focusElement: null, runStep: null }),

  completeReset: () => {
    const { metadata } = get();
    if (!metadata) return;
    set({
      inputs: medianScenario(metadata),
      source: { kind: 'median', label: 'Training-data median' },
      result: null,
      fieldErrors: {},
      error: null,
      lens: 'all',
      activeFactor: null,
      phase: 'ready',
      cameraResetToken: get().cameraResetToken + 1,
    });
  },

  setLens: (lens) => set({ lens }),

  setActiveFactor: (name) =>
    set(
      name
        ? { activeFactor: name, focusElement: FACTOR_ELEMENT[name], cameraMode: 'configure' }
        : { activeFactor: null },
    ),

  setCameraMode: (mode) => set({ cameraMode: mode, focusElement: mode === 'configure' ? get().focusElement : null }),

  resetCamera: () =>
    set((state) => ({ cameraMode: 'overview', focusElement: null, cameraResetToken: state.cameraResetToken + 1 })),

  setHovered: (hovered) => set({ hovered }),

  setDistrictOpen: (districtOpen) => set({ districtOpen }),

  setWebglFailed: () => set({ webglFailed: true }),

  // A lost context is usually recoverable: rebuild the canvas from scratch.
  retryWebgl: () => set((state) => ({ webglFailed: false, canvasKey: state.canvasKey + 1 })),

  beginRun: () => set({ phase: 'validating', runStep: 'validate', error: null, fieldErrors: {}, activeFactor: null }),

  setRunStep: (runStep, phase) =>
    set((state) => ({
      runStep,
      phase: phase ?? state.phase,
      cameraMode: runStep === 'lock' || runStep === 'prepare' || runStep === 'infer' ? 'processing' : state.cameraMode,
    })),

  failRun: (error, fieldErrors = {}) =>
    set({ phase: 'error', runStep: null, error, fieldErrors, cameraMode: 'overview' }),

  completeRun: (partial) => {
    const state = get();
    const index = (state.history[0]?.index ?? 0) + 1;
    const record: RunRecord = { ...partial, id: `run-${index}`, index };
    const history = [record, ...state.history].slice(0, MAX_HISTORY);
    const baselineStillPresent = history.some((run) => run.id === state.baselineId);
    set({
      result: record,
      history,
      baselineId: baselineStillPresent ? state.baselineId : null,
      phase: 'result',
      runStep: 'transition',
      cameraMode: 'result',
      focusElement: 'site',
    });
  },

  dismissError: () => {
    const { result, inputs, metadata } = get();
    const matches = result && inputs && metadata && sameScenario(metadata, result.inputs, inputs);
    set({ error: null, phase: matches ? 'result' : 'modified' });
  },

  setBaseline: (baselineId) => set({ baselineId }),
}));

export const isRunningPhase = (phase: SimulationPhase) => RUNNING_PHASES.includes(phase);

/** True when the inputs no longer match the scenario the current result was computed for. */
export function useResultIsStale(): boolean {
  return useSimulationStore((state) => {
    if (!state.result || !state.inputs || !state.metadata) return false;
    return !sameScenario(state.metadata, state.result.inputs, state.inputs);
  });
}

import type { FactorName, InputFeatures, Place, PredictionResponse, ThemeName, TrainingSample, DistrictSample } from './api';

/** Application states. They drive both the interface and the 3D scene. */
export type SimulationPhase =
  | 'initializing'
  | 'ready'
  | 'modified'
  | 'validating'
  | 'simulating'
  | 'predicting'
  | 'result'
  | 'error'
  | 'resetting';

/** Steps of one RUN SIMULATION sequence, in order. */
export type RunStep = 'validate' | 'lock' | 'prepare' | 'infer' | 'respond' | 'transition' | 'analysis';

export type CameraMode = 'overview' | 'configure' | 'processing' | 'result' | 'inspect';

export type Lens = 'all' | ThemeName;

export type BackendStatus = 'connecting' | 'loading' | 'ready' | 'error' | 'offline';

export type ScenarioSource =
  | { kind: 'median'; label: string }
  | { kind: 'training'; label: string; sample: TrainingSample }
  | { kind: 'district'; label: string; site: DistrictSample }
  | { kind: 'custom'; label: string; basedOn: string; place?: Place | null };

/** One completed, real model run. */
export interface RunRecord {
  id: string;
  index: number;
  at: string;
  sourceLabel: string;
  inputs: InputFeatures;
  response: PredictionResponse;
}

export interface SimulationError {
  title: string;
  message: string;
  kind: 'validation' | 'network' | 'timeout' | 'http' | 'malformed' | 'models';
  detail?: string;
}

export type FieldErrors = Partial<Record<FactorName, string>>;

/** Scene element identifiers used for hover, selection and lens emphasis. */
export type SceneElement = 'site' | 'terrain' | 'stream' | 'road' | 'fault' | 'strata' | 'vegetation' | 'rain';

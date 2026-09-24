import type {
  ApiErrorBody,
  DistrictMeta,
  DistrictSample,
  HealthResponse,
  ModelMetadata,
  PredictionRequest,
  PredictionResponse,
  SensitivityRequest,
  SensitivityResponse,
  TrainingSample,
} from '../types/api';

const BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') ?? '';

export type ApiErrorKind = 'network' | 'timeout' | 'http' | 'malformed';

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;
  readonly code?: string;
  readonly fields?: { field: string; message: string }[];

  constructor(
    kind: ApiErrorKind,
    message: string,
    options: { status?: number; code?: string; fields?: { field: string; message: string }[] } = {},
  ) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = options.status;
    this.code = options.code;
    this.fields = options.fields;
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  return isRecord(value) && isRecord(value.error) && typeof value.error.message === 'string';
}

async function request<T>(
  path: string,
  init: RequestInit & { timeoutMs?: number } = {},
  validate?: (body: unknown) => body is T,
): Promise<T> {
  const { timeoutMs = 20000, ...rest } = init;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...rest,
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...(rest.headers ?? {}) },
    });
  } catch (error) {
    window.clearTimeout(timer);
    if (controller.signal.aborted) {
      throw new ApiError('timeout', `The prediction engine did not answer within ${Math.round(timeoutMs / 1000)} s.`);
    }
    console.error('[predictionApi] network error', path, error);
    throw new ApiError('network', 'Cannot reach the prediction engine. Is the API server running?');
  }
  window.clearTimeout(timer);

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new ApiError('malformed', `The prediction engine returned a non-JSON response (HTTP ${response.status}).`, {
      status: response.status,
    });
  }

  if (!response.ok) {
    if (isApiErrorBody(body)) {
      throw new ApiError('http', body.error.message, {
        status: response.status,
        code: body.error.code,
        fields: body.error.fields,
      });
    }
    throw new ApiError('http', `The prediction engine returned HTTP ${response.status}.`, { status: response.status });
  }

  if (validate && !validate(body)) {
    console.error('[predictionApi] malformed response', path, body);
    throw new ApiError('malformed', 'The prediction engine returned a response in an unexpected format.');
  }
  return body as T;
}

// ---------------------------------------------------------------------------
// response guards — enough structure to render safely, not a full schema
// ---------------------------------------------------------------------------
function isPredictionResponse(body: unknown): body is PredictionResponse {
  if (!isRecord(body) || !isRecord(body.prediction) || !Array.isArray(body.models)) return false;
  const p = body.prediction;
  return (
    isNumber(p.susceptibility) &&
    isNumber(p.threshold) &&
    typeof p.above_threshold === 'boolean' &&
    isRecord(p.class) &&
    typeof p.class.key === 'string' &&
    isNumber(p.full_model) &&
    isNumber(p.disagreement) &&
    (p.confidence === null || isNumber(p.confidence)) &&
    body.models.every((m) => isRecord(m) && typeof m.name === 'string' && isNumber(m.probability)) &&
    Array.isArray(body.warnings)
  );
}

function isMetadata(body: unknown): body is ModelMetadata {
  return (
    isRecord(body) &&
    Array.isArray(body.factors) &&
    body.factors.length === 14 &&
    Array.isArray(body.models) &&
    isRecord(body.threshold) &&
    isNumber(body.threshold.value) &&
    Array.isArray(body.classes)
  );
}

function isHealth(body: unknown): body is HealthResponse {
  return isRecord(body) && typeof body.status === 'string';
}

function isSensitivity(body: unknown): body is SensitivityResponse {
  return (
    isRecord(body) &&
    Array.isArray(body.values) &&
    Array.isArray(body.susceptibility) &&
    body.values.length === body.susceptibility.length &&
    isRecord(body.thematic)
  );
}

function isTrainingSample(body: unknown): body is TrainingSample {
  return isRecord(body) && isRecord(body.factors) && isNumber(body.sample_id);
}

function isDistrictMeta(body: unknown): body is DistrictMeta {
  return isRecord(body) && typeof body.status === 'string' && Array.isArray(body.landslides);
}

function isDistrictSample(body: unknown): body is DistrictSample {
  return isRecord(body) && typeof body.inside === 'boolean';
}

export const predictionApi = {
  health: () => request<HealthResponse>('/api/health', { timeoutMs: 4000 }, isHealth),
  metadata: () => request<ModelMetadata>('/api/metadata', {}, isMetadata),
  predict: (payload: PredictionRequest) =>
    request<PredictionResponse>(
      '/api/predict',
      { method: 'POST', body: JSON.stringify(payload), timeoutMs: 30000 },
      isPredictionResponse,
    ),
  sensitivity: (payload: SensitivityRequest) =>
    request<SensitivityResponse>(
      '/api/sensitivity',
      { method: 'POST', body: JSON.stringify(payload), timeoutMs: 30000 },
      isSensitivity,
    ),
  randomSample: (label?: 0 | 1) =>
    request<TrainingSample>(`/api/samples/random${label === undefined ? '' : `?label=${label}`}`, {}, isTrainingSample),
  districtMeta: () => request<DistrictMeta>('/api/district/meta', {}, isDistrictMeta),
  districtSample: (x: number, y: number) =>
    request<DistrictSample>(`/api/district/sample?x=${x.toFixed(1)}&y=${y.toFixed(1)}`, {}, isDistrictSample),
  districtImageUrl: (layer: string) => `${BASE}/api/district/image/${layer}.png`,
};

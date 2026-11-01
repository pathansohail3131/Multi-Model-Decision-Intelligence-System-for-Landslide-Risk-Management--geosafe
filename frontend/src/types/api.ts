/**
 * Types mirroring the FastAPI responses in api/app/services/.
 * Every value shown in the simulator arrives through one of these shapes.
 */

export type FactorName =
  | 'elevation'
  | 'slope'
  | 'aspect'
  | 'plan_curv'
  | 'profile_curv'
  | 'TWI'
  | 'SPI'
  | 'dist_drainage'
  | 'rainfall'
  | 'lithology'
  | 'dist_fault'
  | 'dist_road'
  | 'LULC'
  | 'NDVI';

export type ThemeName = 'topographic' | 'hydrological' | 'geological' | 'anthropogenic';
export type ModelName = ThemeName | 'full';

/** The 14 raw conditioning factors for one 30 m cell. Categorical factors are class codes. */
export type InputFeatures = Record<FactorName, number>;

export interface TrainingStats {
  min: number;
  p01: number;
  p05: number;
  p25: number;
  median: number;
  p75: number;
  p95: number;
  p99: number;
  max: number;
  median_landslide: number;
  median_non_landslide: number;
}

export interface CategoryLevel {
  code: number;
  key: string;
  label: string;
  n_samples: number;
  n_landslides: number;
  landslide_rate: number;
}

interface FactorSpecBase {
  name: FactorName;
  label: string;
  unit: string;
  theme: ThemeName;
  description: string;
}

export interface NumericFactorSpec extends FactorSpecBase {
  kind: 'numeric';
  bounds: [number, number];
  decimals: number;
  scale: 'linear' | 'log';
  training: TrainingStats;
}

export interface CategoricalFactorSpec extends FactorSpecBase {
  kind: 'categorical';
  levels: CategoryLevel[];
}

export type FactorSpec = NumericFactorSpec | CategoricalFactorSpec;

export interface SpatialCvSummary {
  scheme: string | null;
  block_size_m: number | null;
  n_splits: number | null;
  roc_auc_mean: number | null;
  roc_auc_std: number | null;
  average_precision_mean: number | null;
  average_precision_std: number | null;
}

export interface ModelCard {
  name: ModelName;
  label: string;
  role: 'thematic' | 'benchmark';
  algorithm: string;
  features: string[];
  factors: FactorName[];
  calibrated: boolean;
  created_utc: string | null;
  spatial_cv: SpatialCvSummary | null;
}

export interface ClassBand {
  key: SusceptibilityClassKey;
  label: string;
  range: [number, number];
}

export type SusceptibilityClassKey = 'very_low' | 'low' | 'moderate' | 'high' | 'very_high';
export type DecisionKey = 'act_now' | 'survey_first' | 'monitor' | 'deprioritise';

export interface DecisionInfo {
  key: DecisionKey;
  label: string;
  interpretation: string;
}

export interface ModelMetadata {
  project: { name: string; region: string };
  factors: FactorSpec[];
  models: ModelCard[];
  threshold: {
    value: number;
    method: string | null;
    cost_false_negative: number | null;
    cost_false_positive: number | null;
    oof_metrics: { recall: number; precision: number; f2: number; flagged_rate: number; n: number } | null;
  };
  production: {
    rule: string;
    oof_metrics: { roc_auc: number; average_precision: number; n: number } | null;
  };
  classes: ClassBand[];
  decisions: DecisionInfo[];
  confidence: {
    available: boolean;
    ceiling: number | null;
    split: number | null;
    method: string;
    n_reference: number | null;
  };
  dataset: { n_samples: number; n_landslides: number; cell_size_m: number };
  caveats: string[];
  report_warnings: string[];
  loaded_utc: string;
}

export interface HealthResponse {
  status: 'idle' | 'loading' | 'ready' | 'error';
  stage: string;
  error: string | null;
  elapsed_s: number | null;
  models: ModelName[];
  report_warnings: string[];
  district_map: 'idle' | 'loading' | 'ready' | 'error';
}

export interface PredictionRequest {
  factors: InputFeatures;
  explain?: boolean;
}

export interface ModelOutput {
  name: ModelName;
  label: string;
  role: 'thematic' | 'benchmark';
  probability: number;
  raw_probability: number;
}

export interface ShapFeature {
  feature: string;
  factor: FactorName;
  encoded_value: number | string | null;
  shap: number;
}

export interface Explanation {
  model: ModelName;
  base_log_odds: number;
  output_log_odds: number;
  features: ShapFeature[];
  by_factor: { factor: FactorName; shap: number }[];
}

export interface PredictionWarning {
  factor: FactorName;
  kind: 'outside_training_range' | 'unseen_level' | 'sparse_level' | 'flat_cell' | 'coverage_gap';
  message: string;
}

export interface PredictionResponse {
  prediction: {
    susceptibility: number;
    threshold: number;
    above_threshold: boolean;
    class: ClassBand;
    full_model: number;
    thematic_mean: number;
    disagreement: number;
    confidence: number | null;
    confidence_split: number | null;
    decision: DecisionInfo | null;
  };
  models: ModelOutput[];
  explanations: Record<ModelName, Explanation | null> | null;
  warnings: PredictionWarning[];
  inputs: InputFeatures;
  timing_ms: { inference: number; explanation: number | null };
  generated_utc: string;
}

export interface SensitivityRequest {
  factors: InputFeatures;
  factor: FactorName;
  points?: number;
}

export interface SensitivityResponse {
  factor: FactorName;
  values: number[];
  current_value: number;
  susceptibility: number[];
  full_model: number[];
  thematic: Record<ThemeName, number[]>;
  threshold: number;
  range_basis: string;
}

export interface Place {
  village: string | null;
  inside_village: boolean;
  is_town: boolean;
  taluk: string | null;
  district: string;
  state: string;
  nearest_town: { name: string; distance_m: number; direction_from_town: string } | null;
  label: string;
  sublabel: string;
}

export interface PlaceHit {
  name: string;
  kind: 'village' | 'town' | 'taluk';
  taluk: string | null;
  x: number;
  y: number;
}

export interface TalukOutline {
  name: string;
  rings: [number, number][][];
  label: [number, number];
}

export interface TrainingSample {
  sample_id: number;
  label: 0 | 1;
  taluk: string | null;
  x: number;
  y: number;
  factors: InputFeatures;
  note: string;
  place?: Place | null;
}

export interface DistrictMeta {
  status: 'idle' | 'loading' | 'ready' | 'error';
  error: string | null;
  grid: { crs: string; bounds: [number, number, number, number]; image_size: [number, number] } | null;
  layers: Record<string, { label: string; source: string; partial?: boolean }>;
  landslides: [number, number][];
  taluks?: TalukOutline[];
}

export interface DistrictSample {
  inside: boolean;
  x: number;
  y: number;
  lon?: number | null;
  lat?: number | null;
  factors: InputFeatures | null;
  missing: FactorName[];
  issues: string[];
  nearest_landslide_m?: number | null;
  landslides_within_1km?: number | null;
  place?: Place | null;
}

export interface ApiErrorBody {
  error: { code: string; message: string; fields?: { field: string; message: string }[] };
}

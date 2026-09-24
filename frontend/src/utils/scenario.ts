import type {
  CategoricalFactorSpec,
  FactorName,
  FactorSpec,
  InputFeatures,
  ModelMetadata,
  NumericFactorSpec,
} from '../types/api';
import type { FieldErrors } from '../types/simulation';

export function specFor(meta: ModelMetadata, name: FactorName): FactorSpec {
  const spec = meta.factors.find((f) => f.name === name);
  if (!spec) throw new Error(`No metadata for factor ${name}`);
  return spec;
}

export function numericSpecs(meta: ModelMetadata): NumericFactorSpec[] {
  return meta.factors.filter((f): f is NumericFactorSpec => f.kind === 'numeric');
}

/**
 * The starting scenario: the training-data median of every numeric factor and
 * the most-sampled level of each categorical one. It is a reference point, not
 * a real cell, and is labelled as such in the interface.
 */
export function medianScenario(meta: ModelMetadata): InputFeatures {
  const values = {} as InputFeatures;
  for (const spec of meta.factors) {
    if (spec.kind === 'numeric') {
      values[spec.name] = spec.training.median;
    } else {
      const top = [...spec.levels].sort((a, b) => b.n_samples - a.n_samples)[0];
      values[spec.name] = top.code;
    }
  }
  return values;
}

function tolerance(spec: FactorSpec): number {
  return spec.kind === 'numeric' ? 0.5 * 10 ** -spec.decimals : 0;
}

export function changedFactors(meta: ModelMetadata, a: InputFeatures, b: InputFeatures): FactorName[] {
  return meta.factors.filter((spec) => Math.abs(a[spec.name] - b[spec.name]) > tolerance(spec)).map((s) => s.name);
}

export function sameScenario(meta: ModelMetadata, a: InputFeatures, b: InputFeatures): boolean {
  return changedFactors(meta, a, b).length === 0;
}

export function validateInputs(meta: ModelMetadata, inputs: InputFeatures): FieldErrors {
  const errors: FieldErrors = {};
  for (const spec of meta.factors) {
    const value = inputs[spec.name];
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      errors[spec.name] = 'Enter a number.';
      continue;
    }
    if (spec.kind === 'numeric') {
      const [low, high] = spec.bounds;
      if (value < low || value > high) {
        errors[spec.name] = `Must be between ${low} and ${high}${spec.unit ? ` ${spec.unit}` : ''} — values outside this are not physical terrain.`;
      }
    } else if (!(spec as CategoricalFactorSpec).levels.some((level) => level.code === value)) {
      errors[spec.name] = 'Choose one of the listed classes.';
    }
  }
  return errors;
}

export function trainingRangeStatus(spec: NumericFactorSpec, value: number): 'below' | 'above' | null {
  if (value < spec.training.min) return 'below';
  if (value > spec.training.max) return 'above';
  return null;
}

export function levelLabel(meta: ModelMetadata, name: 'lithology' | 'LULC', code: number): string {
  const spec = specFor(meta, name) as CategoricalFactorSpec;
  return spec.levels.find((level) => level.code === code)?.label ?? `Class ${code}`;
}

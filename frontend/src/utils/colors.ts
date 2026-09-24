import type { DecisionKey, ModelName, SusceptibilityClassKey } from '../types/api';

/** Susceptibility class palette. Must match CLASS_COLOURS in api/district.py. */
export const CLASS_COLOURS: Record<SusceptibilityClassKey, string> = {
  very_low: '#4f7d63',
  low: '#8fae5d',
  moderate: '#d9b44a',
  high: '#d9793a',
  very_high: '#c2412d',
};

export const DECISION_COLOURS: Record<DecisionKey, string> = {
  act_now: '#c2412d',
  survey_first: '#d9a23a',
  monitor: '#6b93c4',
  deprioritise: '#6f8f7b',
};

/** Theme hues are kept away from the red–green hazard ramp so the two never read as one scale. */
export const THEME_COLOURS: Record<ModelName, string> = {
  topographic: '#9d8cf0',
  hydrological: '#4fb6e0',
  geological: '#bba98c',
  anthropogenic: '#e08bb5',
  full: '#d7dde3',
};

export const THEME_LABELS: Record<ModelName, string> = {
  topographic: 'Topographic',
  hydrological: 'Hydrological',
  geological: 'Geological',
  anthropogenic: 'Anthropogenic',
  full: 'Full-feature',
};

export const NEUTRAL_MARKER = '#c9d1d9';

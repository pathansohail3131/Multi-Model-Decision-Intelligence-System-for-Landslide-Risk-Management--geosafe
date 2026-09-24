const COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];

export function compassPoint(degrees: number): string {
  const index = Math.round((((degrees % 360) + 360) % 360) / 22.5) % 16;
  return COMPASS[index];
}

export function formatNumber(value: number, decimals = 0): string {
  if (!Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (decimals > 0 && abs !== 0 && abs < 10 ** -decimals) {
    return value.toExponential(1);
  }
  // Large magnitudes do not need the precision small values of the same factor do.
  if (abs >= 1000) decimals = 0;
  else if (abs >= 100) decimals = Math.min(decimals, 1);
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function formatWithUnit(value: number, decimals: number, unit: string): string {
  const text = formatNumber(value, decimals);
  if (!unit) return text;
  return unit.startsWith('°') ? `${text}${unit}` : `${text} ${unit}`;
}

export function formatDistance(metres: number): string {
  if (!Number.isFinite(metres)) return '—';
  return metres < 1000 ? `${Math.round(metres)} m` : `${(metres / 1000).toFixed(metres < 10000 ? 2 : 1)} km`;
}

export function formatProbability(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined || !Number.isFinite(value) ? '—' : value.toFixed(digits);
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(digits)}%`;
}

export function formatSigned(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return '—';
  const text = Math.abs(value).toFixed(digits);
  if (Number(text) === 0) return `±${text}`;
  return `${value > 0 ? '+' : '−'}${text}`;
}

export function formatLatLon(lat?: number | null, lon?: number | null): string {
  if (lat == null || lon == null) return '';
  return `${lat.toFixed(4)}°N ${lon.toFixed(4)}°E`;
}

export function formatTime(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? ''
    : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

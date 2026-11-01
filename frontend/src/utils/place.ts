import type { DistrictSample, Place, TalukOutline } from '../types/api';
import type { ScenarioSource } from '../types/simulation';
import { formatLatLon } from './format';

/** The place a scenario came from. The typical-values start has none. */
export function sourcePlace(source: ScenarioSource | null | undefined): Place | null {
  if (source?.kind === 'training') return source.sample.place ?? null;
  if (source?.kind === 'district') return source.site.place ?? null;
  if (source?.kind === 'custom') return source.place ?? null;
  return null;
}

/** "in Sirsi" when under 1.5 km from the town, otherwise "9.1 km SE of Ankola". */
export function townHint(place: Place): string {
  const town = place.nearest_town;
  if (!town) return '';
  if (town.distance_m < 1500) return `in ${town.name}`;
  return `${(town.distance_m / 1000).toFixed(1)} km ${town.direction_from_town} of ${town.name}`;
}

/** "Shirur, Ankola taluk". A label that already names the taluk stands alone. */
export function shortPlace(place: Place): string {
  if (!place.taluk || place.label === `${place.taluk} taluk`) return place.label;
  return `${place.label}, ${place.taluk} taluk`;
}

/** The scenario source for a sampled district site. */
export function districtSource(site: DistrictSample): ScenarioSource {
  const where = site.place ? shortPlace(site.place) : formatLatLon(site.lat, site.lon);
  return { kind: 'district', label: `District site · ${where}`, site };
}

/** Even-odd test across all rings, so separate parts and holes both work. */
export function talukAt(taluks: TalukOutline[], x: number, y: number): string | null {
  for (const taluk of taluks) {
    let inside = false;
    for (const ring of taluk.rings) {
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
        const [xi, yi] = ring[i];
        const [xj, yj] = ring[j];
        if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
      }
    }
    if (inside) return taluk.name;
  }
  return null;
}

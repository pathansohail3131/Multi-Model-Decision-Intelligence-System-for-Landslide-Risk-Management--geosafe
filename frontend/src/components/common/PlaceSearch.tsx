import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react';
import { ApiError, predictionApi } from '../../services/predictionApi';
import type { PlaceHit } from '../../types/api';
import { SearchIcon } from './Icons';

const DEBOUNCE_MS = 200;
const MIN_QUERY = 2;

/** Bolds the typed part. A fuzzy match may contain no such part. */
function Highlighted({ name, query }: { name: string; query: string }) {
  const at = name.toLowerCase().indexOf(query.trim().toLowerCase());
  if (at < 0 || !query.trim()) return <strong>{name}</strong>;
  const end = at + query.trim().length;
  return (
    <strong>
      {name.slice(0, at)}
      <mark>{name.slice(at, end)}</mark>
      {name.slice(end)}
    </strong>
  );
}

const describe = (hit: PlaceHit) =>
  hit.kind === 'taluk' ? 'taluk · Uttara Kannada' : `${hit.kind}${hit.taluk ? ` · ${hit.taluk} taluk` : ''}`;

export function PlaceSearch({ onPick, disabled, busy }: { onPick: (hit: PlaceHit) => void; disabled?: boolean; busy?: boolean }) {
  const [query, setQuery] = useState('');
  const [hits, setHits] = useState<PlaceHit[] | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listId = useId();
  const latest = useRef(0);

  useEffect(() => {
    const request = (latest.current += 1);
    const text = query.trim();
    if (text.length < MIN_QUERY) {
      setHits(null);
      setFailure(null);
      return;
    }
    const timer = window.setTimeout(async () => {
      try {
        const found = await predictionApi.searchPlaces(text);
        if (request !== latest.current) return;
        setHits(found);
        setFailure(null);
        setActive(0);
      } catch (error) {
        if (request !== latest.current) return;
        setHits(null);
        setFailure(error instanceof ApiError ? error.message : 'Search is unavailable.');
      }
    }, DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [query]);

  const pick = (hit: PlaceHit) => {
    setQuery('');
    setOpen(false);
    onPick(hit);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Escape') {
      if (open && (hits || failure)) {
        // Close the list only; an enclosing dialog stays open for a second Escape.
        event.stopPropagation();
        setOpen(false);
      }
      return;
    }
    if (!hits?.length) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const step = event.key === 'ArrowDown' ? 1 : -1;
      setActive(open ? (active + step + hits.length) % hits.length : 0);
      setOpen(true);
    } else if (event.key === 'Enter') {
      event.preventDefault();
      if (open) pick(hits[active]);
    }
  };

  const showList = open && (hits !== null || failure !== null);

  return (
    <div className="place-search">
      <div className="place-search__box">
        {busy ? <span className="spinner" /> : <SearchIcon />}
        <input
          type="text"
          role="combobox"
          aria-expanded={showList}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={showList && hits?.length ? `${listId}-${active}` : undefined}
          aria-label="Search a village, town or taluk"
          placeholder="Search a village, town or taluk"
          autoComplete="off"
          spellCheck={false}
          value={query}
          disabled={disabled}
          onChange={(event) => {
            setQuery(event.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setOpen(false)}
          onKeyDown={onKeyDown}
        />
      </div>
      {showList && (
        <ul className="place-search__list" id={listId} role="listbox">
          {failure && <li className="place-search__empty">{failure}</li>}
          {hits?.length === 0 && <li className="place-search__empty">No places found</li>}
          {hits?.map((hit, i) => (
            <li
              key={`${hit.kind}-${hit.name}-${hit.taluk}`}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={i === active}
              className={`place-search__option${i === active ? ' is-active' : ''}`}
              onMouseEnter={() => setActive(i)}
              // Keep focus in the input, so its blur does not close the list before the click lands.
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => pick(hit)}
            >
              <Highlighted name={hit.name} query={query} />
              <span>{describe(hit)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react';
import { ApiError, predictionApi } from '../../services/predictionApi';
import { useSimulationStore } from '../../store/simulationStore';
import type { DistrictMeta, DistrictSample, PlaceHit } from '../../types/api';
import { formatDistance, formatLatLon, formatNumber } from '../../utils/format';
import { districtSource, talukAt, townHint } from '../../utils/place';
import { levelLabel } from '../../utils/scenario';
import { CrossIcon, PinIcon, WarnIcon } from '../common/Icons';
import { PlaceSearch } from '../common/PlaceSearch';

type MetaState = { status: 'loading' } | { status: 'error'; message: string } | { status: 'ready'; meta: DistrictMeta };

export function DistrictLocator() {
  const open = useSimulationStore((s) => s.districtOpen);
  const setOpen = useSimulationStore((s) => s.setDistrictOpen);
  if (!open) return null;
  return <LocatorDialog onClose={() => setOpen(false)} />;
}

function LocatorDialog({ onClose }: { onClose: () => void }) {
  const metadata = useSimulationStore((s) => s.metadata);
  const applyScenario = useSimulationStore((s) => s.applyScenario);
  const [metaState, setMetaState] = useState<MetaState>({ status: 'loading' });
  const [showLandslides, setShowLandslides] = useState(true);
  const [showSusceptibility, setShowSusceptibility] = useState(false);
  const [showTaluks, setShowTaluks] = useState(true);
  const [hoveredTaluk, setHoveredTaluk] = useState<string | null>(null);
  const talukTip = useRef<HTMLDivElement>(null);
  const [pulsedTaluk, setPulsedTaluk] = useState<string | null>(null);
  const pulseTimer = useRef<number | undefined>(undefined);
  const [site, setSite] = useState<DistrictSample | null>(null);
  const [sampling, setSampling] = useState(false);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const [easting, setEasting] = useState('');
  const [northing, setNorthing] = useState('');
  const dialog = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      for (let attempt = 0; attempt < 40 && !cancelled; attempt += 1) {
        try {
          const meta = await predictionApi.districtMeta();
          if (cancelled) return;
          if (meta.status === 'ready' && meta.grid) {
            setMetaState({ status: 'ready', meta });
            return;
          }
          if (meta.status === 'error') {
            setMetaState({ status: 'error', message: meta.error ?? 'The district map could not be built.' });
            return;
          }
        } catch (error) {
          if (!cancelled) setMetaState({ status: 'error', message: error instanceof ApiError ? error.message : 'Map unavailable.' });
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 750));
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    dialog.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      previous?.focus();
    };
  }, [onClose]);

  useEffect(() => () => window.clearTimeout(pulseTimer.current), []);

  const onPlacePick = (hit: PlaceHit) => {
    setEasting(hit.x.toFixed(0));
    setNorthing(hit.y.toFixed(0));
    setPulsedTaluk(hit.taluk);
    window.clearTimeout(pulseTimer.current);
    pulseTimer.current = window.setTimeout(() => setPulsedTaluk(null), 1800);
    void sample(hit.x, hit.y);
  };

  const sample = async (x: number, y: number) => {
    setSampling(true);
    setSampleError(null);
    try {
      setSite(await predictionApi.districtSample(x, y));
    } catch (error) {
      setSampleError(error instanceof ApiError ? error.message : 'Sampling failed.');
    } finally {
      setSampling(false);
    }
  };

  const meta = metaState.status === 'ready' ? metaState.meta : null;
  const grid = meta?.grid ?? null;
  const taluks = useMemo(() => meta?.taluks ?? [], [meta]);

  const eventToGrid = (event: ReactMouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const [left, bottom, right, top] = grid!.bounds;
    return {
      x: left + ((event.clientX - rect.left) / rect.width) * (right - left),
      y: top - ((event.clientY - rect.top) / rect.height) * (top - bottom),
      px: event.clientX - rect.left,
      py: event.clientY - rect.top,
      flip: event.clientX - rect.left > rect.width * 0.6,
    };
  };

  // The tooltip follows the pointer through its ref, so moving the mouse does
  // not re-render the map; state changes only when the pointer crosses a boundary.
  const onMapMove = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (!grid || !showTaluks) return;
    const { x, y, px, py, flip } = eventToGrid(event);
    setHoveredTaluk(talukAt(taluks, x, y));
    if (talukTip.current) {
      talukTip.current.style.left = `${px}px`;
      talukTip.current.style.top = `${py}px`;
      talukTip.current.style.transform = `translate(${flip ? 'calc(-100% - 12px)' : '12px'}, 12px)`;
    }
  };

  const onMapClick = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (!grid) return;
    const { x, y } = eventToGrid(event);
    setEasting(x.toFixed(0));
    setNorthing(y.toFixed(0));
    void sample(x, y);
  };

  const toPercent = (x: number, y: number) => {
    if (!grid) return { left: 0, top: 0 };
    const [left, bottom, right, top] = grid.bounds;
    return { left: ((x - left) / (right - left)) * 100, top: ((top - y) / (top - bottom)) * 100 };
  };

  const applySite = () => {
    if (!site?.factors || site.issues.length) return;
    applyScenario(site.factors, districtSource(site));
    onClose();
  };

  const talukPaths = useMemo(() => {
    if (!grid) return [];
    const [left, bottom, right, top] = grid.bounds;
    return taluks.map((taluk) => ({
      name: taluk.name,
      d: taluk.rings
        .map(
          (ring) =>
            `M${ring
              .map(([x, y]) => `${(((x - left) / (right - left)) * 100).toFixed(3)} ${(((top - y) / (top - bottom)) * 100).toFixed(3)}`)
              .join('L')}Z`,
        )
        .join(''),
    }));
  }, [grid, taluks]);

  const susceptibilityLayer = meta?.layers.susceptibility;
  const marker = site ? toPercent(site.x, site.y) : null;

  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div ref={dialog} className="modal" role="dialog" aria-modal="true" aria-labelledby="district-title" tabIndex={-1}>
        <div className="modal__head">
          <div>
            <h2 className="panel__title" id="district-title">
              Pick a real site
            </h2>
            <p className="muted" style={{ fontSize: 12 }}>
              Click anywhere in Uttara Kannada. The API reads all 14 conditioning-factor rasters at that 30 m cell — the same
              values district inference uses.
            </p>
          </div>
          <button type="button" className="icon-btn" aria-label="Close" onClick={onClose}>
            <CrossIcon />
          </button>
        </div>

        <div className="modal__search">
          <PlaceSearch onPick={onPlacePick} />
        </div>

        <div className="modal__body">
          <div className="district-map">
            {metaState.status === 'loading' && (
              <p className="small-note" style={{ display: 'flex', gap: 8 }}>
                <span className="spinner" /> Preparing the district map…
              </p>
            )}
            {metaState.status === 'error' && <p className="notice notice--danger">{metaState.message}</p>}
            {grid && (
              <div
                className="district-map__frame"
                style={{ ['--map-aspect' as string]: `${grid.image_size[0]} / ${grid.image_size[1]}` }}
                onClick={onMapClick}
                onMouseMove={onMapMove}
                onMouseLeave={() => setHoveredTaluk(null)}
                role="application"
                aria-label="District map. Click to sample a location."
              >
                <img src={predictionApi.districtImageUrl('relief')} alt="Terrain relief of Uttara Kannada" draggable={false} />
                {susceptibilityLayer && showSusceptibility && (
                  <img src={predictionApi.districtImageUrl('susceptibility')} alt="Model susceptibility map" draggable={false} />
                )}
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                  {showTaluks &&
                    talukPaths.map((taluk) => (
                      <path
                        key={taluk.name}
                        className={`taluk-outline${taluk.name === hoveredTaluk ? ' is-hovered' : ''}${taluk.name === pulsedTaluk ? ' is-pulsing' : ''}`}
                        d={taluk.d}
                        fillRule="evenodd"
                        vectorEffect="non-scaling-stroke"
                      />
                    ))}
                  {showLandslides &&
                    meta?.landslides.map(([x, y], i) => {
                      const p = toPercent(x, y);
                      // Outlined so the inventory stays legible over the susceptibility overlay.
                      return (
                        <circle
                          key={i}
                          cx={p.left}
                          cy={p.top}
                          r={0.34}
                          fill="#2a1206"
                          stroke="#ffb27a"
                          strokeWidth={0.18}
                          vectorEffect="non-scaling-stroke"
                        />
                      );
                    })}
                </svg>
                {showTaluks &&
                  taluks.map((taluk) => {
                    const p = toPercent(taluk.label[0], taluk.label[1]);
                    return (
                      <span key={taluk.name} className="taluk-name" style={{ left: `${p.left}%`, top: `${p.top}%` }}>
                        {taluk.name}
                      </span>
                    );
                  })}
                <div ref={talukTip} className="taluk-tip" hidden={!showTaluks || !hoveredTaluk}>
                  {hoveredTaluk} taluk · click to sample
                </div>
                {marker && (
                  <PinIcon
                    style={{
                      position: 'absolute',
                      left: `${marker.left}%`,
                      top: `${marker.top}%`,
                      width: 22,
                      height: 22,
                      transform: 'translate(-50%, -100%)',
                      color: '#e3e7eb',
                      filter: 'drop-shadow(0 1px 2px rgba(0,0,0,.8))',
                      pointerEvents: 'none',
                    }}
                  />
                )}
              </div>
            )}
          </div>

          <div className="district-side">
            <div style={{ display: 'grid', gap: 6 }}>
              <label className="toggle">
                <input type="checkbox" checked={showLandslides} onChange={(e) => setShowLandslides(e.target.checked)} />
                <span className="chip" style={{ background: '#d9793a', borderRadius: '50%' }} />
                Recorded landslides ({meta?.landslides.length ?? 0}, training inventory)
              </label>
              <label className="toggle" style={{ opacity: susceptibilityLayer ? 1 : 0.5 }}>
                <input
                  type="checkbox"
                  checked={showSusceptibility}
                  disabled={!susceptibilityLayer}
                  onChange={(e) => setShowSusceptibility(e.target.checked)}
                />
                {susceptibilityLayer
                  ? `District susceptibility map${susceptibilityLayer.partial ? ' (partial run)' : ''}`
                  : 'District susceptibility map — run pipeline stage 7 to produce it'}
              </label>
              {taluks.length > 0 && (
                <label className="toggle">
                  <input type="checkbox" checked={showTaluks} onChange={(e) => setShowTaluks(e.target.checked)} />
                  Taluk boundaries and names
                </label>
              )}
            </div>

            <form
              style={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: 6, alignItems: 'end' }}
              onSubmit={(event) => {
                event.preventDefault();
                const x = Number(easting);
                const y = Number(northing);
                if (Number.isFinite(x) && Number.isFinite(y)) void sample(x, y);
              }}
            >
              <label className="small-note">
                Easting (m)
                <input className="field-input" inputMode="numeric" value={easting} onChange={(e) => setEasting(e.target.value)} />
              </label>
              <label className="small-note">
                Northing (m)
                <input className="field-input" inputMode="numeric" value={northing} onChange={(e) => setNorthing(e.target.value)} />
              </label>
              <button type="submit" className="btn btn--sm" disabled={!easting || !northing}>
                Sample
              </button>
            </form>
            <p className="small-note">Coordinates in {grid?.crs ?? 'EPSG:32643'} (WGS 84 / UTM 43N).</p>

            {sampling && (
              <p className="small-note" style={{ display: 'flex', gap: 8 }}>
                <span className="spinner" /> Reading 14 rasters…
              </p>
            )}
            {sampleError && <p className="notice notice--danger">{sampleError}</p>}
            {site && !sampling && (
              <div style={{ display: 'grid', gap: 10 }}>
                {site.place ? (
                  <div className="place-card">
                    <strong>{site.place.label}</strong>
                    <span>{site.place.sublabel}</span>
                    {townHint(site.place) && <span>{townHint(site.place)}</span>}
                    {site.lat != null && <small className="mono">{formatLatLon(site.lat, site.lon)}</small>}
                  </div>
                ) : (
                  site.lat != null && <p className="source-card__label">{formatLatLon(site.lat, site.lon)}</p>
                )}
                {site.issues.map((issue) => (
                  <p key={issue} className="notice notice--warn">
                    <WarnIcon />
                    <span>{issue}</span>
                  </p>
                ))}
                {site.factors && metadata && (
                  <dl className="kv">
                    <dt>Slope</dt>
                    <dd>{formatNumber(site.factors.slope, 1)}°</dd>
                    <dt>Elevation</dt>
                    <dd>{formatNumber(site.factors.elevation)} m</dd>
                    <dt>Land cover</dt>
                    <dd>{levelLabel(metadata, 'LULC', site.factors.LULC)}</dd>
                    <dt>Lithology</dt>
                    <dd>{levelLabel(metadata, 'lithology', site.factors.lithology)}</dd>
                    <dt>Road</dt>
                    <dd>{formatDistance(site.factors.dist_road)}</dd>
                    <dt>Rainfall</dt>
                    <dd>{formatNumber(site.factors.rainfall)} mm/yr</dd>
                    {site.nearest_landslide_m != null && (
                      <>
                        <dt>Nearest recorded landslide</dt>
                        <dd>{formatDistance(site.nearest_landslide_m)}</dd>
                      </>
                    )}
                  </dl>
                )}
                <button
                  type="button"
                  className="btn btn--primary"
                  disabled={!site.factors || site.issues.length > 0}
                  onClick={applySite}
                >
                  Use this site
                </button>
              </div>
            )}
            {!site && !sampling && <p className="small-note">No site selected yet.</p>}
          </div>
        </div>
      </div>
    </div>
  );
}

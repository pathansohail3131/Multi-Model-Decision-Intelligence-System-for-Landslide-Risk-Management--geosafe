import { useMemo } from 'react';
import { crossSection as geometry } from './fallbackGeometry';
import { useResultIsStale, useSimulationStore } from '../../store/simulationStore';
import { CLASS_COLOURS } from '../../utils/colors';
import { compassPoint, formatDistance, formatNumber, formatProbability } from '../../utils/format';
import { toSceneParams } from '../../utils/sceneMapping';
import { WarnIcon } from '../common/Icons';

const W = 800;
const H = 360;

/**
 * 2D cross-section used when WebGL is unavailable. It draws the same terrain
 * function as the 3D block, along the fall line through the assessed cell, so
 * prediction and interpretation keep working without a 3D context.
 */
export function Fallback2D({ reason, onRetry }: { reason: string; onRetry?: () => void }) {
  const inputs = useSimulationStore((s) => s.inputs);
  const metadata = useSimulationStore((s) => s.metadata);
  const result = useSimulationStore((s) => s.result);
  const stale = useResultIsStale();

  const profile = useMemo(() => {
    if (!inputs || !metadata) return null;
    const params = toSceneParams(inputs, metadata);
    const samples = geometry.profile(params, 161);
    const heights = samples.map((s) => s.h);
    const min = Math.min(...heights) - 60;
    const max = Math.max(...heights) + 60;
    const span = Math.max(max - min, 1200 * (H / W));
    const mid = (min + max) / 2;
    const x = (u: number) => ((u + 600) / 1200) * W;
    const y = (h: number) => H / 2 - ((h - mid) / span) * H;
    const line = samples.map((s, i) => `${i ? 'L' : 'M'}${x(s.u).toFixed(1)},${y(s.h).toFixed(1)}`).join('');
    return { params, samples, line, x, y, site: geometry.site(params) };
  }, [inputs, metadata]);

  if (!profile || !inputs) return null;
  const prediction = result?.response.prediction;
  const colour = prediction && !stale ? CLASS_COLOURS[prediction.class.key] : '#d0d7de';
  const { x, y, params } = profile;
  const siteY = y(profile.site);

  return (
    <div className="fallback">
      <p className="notice notice--warn" style={{ alignSelf: 'flex-start', alignItems: 'center' }}>
        <WarnIcon />
        <span>3D view unavailable ({reason}). Showing a 2D cross-section; prediction works normally.</span>
        {onRetry && (
          <button type="button" className="btn btn--sm" style={{ marginLeft: 4 }} onClick={onRetry}>
            Try 3D again
          </button>
        )}
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Cross-section of the hillslope through the assessed cell">
        <defs>
          <linearGradient id="fallback-ground" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="#3d5d31" />
            <stop offset="0.06" stopColor="#57432e" />
            <stop offset="1" stopColor="#2a2e33" />
          </linearGradient>
        </defs>
        <path d={`${profile.line}L${W},${H}L0,${H}Z`} fill="url(#fallback-ground)" />
        <path d={profile.line} fill="none" stroke="#9aa5ae" strokeWidth={1.5} />
        {params.roadDistance <= 580 && (
          <g>
            <line x1={x(-params.roadDistance)} x2={x(-params.roadDistance)} y1={y(geometry.height(-params.roadDistance, params)) - 30} y2={y(geometry.height(-params.roadDistance, params))} stroke="#8b9096" strokeDasharray="3 3" />
            <text x={x(-params.roadDistance)} y={y(geometry.height(-params.roadDistance, params)) - 36} fill="#a9b3bc" fontSize="12" textAnchor="middle">
              road · {formatDistance(inputs.dist_road)}
            </text>
          </g>
        )}
        <line x1={x(0)} x2={x(0)} y1={siteY} y2={siteY - 70} stroke="#e7ebef" strokeWidth={2} />
        <rect x={x(0)} y={siteY - 70} width={34} height={20} fill={colour} />
        <circle cx={x(0)} cy={siteY} r={prediction && !stale ? 26 : 6} fill={colour} opacity={prediction && !stale ? 0.3 : 1} />
        <text x={x(0) + 40} y={siteY - 56} fill="#e3e7eb" fontSize="13">
          {prediction && !stale ? `${prediction.class.label} · ${formatProbability(prediction.susceptibility)}` : 'Assessed cell'}
        </text>
        <text x={12} y={20} fill="#76818b" fontSize="12">
          upslope
        </text>
        <text x={W - 12} y={20} fill="#76818b" fontSize="12" textAnchor="end">
          downslope · faces {formatNumber(inputs.aspect)}° {compassPoint(inputs.aspect)}
        </text>
        <text x={12} y={H - 12} fill="#76818b" fontSize="12">
          Slope {formatNumber(inputs.slope, 1)}° · stream {formatDistance(inputs.dist_drainage)} · lineament {formatDistance(inputs.dist_fault)} · 1.2 km section
        </text>
      </svg>
    </div>
  );
}

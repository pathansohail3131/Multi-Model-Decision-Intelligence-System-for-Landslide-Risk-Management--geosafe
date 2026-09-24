import { Suspense, lazy, useMemo } from 'react';
import { useSimulationStore } from '../../store/simulationStore';
import { ErrorBoundary } from '../common/ErrorBoundary';
import { Fallback2D } from './Fallback2D';
import { ViewportHud } from './ViewportHud';

const SimulationCanvas = lazy(() => import('../../three/SimulationCanvas'));

function detectWebGL(): boolean {
  try {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('webgl2') ?? canvas.getContext('webgl');
    // Release the probe context immediately; browsers cap live WebGL contexts.
    context?.getExtension('WEBGL_lose_context')?.loseContext();
    return Boolean(context);
  } catch {
    return false;
  }
}

export function SimulationViewport() {
  const webglFailed = useSimulationStore((s) => s.webglFailed);
  const setWebglFailed = useSimulationStore((s) => s.setWebglFailed);
  const retryWebgl = useSimulationStore((s) => s.retryWebgl);
  const canvasKey = useSimulationStore((s) => s.canvasKey);
  const ready = useSimulationStore((s) => Boolean(s.inputs && s.metadata));
  const supported = useMemo(detectWebGL, []);

  const fallbackReason = !supported
    ? 'WebGL is not available in this browser'
    : 'the 3D context was lost — this usually happens when several tabs render 3D at once';
  const use3D = supported && !webglFailed;

  return (
    <section className="viewport" aria-label="Interactive 3D site model">
      {ready &&
        (use3D ? (
          <ErrorBoundary
            key={canvasKey}
            fallback={<Fallback2D reason="the 3D renderer stopped" onRetry={retryWebgl} />}
            onError={setWebglFailed}
          >
            <Suspense fallback={<div className="fallback" aria-busy="true" />}>
              <SimulationCanvas />
            </Suspense>
            <ViewportHud />
          </ErrorBoundary>
        ) : (
          <Fallback2D reason={fallbackReason} onRetry={supported ? retryWebgl : undefined} />
        ))}
    </section>
  );
}

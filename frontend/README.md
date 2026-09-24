# Simulator frontend

React 19 + TypeScript + three.js (react-three-fiber), built with Vite. It turns
the trained landslide models into an interactive 3D site model: configure one
30 m cell, run it, and the real API scores it.

```bash
npm install
npm run dev        # http://localhost:5173, proxies /api to http://127.0.0.1:8000
npm run build      # emits dist/, which the API then serves at http://127.0.0.1:8000
npm run typecheck
```

The backend must be running: `python -m uvicorn api.app.main:app --port 8000`
from the project root.

## Layout

| Path | Role |
|---|---|
| `src/services/predictionApi.ts` | The only module that calls `fetch`; typed errors and response guards |
| `src/store/simulationStore.ts` | Zustand store: phase, inputs, run history, lens, camera |
| `src/hooks/` | Boot sequence, run sequence, sensitivity sweeps, media queries |
| `src/three/` | 3D scene: terrain model and shader, land cover, overlays, camera rig |
| `src/components/` | Controls, results panels, district locator, viewport HUD |
| `src/utils/sceneMapping.ts` | Documented mapping from each factor to a scene property |

## Rules this app keeps

1. **Nothing is computed in the browser.** Every probability, metric and
   explanation comes from an API response. `sceneMapping.ts` records which
   scene elements are drawn to scale, which are positional schematics, and
   which are qualitative.
2. **State that animates lives outside React.** Scene parameters are held in a
   mutable runtime ref and interpolated inside `useFrame`, so dragging a slider
   does not re-render the component tree per frame.
3. **Degrade honestly.** No WebGL → a 2D cross-section of the same terrain
   function, with prediction still working. API down → the startup screen shows
   the command to start it and keeps retrying.

## Environment

| Variable | When | Purpose |
|---|---|---|
| `VITE_API_TARGET` | `npm run dev` | Backend the dev-server proxy forwards `/api` to (default `http://127.0.0.1:8000`) |
| `VITE_API_BASE` | `npm run build` | API origin when the frontend is hosted separately; leave unset for same-origin |

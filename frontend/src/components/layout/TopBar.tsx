import { useSimulationStore } from '../../store/simulationStore';
import { MountainMark } from '../common/Icons';
import { EngineStatus, StatusBadge } from '../common/StatusBadge';

export function TopBar() {
  const metadata = useSimulationStore((s) => s.metadata);
  const models = metadata?.models.length ?? 5;
  return (
    <header className="topbar">
      <div className="brand">
        <MountainMark className="brand__mark" />
        <div className="brand__text">
          <h1 className="brand__title">Landslide Susceptibility Simulator</h1>
          <p className="brand__subtitle">
            Uttara Kannada, Western Ghats · {models} calibrated CatBoost models
          </p>
        </div>
      </div>
      <div className="topbar__mode" aria-hidden="true">
        ML PREDICTION SIMULATOR
      </div>
      <div className="topbar__status">
        <EngineStatus />
        <StatusBadge />
      </div>
    </header>
  );
}

import { useState } from 'react';
import { BootScreen } from './components/boot/BootScreen';
import { LiveAnnouncer } from './components/common/LiveAnnouncer';
import { ParameterPanel } from './components/controls/ParameterPanel';
import { DistrictLocator } from './components/district/DistrictLocator';
import { ActionBar } from './components/layout/ActionBar';
import { TopBar } from './components/layout/TopBar';
import { ModelCard } from './components/results/ModelCard';
import { AnalysisContent, ResultPanel } from './components/results/ResultPanel';
import { SimulationViewport } from './components/simulation/SimulationViewport';
import { useBootSequence } from './hooks/useBootSequence';
import { useCompactLayout } from './hooks/useMediaQuery';
import { useRunSimulation } from './hooks/useRunSimulation';
import { useSimulationStore } from './store/simulationStore';

type Tab = 'parameters' | 'analysis' | 'model';

function CompactPanels() {
  const [tab, setTab] = useState<Tab>('parameters');
  const hasResult = useSimulationStore((s) => Boolean(s.result));
  return (
    <div className="panel">
      <div className="mobile-tabs segmented" role="tablist" aria-label="Panels">
        {(['parameters', 'analysis', 'model'] as Tab[]).map((key) => (
          <button key={key} type="button" role="tab" aria-selected={tab === key} onClick={() => setTab(key)}>
            {key === 'parameters' ? 'Parameters' : key === 'analysis' ? `Analysis${hasResult ? ' •' : ''}` : 'Model'}
          </button>
        ))}
      </div>
      <div role="tabpanel">
        {tab === 'parameters' && <ParameterPanel embedded />}
        {tab === 'analysis' && <AnalysisContent />}
        {tab === 'model' && <ModelCard defaultOpen />}
      </div>
    </div>
  );
}

export default function App() {
  const { retry } = useBootSequence();
  const run = useRunSimulation();
  const compact = useCompactLayout();
  const phase = useSimulationStore((s) => s.phase);

  return (
    <div className="app" data-phase={phase}>
      <TopBar />
      <main className="workspace">
        {compact ? (
          <>
            <SimulationViewport />
            <CompactPanels />
          </>
        ) : (
          <>
            <aside className="panel panel--left" aria-label="Simulation parameters">
              <ParameterPanel />
            </aside>
            <SimulationViewport />
            <aside className="panel panel--right" aria-label="Prediction analysis">
              <ResultPanel />
            </aside>
          </>
        )}
      </main>
      <ActionBar onRun={() => void run()} />
      <BootScreen onRetry={() => void retry()} />
      <DistrictLocator />
      <LiveAnnouncer />
    </div>
  );
}

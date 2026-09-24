import { useEffect, useState } from 'react';
import { useResultIsStale, useSimulationStore } from '../../store/simulationStore';
import { InfoIcon, WarnIcon } from '../common/Icons';
import { DecisionCard } from './DecisionCard';
import { ExplanationChart } from './ExplanationChart';
import { ModelCard } from './ModelCard';
import { PipelineStrip } from './PipelineStrip';
import { PredictionSummary } from './PredictionSummary';
import { ScenarioCompare } from './ScenarioCompare';
import { SensitivityChart } from './SensitivityChart';
import { ThemeEvidence } from './ThemeEvidence';

export function AnalysisContent() {
  const metadata = useSimulationStore((s) => s.metadata);
  const result = useSimulationStore((s) => s.result);
  const runStep = useSimulationStore((s) => s.runStep);
  const phase = useSimulationStore((s) => s.phase);
  const stale = useResultIsStale();
  const [entering, setEntering] = useState(false);

  useEffect(() => {
    if (!result) return;
    setEntering(true);
    const timer = window.setTimeout(() => setEntering(false), 900);
    return () => window.clearTimeout(timer);
  }, [result]);

  if (!metadata) return null;
  const revealed = result && (runStep === 'analysis' || runStep === null || phase !== 'result');

  return (
    <>
      <PipelineStrip />
      {!result && (
        <div className="empty-state">
          <p>
            No prediction yet. The analysis appears here once the real model has scored the scenario.
          </p>
          <ol>
            <li>Choose a scenario — a real site from the district map, a recorded landslide, or the median.</li>
            <li>Adjust parameters and watch the site model respond.</li>
            <li>Press Run simulation to score it with the five calibrated models.</li>
          </ol>
        </div>
      )}
      {result && !revealed && (
        <div className="empty-state">
          <p style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span className="spinner" /> Result received — updating the site model…
          </p>
        </div>
      )}
      {result && revealed && (
        <div className={`analysis${stale ? ' is-stale' : ''}${entering ? ' is-entering' : ''}`}>
          {stale && (
            <div className="section" style={{ paddingBottom: 0, borderBottom: 0 }}>
              <p className="notice notice--warn">
                <WarnIcon />
                <span>
                  These results are for run #{result.index}. The parameters have changed since — run again to update them.
                </span>
              </p>
            </div>
          )}
          <PredictionSummary run={result} metadata={metadata} />
          <DecisionCard run={result} metadata={metadata} />
          <ThemeEvidence run={result} metadata={metadata} />
          {result.response.warnings.length > 0 && (
            <section className="section" aria-label="Input warnings">
              <div style={{ display: 'grid', gap: 6 }}>
                {result.response.warnings.map((warning) => (
                  <p
                    key={`${warning.kind}-${warning.factor}`}
                    className={`notice ${warning.kind === 'outside_training_range' || warning.kind === 'unseen_level' ? 'notice--warn' : 'notice--info'}`}
                  >
                    {warning.kind === 'outside_training_range' || warning.kind === 'unseen_level' ? <WarnIcon /> : <InfoIcon />}
                    <span>{warning.message}</span>
                  </p>
                ))}
              </div>
            </section>
          )}
          <ExplanationChart run={result} metadata={metadata} />
          <SensitivityChart run={result} metadata={metadata} />
          <ScenarioCompare current={result} metadata={metadata} />
        </div>
      )}
    </>
  );
}

export function ResultPanel() {
  return (
    <>
      <div className="panel__header">
        <div>
          <h2 className="panel__title">Prediction analysis</h2>
          <p className="muted" style={{ fontSize: 11.5 }}>
            Every value below comes from the prediction API
          </p>
        </div>
      </div>
      <div className="panel__scroll">
        <AnalysisContent />
        <ModelCard />
      </div>
    </>
  );
}

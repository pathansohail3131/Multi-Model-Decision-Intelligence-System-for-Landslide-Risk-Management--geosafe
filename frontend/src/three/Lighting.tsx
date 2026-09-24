import { Environment, Lightformer } from '@react-three/drei';

export function Lighting({ shadows }: { shadows: boolean }) {
  return (
    <>
      <hemisphereLight args={['#c7d5e3', '#2e2a22', 0.65]} />
      <directionalLight
        position={[-38, 62, 34]}
        intensity={2.4}
        color="#fff4e2"
        castShadow={shadows}
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-46}
        shadow-camera-right={46}
        shadow-camera-top={46}
        shadow-camera-bottom={-46}
        shadow-camera-near={10}
        shadow-camera-far={220}
        shadow-bias={-0.0004}
        shadow-normalBias={0.05}
      />
      <directionalLight position={[44, 18, -36]} intensity={0.45} color="#9db8d2" />
      {/* Procedural environment for soft reflections on water; no external HDR is fetched. */}
      <Environment resolution={32} frames={1}>
        <Lightformer form="rect" intensity={1.2} color="#9fb6cc" position={[0, 12, -18]} scale={[40, 12, 1]} />
        <Lightformer form="rect" intensity={0.5} color="#e8d9c2" position={[-20, 8, 16]} scale={[20, 8, 1]} />
      </Environment>
    </>
  );
}

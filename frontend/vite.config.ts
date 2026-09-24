import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The simulator talks to the FastAPI backend under /api. In development Vite
// proxies those calls; in production FastAPI serves the built files itself.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 1600,
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes('node_modules/three/')) return 'three';
          if (id.includes('node_modules/@react-three/') || id.includes('node_modules/three-stdlib')) return 'r3f';
          return undefined;
        },
      },
    },
  },
});

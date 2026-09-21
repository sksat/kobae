import { defineConfig } from "vite";

// Dev: `pnpm dev` proxies /api and /ws to the Python server on :8765.
// Build: output goes to ../src/kobae/viewer/dist and is served by the Python server.
export default defineConfig({
  build: { outDir: "../src/kobae/viewer/dist", emptyOutDir: true },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
    },
  },
});

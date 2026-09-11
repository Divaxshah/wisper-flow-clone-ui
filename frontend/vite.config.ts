import { defineConfig } from "vite";

const api = process.env.VITE_API_TARGET || "http://127.0.0.1:8000";

export default defineConfig({
  server: {
    host: "0.0.0.0",
    port: 5173,
    allowedHosts: [".ngrok-free.dev", ".ngrok.io", ".ngrok.app"],
    proxy: {
      "/api": {
        target: api,
        changeOrigin: true,
      },
      "/ws": {
        target: api,
        ws: true,
        changeOrigin: true,
        timeout: 0,
        proxyTimeout: 0,
      },
    },
  },
});

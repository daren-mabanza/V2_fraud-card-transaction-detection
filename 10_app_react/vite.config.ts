import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

export default defineConfig({
  plugins: [react()],
  resolve: {
    // Les onglets sont conservés dans 03_src, les dépendances dans 10_app_react.
    alias: [
      {
        find: /^react-dom(\/.*)?$/,
        replacement: fileURLToPath(
          new URL("./node_modules/react-dom$1", import.meta.url),
        ),
      },
      {
        find: /^react(\/.*)?$/,
        replacement: fileURLToPath(
          new URL("./node_modules/react$1", import.meta.url),
        ),
      },
      {
        find: "lucide-react",
        replacement: fileURLToPath(
          new URL(
            "./node_modules/lucide-react/dist/esm/lucide-react.mjs",
            import.meta.url,
          ),
        ),
      },
      {
        find: "plotly.js-dist-min",
        replacement: fileURLToPath(
          new URL(
            "./node_modules/plotly.js-dist-min/plotly.min.js",
            import.meta.url,
          ),
        ),
      },
    ],
  },
  server: {
    fs: {
      allow: [
        fileURLToPath(new URL(".", import.meta.url)),
        fileURLToPath(new URL("../03_src/ReactTabs", import.meta.url)),
      ],
    },
  },
  build: { chunkSizeWarningLimit: 5500 },
});

import { defineConfig } from "vite";
import { resolve } from "node:path";

// TP2b (idée 3 : distribution / MDN). Le code du site vit dans TP2b/front-end,
// node_modules reste à la racine, et les artefacts du modèle (TP2b/model) sont
// servis tels quels comme fichiers statiques.
export default defineConfig({
  root: resolve(__dirname, "TP2b/front-end"),
  publicDir: resolve(__dirname, "TP2b/model"),
  build: {
    outDir: resolve(__dirname, "TP2b/front-end/dist"),
    emptyOutDir: true,
  },
  server: {
    port: 5174,
    open: true,
  },
});

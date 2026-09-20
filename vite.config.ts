import { defineConfig } from "vite";
import { resolve } from "node:path";

// Le code du site vit dans TP2/front-end, mais node_modules reste à la racine.
// Les artefacts du modèle (TP2/model) sont servis tels quels comme fichiers statiques.
export default defineConfig({
  root: resolve(__dirname, "TP2/front-end"),
  publicDir: resolve(__dirname, "TP2/model"),
  build: {
    outDir: resolve(__dirname, "TP2/front-end/dist"),
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    open: true,
  },
});

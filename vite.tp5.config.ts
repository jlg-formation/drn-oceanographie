import { defineConfig } from "vite";
import { resolve } from "node:path";

// TP5 (MaréeLab, prévision des marées par RNN/GRU/LSTM). Le code du site vit
// dans TP5/front-end, node_modules reste à la racine, et les artefacts du
// pipeline Python (TP5/model : modèles ONNX, série, métadonnées) sont servis
// tels quels comme fichiers statiques.
export default defineConfig({
  root: resolve(__dirname, "TP5/front-end"),
  publicDir: resolve(__dirname, "TP5/model"),
  build: {
    outDir: resolve(__dirname, "TP5/front-end/dist"),
    emptyOutDir: true,
  },
  server: {
    port: 5177,
    open: true,
  },
});

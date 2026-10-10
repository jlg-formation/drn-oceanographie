import { defineConfig } from "vite";
import { resolve } from "node:path";

// TP4 (segmentation terre/mer). Le code du site vit dans TP4/front-end,
// node_modules reste à la racine, et les artefacts du pipeline Python
// (TP4/model : modèle ONNX, métadonnées, images d'exemple) sont servis tels
// quels comme fichiers statiques.
export default defineConfig({
  root: resolve(__dirname, "TP4/front-end"),
  publicDir: resolve(__dirname, "TP4/model"),
  // Le worker d'inférence importe onnxruntime-web : format ES requis.
  worker: { format: "es" },
  build: {
    outDir: resolve(__dirname, "TP4/front-end/dist"),
    emptyOutDir: true,
  },
  server: {
    port: 5176,
    open: true,
  },
});

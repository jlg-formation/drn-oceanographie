import { defineConfig } from "vite";
import { resolve } from "node:path";

// TP6 (Ocean Terrain Lab, reliefs sous-marins générés par VAE et GAN). Le code
// du site vit dans TP6/front-end, node_modules reste à la racine, et les
// artefacts du pipeline Python (TP6/model : modèles ONNX, patches réels,
// métadonnées) sont servis tels quels comme fichiers statiques.
export default defineConfig({
  root: resolve(__dirname, "TP6/front-end"),
  publicDir: resolve(__dirname, "TP6/model"),
  build: {
    outDir: resolve(__dirname, "TP6/front-end/dist"),
    emptyOutDir: true,
    // three.js + onnxruntime-web : un seul paquet d'environ 600 ko, attendu.
    chunkSizeWarningLimit: 800,
  },
  server: {
    port: 5178,
    open: true,
  },
});

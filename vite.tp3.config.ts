import { defineConfig } from "vite";
import { resolve } from "node:path";

// TP3 (bathymétrie GEBCO). Le code du site vit dans TP3/front-end,
// node_modules reste à la racine, et les exports du pipeline Python
// (TP3/output) sont servis tels quels comme fichiers statiques.
export default defineConfig({
  root: resolve(__dirname, "TP3/front-end"),
  publicDir: resolve(__dirname, "TP3/output"),
  build: {
    outDir: resolve(__dirname, "TP3/front-end/dist"),
    emptyOutDir: true,
  },
  server: {
    port: 5175,
    open: true,
  },
});

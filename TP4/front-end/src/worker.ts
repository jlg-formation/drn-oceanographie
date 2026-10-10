/**
 * Worker d'inférence : exécute le U-Net ONNX hors du fil principal pour que
 * l'interface reste fluide pendant l'analyse (barre de progression, NF07).
 */
import * as ort from "onnxruntime-web/webgpu";

import { extraireFenetre, fenetres, poidsFenetre } from "./tiling";
import type { InferenceParams, WorkerRequest, WorkerResponse } from "./protocol";

// Les binaires WASM d'onnxruntime-web sont servis depuis le CDN jsDelivr,
// à la version installée dans node_modules (voir package.json).
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const ctx = self as unknown as {
  postMessage(message: WorkerResponse, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null;
};

let session: ort.InferenceSession | null = null;
let backend = "";
let params: InferenceParams | null = null;

async function creerSession(url: string): Promise<void> {
  // WebGPU si le navigateur le permet (beaucoup plus rapide), sinon WebAssembly.
  if ("gpu" in navigator) {
    try {
      session = await ort.InferenceSession.create(url, { executionProviders: ["webgpu"] });
      backend = "WebGPU";
      return;
    } catch (err) {
      console.warn("WebGPU indisponible, repli sur WebAssembly :", err);
    }
  }
  session = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
  backend = "WebAssembly (CPU)";
}

async function analyser(id: number, largeur: number, hauteur: number, rgba: Uint8ClampedArray): Promise<void> {
  if (!session || !params) throw new Error("Modèle non chargé.");
  const { patch_px: taille, pas_px: pas, rampe_px: rampe } = params;
  const t0 = performance.now();
  const liste = fenetres(largeur, hauteur, taille, pas);
  const poids = poidsFenetre(taille, rampe);
  const somme = new Float32Array(largeur * hauteur);
  const sommePoids = new Float32Array(largeur * hauteur);
  const lot = backend === "WebGPU" ? 8 : 4;
  const plan = taille * taille;

  for (let i = 0; i < liste.length; i += lot) {
    const groupe = liste.slice(i, i + lot);
    const entree = new Float32Array(groupe.length * 3 * plan);
    groupe.forEach((f, k) => extraireFenetre(rgba, largeur, hauteur, f, taille, entree, k * 3 * plan));
    const tenseur = new ort.Tensor("float32", entree, [groupe.length, 3, taille, taille]);
    const sorties = await session.run({ [params.entree]: tenseur });
    const prob = (await sorties[params.sortie].getData()) as Float32Array;
    tenseur.dispose();

    groupe.forEach((f, k) => {
      const base = k * plan;
      const yMax = Math.min(taille, hauteur - f.r);
      const xMax = Math.min(taille, largeur - f.c);
      for (let y = 0; y < yMax; y++) {
        const ligne = (f.r + y) * largeur + f.c;
        for (let x = 0; x < xMax; x++) {
          const w = poids[y * taille + x];
          somme[ligne + x] += prob[base + y * taille + x] * w;
          sommePoids[ligne + x] += w;
        }
      }
    });
    ctx.postMessage({ type: "progress", id, fait: Math.min(i + lot, liste.length), total: liste.length });
  }

  for (let i = 0; i < somme.length; i++) somme[i] /= sommePoids[i];
  const ms = performance.now() - t0;
  ctx.postMessage({ type: "result", id, prob: somme, ms, fenetres: liste.length, backend }, [somme.buffer]);
}

ctx.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      params = msg.params;
      await creerSession(msg.modelUrl);
      ctx.postMessage({ type: "ready", backend });
    } else if (msg.type === "run") {
      await analyser(msg.id, msg.largeur, msg.hauteur, new Uint8ClampedArray(msg.pixels));
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    ctx.postMessage({ type: "error", id: msg.type === "run" ? msg.id : undefined, phase: msg.type, message });
  }
};

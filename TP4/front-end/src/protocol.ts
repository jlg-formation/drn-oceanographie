/** Messages échangés entre le fil principal et le worker d'inférence. */

export interface InferenceParams {
  entree: string;
  sortie: string;
  patch_px: number;
  pas_px: number;
  rampe_px: number;
}

export type WorkerRequest =
  | { type: "init"; modelUrl: string; params: InferenceParams }
  | { type: "run"; id: number; largeur: number; hauteur: number; pixels: ArrayBuffer };

export type WorkerResponse =
  | { type: "ready"; backend: string }
  | { type: "progress"; id: number; fait: number; total: number }
  | { type: "result"; id: number; prob: Float32Array; ms: number; fenetres: number; backend: string }
  | { type: "error"; id?: number; phase: string; message: string };

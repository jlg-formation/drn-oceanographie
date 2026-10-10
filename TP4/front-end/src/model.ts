/** Chargement des fichiers produits par le pipeline Python et pilotage du worker. */

import type { InferenceParams, WorkerRequest, WorkerResponse } from "./protocol";

export interface Bornes {
  west: number;
  south: number;
  east: number;
  north: number;
}

export interface Metadata {
  modele: {
    fichier: string;
    architecture: string;
    parametres: number;
    entree: { nom: string };
    sortie: { nom: string };
  };
  inference: {
    patch_px: number;
    pas_px: number;
    rampe_px: number;
    seuil_terre: number;
    incertitude: { min: number; max: number };
  };
  couleurs: { terre: string; mer: string; ignore: string; erreur: string };
  limites: { taille_fichier_mo: number; cote_max_px: number; pixels_max: number };
  entrainement: {
    date_creation: string;
    dataset: string;
    epoque: number;
    validation: { miou: number };
    test: { miou: number; iou_terre: number; iou_mer: number; exactitude: number } | null;
  };
  sources: { images: string; masques: string };
}

export interface Exemple {
  id: string;
  titre: string;
  configuration: string;
  fichier: string;
  largeur_px: number;
  hauteur_px: number;
  bornes: Bornes;
  origine: string;
}

/** Erreur lisible : quel fichier, pourquoi, et comment le régénérer. */
export class DataError extends Error {
  constructor(fichier: string, raison: string) {
    super(
      `Impossible de charger « ${fichier} » (${raison}). ` +
        "Lancez `mise run tp4` (ou au moins tp4_02_dataset et tp4_05_export) puis rechargez la page.",
    );
  }
}

async function fetchJson<T>(fichier: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/${fichier}`);
  } catch {
    throw new DataError(fichier, "serveur injoignable");
  }
  if (!res.ok) throw new DataError(fichier, `HTTP ${res.status}`);
  // Le serveur de dev Vite répond index.html (200) pour un fichier inexistant.
  if (res.headers.get("content-type")?.includes("text/html")) {
    throw new DataError(fichier, "fichier absent");
  }
  return (await res.json()) as T;
}

export const loadMetadata = (): Promise<Metadata> => fetchJson<Metadata>("metadata.json");
export const loadExemples = (): Promise<Exemple[]> => fetchJson<Exemple[]>("examples.json");

export interface Resultat {
  prob: Float32Array;
  ms: number;
  fenetres: number;
  backend: string;
}

/** Façade du worker : chargement du modèle et segmentation d'une image. */
export class Segmenteur {
  private readonly worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  private prochainId = 1;
  private enCours = new Map<
    number,
    { resolve: (r: Resultat) => void; reject: (e: Error) => void; progression: (f: number, t: number) => void }
  >();

  private constructor() {}

  static charger(meta: Metadata): Promise<{ segmenteur: Segmenteur; backend: string }> {
    const s = new Segmenteur();
    const params: InferenceParams = {
      entree: meta.modele.entree.nom,
      sortie: meta.modele.sortie.nom,
      patch_px: meta.inference.patch_px,
      pas_px: meta.inference.pas_px,
      rampe_px: meta.inference.rampe_px,
    };
    return new Promise((resolve, reject) => {
      s.worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
        const msg = e.data;
        if (msg.type === "ready") {
          s.worker.onmessage = (ev: MessageEvent<WorkerResponse>) => s.recevoir(ev.data);
          resolve({ segmenteur: s, backend: msg.backend });
        } else if (msg.type === "error") {
          reject(new Error(`Chargement du modèle ONNX impossible : ${msg.message}`));
        }
      };
      s.worker.onerror = (e) => reject(new Error(`Worker d'inférence en erreur : ${e.message}`));
      const init: WorkerRequest = { type: "init", modelUrl: `/${meta.modele.fichier}`, params };
      s.worker.postMessage(init);
    });
  }

  private recevoir(msg: WorkerResponse): void {
    if (msg.type === "ready" || msg.id === undefined) return;
    const attente = this.enCours.get(msg.id);
    if (!attente) return;
    if (msg.type === "progress") attente.progression(msg.fait, msg.total);
    else if (msg.type === "result") {
      this.enCours.delete(msg.id);
      attente.resolve({ prob: msg.prob, ms: msg.ms, fenetres: msg.fenetres, backend: msg.backend });
    } else if (msg.type === "error") {
      this.enCours.delete(msg.id);
      attente.reject(new Error(msg.message));
    }
  }

  segmenter(image: ImageData, progression: (fait: number, total: number) => void): Promise<Resultat> {
    const id = this.prochainId++;
    // Copie des pixels : le tampon est transféré au worker (sans recopie supplémentaire).
    const pixels = image.data.slice().buffer;
    return new Promise((resolve, reject) => {
      this.enCours.set(id, { resolve, reject, progression });
      const req: WorkerRequest = { type: "run", id, largeur: image.width, hauteur: image.height, pixels };
      this.worker.postMessage(req, [pixels]);
    });
  }
}

import * as ort from "onnxruntime-web";

// Les binaires WASM d'onnxruntime-web sont servis depuis le CDN jsDelivr,
// à la version installée dans node_modules (voir package.json).
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

export interface Mixture {
  weights: number[];
  means: number[];
  scales: number[];
}

export interface Normalization {
  temp_mean: number;
  temp_std: number;
  temp_min: number;
  temp_max: number;
}

export interface Metrics {
  val_nll_phys: number;
  kl_hist_modele: number;
  wasserstein1: number;
}

export interface Metadata {
  variable: string;
  input_unit: string;
  num_components: number;
  normalization: Normalization;
  mixture_phys_space: Mixture;
  metrics: Metrics;
}

export interface Histogram {
  n: number;
  bin_edges: number[];
  bin_centers: number[];
  bin_width: number;
  density: number[];
  counts: number[];
}

/** Modèle de densité (MDN) chargé en ONNX + ses paramètres de mélange. */
export class DensityModel {
  private constructor(
    private readonly session: ort.InferenceSession,
    readonly meta: Metadata,
  ) {}

  static async load(onnxUrl: string, metaUrl: string): Promise<DensityModel> {
    const [session, meta] = await Promise.all([
      ort.InferenceSession.create(onnxUrl, { executionProviders: ["wasm"] }),
      fetch(metaUrl).then((r) => r.json() as Promise<Metadata>),
    ]);
    return new DensityModel(session, meta);
  }

  /** Densité p(T) pour un lot de températures (°C), via inférence ONNX. */
  async density(temps: Float32Array): Promise<Float32Array> {
    const tensor = new ort.Tensor("float32", temps, [temps.length, 1]);
    const outputs = await this.session.run({ temperature: tensor });
    return outputs.density.data as Float32Array;
  }

  /** Tire ``n`` températures selon le mélange appris (échantillonnage client). */
  sample(n: number): Float32Array {
    const { weights, means, scales } = this.meta.mixture_phys_space;
    const cumul: number[] = [];
    let acc = 0;
    for (const w of weights) {
      acc += w;
      cumul.push(acc);
    }
    const total = acc;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const r = Math.random() * total;
      let k = 0;
      while (k < cumul.length - 1 && r > cumul[k]) k++;
      out[i] = means[k] + scales[k] * gaussian();
    }
    return out;
  }
}

/** Tirage normal centré-réduit (Box-Muller). */
function gaussian(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

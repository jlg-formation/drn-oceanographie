/**
 * Vue 3D interactive d'un relief (F30), calculée à partir des valeurs générées.
 *
 * Surface de 64 × 64 sommets : x et y en kilomètres (pixel ≈ 1,85 km), z = relief
 * en kilomètres × exagération verticale. Couleurs : même palette et même échelle
 * que la carte 2D. Rotation, zoom et déplacement à la souris (OrbitControls).
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { TAILLE } from "./donnees";
import { couleur, type Echelle } from "./rendu2d";

export class Vue3D {
  readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controles: OrbitControls;
  private readonly geometrie: THREE.PlaneGeometry;
  private readonly socle: THREE.Mesh;
  private relief: Float32Array | null = null;
  private echelle: Echelle = { min: -1500, max: 1500 };
  private exageration = 10;
  private readonly cote: number;
  private enAttente = false;

  constructor(private readonly conteneur: HTMLElement, pixelKm: number) {
    this.cote = (TAILLE - 1) * pixelKm;
    // preserveDrawingBuffer : nécessaire à la capture PNG de la vue (F36).
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(0xf8fafc);
    conteneur.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(40, 1, 1, 5000);
    this.camera.up.set(0, 0, 1);
    this.controles = new OrbitControls(this.camera, this.renderer.domElement);
    this.controles.enableDamping = true;
    this.controles.maxPolarAngle = Math.PI / 2 - 0.02;
    this.controles.minDistance = 20;
    this.controles.maxDistance = 900;
    this.controles.addEventListener("change", () => this.rendre());

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x334155, 1.2));
    const soleil = new THREE.DirectionalLight(0xffffff, 1.6);
    soleil.position.set(-120, 160, 220);
    this.scene.add(soleil);

    this.geometrie = new THREE.PlaneGeometry(this.cote, this.cote, TAILLE - 1, TAILLE - 1);
    this.geometrie.setAttribute("color", new THREE.BufferAttribute(new Float32Array(TAILLE * TAILLE * 3), 3));
    const surface = new THREE.Mesh(
      this.geometrie,
      new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0, side: THREE.DoubleSide }),
    );
    this.scene.add(surface);
    this.socle = new THREE.Mesh(
      new THREE.PlaneGeometry(this.cote, this.cote),
      new THREE.MeshBasicMaterial({ color: 0xcbd5e1, transparent: true, opacity: 0.35 }),
    );
    this.scene.add(this.socle);
    // Repère « nord » : petite flèche au bord nord de la surface.
    const nord = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, this.cote / 2 + 4, 0), 12,
      0x0f172a, 5, 4);
    this.scene.add(nord);

    new ResizeObserver(() => this.redimensionner()).observe(conteneur);
    this.reinitialiserCamera();
    this.redimensionner();
  }

  /** Remplace le relief affiché (m) ; l'échelle de couleurs est celle de la carte 2D. */
  afficher(relief: Float32Array, echelle: Echelle): void {
    this.relief = relief;
    this.echelle = echelle;
    this.mettreAJour();
  }

  definirExageration(valeur: number): void {
    this.exageration = valeur;
    this.mettreAJour();
  }

  reinitialiserCamera(): void {
    this.camera.position.set(0, -this.cote * 1.55, this.cote * 1.2);
    this.controles.target.set(0, 0, 0);
    this.controles.update();
    this.rendre();
  }

  vueDeDessus(): void {
    this.camera.position.set(0, -0.01, this.cote * 2.3);
    this.controles.target.set(0, 0, 0);
    this.controles.update();
    this.rendre();
  }

  vueRasante(): void {
    this.camera.position.set(-this.cote * 1.6, -this.cote * 0.8, this.cote * 0.3);
    this.controles.target.set(0, 0, 0);
    this.controles.update();
    this.rendre();
  }

  /** Capture PNG de la vue courante. */
  capture(): Promise<Blob | null> {
    this.renderer.render(this.scene, this.camera);
    return new Promise((ok) => this.renderer.domElement.toBlob(ok, "image/png"));
  }

  private mettreAJour(): void {
    if (!this.relief) return;
    const pos = this.geometrie.attributes.position as THREE.BufferAttribute;
    const col = this.geometrie.attributes.color as THREE.BufferAttribute;
    const { min, max } = this.echelle;
    // PlaneGeometry : sommets rangés ligne par ligne du haut (+y = nord) vers le bas,
    // de l'ouest vers l'est : même ordre que la grille (ligne 0 = nord).
    for (let i = 0; i < TAILLE * TAILLE; i++) {
      const v = this.relief[i];
      pos.setZ(i, (v / 1000) * this.exageration);
      const c = couleur((v - min) / (max - min));
      col.setXYZ(i, c[0] / 255, c[1] / 255, c[2] / 255);
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
    // Les couleurs de la palette sont en sRGB ; three.js travaille en linéaire.
    for (let i = 0; i < col.count * 3; i++) (col.array as Float32Array)[i] = srgbVersLineaire(col.array[i]);
    this.geometrie.computeVertexNormals();
    this.geometrie.computeBoundingSphere();
    this.socle.position.z = (min / 1000) * this.exageration - 0.5;
    this.rendre();
  }

  private redimensionner(): void {
    const l = this.conteneur.clientWidth || 300;
    const h = this.conteneur.clientHeight || 300;
    this.renderer.setSize(l, h, false);
    this.camera.aspect = l / h;
    this.camera.updateProjectionMatrix();
    this.rendre();
  }

  /** Rendu groupé sur la prochaine image (et prolongé pendant l'amortissement des contrôles). */
  private rendre(): void {
    if (this.enAttente) return;
    this.enAttente = true;
    requestAnimationFrame(() => {
      this.enAttente = false;
      if (this.controles.update()) this.rendre();
      this.renderer.render(this.scene, this.camera);
    });
  }
}

function srgbVersLineaire(c: number): number {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

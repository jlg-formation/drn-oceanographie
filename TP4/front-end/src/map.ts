/**
 * Visualisation cartographique des exemples géoréférencés (F26, facultatif).
 *
 * Les images d'exemple sont des assemblages de tuiles Web Mercator : posées
 * avec `L.imageOverlay` sur leurs bornes, elles s'alignent exactement avec le
 * fond de carte (Leaflet étire l'image linéairement en EPSG:3857).
 */
import L from "leaflet";
import "leaflet/dist/leaflet.css";

import type { Bornes } from "./model";

export class CarteExemple {
  private readonly map: L.Map;
  private readonly controle: L.Control.Layers;
  private image: L.ImageOverlay | null = null;
  private masque: L.ImageOverlay | null = null;

  constructor(element: HTMLElement) {
    this.map = L.map(element, { zoomControl: true, attributionControl: true });
    const osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      subdomains: "abc",
      maxZoom: 18,
    }).addTo(this.map);
    this.controle = L.control.layers({ OpenStreetMap: osm }, {}, { collapsed: false }).addTo(this.map);
    this.map.setView([48, -3], 6);
  }

  /** Affiche une image d'exemple ; retire le masque de l'analyse précédente. */
  afficherImage(url: string, b: Bornes): void {
    this.retirerMasque();
    if (this.image) {
      this.controle.removeLayer(this.image);
      this.image.remove();
    }
    const bounds = L.latLngBounds([b.south, b.west], [b.north, b.east]);
    this.image = L.imageOverlay(url, bounds).addTo(this.map);
    this.controle.addOverlay(this.image, "Image satellite");
    this.map.invalidateSize();
    this.map.fitBounds(bounds.pad(0.15));
  }

  /** Superpose le masque prédit (data URL PNG) sur l'emprise de l'image. */
  afficherMasque(url: string, opacite: number): void {
    if (!this.image) return;
    this.retirerMasque();
    this.masque = L.imageOverlay(url, this.image.getBounds(), { opacity: opacite }).addTo(this.map);
    this.controle.addOverlay(this.masque, "Masque prédit");
  }

  setOpaciteMasque(opacite: number): void {
    this.masque?.setOpacity(opacite);
  }

  retirerMasque(): void {
    if (!this.masque) return;
    this.controle.removeLayer(this.masque);
    this.masque.remove();
    this.masque = null;
  }

  rafraichir(): void {
    this.map.invalidateSize();
  }
}

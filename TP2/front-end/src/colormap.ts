/** Palette thermique (froid -> chaud) pour colorer le champ de température. */

interface Stop {
  t: number;
  rgb: [number, number, number];
}

// Dégradé type "thermique océan" : bleu profond -> cyan -> vert -> jaune -> rouge.
const STOPS: Stop[] = [
  { t: 0.0, rgb: [12, 44, 132] },
  { t: 0.25, rgb: [34, 168, 205] },
  { t: 0.5, rgb: [96, 200, 120] },
  { t: 0.75, rgb: [250, 205, 60] },
  { t: 1.0, rgb: [200, 40, 40] },
];

/** Renvoie la couleur (r,g,b) pour une valeur normalisée t dans [0, 1]. */
export function thermalColor(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  for (let i = 1; i < STOPS.length; i++) {
    const a = STOPS[i - 1];
    const b = STOPS[i];
    if (clamped <= b.t) {
      const f = (clamped - a.t) / (b.t - a.t);
      return [
        Math.round(a.rgb[0] + f * (b.rgb[0] - a.rgb[0])),
        Math.round(a.rgb[1] + f * (b.rgb[1] - a.rgb[1])),
        Math.round(a.rgb[2] + f * (b.rgb[2] - a.rgb[2])),
      ];
    }
  }
  return STOPS[STOPS.length - 1].rgb;
}

/** Construit un dégradé CSS linéaire pour la légende. */
export function cssGradient(): string {
  const parts = STOPS.map(
    (s) => `rgb(${s.rgb[0]}, ${s.rgb[1]}, ${s.rgb[2]}) ${Math.round(s.t * 100)}%`,
  );
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}

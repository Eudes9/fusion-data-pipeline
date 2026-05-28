"""
Générateur de données synthétiques de diagnostics tokamak.

Simule des décharges plasma multi-capteurs avec :
- Plusieurs grandeurs physiques (courant plasma, densité, température, tension boucle)
- Hétérogénéité contrôlée : formats (CSV/JSON), unités, fréquences d'échantillonnage,
  noms de colonnes, valeurs manquantes, événements de disruption
- Métadonnées par décharge (YAML)

Usage :
    python -m src.ingestion.generate_data
    python -m src.ingestion.generate_data --n-shots 300 --output data/raw
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RNG_SEED = 42


@dataclass
class ShotConfig:
    """Configuration d'une décharge simulée."""

    shot_id: int
    duration_s: float
    sampling_hz: int
    disruption: bool
    disruption_time_s: float | None
    file_format: str
    current_unit: str
    column_naming: str
    operator: str
    machine_config: str


COLUMN_NAMING_SCHEMES = {
    "canonical": {
        "time": "time_s",
        "current": "plasma_current",
        "density": "electron_density",
        "temperature": "electron_temperature",
        "vloop": "loop_voltage",
    },
    "raw": {
        "time": "t",
        "current": "Ip",
        "density": "ne",
        "temperature": "Te",
        "vloop": "Vl",
    },
    "verbose": {
        "time": "timestamp_seconds",
        "current": "Ip_plasma_MA",
        "density": "ne_electron_1e19_m3",
        "temperature": "Te_electron_keV",
        "vloop": "V_loop_volts",
    },
}


def generate_plasma_current(t, duration, disruption, disruption_time):
    """Courant plasma : rampe / plateau / rampe descendante, avec disruption optionnelle."""
    peak_current = np.random.uniform(1.0, 1.8)
    ramp_up_end = 0.5
    ramp_down_start = duration - 0.5

    current = np.zeros_like(t)
    ramp_up_mask = t < ramp_up_end
    current[ramp_up_mask] = peak_current * (t[ramp_up_mask] / ramp_up_end)

    plateau_mask = (t >= ramp_up_end) & (t < ramp_down_start)
    current[plateau_mask] = peak_current + np.random.normal(0, 0.02, size=plateau_mask.sum())

    ramp_down_mask = t >= ramp_down_start
    current[ramp_down_mask] = peak_current * (1 - (t[ramp_down_mask] - ramp_down_start) / 0.5)

    if disruption and disruption_time is not None:
        idx_disrupt = np.argmax(t >= disruption_time)
        disrupt_mask = t >= disruption_time
        decay = np.exp(-(t[disrupt_mask] - disruption_time) / 0.02)
        current[disrupt_mask] = current[idx_disrupt] * decay

    current += np.random.normal(0, 0.01, size=t.shape)
    return np.maximum(current, 0)


def generate_density(t, current):
    """Densité électronique (1e19 m^-3) corrélée au courant."""
    base = 3.5 * current + np.random.uniform(0.5, 1.5)
    return np.maximum(base + np.random.normal(0, 0.1, size=t.shape), 0)


def generate_temperature(t, current):
    """Température électronique (keV) corrélée au courant."""
    base = 2.5 * current + np.random.uniform(-0.3, 0.3)
    return np.maximum(base + np.random.normal(0, 0.08, size=t.shape), 0.1)


def generate_loop_voltage(t, current):
    """Tension boucle (V) : terme inductif + résistif."""
    di_dt = np.gradient(current)
    vloop = 1.5 - 0.5 * current + 50 * di_dt
    vloop += np.random.normal(0, 0.1, size=t.shape)
    return vloop


def inject_missing_values(signal, missing_fraction=0.02):
    """Trous par paquets de 50 à 200 points (panne capteur, pas point isolé)."""
    signal = signal.copy()
    n = len(signal)
    n_missing = int(n * missing_fraction)
    while n_missing > 0:
        gap_size = min(np.random.randint(50, 200), n_missing)
        start = np.random.randint(0, max(1, n - gap_size))
        signal[start:start + gap_size] = np.nan
        n_missing -= gap_size
    return signal


def make_shot_config(shot_id):
    """Configuration aléatoire d'une décharge."""
    disruption = random.random() < 0.20
    duration = np.random.uniform(4.0, 6.0)
    disruption_time = np.random.uniform(1.5, duration - 0.8) if disruption else None
    return ShotConfig(
        shot_id=shot_id,
        duration_s=duration,
        sampling_hz=random.choice([1000, 2000, 5000]),
        disruption=disruption,
        disruption_time_s=disruption_time,
        file_format=random.choice(["csv", "csv", "json"]),
        current_unit=random.choice(["MA", "MA", "A"]),
        column_naming=random.choice(list(COLUMN_NAMING_SCHEMES.keys())),
        operator=random.choice(["Dupont", "Martin", "Bernard", "Petit", "Robert"]),
        machine_config=random.choice(["WEST-LH", "WEST-IC", "WEST-ECRH"]),
    )


def generate_shot(config):
    """Génère le DataFrame brut d'une décharge."""
    n_points = int(config.duration_s * config.sampling_hz)
    t = np.linspace(0, config.duration_s, n_points)

    current_MA = generate_plasma_current(t, config.duration_s, config.disruption, config.disruption_time_s)
    density = inject_missing_values(generate_density(t, current_MA), 0.03)
    temperature = inject_missing_values(generate_temperature(t, current_MA), 0.02)
    vloop = generate_loop_voltage(t, current_MA)

    current_out = current_MA * 1e6 if config.current_unit == "A" else current_MA

    names = COLUMN_NAMING_SCHEMES[config.column_naming]
    return pd.DataFrame({
        names["time"]: t,
        names["current"]: current_out,
        names["density"]: density,
        names["temperature"]: temperature,
        names["vloop"]: vloop,
    })


def save_shot(df, config, output_dir):
    """Sauvegarde signaux + metadata.yaml."""
    shot_dir = output_dir / f"shot_{config.shot_id:04d}"
    shot_dir.mkdir(parents=True, exist_ok=True)

    if config.file_format == "csv":
        df.to_csv(shot_dir / "signals.csv", index=False)
    else:
        df.to_json(shot_dir / "signals.json", orient="records", indent=2)

    metadata = {
        "shot_id": config.shot_id,
        "duration_s": round(config.duration_s, 3),
        "sampling_hz": config.sampling_hz,
        "disruption": config.disruption,
        "disruption_time_s": round(config.disruption_time_s, 3) if config.disruption_time_s else None,
        "file_format": config.file_format,
        "current_unit": config.current_unit,
        "column_naming_scheme": config.column_naming,
        "operator": config.operator,
        "machine_config": config.machine_config,
    }
    with open(shot_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


def generate_dataset(n_shots=200, output_dir=Path("data/raw")):
    """Génère n_shots décharges et un résumé global."""
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_disruptions = 0
    formats_count = {"csv": 0, "json": 0}
    units_count = {"MA": 0, "A": 0}
    naming_count = {k: 0 for k in COLUMN_NAMING_SCHEMES}

    for shot_id in range(1, n_shots + 1):
        config = make_shot_config(shot_id)
        df = generate_shot(config)
        save_shot(df, config, output_dir)

        if config.disruption:
            n_disruptions += 1
        formats_count[config.file_format] += 1
        units_count[config.current_unit] += 1
        naming_count[config.column_naming] += 1

        if shot_id % 50 == 0:
            print(f"  {shot_id}/{n_shots} décharges générées")

    summary = {
        "total_shots": n_shots,
        "disruptions": n_disruptions,
        "disruption_rate": round(n_disruptions / n_shots, 3),
        "formats": formats_count,
        "current_units": units_count,
        "column_naming_schemes": naming_count,
    }

    with open(output_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Génère des données synthétiques de diagnostics tokamak")
    parser.add_argument("--n-shots", type=int, default=200, help="Nombre de décharges")
    parser.add_argument("--output", type=Path, default=Path("data/raw"), help="Dossier de sortie")
    args = parser.parse_args()

    print(f"Génération de {args.n_shots} décharges dans {args.output}/")
    summary = generate_dataset(args.n_shots, args.output)

    print("\nDataset généré :")
    print(f"  Total décharges     : {summary['total_shots']}")
    print(f"  Disruptions         : {summary['disruptions']} ({summary['disruption_rate'] * 100:.1f} %)")
    print(f"  Formats             : {summary['formats']}")
    print(f"  Unités courant      : {summary['current_units']}")
    print(f"  Schémas de nommage  : {summary['column_naming_schemes']}")
    print(f"\nRésumé dans : {args.output}/dataset_summary.json")


if __name__ == "__main__":
    main()

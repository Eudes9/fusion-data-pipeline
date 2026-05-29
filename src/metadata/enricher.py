"""
Module d'enrichissement en métadonnées.

Pour chaque décharge, fusionne :
  1. Les métadonnées brutes (lues du metadata.yaml original)
  2. Les indicateurs de qualité (issus du QualityReport du nettoyage)
  3. Les grandeurs dérivées calculées sur les signaux (max courant, durée, ...)
  4. La traçabilité technique (schéma appliqué, version pipeline, timestamp)

Produit un DataFrame plat : une ligne par décharge, prêt pour la sérialisation
en Parquet (table de dimensions du data mart).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from src.cleaning.cleaner import QualityReport


# Version du pipeline. À incrémenter en cas de changement de méthode de
# traitement, pour assurer la traçabilité des datasets produits.
PIPELINE_VERSION = "1.0.0"


# =============================================================================
# Calcul des grandeurs dérivées à partir des signaux nettoyés
# =============================================================================

@dataclass
class DerivedFeatures:
    """Grandeurs calculées sur les signaux nettoyés d'une décharge."""

    duration_effective_s: float        # durée effective (basée sur le temps)
    n_samples: int                     # nombre de points après nettoyage
    plasma_current_max_MA: float       # pic de courant
    plasma_current_mean_MA: float      # courant moyen sur plateau
    electron_density_mean: float       # densité moyenne
    electron_temperature_mean: float   # température moyenne
    disruption_detected: bool          # disruption détectée dans les données ?
    disruption_time_detected_s: float | None  # instant détecté


def compute_derived_features(df: pd.DataFrame) -> DerivedFeatures:
    """
    Calcule les grandeurs dérivées à partir d'un DataFrame canonique nettoyé.

    La détection de disruption est faite par une règle simple :
    chute du courant plasma supérieure à 50% de son pic sur moins de 50 ms.
    """
    time = df["time_s"].values
    current = df["plasma_current_MA"].values
    density = df["electron_density"].values
    temperature = df["electron_temperature"].values

    duration = float(time[-1] - time[0]) if len(time) > 1 else 0.0
    n_samples = len(df)

    # Statistiques sur le courant
    current_max = float(np.nanmax(current)) if n_samples > 0 else 0.0

    # Plateau : on prend la moyenne sur la zone où le courant > 70% du max
    plateau_mask = current > 0.7 * current_max if current_max > 0 else np.zeros_like(current, dtype=bool)
    current_mean_plateau = (
        float(np.nanmean(current[plateau_mask])) if plateau_mask.any() else 0.0
    )

    density_mean = float(np.nanmean(density[plateau_mask])) if plateau_mask.any() else 0.0
    temperature_mean = float(np.nanmean(temperature[plateau_mask])) if plateau_mask.any() else 0.0

    # Détection de disruption : chute brutale du courant
    disruption_detected, disruption_time = _detect_disruption(time, current)

    return DerivedFeatures(
        duration_effective_s=round(duration, 4),
        n_samples=n_samples,
        plasma_current_max_MA=round(current_max, 4),
        plasma_current_mean_MA=round(current_mean_plateau, 4),
        electron_density_mean=round(density_mean, 4),
        electron_temperature_mean=round(temperature_mean, 4),
        disruption_detected=disruption_detected,
        disruption_time_detected_s=(
            round(disruption_time, 4) if disruption_time is not None else None
        ),
    )


def _detect_disruption(
    time: np.ndarray, current: np.ndarray, drop_fraction: float = 0.5,
    window_s: float = 0.05,
) -> tuple[bool, float | None]:
    """
    Détecte une chute brutale du courant plasma (disruption).

    On parcourt le signal et on cherche un instant où le courant a chuté de plus
    de `drop_fraction` de son pic en moins de `window_s` secondes, avant la fin
    programmée de la décharge.
    """
    if len(time) < 10 or np.all(np.isnan(current)):
        return False, None

    peak = np.nanmax(current)
    if peak < 0.1:
        return False, None

    dt = time[1] - time[0] if len(time) > 1 else 0.001
    window_pts = max(1, int(window_s / dt))

    # On ignore les 200 dernières ms (fin normale)
    end_margin_pts = max(1, int(0.2 / dt))
    search_end = max(window_pts + 1, len(current) - end_margin_pts)

    for i in range(window_pts, search_end):
        if np.isnan(current[i]) or np.isnan(current[i - window_pts]):
            continue
        drop = current[i - window_pts] - current[i]
        if drop > drop_fraction * peak and current[i - window_pts] > 0.5 * peak:
            return True, float(time[i])

    return False, None


# =============================================================================
# Construction de l'enregistrement enrichi pour une décharge
# =============================================================================

def enrich_shot(
    raw_metadata: dict,
    quality_report: QualityReport,
    cleaned_df: pd.DataFrame,
) -> dict:
    """
    Construit l'enregistrement plat (1 ligne) d'une décharge enrichie,
    en fusionnant métadonnées brutes, rapport qualité et features dérivées.
    """
    features = compute_derived_features(cleaned_df)

    record = {
        # --- Identité & contexte ---
        "shot_id": raw_metadata["shot_id"],
        "operator": raw_metadata.get("operator"),
        "machine_config": raw_metadata.get("machine_config"),

        # --- Caractéristiques d'origine ---
        "source_format": raw_metadata.get("file_format"),
        "source_current_unit": raw_metadata.get("current_unit"),
        "source_column_naming": raw_metadata.get("column_naming_scheme"),
        "source_sampling_hz": raw_metadata.get("sampling_hz"),
        "declared_disruption": raw_metadata.get("disruption"),
        "declared_disruption_time_s": raw_metadata.get("disruption_time_s"),
        "declared_duration_s": raw_metadata.get("duration_s"),

        # --- Indicateurs de qualité ---
        "quality_flag": quality_report.quality_flag,
        "overall_completeness": round(quality_report.overall_completeness, 4),
        "completeness_plasma_current": round(
            quality_report.completeness_per_signal.get("plasma_current_MA", 0.0), 4
        ),
        "completeness_density": round(
            quality_report.completeness_per_signal.get("electron_density", 0.0), 4
        ),
        "completeness_temperature": round(
            quality_report.completeness_per_signal.get("electron_temperature", 0.0), 4
        ),
        "completeness_vloop": round(
            quality_report.completeness_per_signal.get("loop_voltage", 0.0), 4
        ),
        "outliers_plasma_current": quality_report.outliers_per_signal.get(
            "plasma_current_MA", 0
        ),
        "outliers_density": quality_report.outliers_per_signal.get(
            "electron_density", 0
        ),
        "outliers_temperature": quality_report.outliers_per_signal.get(
            "electron_temperature", 0
        ),
        "outliers_vloop": quality_report.outliers_per_signal.get("loop_voltage", 0),

        # --- Grandeurs dérivées (calculées sur les signaux) ---
        "duration_effective_s": features.duration_effective_s,
        "n_samples_after_cleaning": features.n_samples,
        "plasma_current_max_MA": features.plasma_current_max_MA,
        "plasma_current_mean_MA": features.plasma_current_mean_MA,
        "electron_density_mean": features.electron_density_mean,
        "electron_temperature_mean": features.electron_temperature_mean,
        "disruption_detected": features.disruption_detected,
        "disruption_time_detected_s": features.disruption_time_detected_s,

        # --- Traçabilité ---
        "schema_applied": raw_metadata.get("_schema_applied"),
        "pipeline_version": PIPELINE_VERSION,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    return record


def build_metadata_table(records: Iterable[dict]) -> pd.DataFrame:
    """
    Assemble une liste d'enregistrements enrichis en un DataFrame
    (la table de dimensions du data mart).
    """
    df = pd.DataFrame(list(records))

    # Tri logique : par shot_id
    if "shot_id" in df.columns:
        df = df.sort_values("shot_id").reset_index(drop=True)

    return df


# =============================================================================
# CLI de démonstration
# =============================================================================

def main():
    """
    Démonstration : ingère, nettoie, enrichit et affiche la table de métadonnées.
    """
    import argparse
    from src.ingestion.loader import IngestionConfig, load_all_shots
    from src.cleaning.cleaner import CleaningConfig, clean_shot

    parser = argparse.ArgumentParser(
        description="Enrichissement en métadonnées des décharges"
    )
    parser.add_argument("--config", default="config/sources.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()

    print("1. Ingestion...")
    ing_config = IngestionConfig.from_yaml(args.config)
    shots = load_all_shots(args.raw_dir, ing_config)
    print(f"   {len(shots)} décharges ingérées")

    print("2. Nettoyage...")
    cleaning_config = CleaningConfig()
    cleaned_shots = []
    for s in shots:
        df_clean, report = clean_shot(s["df"], s["shot_id"], cleaning_config)
        cleaned_shots.append({
            "raw_metadata": s["metadata"],
            "quality_report": report,
            "cleaned_df": df_clean,
        })

    print("3. Enrichissement en métadonnées...")
    records = [
        enrich_shot(s["raw_metadata"], s["quality_report"], s["cleaned_df"])
        for s in cleaned_shots
    ]
    df_meta = build_metadata_table(records)

    print(f"\nTable de métadonnées construite : {len(df_meta)} lignes × {len(df_meta.columns)} colonnes")
    print(f"\nColonnes :")
    for c in df_meta.columns:
        print(f"  - {c}")

    print(f"\nAperçu (3 premières décharges, colonnes sélectionnées) :")
    cols_apercu = [
        "shot_id", "machine_config", "quality_flag",
        "overall_completeness", "declared_disruption",
        "disruption_detected", "plasma_current_max_MA",
    ]
    print(df_meta[cols_apercu].head(3).to_string(index=False))

    # Quelques statistiques utiles
    print("\nStatistiques globales :")
    print(f"  Flags qualité           : {df_meta['quality_flag'].value_counts().to_dict()}")
    print(f"  Disruptions déclarées   : {int(df_meta['declared_disruption'].sum())}")
    print(f"  Disruptions détectées   : {int(df_meta['disruption_detected'].sum())}")
    print(f"  Accord déclaré/détecté  : {int((df_meta['declared_disruption'] == df_meta['disruption_detected']).sum())} / {len(df_meta)}")
    print(f"  Complétude moyenne      : {df_meta['overall_completeness'].mean() * 100:.2f} %")


if __name__ == "__main__":
    main()
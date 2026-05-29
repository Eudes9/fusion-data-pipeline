"""
Module de nettoyage générique.

Applique sur n'importe quelle décharge ingérée (au format canonique) :
  1. Resynchronisation temporelle sur une grille uniforme
  2. Détection et marquage des valeurs aberrantes (méthode MAD)
  3. Imputation des valeurs manquantes par interpolation
  4. Calcul d'indicateurs de qualité par décharge

Le code est totalement indépendant du domaine : il opère sur des séries
temporelles numériques et ne sait rien de la physique des plasmas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# Configuration du nettoyage (valeurs par défaut raisonnables)
# =============================================================================

@dataclass
class CleaningConfig:
    """Paramètres du nettoyage."""

    # Grille temporelle commune (Hz) sur laquelle tout est rééchantillonné
    target_sampling_hz: int = 1000

    # Seuil de détection d'aberrants en nombre d'écarts MAD
    outlier_mad_threshold: float = 6.0

    # Longueur max d'un trou qu'on accepte d'imputer (en points, à la fréquence
    # cible). Au-delà, on laisse en NaN car l'imputation deviendrait trompeuse.
    max_gap_to_impute: int = 100

    # Colonnes de signaux à nettoyer (toutes sauf le temps)
    signal_columns: tuple[str, ...] = (
        "plasma_current_MA",
        "electron_density",
        "electron_temperature",
        "loop_voltage",
    )

    time_column: str = "time_s"


@dataclass
class QualityReport:
    """Indicateurs de qualité d'une décharge après nettoyage."""

    shot_id: int
    n_points_before: int
    n_points_after: int
    completeness_per_signal: dict = field(default_factory=dict)  # % non-NaN
    outliers_per_signal: dict = field(default_factory=dict)      # nombre détecté
    overall_completeness: float = 0.0
    quality_flag: str = "OK"  # "OK", "DEGRADED", "REJECTED"

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "n_points_before": self.n_points_before,
            "n_points_after": self.n_points_after,
            "overall_completeness": round(self.overall_completeness, 4),
            "quality_flag": self.quality_flag,
            "completeness_per_signal": {
                k: round(v, 4) for k, v in self.completeness_per_signal.items()
            },
            "outliers_per_signal": self.outliers_per_signal,
        }


# =============================================================================
# 1. Resynchronisation temporelle
# =============================================================================

def resample_to_grid(
    df: pd.DataFrame, time_col: str, target_hz: int
) -> pd.DataFrame:
    """
    Rééchantillonne le DataFrame sur une grille temporelle uniforme à target_hz.

    Utilise np.interp pour l'interpolation linéaire, qui étend automatiquement
    par les valeurs aux bornes (pas de NaN d'extrémité). Les NaN à l'intérieur
    du signal source sont filtrés avant interpolation pour ne pas les propager
    artificiellement ; ils sont traités par la suite par impute_short_gaps.
    """
    if df.empty:
        return df.copy()

    t_orig = df[time_col].values
    t_min = t_orig[0]
    t_max = t_orig[-1]
    n_target = int(round((t_max - t_min) * target_hz)) + 1
    new_time = np.linspace(t_min, t_max, n_target)

    result = pd.DataFrame({time_col: new_time})

    for col in df.columns:
        if col == time_col:
            continue
        values = df[col].values.astype(float)

        # np.interp ne sait pas gérer les NaN dans la source : on interpole
        # uniquement sur les points valides. Les zones manquantes seront
        # détectées et traitées par impute_short_gaps en aval.
        valid = ~np.isnan(values)

        if valid.sum() < 2:
            # Pas assez de points valides pour interpoler : on remplit de NaN
            result[col] = np.full(n_target, np.nan)
            continue

        # Interpolation linéaire avec extension constante aux bornes
        # (comportement par défaut de np.interp : left=values[valid][0], right=values[valid][-1])
        result[col] = np.interp(new_time, t_orig[valid], values[valid])

    return result
# =============================================================================
# 2. Détection des aberrants (méthode MAD - Median Absolute Deviation)
# =============================================================================

def detect_outliers_mad(
    signal: np.ndarray, threshold: float = 6.0, window: int = 101
) -> np.ndarray:
    """
    Détecte les valeurs aberrantes par la méthode MAD glissante.

    Pour chaque point, calcule la médiane et le MAD sur une fenêtre locale
    centrée. Un point est marqué aberrant s'il s'écarte fortement de cette
    tendance locale. Adapté aux signaux structurés (rampes, plateaux) où
    une MAD globale ne marcherait pas.

    Args:
        signal : série temporelle 1D
        threshold : seuil en modified z-scores (6.0 = très conservateur)
        window : taille de la fenêtre glissante (impaire, ~100 ms à 1 kHz)

    Retourne un masque booléen : True = aberrant.
    """
    valid = ~np.isnan(signal)
    if valid.sum() < window:
        return np.zeros_like(signal, dtype=bool)

    # Médiane glissante (tendance locale)
    s = pd.Series(signal)
    local_median = s.rolling(window=window, center=True, min_periods=1).median()

    # Écart absolu à la tendance locale
    abs_dev = (s - local_median).abs()

    # MAD glissant (échelle locale du bruit)
    local_mad = abs_dev.rolling(window=window, center=True, min_periods=1).median()

    # Modified z-score local
    # On évite la division par zéro en ajoutant un epsilon minuscule
    eps = 1e-9
    modified_z = 0.6745 * (s - local_median) / (local_mad + eps)

    mask = modified_z.abs() > threshold
    return mask.values
# =============================================================================
# 3. Imputation des valeurs manquantes
# =============================================================================

def impute_short_gaps(
    signal: np.ndarray, max_gap: int = 100
) -> tuple[np.ndarray, int]:
    """
    Impute par interpolation linéaire les trous de NaN plus courts que max_gap.
    Les trous plus longs sont laissés en NaN (l'imputation serait trompeuse).

    Retourne (signal imputé, nombre de NaN restants).
    """
    s = pd.Series(signal)
    # Identification des trous et de leur longueur
    is_nan = s.isna()
    if not is_nan.any():
        return signal, 0

    # Groupes contigus de NaN
    gap_id = (is_nan != is_nan.shift()).cumsum()
    gap_sizes = is_nan.groupby(gap_id).transform("sum")
    short_gaps = is_nan & (gap_sizes <= max_gap)

    # Interpolation seulement sur les trous courts
    s_interp = s.interpolate(method="linear", limit_direction="both")
    s.loc[short_gaps] = s_interp.loc[short_gaps]

    return s.values, int(s.isna().sum())


# =============================================================================
# 4. Pipeline complet de nettoyage d'une décharge
# =============================================================================

def clean_shot(
    df: pd.DataFrame, shot_id: int, config: CleaningConfig
) -> tuple[pd.DataFrame, QualityReport]:
    """
    Nettoie une décharge canonique et produit un rapport de qualité.

    Étapes :
      1. Resampling sur la grille commune
      2. Détection des aberrants (les remplace par NaN pour imputation)
      3. Imputation des courts trous
      4. Calcul du rapport de qualité
    """
    n_before = len(df)

    # 1. Rééchantillonnage
    df_clean = resample_to_grid(df, config.time_column, config.target_sampling_hz)

    report = QualityReport(
        shot_id=shot_id,
        n_points_before=n_before,
        n_points_after=len(df_clean),
    )

    # 2. + 3. Détection des aberrants puis imputation, signal par signal
    for col in config.signal_columns:
        if col not in df_clean.columns:
            continue
        signal = df_clean[col].values.astype(float)

        # Détection des aberrants -> remplacés par NaN
        outliers_mask = detect_outliers_mad(signal, config.outlier_mad_threshold)
        n_outliers = int(outliers_mask.sum())
        signal[outliers_mask] = np.nan

        # Imputation des courts trous
        signal, n_remaining_nan = impute_short_gaps(
            signal, config.max_gap_to_impute
        )
        df_clean[col] = signal

        # Indicateurs
        completeness = 1.0 - (n_remaining_nan / len(signal))
        report.completeness_per_signal[col] = completeness
        report.outliers_per_signal[col] = n_outliers

    # 4. Score global et flag de qualité
    if report.completeness_per_signal:
        report.overall_completeness = float(
            np.mean(list(report.completeness_per_signal.values()))
        )
    else:
        report.overall_completeness = 0.0

    # Règle métier simple : flag selon le seuil de complétude
    if report.overall_completeness >= 0.98:
        report.quality_flag = "OK"
    elif report.overall_completeness >= 0.90:
        report.quality_flag = "DEGRADED"
    else:
        report.quality_flag = "REJECTED"

    return df_clean, report


# =============================================================================
# CLI de démonstration
# =============================================================================

def main():
    """Démontre le nettoyage sur toutes les décharges ingérées."""
    import argparse
    from src.ingestion.loader import IngestionConfig, load_all_shots

    parser = argparse.ArgumentParser(description="Nettoyage générique des décharges")
    parser.add_argument("--config", default="config/sources.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()

    print("Chargement de la configuration d'ingestion...")
    ing_config = IngestionConfig.from_yaml(args.config)

    print(f"Ingestion depuis {args.raw_dir}/")
    shots = load_all_shots(args.raw_dir, ing_config)
    print(f"  {len(shots)} décharges ingérées\n")

    print("Nettoyage générique en cours...")
    cleaning_config = CleaningConfig()
    reports = []
    for s in shots:
        df_clean, report = clean_shot(s["df"], s["shot_id"], cleaning_config)
        reports.append(report)

    # Synthèse globale
    flags = {"OK": 0, "DEGRADED": 0, "REJECTED": 0}
    for r in reports:
        flags[r.quality_flag] += 1

    avg_completeness = np.mean([r.overall_completeness for r in reports])
    total_outliers = sum(
        sum(r.outliers_per_signal.values()) for r in reports
    )

    print(f"\nNettoyage terminé sur {len(reports)} décharges.")
    print(f"  Complétude moyenne   : {avg_completeness * 100:.2f} %")
    print(f"  Aberrants détectés   : {total_outliers}")
    print(f"  Flags qualité        : {flags}")

    # Échantillon d'un rapport
    print(f"\nExemple - rapport décharge {reports[0].shot_id} :")
    import json
    print(json.dumps(reports[0].to_dict(), indent=2))


if __name__ == "__main__":
    main()
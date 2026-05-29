"""
Tests du module de nettoyage générique.
"""

import numpy as np
import pandas as pd

from src.cleaning.cleaner import (
    CleaningConfig,
    clean_shot,
    detect_outliers_mad,
    impute_short_gaps,
    resample_to_grid,
)


def test_detect_outliers_mad_finds_clear_outlier():
    """Un point très loin de la médiane doit être détecté."""
    signal = np.ones(1000)
    signal[500] = 100.0  # aberrant évident
    mask = detect_outliers_mad(signal, threshold=6.0)
    assert mask[500]
    assert mask.sum() == 1


def test_detect_outliers_mad_ignores_normal_noise():
    """Du bruit gaussien normal ne doit pas être détecté massivement."""
    np.random.seed(0)
    signal = np.random.normal(0, 1, 1000)
    mask = detect_outliers_mad(signal, threshold=6.0)
    # Moins de 1% détecté à 6 sigmas (très permissif)
    assert mask.sum() < 10


def test_impute_short_gaps_fills_small_holes():
    """Un trou court doit être imputé linéairement."""
    signal = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
    imputed, n_remaining = impute_short_gaps(signal, max_gap=10)
    assert n_remaining == 0
    assert not np.isnan(imputed).any()


def test_impute_short_gaps_leaves_long_holes():
    """Un trou plus long que max_gap doit rester NaN."""
    signal = np.array([1.0] + [np.nan] * 50 + [2.0])
    imputed, n_remaining = impute_short_gaps(signal, max_gap=10)
    assert n_remaining == 50


def test_resample_to_grid_produces_uniform_spacing():
    """Le rééchantillonnage doit produire une grille uniforme."""
    df = pd.DataFrame({
        "time_s": [0.0, 0.003, 0.007, 0.012, 0.020],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    df_resampled = resample_to_grid(df, "time_s", target_hz=1000)
    # La grille doit être uniforme
    diffs = np.diff(df_resampled["time_s"].values)
    assert np.allclose(diffs, diffs[0])


def test_clean_shot_produces_quality_report():
    """Test bout-en-bout : nettoyage d'une décharge avec rapport."""
    np.random.seed(0)
    n = 5000
    df = pd.DataFrame({
        "time_s": np.linspace(0, 5, n),
        "plasma_current_MA": np.full(n, 1.5) + np.random.normal(0, 0.02, n),
        "electron_density": np.full(n, 4.0) + np.random.normal(0, 0.1, n),
        "electron_temperature": np.full(n, 3.0) + np.random.normal(0, 0.05, n),
        "loop_voltage": np.full(n, 0.5) + np.random.normal(0, 0.1, n),
    })
    # On injecte quelques aberrants et NaN
    df.loc[100, "plasma_current_MA"] = 50.0
    df.loc[1000:1050, "electron_density"] = np.nan

    df_clean, report = clean_shot(df, shot_id=1, config=CleaningConfig())

    assert report.shot_id == 1
    assert report.overall_completeness > 0.95
    assert report.outliers_per_signal["plasma_current_MA"] >= 1
    assert report.quality_flag in {"OK", "DEGRADED", "REJECTED"}
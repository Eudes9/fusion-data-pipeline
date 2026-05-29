"""
Tests du module d'enrichissement en métadonnées.
"""

import numpy as np
import pandas as pd

from src.cleaning.cleaner import QualityReport
from src.metadata.enricher import (
    PIPELINE_VERSION,
    build_metadata_table,
    compute_derived_features,
    enrich_shot,
    _detect_disruption,
)


# Fixture utilitaire : un DataFrame canonique propre simulant une décharge sans disruption
def _make_clean_shot(n=5000, duration=5.0, peak_current=1.5):
    t = np.linspace(0, duration, n)
    # Forme rampe / plateau / rampe
    current = np.full(n, peak_current)
    ramp_pts = int(0.5 / (duration / n))
    current[:ramp_pts] = np.linspace(0, peak_current, ramp_pts)
    current[-ramp_pts:] = np.linspace(peak_current, 0, ramp_pts)

    return pd.DataFrame({
        "time_s": t,
        "plasma_current_MA": current,
        "electron_density": 3.5 * current + 0.5,
        "electron_temperature": 2.5 * current + 0.3,
        "loop_voltage": 1.0 - 0.3 * current,
    })


def test_compute_derived_features_basic():
    """Vérifie les grandeurs dérivées sur un signal propre."""
    df = _make_clean_shot()
    features = compute_derived_features(df)

    assert features.n_samples == 5000
    assert features.duration_effective_s > 4.9
    assert 1.4 < features.plasma_current_max_MA < 1.6
    assert features.disruption_detected is False


def test_detect_disruption_finds_real_drop():
    """Une chute brutale du courant doit être détectée."""
    n = 5000
    t = np.linspace(0, 5, n)
    current = np.full(n, 1.5)
    # Disruption à t=2s : chute brutale en quelques ms
    drop_idx = int(2.0 / 5.0 * n)
    current[drop_idx:drop_idx + 20] = np.linspace(1.5, 0.1, 20)
    current[drop_idx + 20:] = 0.1

    detected, time = _detect_disruption(t, current)
    assert detected
    assert 1.95 < time < 2.05


def test_detect_disruption_normal_shot():
    """Une décharge propre sans chute brutale ne doit pas déclencher."""
    df = _make_clean_shot()
    detected, _ = _detect_disruption(df["time_s"].values, df["plasma_current_MA"].values)
    assert detected is False


def test_enrich_shot_produces_complete_record():
    """L'enrichissement doit produire un dict avec toutes les clés attendues."""
    df = _make_clean_shot()
    raw_meta = {
        "shot_id": 42,
        "operator": "Dupont",
        "machine_config": "WEST-LH",
        "file_format": "csv",
        "current_unit": "MA",
        "column_naming_scheme": "canonical",
        "sampling_hz": 1000,
        "disruption": False,
        "disruption_time_s": None,
        "duration_s": 5.0,
        "_schema_applied": "canonical_MA",
    }
    quality = QualityReport(
        shot_id=42,
        n_points_before=5000,
        n_points_after=5000,
        completeness_per_signal={
            "plasma_current_MA": 1.0,
            "electron_density": 0.98,
            "electron_temperature": 0.99,
            "loop_voltage": 1.0,
        },
        outliers_per_signal={
            "plasma_current_MA": 0,
            "electron_density": 2,
            "electron_temperature": 1,
            "loop_voltage": 0,
        },
        overall_completeness=0.9925,
        quality_flag="OK",
    )

    record = enrich_shot(raw_meta, quality, df)

    # Identité
    assert record["shot_id"] == 42
    assert record["operator"] == "Dupont"
    # Qualité
    assert record["quality_flag"] == "OK"
    assert record["completeness_density"] == 0.98
    # Dérivées
    assert "plasma_current_max_MA" in record
    assert record["disruption_detected"] is False
    # Traçabilité
    assert record["pipeline_version"] == PIPELINE_VERSION
    assert record["schema_applied"] == "canonical_MA"
    assert "processed_at_utc" in record


def test_build_metadata_table_sorts_by_shot_id():
    """La table doit être triée par shot_id."""
    df = _make_clean_shot()
    quality = QualityReport(
        shot_id=0, n_points_before=5000, n_points_after=5000,
        completeness_per_signal={}, outliers_per_signal={},
        overall_completeness=1.0, quality_flag="OK",
    )
    raw_meta_base = {
        "operator": "X", "machine_config": "Y", "file_format": "csv",
        "current_unit": "MA", "column_naming_scheme": "canonical",
        "sampling_hz": 1000, "disruption": False, "disruption_time_s": None,
        "duration_s": 5.0, "_schema_applied": "canonical_MA",
    }
    records = []
    for sid in [3, 1, 2]:
        meta = {**raw_meta_base, "shot_id": sid}
        quality.shot_id = sid
        records.append(enrich_shot(meta, quality, df))

    table = build_metadata_table(records)
    assert list(table["shot_id"]) == [1, 2, 3]
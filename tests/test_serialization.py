"""
Tests du module de sérialisation Parquet.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.serialization.writer import (
    SerializationConfig,
    read_metadata_table,
    read_signals,
    serialize_data_mart,
    write_dataset_manifest,
    write_metadata_table,
    write_signals_partitioned,
)


def _make_meta_df(n=5):
    """Mini table de métadonnées pour les tests."""
    return pd.DataFrame({
        "shot_id": list(range(1, n + 1)),
        "operator": ["Dupont"] * n,
        "machine_config": ["WEST-LH"] * n,
        "quality_flag": ["OK"] * (n - 1) + ["DEGRADED"],
        "overall_completeness": [0.99] * n,
        "declared_disruption": [False, True, False, True, False][:n],
        "disruption_detected": [False, True, False, True, False][:n],
    })


def _make_signals(n_shots=5, n_points=100):
    """Mini liste de décharges pour les tests."""
    shots = []
    for sid in range(1, n_shots + 1):
        df = pd.DataFrame({
            "time_s": np.linspace(0, 1, n_points),
            "plasma_current_MA": np.full(n_points, 1.5),
            "electron_density": np.full(n_points, 4.0),
            "electron_temperature": np.full(n_points, 3.0),
            "loop_voltage": np.full(n_points, 0.5),
        })
        shots.append({"shot_id": sid, "df": df})
    return shots


def test_write_metadata_table(tmp_path):
    """La table de métadonnées doit être écrite et relisible à l'identique."""
    df = _make_meta_df()
    config = SerializationConfig(output_dir=tmp_path)
    path = write_metadata_table(df, config)
    assert path.exists()

    df_back = pd.read_parquet(path)
    assert len(df_back) == len(df)
    assert list(df_back.columns) == list(df.columns)


def test_write_signals_partitioned_creates_one_dir_per_shot(tmp_path):
    """Chaque décharge doit avoir sa partition shot_id=N."""
    shots = _make_signals(n_shots=3)
    config = SerializationConfig(output_dir=tmp_path)
    signals_path = write_signals_partitioned(shots, config)

    assert signals_path.exists()
    partitions = sorted(p.name for p in signals_path.iterdir() if p.is_dir())
    assert partitions == ["shot_id=1", "shot_id=2", "shot_id=3"]
    # Chaque partition doit contenir un fichier Parquet
    for p in signals_path.iterdir():
        assert (p / "part-0.parquet").exists()


def test_read_signals_with_predicate_pushdown(tmp_path):
    """Le filtre sur shot_id ne doit retourner que les décharges demandées."""
    shots = _make_signals(n_shots=5, n_points=50)
    config = SerializationConfig(output_dir=tmp_path)
    write_signals_partitioned(shots, config)

    # On ne lit que les décharges 2 et 4
    df = read_signals(tmp_path, shot_ids=[2, 4])
    assert set(df["shot_id"].unique()) == {2, 4}
    assert len(df) == 2 * 50


def test_dataset_manifest_is_valid_json(tmp_path):
    """Le manifest doit être un JSON valide avec les sections attendues."""
    df = _make_meta_df()
    config = SerializationConfig(output_dir=tmp_path)
    path = write_dataset_manifest(df, config)

    with open(path) as f:
        manifest = json.load(f)

    assert "pipeline_version" in manifest
    assert "tables" in manifest
    assert "metadata" in manifest["tables"]
    assert "signals" in manifest["tables"]
    assert "statistics" in manifest
    assert manifest["statistics"]["total_shots"] == len(df)


def test_serialize_data_mart_end_to_end(tmp_path):
    """Pipeline complet de sérialisation + relecture."""
    df_meta = _make_meta_df()
    shots = _make_signals(n_shots=5)
    config = SerializationConfig(output_dir=tmp_path)

    artefacts = serialize_data_mart(df_meta, shots, config)

    # Tous les artefacts existent
    assert all(_path_exists(p) for p in artefacts.values())

    # Lecture complète via les helpers
    df_meta_back = read_metadata_table(tmp_path)
    df_signals_back = read_signals(tmp_path)
    assert len(df_meta_back) == 5
    assert len(df_signals_back) == 5 * 100  # 5 shots × 100 points


def _path_exists(p):
    from pathlib import Path
    return Path(p).exists()
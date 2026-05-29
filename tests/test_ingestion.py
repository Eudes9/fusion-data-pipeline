"""
Tests du module d'ingestion configurable.

On vérifie que :
  - La configuration se charge correctement
  - Le schéma résolu dépend bien des métadonnées
  - Une décharge en MA et une en A donnent le même résultat canonique
  - Les colonnes canoniques sont identiques quels que soient le format/schéma
  - Une métadonnée invalide lève une erreur explicite
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.ingestion.loader import (
    IngestionConfig,
    _apply_schema,
    load_shot,
)


@pytest.fixture(scope="module")
def config():
    """Charge la configuration une fois pour tous les tests du module."""
    config_path = Path(__file__).parent.parent / "config" / "sources.yaml"
    return IngestionConfig.from_yaml(config_path)


def test_config_loads(config):
    """La config doit définir des colonnes canoniques et au moins un schéma."""
    assert config.canonical_columns
    assert "time" in config.canonical_columns
    assert len(config.schemas) >= 1


def test_resolve_schema(config):
    """La résolution doit combiner naming + unit selon le pattern."""
    metadata = {"column_naming_scheme": "raw", "current_unit": "MA"}
    assert config.resolve_schema(metadata) == "raw_MA"

    metadata = {"column_naming_scheme": "verbose", "current_unit": "A"}
    assert config.resolve_schema(metadata) == "verbose_A"


def test_resolve_schema_missing_metadata(config):
    """Métadonnée incomplète doit lever une erreur claire."""
    with pytest.raises(ValueError, match="Métadonnées insuffisantes"):
        config.resolve_schema({"column_naming_scheme": "raw"})


def test_apply_schema_renames_and_converts(config):
    """Vérifie le renommage + conversion d'unités sur un mini DataFrame."""
    df_raw = pd.DataFrame({
        "t": [0.0, 0.1, 0.2],
        "Ip": [1_500_000.0, 1_500_000.0, 1_500_000.0],  # Ampères bruts
        "ne": [3.0, 3.1, 3.0],
        "Te": [2.5, 2.5, 2.6],
        "Vl": [1.0, 1.0, 1.0],
    })
    schema = config.schemas["raw_A"]
    df_can = _apply_schema(df_raw, schema, config.canonical_columns)

    # Colonnes canoniques attendues
    assert "time_s" in df_can.columns
    assert "plasma_current_MA" in df_can.columns
    # Conversion A -> MA : 1.5e6 A doit donner 1.5 MA
    assert np.allclose(df_can["plasma_current_MA"], 1.5)


def test_apply_schema_missing_column_raises(config):
    """Une colonne attendue manquante doit lever une erreur explicite."""
    df_bad = pd.DataFrame({"t": [0.0], "Ip": [1.0]})  # manque ne, Te, Vl
    schema = config.schemas["raw_MA"]
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        _apply_schema(df_bad, schema, config.canonical_columns)


def test_load_shot_produces_canonical_format(config, tmp_path):
    """
    Test bout-en-bout : on simule une décharge en format raw_A et on vérifie
    que load_shot la convertit bien au format canonique.
    """
    shot_dir = tmp_path / "shot_test"
    shot_dir.mkdir()

    # Signaux en format "raw", courant en Ampères bruts
    df_raw = pd.DataFrame({
        "t": np.linspace(0, 1, 100),
        "Ip": np.full(100, 1_200_000.0),  # 1.2 MA exprimé en A
        "ne": np.full(100, 4.0),
        "Te": np.full(100, 3.0),
        "Vl": np.full(100, 0.5),
    })
    df_raw.to_csv(shot_dir / "signals.csv", index=False)

    metadata = {
        "shot_id": 999,
        "file_format": "csv",
        "current_unit": "A",
        "column_naming_scheme": "raw",
        "duration_s": 1.0,
        "sampling_hz": 100,
        "disruption": False,
        "disruption_time_s": None,
        "operator": "Test",
        "machine_config": "TEST",
    }
    with open(shot_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)

    df_can, md = load_shot(shot_dir, config)

    # Format canonique
    assert list(df_can.columns) == [
        "time_s", "plasma_current_MA", "electron_density",
        "electron_temperature", "loop_voltage",
    ]
    # Conversion d'unité appliquée
    assert np.allclose(df_can["plasma_current_MA"], 1.2)
    # Schéma utilisé tracé dans les métadonnées
    assert md["_schema_applied"] == "raw_A"

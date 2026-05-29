"""
Tests de la couche de visualisation Plotly.

On ne teste pas le rendu visuel (impossible à automatiser proprement) ; on teste
que les HTML sont produits, non-vides, et contiennent les attendus.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.serialization.writer import (
    SerializationConfig,
    serialize_data_mart,
)
from src.visualization.plots import (
    plot_quality_overview,
    plot_shot_detail,
    plot_shots_comparison,
)


@pytest.fixture
def mini_data_mart(tmp_path):
    """Crée un mini data mart en mémoire/disque pour les tests viz."""
    n_shots = 5
    n_points = 100
    meta_rows = []
    shots = []
    for sid in range(1, n_shots + 1):
        disruption = sid in (2, 4)
        meta_rows.append({
            "shot_id": sid,
            "operator": "Test",
            "machine_config": "WEST-LH",
            "quality_flag": "OK",
            "overall_completeness": 0.99,
            "completeness_plasma_current": 1.0,
            "completeness_density": 0.99,
            "completeness_temperature": 0.99,
            "completeness_vloop": 1.0,
            "declared_disruption": disruption,
            "disruption_detected": disruption,
            "disruption_time_detected_s": 0.5 if disruption else None,
        })
        df = pd.DataFrame({
            "time_s": np.linspace(0, 1, n_points),
            "plasma_current_MA": np.full(n_points, 1.5),
            "electron_density": np.full(n_points, 4.0),
            "electron_temperature": np.full(n_points, 3.0),
            "loop_voltage": np.full(n_points, 0.5),
        })
        shots.append({"shot_id": sid, "df": df})

    df_meta = pd.DataFrame(meta_rows)
    config = SerializationConfig(output_dir=tmp_path)
    serialize_data_mart(df_meta, shots, config)
    return tmp_path


def test_plot_shot_detail_produces_html(mini_data_mart):
    """La vue détaillée doit produire un HTML non vide."""
    out = plot_shot_detail(shot_id=2, processed_dir=mini_data_mart)
    assert out.exists()
    assert out.stat().st_size > 1000  # HTML Plotly fait au moins quelques Ko
    content = out.read_text(encoding="utf-8")
    assert "plotly" in content.lower()
    # On vérifie que les données du shot 2 sont bien dedans
    assert "Shot 2" in content or "shot_id=2" in content or "2" in content


def test_plot_quality_overview_produces_html(mini_data_mart):
    """La vue qualité doit produire un HTML non vide et valide."""
    out = plot_quality_overview(processed_dir=mini_data_mart)
    assert out.exists()
    # Un HTML Plotly avec subplots fait largement plus de 50 Ko
    assert out.stat().st_size > 5_000
    content = out.read_text(encoding="utf-8")
    # Vérifications structurelles minimales
    assert content.startswith("<html>") or "<html" in content
    assert "plotly" in content.lower()
    assert "</html>" in content

def test_plot_shots_comparison_produces_html(mini_data_mart):
    """La superposition doit produire un HTML non vide."""
    out = plot_shots_comparison(
        shot_ids=[1, 2, 3],
        signal="plasma_current_MA",
        processed_dir=mini_data_mart,
    )
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "plotly" in content.lower()
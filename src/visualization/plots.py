"""
Module de visualisation Plotly.

Produit trois visualisations interactives en HTML autonome, lisibles dans
n'importe quel navigateur (pas besoin de Python ou de serveur) :

  1. shot_detail.html   : vue détaillée d'une décharge (4 signaux + disruption)
  2. quality_overview.html : vue agrégée qualité du dataset
  3. shots_comparison.html : superposition de plusieurs décharges

Les fichiers sont écrits dans data/processed/visualizations/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.serialization.writer import read_metadata_table, read_signals


# =============================================================================
# 1. Vue détaillée d'une décharge
# =============================================================================

def plot_shot_detail(
    shot_id: int,
    processed_dir: str | Path = "data/processed",
    output_path: str | Path | None = None,
) -> Path:
    """
    Vue détaillée d'une décharge : 4 signaux superposés en sous-graphiques
    partageant l'axe temporel, avec marquage de la disruption si elle existe.
    """
    processed_dir = Path(processed_dir)
    df_signals = read_signals(processed_dir, shot_ids=[shot_id])
    df_meta = read_metadata_table(processed_dir)
    meta = df_meta[df_meta["shot_id"] == shot_id].iloc[0].to_dict()

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "Courant plasma (MA)",
            "Densité électronique (10¹⁹ m⁻³)",
            "Température électronique (keV)",
            "Tension boucle (V)",
        ],
    )

    signals_to_plot = [
        ("plasma_current_MA", "#1f77b4"),
        ("electron_density", "#2ca02c"),
        ("electron_temperature", "#d62728"),
        ("loop_voltage", "#9467bd"),
    ]

    for i, (col, color) in enumerate(signals_to_plot, start=1):
        fig.add_trace(
            go.Scatter(
                x=df_signals["time_s"],
                y=df_signals[col],
                mode="lines",
                line=dict(color=color, width=1.2),
                name=col,
                showlegend=False,
                hovertemplate="t=%{x:.3f}s<br>%{y:.3f}<extra></extra>",
            ),
            row=i, col=1,
        )

    # Marquage de la disruption si elle existe
    if meta.get("disruption_detected") and meta.get("disruption_time_detected_s"):
        t_disrupt = meta["disruption_time_detected_s"]
        for i in range(1, 5):
            fig.add_vline(
                x=t_disrupt,
                line_dash="dash",
                line_color="red",
                row=i, col=1,
            )
        fig.add_annotation(
            x=t_disrupt, y=1.0, xref="x", yref="paper",
            text=f"Disruption détectée à t={t_disrupt:.3f}s",
            showarrow=True, arrowhead=2, ax=40, ay=-30,
            font=dict(color="red", size=12),
        )

    fig.update_layout(
        title=(
            f"Décharge {shot_id} — {meta.get('machine_config', '?')} "
            f"(qualité: {meta.get('quality_flag', '?')}, "
            f"complétude: {meta.get('overall_completeness', 0) * 100:.1f} %)"
        ),
        height=800,
        hovermode="x unified",
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Temps (s)", row=4, col=1)

    if output_path is None:
        output_path = processed_dir / "visualizations" / f"shot_{shot_id:04d}.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


# =============================================================================
# 2. Vue qualité globale du dataset
# =============================================================================

def plot_quality_overview(
    processed_dir: str | Path = "data/processed",
    output_path: str | Path | None = None,
) -> Path:
    """
    Vue agrégée de la qualité du dataset :
      - distribution des flags qualité
      - complétude moyenne par capteur
      - taux de disruption par configuration machine
      - distribution de la complétude globale
    """
    processed_dir = Path(processed_dir)
    df_meta = read_metadata_table(processed_dir)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Distribution des flags qualité",
            "Complétude moyenne par capteur (%)",
            "Taux de disruption par configuration machine",
            "Distribution de la complétude globale",
        ],
        specs=[
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "histogram"}],
        ],
    )

    # 2.1 - Pie chart des flags
    flags = df_meta["quality_flag"].value_counts()
    flag_colors = {"OK": "#2ca02c", "DEGRADED": "#ff7f0e", "REJECTED": "#d62728"}
    fig.add_trace(
        go.Pie(
            labels=flags.index, values=flags.values,
            marker=dict(colors=[flag_colors.get(f, "#cccccc") for f in flags.index]),
            hole=0.4,
        ),
        row=1, col=1,
    )

    # 2.2 - Complétude par capteur (barres)
    completeness_cols = [
        "completeness_plasma_current",
        "completeness_density",
        "completeness_temperature",
        "completeness_vloop",
    ]
    completeness_labels = ["Courant", "Densité", "Température", "Tension"]
    means = [df_meta[c].mean() * 100 for c in completeness_cols]
    fig.add_trace(
        go.Bar(
            x=completeness_labels, y=means,
            marker_color="#1f77b4",
            text=[f"{m:.1f}%" for m in means],
            textposition="outside",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # 2.3 - Taux de disruption par machine_config
    if "machine_config" in df_meta.columns:
        disrupt_by_config = (
            df_meta.groupby("machine_config")["declared_disruption"]
            .agg(["sum", "count"])
        )
        disrupt_by_config["rate"] = (
            disrupt_by_config["sum"] / disrupt_by_config["count"] * 100
        )
        fig.add_trace(
            go.Bar(
                x=disrupt_by_config.index.tolist(),
                y=disrupt_by_config["rate"].tolist(),
                marker_color="#d62728",
                text=[f"{r:.1f}%" for r in disrupt_by_config["rate"]],
                textposition="outside",
                showlegend=False,
            ),
            row=2, col=1,
        )

    # 2.4 - Histogramme de la complétude globale
    fig.add_trace(
        go.Histogram(
            x=df_meta["overall_completeness"] * 100,
            nbinsx=30,
            marker_color="#2ca02c",
            showlegend=False,
        ),
        row=2, col=2,
    )

    fig.update_yaxes(title_text="Complétude (%)", range=[0, 105], row=1, col=2)
    fig.update_yaxes(title_text="Taux (%)", row=2, col=1)
    fig.update_xaxes(title_text="Complétude (%)", row=2, col=2)
    fig.update_yaxes(title_text="Nb décharges", row=2, col=2)

    fig.update_layout(
        title=f"Vue qualité du dataset — {len(df_meta)} décharges",
        height=800,
        template="plotly_white",
    )

    if output_path is None:
        output_path = processed_dir / "visualizations" / "quality_overview.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


# =============================================================================
# 3. Superposition de plusieurs décharges (recherche de patterns)
# =============================================================================

def plot_shots_comparison(
    shot_ids: list[int],
    signal: str = "plasma_current_MA",
    processed_dir: str | Path = "data/processed",
    output_path: str | Path | None = None,
    title_suffix: str = "",
) -> Path:
    """
    Superpose le signal choisi pour plusieurs décharges.

    Utile par exemple pour comparer toutes les décharges avec disruption,
    ou pour repérer un pattern récurrent sur le courant plasma.
    """
    processed_dir = Path(processed_dir)
    df_signals = read_signals(processed_dir, shot_ids=shot_ids)
    df_meta = read_metadata_table(processed_dir)
    df_meta = df_meta[df_meta["shot_id"].isin(shot_ids)]

    fig = go.Figure()

    # On affecte une couleur par flag qualité pour lecture immédiate
    color_by_flag = {"OK": "#2ca02c", "DEGRADED": "#ff7f0e", "REJECTED": "#d62728"}

    for sid in shot_ids:
        df_shot = df_signals[df_signals["shot_id"] == sid].sort_values("time_s")
        meta_row = df_meta[df_meta["shot_id"] == sid]
        if meta_row.empty:
            continue
        flag = meta_row["quality_flag"].iloc[0]
        disruption = meta_row["declared_disruption"].iloc[0]

        fig.add_trace(
            go.Scatter(
                x=df_shot["time_s"],
                y=df_shot[signal],
                mode="lines",
                line=dict(color=color_by_flag.get(flag, "#999999"), width=1),
                opacity=0.6,
                name=f"Shot {sid} ({flag}{', disrupt' if disruption else ''})",
                hovertemplate=(
                    f"Shot {sid}<br>"
                    "t=%{x:.3f}s<br>"
                    f"{signal}=%{{y:.3f}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            f"Comparaison multi-décharges — {signal}"
            + (f" ({title_suffix})" if title_suffix else "")
        ),
        xaxis_title="Temps (s)",
        yaxis_title=signal,
        hovermode="closest",
        height=600,
        template="plotly_white",
    )

    if output_path is None:
        output_path = processed_dir / "visualizations" / "shots_comparison.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


# =============================================================================
# CLI : produit les trois visualisations standard
# =============================================================================

def main():
    """
    Génère les trois visualisations par défaut :
      - shot_detail pour la première décharge avec disruption
      - quality_overview du dataset complet
      - shots_comparison sur les 10 premières décharges avec disruption
    """
    import argparse

    parser = argparse.ArgumentParser(description="Visualisations Plotly du data mart")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    print("Chargement du data mart...")
    df_meta = read_metadata_table(args.processed_dir)
    print(f"  {len(df_meta)} décharges disponibles")

    # 1. Vue détaillée : on choisit une décharge avec disruption pour avoir
    # quelque chose d'intéressant à montrer, sinon la première décharge.
    disrupt_shots = df_meta[df_meta["declared_disruption"]]["shot_id"].tolist()
    detail_shot = disrupt_shots[0] if disrupt_shots else int(df_meta["shot_id"].iloc[0])
    print(f"\n1. Vue détaillée (décharge {detail_shot}) :")
    p1 = plot_shot_detail(detail_shot, args.processed_dir)
    print(f"   → {p1}")

    # 2. Vue qualité globale
    print("\n2. Vue qualité globale :")
    p2 = plot_quality_overview(args.processed_dir)
    print(f"   → {p2}")

    # 3. Comparaison multi-décharges : les 10 premières avec disruption
    comparison_shots = disrupt_shots[:10] if len(disrupt_shots) >= 10 else disrupt_shots
    if comparison_shots:
        print(f"\n3. Comparaison multi-décharges ({len(comparison_shots)} décharges avec disruption) :")
        p3 = plot_shots_comparison(
            shot_ids=comparison_shots,
            signal="plasma_current_MA",
            processed_dir=args.processed_dir,
            title_suffix="décharges avec disruption",
        )
        print(f"   → {p3}")

    print("\nVisualisations produites. Ouvrir les fichiers .html dans un navigateur.")


if __name__ == "__main__":
    main()
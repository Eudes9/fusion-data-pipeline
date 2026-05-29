"""
Module de sérialisation en data mart Parquet.

Architecture produite :
    data/processed/
    ├── metadata.parquet         (table de dimensions, 1 ligne / décharge)
    ├── signals/                 (table de faits, partitionnée par shot_id)
    │   ├── shot_id=1/part-0.parquet
    │   ├── shot_id=2/part-0.parquet
    │   ...
    └── dataset_manifest.json    (data card : métadonnées globales du dataset)

Le format Parquet est colonnaire, compressé et typé. Il est le standard des
workflows IA (lisible nativement par Pandas, Polars, PyArrow, Spark, PyTorch
via webdataset, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.metadata.enricher import PIPELINE_VERSION


# =============================================================================
# Configuration de la sérialisation
# =============================================================================

@dataclass
class SerializationConfig:
    """Paramètres de sérialisation du data mart."""

    output_dir: Path = Path("data/processed")
    compression: str = "snappy"   # "snappy" (rapide) ou "zstd" (plus compressé)
    overwrite: bool = True        # écrase un data mart existant


# =============================================================================
# Écriture de la table de dimensions (métadonnées)
# =============================================================================

def write_metadata_table(
    df_meta: pd.DataFrame, config: SerializationConfig
) -> Path:
    """
    Écrit la table de métadonnées en Parquet unique.
    Retourne le chemin du fichier produit.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "metadata.parquet"

    if path.exists() and config.overwrite:
        path.unlink()

    df_meta.to_parquet(
        path,
        engine="pyarrow",
        compression=config.compression,
        index=False,
    )
    return path


# =============================================================================
# Écriture de la table de faits (signaux), partitionnée par shot_id
# =============================================================================

def write_signals_partitioned(
    cleaned_shots: Iterable[dict], config: SerializationConfig
) -> Path:
    """
    Écrit les signaux nettoyés sous forme partitionnée par shot_id.

    Chaque décharge est stockée dans son propre dossier `shot_id=NNN/`,
    ce qui permet à un lecteur Parquet de ne charger que les décharges
    voulues sans lire les autres (predicate pushdown).

    cleaned_shots : itérable de dicts {shot_id, df}
    """
    signals_dir = config.output_dir / "signals"
    if signals_dir.exists() and config.overwrite:
        # Nettoyage propre des anciennes partitions
        import shutil
        shutil.rmtree(signals_dir)
    signals_dir.mkdir(parents=True, exist_ok=True)

    for shot in cleaned_shots:
        shot_id = shot["shot_id"]
        df = shot["df"].copy()
        # Ajout de la colonne de partition (convention Hive)
        df["shot_id"] = shot_id

        partition_dir = signals_dir / f"shot_id={shot_id}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        # On retire la colonne shot_id du fichier lui-même : elle est
        # déjà portée par le nom du dossier (style Hive partitioning)
        df_to_write = df.drop(columns=["shot_id"])

        df_to_write.to_parquet(
            partition_dir / "part-0.parquet",
            engine="pyarrow",
            compression=config.compression,
            index=False,
        )

    return signals_dir


# =============================================================================
# Manifest du dataset (data card)
# =============================================================================

def write_dataset_manifest(
    df_meta: pd.DataFrame, config: SerializationConfig
) -> Path:
    """
    Écrit un manifest JSON qui documente le dataset dans son ensemble.

    Ce fichier est l'équivalent d'un "data card" : il décrit le dataset
    pour qu'un utilisateur sache ce qu'il manipule sans avoir à le lire.
    """
    n_total = len(df_meta)
    flags = df_meta["quality_flag"].value_counts().to_dict() if "quality_flag" in df_meta.columns else {}

    manifest = {
        "name": "fusion-data-mart",
        "description": (
            "Data mart de diagnostics multi-capteurs de décharges plasma "
            "(données synthétiques inspirées de la fusion par confinement magnétique). "
            "Préparé pour des workflows IA de prédiction de disruption."
        ),
        "pipeline_version": PIPELINE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),

        "tables": {
            "metadata": {
                "path": "metadata.parquet",
                "rows": n_total,
                "columns": list(df_meta.columns) if n_total > 0 else [],
                "description": "Table de dimensions : une ligne par décharge",
            },
            "signals": {
                "path": "signals/",
                "partitioning": "shot_id",
                "description": (
                    "Table de faits : séries temporelles canoniques nettoyées, "
                    "partitionnées par shot_id pour permettre le predicate pushdown."
                ),
                "canonical_columns": [
                    "time_s",
                    "plasma_current_MA",
                    "electron_density",
                    "electron_temperature",
                    "loop_voltage",
                ],
            },
        },

        "statistics": {
            "total_shots": n_total,
            "quality_flags": flags,
            "disruption_rate_declared": (
                round(float(df_meta["declared_disruption"].mean()), 3)
                if "declared_disruption" in df_meta.columns and n_total > 0
                else None
            ),
            "disruption_rate_detected": (
                round(float(df_meta["disruption_detected"].mean()), 3)
                if "disruption_detected" in df_meta.columns and n_total > 0
                else None
            ),
            "mean_completeness": (
                round(float(df_meta["overall_completeness"].mean()), 4)
                if "overall_completeness" in df_meta.columns and n_total > 0
                else None
            ),
        },

        "usage": {
            "load_metadata": "pd.read_parquet('data/processed/metadata.parquet')",
            "load_one_shot": "pd.read_parquet('data/processed/signals/shot_id=1/')",
            "load_filtered_signals": (
                "import pyarrow.dataset as ds; "
                "ds.dataset('data/processed/signals', partitioning='hive')"
                ".to_table(filter=ds.field('shot_id').isin([1, 5, 7]))"
            ),
        },
    }

    path = config.output_dir / "dataset_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


# =============================================================================
# Pipeline complet de sérialisation
# =============================================================================

def serialize_data_mart(
    df_meta: pd.DataFrame,
    cleaned_shots: Iterable[dict],
    config: SerializationConfig | None = None,
) -> dict:
    """
    Écrit l'ensemble du data mart sur disque.

    Retourne un dict avec les chemins des artefacts produits.
    """
    config = config or SerializationConfig()

    meta_path = write_metadata_table(df_meta, config)
    signals_path = write_signals_partitioned(cleaned_shots, config)
    manifest_path = write_dataset_manifest(df_meta, config)

    return {
        "metadata_table": str(meta_path),
        "signals_dir": str(signals_path),
        "manifest": str(manifest_path),
    }


# =============================================================================
# Helpers de lecture (pour valider que ce qu'on écrit est ce qu'on relit)
# =============================================================================

def read_metadata_table(processed_dir: str | Path = "data/processed") -> pd.DataFrame:
    """Charge la table de métadonnées du data mart."""
    return pd.read_parquet(Path(processed_dir) / "metadata.parquet")


def read_signals(
    processed_dir: str | Path = "data/processed",
    shot_ids: list[int] | None = None,
) -> pd.DataFrame:
    """
    Charge les signaux de tout ou partie des décharges.

    Si shot_ids est fourni, ne lit que les partitions correspondantes
    (predicate pushdown : très efficace, ne touche pas aux autres fichiers).
    """
    import pyarrow.dataset as ds

    dataset = ds.dataset(
        Path(processed_dir) / "signals",
        partitioning="hive",
        format="parquet",
    )

    if shot_ids is None:
        return dataset.to_table().to_pandas()

    return dataset.to_table(filter=ds.field("shot_id").isin(shot_ids)).to_pandas()


# =============================================================================
# CLI de démonstration
# =============================================================================

def main():
    """
    Pipeline complet : ingestion → nettoyage → enrichissement → sérialisation.
    Vérifie que le data mart est relisible et exploitable.
    """
    import argparse
    from src.ingestion.loader import IngestionConfig, load_all_shots
    from src.cleaning.cleaner import CleaningConfig, clean_shot
    from src.metadata.enricher import build_metadata_table, enrich_shot

    parser = argparse.ArgumentParser(description="Sérialise le data mart Parquet")
    parser.add_argument("--config", default="config/sources.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    print("1. Ingestion...")
    ing_config = IngestionConfig.from_yaml(args.config)
    shots = load_all_shots(args.raw_dir, ing_config)
    print(f"   {len(shots)} décharges ingérées")

    print("2. Nettoyage...")
    cleaning_config = CleaningConfig()
    cleaned_shots = []
    enriched_records = []
    for s in shots:
        df_clean, report = clean_shot(s["df"], s["shot_id"], cleaning_config)
        cleaned_shots.append({"shot_id": s["shot_id"], "df": df_clean})
        enriched_records.append(
            enrich_shot(s["metadata"], report, df_clean)
        )
    df_meta = build_metadata_table(enriched_records)
    print(f"   Table de métadonnées : {len(df_meta)} lignes × {len(df_meta.columns)} colonnes")

    print("3. Sérialisation Parquet...")
    config = SerializationConfig(output_dir=Path(args.output_dir))
    artefacts = serialize_data_mart(df_meta, cleaned_shots, config)
    for name, path in artefacts.items():
        print(f"   {name:20s} : {path}")

    # Vérification : on relit ce qu'on vient d'écrire
    print("\n4. Vérification (relecture)...")
    df_meta_back = read_metadata_table(args.output_dir)
    print(f"   metadata.parquet         : {len(df_meta_back)} lignes")

    sample_ids = df_meta_back["shot_id"].head(3).tolist()
    df_signals_sample = read_signals(args.output_dir, shot_ids=sample_ids)
    print(f"   signals (3 décharges)   : {len(df_signals_sample)} points chargés")
    print(f"   colonnes signals        : {list(df_signals_sample.columns)}")

    # Taille du data mart
    total_size_mb = sum(
        f.stat().st_size for f in Path(args.output_dir).rglob("*.parquet")
    ) / (1024 ** 2)
    print(f"\nTaille totale du data mart : {total_size_mb:.2f} Mo")


if __name__ == "__main__":
    main()
"""
Module d'ingestion configurable.

Lit n'importe quelle décharge en s'appuyant sur le fichier de configuration
config/sources.yaml. Le code ne contient AUCUNE logique spécifique à un format
ou à un nom de colonne : tout est piloté par la configuration.

Ajouter une nouvelle source = ajouter une entrée dans le YAML, sans toucher au code.

Usage :
    from src.ingestion.loader import IngestionConfig, load_shot

    config = IngestionConfig.from_yaml("config/sources.yaml")
    df, metadata = load_shot("data/raw/shot_0001", config)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IngestionConfig:
    """Configuration d'ingestion chargée depuis le YAML."""

    canonical_columns: dict          # noms canoniques (time, current, ...)
    schemas: dict                    # définition de chaque schéma
    resolution_pattern: str          # template pour résoudre le nom de schéma

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IngestionConfig":
        """Charge la configuration depuis un fichier YAML."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            canonical_columns=data["canonical_columns"],
            schemas=data["schemas"],
            resolution_pattern=data["resolution"]["pattern"],
        )

    def resolve_schema(self, metadata: dict) -> str:
        """Détermine le nom du schéma applicable à partir des métadonnées."""
        try:
            return self.resolution_pattern.format(**metadata)
        except KeyError as e:
            raise ValueError(
                f"Métadonnées insuffisantes pour résoudre le schéma : {e}"
            ) from e


# =============================================================================
# Chargement bas niveau
# =============================================================================

def _read_signal_file(shot_dir: Path, file_format: str) -> pd.DataFrame:
    """Lit le fichier de signaux brut selon son format."""
    if file_format == "csv":
        path = shot_dir / "signals.csv"
        return pd.read_csv(path)
    elif file_format == "json":
        path = shot_dir / "signals.json"
        return pd.read_json(path)
    else:
        raise ValueError(f"Format de fichier non supporté : {file_format}")


def _read_metadata(shot_dir: Path) -> dict:
    """Lit le fichier metadata.yaml d'une décharge."""
    with open(shot_dir / "metadata.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# Application du schéma : renommage + conversions d'unités
# =============================================================================

def _apply_schema(
    df_raw: pd.DataFrame, schema: dict, canonical_columns: dict
) -> pd.DataFrame:
    """
    Applique un schéma d'ingestion à un DataFrame brut :
      1. Vérifie que les colonnes attendues sont présentes
      2. Renomme les colonnes vers les noms internes (time, current, ...)
      3. Applique les conversions d'unités
      4. Renomme vers les noms canoniques de sortie
    """
    # 1. Vérification de la présence des colonnes
    expected = schema["expected_columns"]
    missing = [c for c in expected if c not in df_raw.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans la source : {missing}. "
            f"Trouvées : {list(df_raw.columns)}"
        )

    # 2. Renommage brut -> noms internes (time, current, density, ...)
    df = df_raw.rename(columns=schema["column_mapping"]).copy()

    # 3. Conversions d'unités sur les noms internes
    for internal_name, factor in schema.get("unit_conversions", {}).items():
        if internal_name in df.columns:
            df[internal_name] = df[internal_name] * factor

    # 4. Renommage noms internes -> noms canoniques de sortie
    df = df.rename(columns=canonical_columns)

    # 5. Tri par temps + reset_index pour propreté
    time_col = canonical_columns["time"]
    df = df.sort_values(time_col).reset_index(drop=True)

    return df


# =============================================================================
# API publique
# =============================================================================

def load_shot(
    shot_dir: str | Path, config: IngestionConfig
) -> tuple[pd.DataFrame, dict]:
    """
    Charge une décharge et retourne (DataFrame canonique, métadonnées).

    Le DataFrame en sortie est garanti dans le format canonique :
        time_s, plasma_current_MA, electron_density,
        electron_temperature, loop_voltage
    quels que soient le format d'entrée, les noms de colonnes ou les unités.
    """
    shot_dir = Path(shot_dir)
    metadata = _read_metadata(shot_dir)
    schema_name = config.resolve_schema(metadata)

    if schema_name not in config.schemas:
        raise ValueError(
            f"Schéma '{schema_name}' inconnu. Schémas disponibles : "
            f"{list(config.schemas.keys())}"
        )

    schema = config.schemas[schema_name]
    df_raw = _read_signal_file(shot_dir, metadata["file_format"])
    df_canonical = _apply_schema(df_raw, schema, config.canonical_columns)

    # On enrichit les métadonnées avec le schéma utilisé (traçabilité)
    metadata["_schema_applied"] = schema_name

    return df_canonical, metadata


def load_all_shots(
    raw_dir: str | Path, config: IngestionConfig
) -> list[dict]:
    """
    Ingère toutes les décharges d'un dossier et retourne une liste de
    dictionnaires {shot_id, df, metadata}.

    Robuste aux erreurs : les décharges qui échouent sont signalées mais
    n'interrompent pas le traitement (typique du data engineering en
    production).
    """
    raw_dir = Path(raw_dir)
    shot_dirs = sorted(raw_dir.glob("shot_*"))

    results = []
    errors = []
    for shot_dir in shot_dirs:
        try:
            df, metadata = load_shot(shot_dir, config)
            results.append({
                "shot_id": metadata["shot_id"],
                "df": df,
                "metadata": metadata,
            })
        except Exception as e:
            errors.append({"shot_dir": str(shot_dir), "error": str(e)})

    if errors:
        print(f"  /!\\  {len(errors)} décharge(s) en erreur :")
        for err in errors[:5]:
            print(f"    {err['shot_dir']} : {err['error']}")

    return results


# =============================================================================
# CLI de démonstration
# =============================================================================

def main():
    """Démonstration : charge toutes les décharges et affiche un résumé."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingestion configurable des décharges")
    parser.add_argument("--config", default="config/sources.yaml", help="Fichier de config")
    parser.add_argument("--raw-dir", default="data/raw", help="Dossier des décharges brutes")
    args = parser.parse_args()

    print(f"Chargement de la configuration : {args.config}")
    config = IngestionConfig.from_yaml(args.config)
    print(f"  {len(config.schemas)} schémas définis : {list(config.schemas.keys())}\n")

    print(f"Ingestion des décharges depuis {args.raw_dir}/")
    shots = load_all_shots(args.raw_dir, config)
    print(f"  {len(shots)} décharges ingérées avec succès")

    if shots:
        # Statistiques sur les schémas effectivement appliqués
        schemas_used = {}
        for s in shots:
            sch = s["metadata"]["_schema_applied"]
            schemas_used[sch] = schemas_used.get(sch, 0) + 1
        print(f"\nRépartition des schémas appliqués :")
        for sch, n in sorted(schemas_used.items()):
            print(f"  {sch:20s} : {n:4d} décharges")

        # Vérification : toutes les décharges ont le même format canonique
        first_cols = list(shots[0]["df"].columns)
        all_same = all(list(s["df"].columns) == first_cols for s in shots)
        print(f"\nFormat canonique homogène : {all_same}")
        print(f"Colonnes canoniques        : {first_cols}")

        # Exemple
        example = shots[0]
        print(f"\nExemple - shot {example['shot_id']} ({example['metadata']['_schema_applied']}) :")
        print(example["df"].head(3))
        print(f"  ({len(example['df'])} lignes)")


if __name__ == "__main__":
    main()

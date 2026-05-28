# Fusion Data Pipeline

Pipeline générique d'ingestion et de préparation de données de diagnostics multi-capteurs, inspiré des problématiques de data engineering en recherche sur la fusion par confinement magnétique.

## Contexte

Les expériences de fusion produisent des données de diagnostics hétérogènes : séries temporelles à fréquences différentes, formats variés (CSV, JSON), unités non normalisées, valeurs manquantes, métadonnées dispersées. Pour rendre ces données exploitables par des workflows d'intelligence artificielle, il faut un outil capable d'ingérer ces sources de manière configurable, de les nettoyer de façon générique et de les sérialiser en data marts.

Ce projet implémente un pipeline complet répondant à ce besoin, sur des données synthétiques inspirées de diagnostics tokamak (courant plasma, densité électronique, température électronique, tension boucle).

## Architecture

```
fusion-data-pipeline/
├── config/                # Fichiers de configuration YAML
├── data/
│   ├── raw/               # Données brutes simulées (CSV, JSON)
│   └── processed/         # Data marts Parquet
├── src/
│   ├── ingestion/         # Ingestion configurable par YAML
│   ├── cleaning/          # Nettoyage générique
│   ├── metadata/          # Enrichissement en métadonnées
│   ├── serialization/     # Sérialisation Parquet
│   ├── visualization/     # Visualisation Plotly
│   ├── demo/              # Démonstrateur IA (Random Forest)
│   └── llm_extraction/    # Extraction de métadonnées par LLM
├── tests/                 # Tests unitaires
├── notebooks/             # Notebooks de démonstration
└── .github/workflows/     # CI GitHub Actions
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Utilisation

```bash
# 1. Générer les données synthétiques
python -m src.ingestion.generate_data

# 2. Exécuter le pipeline complet
python -m src.demo.run_pipeline --config config/sources.yaml
```

## Fonctionnalités clés

- **Ingestion configurable** : ajout d'une nouvelle source via une ligne de YAML, sans modifier le code
- **Nettoyage générique** : resynchronisation temporelle, gestion des valeurs manquantes, détection d'aberrants, normalisation des unités
- **Indicateurs de qualité** : score de complétude par décharge et par capteur
- **Sérialisation Parquet** : data mart prêt pour PyTorch / scikit-learn
- **Démonstrateur** : Random Forest de prédiction de disruption
- **Tests unitaires + CI** : GitHub Actions

## Tests

```bash
pytest tests/
```

## Licence

MIT

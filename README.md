# Fusion Data Pipeline

Pipeline générique d'ingestion et de préparation de données de diagnostics multi-capteurs, inspiré des problématiques de data engineering en recherche sur la fusion par confinement magnétique.

## Contexte

Les expériences de fusion produisent des données de diagnostics hétérogènes : séries temporelles à fréquences différentes, formats variés (CSV, JSON), unités non normalisées, valeurs manquantes, métadonnées dispersées. Pour rendre ces données exploitables par des workflows d'intelligence artificielle, il faut un outil capable d'ingérer ces sources de manière configurable, de les nettoyer de façon générique et de les sérialiser en data marts.

Ce projet implémente un pipeline complet répondant à ce besoin, sur des données synthétiques inspirées de diagnostics tokamak (courant plasma, densité électronique, température électronique, tension boucle).

# Fusion Data Pipeline

[![CI](https://github.com/Eudes9/fusion-data-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Eudes9/fusion-data-pipeline/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-31%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

> **Pipeline générique d'ingestion, de préparation et de valorisation de données expérimentales multi-capteurs**, inspiré des problématiques de data engineering en recherche sur la fusion par confinement magnétique.

---

## Pourquoi ce projet ?

Les expériences de fusion produisent des données de diagnostics fortement **hétérogènes** : formats variés (CSV, JSON), unités non normalisées, noms de colonnes incohérents entre équipes, valeurs manquantes, fréquences d'échantillonnage différentes, métadonnées dispersées.

Pour rendre ces données exploitables par des **workflows IA** (par exemple la prédiction de disruption), il faut un outil capable de :

- ingérer ces sources de manière **configurable** (sans modifier le code à chaque nouveau format),
- les nettoyer de façon **générique** (resynchronisation, aberrants, manquants),
- les enrichir en **métadonnées tracées** (qualité, lignage, version du pipeline),
- les sérialiser dans un **data mart** standard, prêt pour le ML.

Ce projet implémente un tel pipeline de bout en bout, sur des données synthétiques inspirées de diagnostics tokamak (courant plasma, densité électronique, température, tension boucle).

---

## Aperçu de l'architecture

```
Données brutes hétérogènes        (CSV/JSON, MA/A, fréquences variables)
        │
        ▼
[1] Ingestion configurable        config/sources.yaml → format canonique
        │
        ▼
[2] Nettoyage générique           resampling + MAD glissant + imputation
        │
        ▼
[3] Enrichissement métadonnées    qualité, features dérivées, traçabilité
        │
        ▼
[4] Data mart Parquet             metadata.parquet + signals/ partitionné
        │
        ├──▶ [5] Visualisations Plotly  (3 HTML interactifs)
        │
        └──▶ [6] Démonstrateur ML       Random Forest + CV stratifiée
```

---

## Exemple d'utilisation

Le data mart produit est conçu pour être consommé en quelques lignes par n'importe quel framework ML :

```python
import pandas as pd
from src.serialization.writer import read_metadata_table, read_signals

# Charger la table de dimensions (1 ligne par décharge)
meta = read_metadata_table("data/processed")

# Filtrer les décharges utilisables (flag qualité OK) avec disruption
shots = meta.query("quality_flag == 'OK' and declared_disruption").shot_id.tolist()

# Charger seulement ces décharges (predicate pushdown : pas de lecture inutile)
signals = read_signals("data/processed", shot_ids=shots)
```

Pour lancer le pipeline complet :

```bash
python -m src.ingestion.generate_data        # génère 200 décharges synthétiques
python -m src.serialization.writer            # ingestion → nettoyage → data mart
python -m src.visualization.plots             # 3 HTML interactifs
python -m src.demo.predict_disruption         # démonstrateur Random Forest
```

---

## Couverture des missions du poste

Ce projet a été conçu en miroir des missions d'une alternance en data engineering scientifique au CEA. Chaque module correspond directement à une mission :

| Mission de l'annonce | Module du projet |
|---|---|
| Ingestion automatisée par fichiers de configuration | `src/ingestion/` + `config/sources.yaml` |
| Méthodes génériques de nettoyage et préparation | `src/cleaning/cleaner.py` |
| Outils de visualisation adaptés aux utilisateurs | `src/visualization/plots.py` |
| Noyau de sérialisation vers formats compatibles IA | `src/serialization/writer.py` (Parquet) |
| Annotation et enrichissement en métadonnées | `src/metadata/enricher.py` |
| Tests + intégration continue | `tests/` + `.github/workflows/ci.yml` |

---

## Choix techniques

**Pourquoi une ingestion pilotée par YAML.** Le code ne contient aucune logique métier sur les sources : tous les mappings de colonnes et conversions d'unités sont déclarés dans `config/sources.yaml`. Ajouter une nouvelle source ne demande aucune modification de code, seulement quelques lignes de YAML. C'est une architecture *configuration-driven*.

**Pourquoi MAD glissant et pas MAD global.** La méthode MAD classique suppose que les données sont distribuées autour d'une médiane unique. Or les signaux de tokamak ont une structure temporelle (rampe → plateau → descente) : un MAD global classerait comme aberrants tous les points hors du plateau. Le MAD glissant calcule la médiane et l'échelle sur une fenêtre locale, ce qui est robuste pour des séries temporelles non stationnaires.

**Pourquoi Parquet partitionné par `shot_id`.** Parquet est colonnaire, compressé, typé — le format standard des workflows IA modernes. Le partitionnement par identifiant de décharge permet du *predicate pushdown* : un utilisateur qui veut analyser 10 décharges sur 10 000 ne lit que les 10 fichiers concernés. Même logique qu'en Spark, sans Spark.

**Pourquoi une cross-validation stratifiée pour le démonstrateur.** Avec une classe minoritaire (~16% de disruptions), un simple train/test split donne des scores variables d'un run à l'autre. La CV stratifiée k=5 donne une moyenne ET un écart-type, ce qui mesure aussi la stabilité du modèle.

---

## Structure du dépôt

```
fusion-data-pipeline/
├── config/
│   └── sources.yaml              Schémas d'ingestion (mappings, conversions)
├── data/
│   ├── raw/                      Données brutes (générées par script)
│   └── processed/                Data mart Parquet (généré)
├── src/
│   ├── ingestion/
│   │   ├── generate_data.py      Générateur synthétique
│   │   └── loader.py             Ingestion configurable
│   ├── cleaning/
│   │   └── cleaner.py            Nettoyage générique
│   ├── metadata/
│   │   └── enricher.py           Enrichissement + traçabilité
│   ├── serialization/
│   │   └── writer.py             Sérialisation Parquet + manifest
│   ├── visualization/
│   │   └── plots.py              3 visualisations Plotly
│   └── demo/
│       └── predict_disruption.py Random Forest + cross-validation
├── tests/                        31 tests unitaires
└── .github/workflows/ci.yml      CI Python 3.10 et 3.11
```

---

## Installation

```bash
git clone https://github.com/Eudes9/fusion-data-pipeline.git
cd fusion-data-pipeline
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

## Tests

```bash
pytest tests/ -v
```

31 tests unitaires couvrent l'ingestion, le nettoyage, l'enrichissement, la sérialisation, la visualisation et le démonstrateur. La CI les exécute automatiquement sur Python 3.10 et 3.11 à chaque push.

---

## Limites assumées

- **Données synthétiques** : aucun dataset public de diagnostics tokamak n'étant disponible sur Kaggle, j'ai simulé des signaux physiquement plausibles avec hétérogénéité contrôlée. Le pipeline est conçu pour transposer à des données réelles sans modification.
- **AUC ≈ 1.0** du démonstrateur : c'est un effet attendu des données synthétiques très séparables, pas une performance modèle. Sur des données réelles bruitées, on serait plutôt autour de 0.85-0.92.

---

## Licence

MIT

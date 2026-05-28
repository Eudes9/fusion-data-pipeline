"""
Tests smoke : vérifient que la structure du projet est en place
et que les modules sont importables. Sera complété au fur et à mesure
de l'implémentation des fonctionnalités.
"""


def test_project_structure_imports():
    """Tous les sous-modules src/ doivent être importables."""
    import src
    import src.ingestion
    import src.cleaning
    import src.metadata
    import src.serialization
    import src.visualization
    import src.demo
    import src.llm_extraction

    assert src is not None


def test_python_version():
    """Le projet cible Python 3.10+."""
    import sys

    assert sys.version_info >= (3, 10), "Python 3.10+ requis"

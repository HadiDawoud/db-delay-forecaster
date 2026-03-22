# db-delay-forecaster

## English

Forecast train delays using [piebro/deutsche-bahn-data](https://huggingface.co/datasets/piebro/deutsche-bahn-data) (CC BY 4.0). Python 3.10+, dependencies in `requirements.txt`.

Designed to run **locally** from the **repository root** (venv recommended). Code resolves paths from the project directory. To use **Jupyter**, reuse the imports and logic from `eda.py` / `main.py`; set the notebook working directory to the repo root or update paths / `sys.path` as needed.

```bash
pip install -r requirements.txt
python data/download.py
python eda.py
python main.py
```

Stdout: metrics. Figures: `outputs/plots/`.

---

## Deutsch

Verspätungsprognose mit Daten von [piebro/deutsche-bahn-data](https://huggingface.co/datasets/piebro/deutsche-bahn-data) (CC BY 4.0). Abhängigkeiten: `requirements.txt`.

**Lokal:** Befehle im **Projektroot** ausführen (venv). Pfade beziehen sich auf das Repo. **Jupyter:** gleiche Schritte wie in den Skripten; Arbeitsverzeichnis auf den Projektroot setzen oder Pfade/`sys.path` anpassen.

```bash
pip install -r requirements.txt
python data/download.py
python eda.py
python main.py
```

Ausgabe: Metriken im Terminal, Grafiken unter `outputs/plots/`.

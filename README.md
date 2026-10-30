# DC ThermalGraph Risk OS

Python implementation of the dissertation prototype for continuous data-center temperature-field forecasting, uncertainty-aware thermal-risk inference, spatial localization, and inspection recommendations.

## Application modules

- **Thermal Field Situation** — 3D mapping of 12 racks and 72 monitoring nodes, archived replay, frozen 1–60 min temperature trajectories, and candidate RC graph edges.
- **Risk Inference** — empirical residual risk-confidence mapping, four independent observation references, and node-level spatial review.
- **Recommended Actions** — evidence-linked, human-in-the-loop inspection recommendations and CSV export.

## Research boundary

The included replay package is a limited TDC2.0 historical demonstration window. Risk values are empirical research indicators derived from frozen forecasts and archived forecast residuals. They are not ground-truth fault labels and must not directly trigger equipment control.

## Run locally

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:8050`.

## Deploy from GitHub

The repository includes `render.yaml`. In Render, create a Blueprint from this repository. Render installs the Python dependencies and starts the Dash application with Gunicorn. Every push to `main` can be deployed automatically.

## Source policy

All authored application logic is Python. Dash and Plotly generate the browser runtime required for interactive web rendering.

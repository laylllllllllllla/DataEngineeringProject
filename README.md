# NYC Housing & Crashes

Data engineering project that combines NYC housing data with motor-vehicle crash data
at the community-district (CD) level, and shows the results in a Dash website.

## Project layout

```
├── notebook/
│   └── DataEngineering.ipynb   # full pipeline: ingest → clean → integrate → analyse
├── website/
│   ├── app.py                  # Dash app
│   ├── charts.py               # build_q1() … build_q12()
│   └── data/                   # small files exported by the notebook
│       ├── integrated_cd_year.csv
│       ├── housing_monthly.parquet
│       ├── housing_nta.parquet
│       ├── cd_centroids.csv
│       └── crashes.parquet
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

1. Run `notebook/DataEngineering.ipynb` to produce the files in `website/data/`.
2. Start the website:

   ```bash
   cd website
   python app.py
   ```

   Then open http://127.0.0.1:8050.

# Spotify Hit Prediction - End-to-End MLOps Pipeline

An MLOps project that predicts whether a Spotify track will be a **hit** (popularity >= 50) from its audio
features and genre. It is the Spotify version of the *Telco Churn* MLOps template: the same pipeline
(validation -> preprocessing -> MLflow training/registry -> evaluation -> recall quality gate -> lifecycle
promotion -> deployment -> drift monitoring -> retraining) applied to a different dataset.

## Problem framing (what changed from the churn template)

Your `dataset.csv` is a track catalogue, not a customer table, so there is no churn label. The project therefore
defines an equivalent binary problem:

| Churn template | This project | Notes |
|---|---|---|
| `Churn` = Yes/No | `Hit` = Yes/No | `Yes` when `popularity >= 50` (`data.hit_threshold` in `config.yaml`) |
| `customerID` | `track_id` | unique per row after de-duplication |
| 26% churners | 23.7% hits | similar class imbalance, so `class_weight=balanced` and the recall gate still make sense |
| Telco service features | audio features + `track_genre` + `explicit` | 15 model inputs, 139 columns after encoding |
| `Telco_Churn_Production_Model` | `Spotify_Hit_Production_Model` | MLflow registry name |

Decisions you may want to revisit (all in `src/prepare_spotify_data.py` / `config.yaml`):

1. **`popularity` is never a feature.** The target is derived from it; using it would be leakage. It stays in the raw CSV for auditing and is dropped in `clean_dataframe` (`config.yaml -> data.leakage_columns`).
2. **One row per track.** The original file lists a track once per genre (114,000 rows, 89,741 unique `track_id`s). Keeping duplicates would put copies of the same song in both train and test. We keep the first genre listed.
3. **Artist / album / track names are not features** (too high-cardinality for one-hot encoding); they are kept for display in inference output.
4. `key` and `time_signature` are integer *codes*, so they are one-hot encoded rather than scaled.

## Results on the held-out test set (16,154 tracks)

RandomForest with the settings in `config.yaml`:

| Metric | Value |
|---|---|
| Recall (hits caught) | **0.774** |
| Precision | 0.420 |
| ROC-AUC | 0.806 |
| F1 | 0.544 |
| Accuracy | 0.693 |

Read these honestly: audio features and genre carry real but limited signal. The model catches ~77% of hits but
~58% of the tracks it flags are not hits, and accuracy is below the 76% you would get by always answering "No"
(the model is deliberately tuned for recall). The quality gate is `recall >= 0.70`, which it passes.
The notebook also shows that HistGradientBoosting (ROC-AUC 0.85) and Logistic Regression (0.83) outperform the
RandomForest; the pipeline keeps RandomForest to stay faithful to the template - swapping it is the obvious next step.

## Repository layout

```
config.yaml                     all paths, target/id columns, model params, quality gate
data/raw/dataset.csv            your original upload (114,000 rows)
data/raw/spotify_tracks.csv     generated: cleaned training dataset (80,766 tracks)
data/staging/new_tracks.csv     generated: 8,974 unseen tracks = the "incoming batch"
src/prepare_spotify_data.py     dataset.csv -> spotify_tracks.csv + new_tracks.csv
src/mlops_pipeline/             01..09 pipeline stages, 10 failure injection, 11 deployment prep, 12_* drift/retraining
  validation.py                   strict schema (pandera if installed, pandas fallback)
  spotify_drift.py                simulated feature + concept drift used by the 12_create_* scripts
  spotify_hit_pipeline.py         orchestrator (runs stages 01-09 in order)
src/train_local.py              trains the same model WITHOUT MLflow (handy for a quick check)
src/*.py                        early-lab scripts (preprocess, train, evaluate, validate, corruption simulation)
api/                            FastAPI service (TrackInput schema, /predict, /predict/batch, /health, /metadata)
tests/                          API, performance and local-vs-container tests
notebooks/spotify_hit_prediction.ipynb   EDA + model comparison + threshold analysis
k8s/, Dockerfile, .github/      container, Kubernetes and CI definitions
docs/, lab11_*, GIT_WORKFLOW.md lab reports carried over from the template (see note below)
```

## Setup

Python 3.11 is recommended (the Dockerfile uses `python:3.11-slim`).

```bash
python -m venv venv
source venv/bin/activate            # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt     # full environment incl. mlflow, pandera, fastapi, xgboost
export PYTHONPATH=.                 # Windows PowerShell: $env:PYTHONPATH="."
```

Train and serve with the same scikit-learn version (the pins in `requirements*.txt`): pickled preprocessors and
models are not guaranteed to load across versions.

## Running

```bash
# 0. build the datasets from your dataset.csv (already done once; re-run to regenerate)
python src/prepare_spotify_data.py

# 1. first full run. --force is needed because no new batch was ingested yet
python src/mlops_pipeline/spotify_hit_pipeline.py --force
```

Stages: load -> validate -> preprocess -> validate outputs -> **train + register in MLflow** -> evaluate ->
quality gate -> lifecycle (promote to Production only if better) -> deploy-if-promoted -> inference.

Stage 05 / 08 / 09 need MLflow. For a quick MLflow-free check of the model:

```bash
python src/mlops_pipeline/03_preprocessing.py && python src/train_local.py \
  && python src/mlops_pipeline/06_model_evaluation.py && python src/mlops_pipeline/07_quality_gate.py
```

### Ingestion, drift and retraining demo

```bash
cp data/staging/new_tracks.csv data/raw/new_tracks.csv      # drop a new batch next to the dataset
python src/mlops_pipeline/spotify_hit_pipeline.py            # ingests (archives previous version, updates lineage), retrains
python src/mlops_pipeline/12_create_evolved_dataset.py       # simulated market shift (feature drift)
python src/mlops_pipeline/12_create_concept_drift_dataset.py # simulated label shift (concept drift)
python src/mlops_pipeline/12_data_drift.py                   # KS / Jensen-Shannon report
python src/mlops_pipeline/12_monitor_trigger.py              # drift + new batch -> retrain decision
```

Ingestion rules (unchanged from the template): the batch must pass schema validation, must not contain `track_id`s already
present, and is tracked by SHA-256 so the same file is never ingested twice. Archived versions are `data/raw/archive/tracks_vNN.csv`.

The drift scenarios are **simulated** from the existing data (`spotify_drift.py`): the catalogue is a static snapshot,
so there is no real time dimension to drift along.

### API

```bash
uvicorn api.main:app --reload     # needs a deployed model: stage 11 / deploy_if_promoted writes deployment/
pytest tests/test_api.py
```

Example request body for `POST /predict`:

```json
{"track_id": "demo1", "track_genre": "pop", "explicit": "No", "duration_ms": 210000,
 "danceability": 0.72, "energy": 0.65, "key": 5, "loudness": -6.5, "mode": 1, "speechiness": 0.05,
 "acousticness": 0.12, "instrumentalness": 0.0, "liveness": 0.11, "valence": 0.60, "tempo": 118.0,
 "time_signature": 4}
```

`POST /predict/batch` takes `{"tracks": [ ... ]}`. Responses contain `prediction` (0/1) and `hit_probability`.
A genre not seen in training does not crash the service; it is encoded as all zeros.

## What was verified, and what was not

Run and passing in the environment used to build this project (Python 3.12, pandas 3.0, scikit-learn 1.9.1, no network):

- `prepare_spotify_data.py`, stages 01, 02, 03, 04, 06, 07, `train_local.py`
- schema validation catches corrupted data (bad dtype, out-of-range, missing/extra column, null, bad label)
- `12_create_evolved_dataset`, `12_create_concept_drift_dataset`, `12_create_updated_dataset`, `12_data_drift`
- early-lab chain `preprocess_pipeline.py -> train.py -> evaluate.py`, `simulate_data_corruption.py`
- training/serving parity: the API's `clean_prediction_data` + saved preprocessor reproduces the training features exactly (max difference 0.0)
- the notebook code runs end to end

**Edited for Spotify but not executed here** (the sandbox has no `mlflow`, `fastapi`, `pydantic`, `pytest`, Docker or Kubernetes):
stages 05, 08, 09, 10, 11, `deploy_if_promoted`, the remaining `12_*` scripts (monitor trigger, drift analysis,
concept/prediction drift, retraining, MLflow registry update), `api/`, `tests/`, Dockerfile, k8s manifests and the GitHub workflows.
Run `pytest` and the full pipeline once on your machine and expect to fix small things.

## Notes on carried-over documents

`docs/`, `lab11_*`, `GIT_WORKFLOW.md`, `lab11_commands/` and `reports/lab10_summary.md` are the lab write-ups from the churn
project with names updated to this project. Their narrative, screenshots-style results and numbers (for example
MLflow model version 17, placeholder Windows paths such as `C:\path\to\spotify-hit-prediction` and `your-runner-name` in the workflows, which you must set to your own) describe the *original* run:
replace them with the results of your own run before submitting.

## Known limitations

- `popularity` is a snapshot of one moment; hits change over time and the dataset has no timestamp, so it cannot be split chronologically.
- No artist, release-date or playlist information, which strongly influences popularity in practice.
- The same song can exist under different `track_id`s (single vs album vs remaster); these near-duplicates are not removed and may inflate test scores slightly.
- Precision is low at the default 0.5 threshold; choose the threshold from the precision/recall trade-off in the notebook for your use case.

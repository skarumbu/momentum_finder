# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

NBA Momentum Finder — a FastAPI REST API that analyzes NBA play-by-play data to detect momentum shifts during games using a pre-trained logistic regression model.

## Common Commands

**Run locally:**
```bash
pip install -r docker/requirements.txt
uvicorn server:app --host 0.0.0.0 --port 80
```

**Build Docker image:**
```bash
docker build -t nba-momentum-api:latest -f docker/dockerfile .
```

**Retrain the model manually (triggers the Azure Container App Job):**
```bash
az containerapp job start \
  --name momentum-finder-retrain-prod \
  --resource-group my-website-prod-rg
```
The job runs automatically every Monday at 6am UTC. It trains on all games from the current season, validates ROC-AUC ≥ 0.60, and uploads the models to Azure Blob Storage. The server downloads them on next cold start.

## Architecture

Three source files, no test suite:

- **server.py** — FastAPI app with three endpoints:
  - `POST /get-momentum` — accepts `{team1, team2, date}`, returns momentum shift events
  - `GET /get-current-games` — returns live NBA game scores
  - `GET /health` — health check
  - CORS is restricted to `https://www.quixotry.me` and `http://localhost:3000`

- **retriever.py** — fetches play-by-play data from the `nba_api` library and engineers features (Home_Lead, Lead_Change, Score_Change, Time_Since_Last_Score)

- **model_trainer.py** — trains the logistic regression model (scikit-learn, balanced class weights) and serializes it to `momentum_model.pkl`

- **momentum_model.pkl** — pre-trained model loaded at server startup; do not delete

## Deployment

Pushes to `main` trigger a GitHub Actions workflow (`.github/workflows/deploy.yml`) that builds the Docker image, pushes it to Azure Container Registry (ACR), and updates the Azure Container App (`momentum-finder-prod` in resource group `my-website-prod-rg`).

Infrastructure is defined in the `azure-infrastructure` repo (`modules/momentumfinder.bicep`). The Container App runs on port 80 with `minReplicas: 0` (expect cold starts).

**Required GitHub Actions secrets:**
- `AZURE_CREDENTIALS` — service principal JSON for `azure/login`
- `ACR_NAME` — Azure Container Registry name
- `ACR_LOGIN_SERVER` — ACR login server (e.g., `<registryname>.azurecr.io`)

**Legacy (AWS):** The `CFN/` directory and `task-definition.json` contain the old AWS ECS infrastructure (no longer active).

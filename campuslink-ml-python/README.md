# CampusLink Python ML API

This is a Python implementation of the existing CampusLink model/API. It preserves the four POST endpoints, the six placement inputs, the request/response field names, bearer-token authentication, and the `unverified_demo` warning. The Node/TypeScript API remains unchanged.

## Python stack

- FastAPI and Pydantic for the HTTP API and JSON validation.
- scikit-learn LogisticRegression for placement probability, with the same six features scaled from 0–100 to 0–1.
- scikit-learn TF-IDF and cosine similarity for lexical job matching.
- joblib for the saved model artifact.

Resume extraction and interview feedback remain rule/rubric based and are clearly marked `trained: false`; no new AI capability is claimed for those routes.

## Install and train

Use PowerShell from this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_placement.py --input "C:\Users\rksah\Documents\Codex\2026-10-02\c\outputs\campuslink-ml\models-demo\placement-training-demo.jsonl" --output models-demo --provenance unverified_demo
```

The demo trainer accepts the normalized JSONL generated earlier. It also reads the original trainer-shaped JSONL fields (`student_id`, `placement_status`) and maps them to the unchanged canonical training schema. It discards rows with invalid 0–100 values and prints the exclusion count. Historical training requires `outcomeSource: "observed"` and observed binary labels. Never label generated/demo data as historical.

The trainer holds out the latest cohort, removes any earlier rows for students in the test cohort, and compares held-out metrics to the same constant-prevalence baseline. It writes `placement.model.joblib`, `placement.metadata.json`, and `placement.preprocessing.json`.

## Start the local API

```powershell
$env:CAMPUSLINK_MODEL_DIR = "models-demo"
$env:CAMPUSLINK_ALLOW_UNVERIFIED_DEMO = "true"
$env:CAMPUSLINK_ML_DEV_TOKEN = "local-demo-only"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8010
```

Open `http://127.0.0.1:8010/health` to check readiness. The API requires a bearer token for all `/v1/models/*` routes. Do not place this token in browser-side Next.js code.

## Existing API contracts (unchanged)

All inference routes use the `POST` method and are mounted below `/v1/models`:

1. `/placement/predict` takes `evidence` containing `verifiedSkills`, `academics`, `projects`, `aptitude`, `communication`, and `interview`. Response includes `outcomeProbability`, `modelVersion`, `provenance`, held-out `metrics`, `limitations`, and `latencyMs`.
2. `/resume/extract` takes `resumeText` and optional `language: "en"`; returns skills, education, experience, projects, extraction method, confidence and review flag.
3. `/jobs/match` takes `studentProfile` and `job`; returns relevance score, matched skills, gaps, evidence, method, and `trained: false`.
4. `/interviews/score` takes `responses` and `rubric`; returns rubric scores, answer evidence, generated preparation advice, and `trained: false`.

Request fields and response concepts are the same as the TypeScript service. Invalid input returns HTTP 400, missing/invalid bearer token returns 401, a resume over 100,000 characters returns 413, and a missing/disallowed model returns 503.

## Demo limitations

The provided demo labels and scores have not been verified as real historical outcomes. The demo model is illustrative and must not make hiring decisions. Keep the generated API token server-side. The Node backend should call this Python API; the Next.js browser client should call the Node backend.

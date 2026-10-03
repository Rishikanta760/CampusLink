# CampusLink ML package

Standalone TypeScript package for the CampusLink Node/Express project. It provides four approved contracts. The placement endpoint uses the historical logistic-regression model after training. Resume extraction, job similarity, and interview text feedback reuse local, rule-based/TF-IDF methods from the supplied helper code and are explicitly labelled as untrained. Mount the router after your existing authentication middleware; it does not authenticate users itself.

## Important data requirement

The production trainer accepts only observed historical outcomes matching `schema/placement-training-row.schema.json`. The file `placement_ai_trainer_dataset.jsonl` can be run in an explicitly unverified demo mode: it maps its provided `Placed`/`Unplaced` values to labels, rejects records with scores outside 0..100, and tags the saved artifact `unverified_demo`. This mode is for demonstrating the code path only. It does not establish that the input labels or scores are real, and its predictions must not be used for real placement decisions. Do not use it as a historical model.

Use the exact format in `schema/placement-training-row.schema.json`. Each JSONL row needs a stable pseudonymous `studentId`, cohort, evidence and outcome timestamps, the six evidence scores, `outcomeSource:"observed"`, and the observed `placed` label. Use `studentId` only for group-safe splitting. The trainer holds out the latest cohort, excludes all earlier rows belonging to students in that held-out cohort, and reports how many rows it purged to prevent leakage. Define one placement outcome window and apply it consistently. A student without a complete outcome observation window is not a negative example. Define and document rubrics for all 0–100 scores before labelling. Do not manufacture labels or copy illustrative scores into training data.

## Install and train

Use a Node.js version supported by your CampusLink deployment. Exact dependency versions and the generated `package-lock.json` are included.

```powershell
npm ci
npm run typecheck
npm run train:placement -- path/to/placement-training.jsonl models
```

For the supplied trainer-shaped demo JSONL only, run:

```powershell
npm run train:placement:demo -- "C:\full\path\to\placement_ai_trainer_dataset.jsonl" models-demo
```

This saves a normalized, filtered demo JSONL and a model tagged `unverified_demo` in `models-demo`. Rows with missing/malformed fields, dates not ordered before the outcome, unknown outcome labels, or any score outside 0..100 are excluded and counted in the report. The API rejects this artifact by default. To demonstrate it locally, set `CAMPUSLINK_MODEL_DIR=models-demo` and `CAMPUSLINK_ALLOW_UNVERIFIED_DEMO=true`; the response will identify the unverified provenance. Never set this flag in a deployed production service.

Training writes `models/placement.model.json` and `models/placement.preprocessing.json`. The model records a data fingerprint, version, held-out cohort, train/test student counts, purged rows, and metrics. The baseline is the training-set placement prevalence used as a constant probability and a 0.5 classification threshold. The unseen latest cohort is evaluated for accuracy, precision, recall, F1, and Brier score. Training fails if the train or held-out set lacks either observed class.

The model version is `campuslink-logistic-v2-<data fingerprint>`. Preprocessing is fixed and saved: the six 0–100 evidence scores are divided by 100 in the listed feature order; missing and out-of-range values are rejected.

## Mount in Express

Install the package dependencies, then mount the router after the CampusLink authentication middleware:

```ts
import { createMlRouter } from './ml-router.js';
app.use(express.json({ limit: '256kb' }));
app.use('/v1/models', createMlRouter(requireAuthenticatedUser));
```

Set `CAMPUSLINK_MODEL_DIR` to the model directory. The router returns `401` from the supplied auth middleware, `400` for invalid input, `413` for oversized resume text, and `503` if no historical model is trained. Every successful inference response includes measured model-function `latencyMs`; it excludes model loading, JSON parsing/serialization, network, and authentication time. No route returns a hire/reject decision.

For local API exploration only, set `CAMPUSLINK_ML_DEV_TOKEN` to a random local token and run:

```powershell
$env:CAMPUSLINK_ML_DEV_TOKEN = 'local-development-token'
npm run dev
```

The local example uses a bearer token and is not a replacement for CampusLink's user authentication.

## Deploy the hackathon demo to Render

`render.yaml` configures a Node web service that builds with `npm ci --include=dev && npm run build`, starts with `npm start`, checks `/health`, loads `models-demo`, and generates a private bearer token. This deploy intentionally enables the `unverified_demo` model. The public response continues to identify that provenance and must not be used for actual placement decisions.

Put this package at the root of a GitHub repository, push the files, then in Render choose **New > Blueprint** and connect that repository. Render reads `render.yaml`, prompts to create the service, and provides the public `onrender.com` address when deployment succeeds. Render deploys services from a connected Git repository or other supported source; it does not deploy directly from this local folder. See [Render's deploy documentation](https://render.com/docs/deploys) and [Blueprint reference](https://render.com/docs/blueprint-spec).

The public API address will be `https://<your-service>.onrender.com/v1/models/placement/predict`; check readiness at `/health`. The generated `CAMPUSLINK_ML_DEV_TOKEN` is a secret. Read it from the Render dashboard and send requests from your Node.js backend using `Authorization: Bearer <token>`. Do not put this token in browser-side Next.js code. The endpoint is not a webpage or Swagger UI and requires a POST JSON request. `CAMPUSLINK_ALLOW_UNVERIFIED_DEMO=true` is only for this professor-facing demonstration; unset it and train on validated observed historical data before any real-world use.

Error bodies use a stable shape, for example `{"error":{"code":"MODEL_UNAVAILABLE","message":"Historical placement model is not trained."}}`. The router returns `400` for invalid JSON or fields, `401` for auth failures, `413` for an oversized body, and `503` if the historical artifact is missing or invalid.

## Contracts

All four endpoints are mounted below `/v1/models`.

### 1. Placement outcome probability

Request:

```json
{"evidence":{"verifiedSkills":70,"academics":82,"projects":65,"aptitude":74,"communication":68,"interview":72}}
```

Response shape after valid historical training (values are placeholders, not evaluation results or a measured prediction):

```json
{"outcomeProbability":"<0..1>","modelVersion":"<generated from historical data fingerprint>","provenance":"historical","metrics":"<held-out metrics generated by training>","limitations":["Association in historical data; preparation and human review only."],"latencyMs":"<measured per request>"}
```

`studentId`, `cohort`, and `placed` are training/evaluation fields and are not prediction features. The returned historical metrics come from the held-out cohort and are not guarantees for an individual.

### 2. Resume extraction

Request:

```json
{"resumeText":"EDUCATION\nB.Tech in CSE from Example Institute 2022-2026\nSKILLS\nReact, TypeScript\nPROJECTS\nCampus app - Built with React","language":"en"}
```

Response includes `skills`, `education`, `experience`, and `projects`; each extracted item carries source evidence and heuristic confidence. For this sample, the skill entries are `React` and `TypeScript`, both with evidence taken from the resume text. `method` identifies the dictionary/section rules and `reviewRequired` is true. Confidence values are heuristic and not calibrated. The current parser accepts English only. Raw resume text is not persisted by this package.

### 3. Job match relevance

Request:

```json
{"studentProfile":{"skills":["React","TypeScript"],"education":["BSc Computer Science"],"experience":["Built a React dashboard"],"projects":["Campus events app with TypeScript"]},"job":{"title":"Frontend developer","description":"Build accessible web interfaces","requiredSkills":["React","JavaScript"]}}
```

Response includes `relevanceScore` (TF-IDF cosine similarity, from 0 to 100), `matchedSkills`, `skillGaps`, evidence excerpts, `method:"tfidf-keyword-v1"`, and `trained:false`. Its numeric result depends on the supplied text, so no benchmark score is claimed here. It is a lexical relevance baseline, not a trained or semantic model.

### 4. Interview preparation feedback

Request:

```json
{"responses":[{"question":"Describe a project challenge","answer":"I tested the change and measured the result."}],"rubric":[{"name":"evidence","expectedTerms":["tested","measured"]},{"name":"project context","expectedTerms":["project","challenge"]}]}
```

Response contains per-criterion scores and supporting answer excerpts, followed by `generatedPreparationAdvice`. `method:"rubric-keyword-heuristic-v1"` and `trained:false` distinguish this feedback from a trained assessment. It does not assess factual correctness, vocal delivery, or hiring suitability.

## Supabase dataset

`supabase/migrations/202610020001_placement_training_evidence.sql` defines a restricted table with stable student grouping keys, evidence and outcome times, observed labels, and score range checks. Keep RLS enabled and export training rows from a trusted server-side job. Never put the Supabase service-role key in Next.js client code.

## Review of the supplied `server/ml.ts`

The supplied logistic trainer is a sound small baseline and is reused conceptually. The new version adds stable student-group isolation when holding out the latest cohort, checks evidence timestamps precede outcomes, validates score bounds, fingerprints the data, saves preprocessing, and compares against a prevalence baseline. The supplied skill dictionary, TF-IDF similarity, resume suggestions, and interview heuristics are retained as rule-based methods rather than relabelled as trained models. Expand their dictionaries/section parsers with CampusLink's real data conventions before treating extraction as complete.

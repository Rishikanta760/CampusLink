import { Router, type ErrorRequestHandler, type Request, type RequestHandler, type Response } from 'express';
import path from 'node:path';
import {
  extractResume, loadPlacementModel, matchJob, measured, predictPlacement,
  scoreInterviewWithRubric, validatePlacementEvidence, type PlacementEvidence, type PlacementModel,
} from './ml.js';

const fail = (res: Response, status: number, code: string, message: string) =>
  res.status(status).json({ error: { code, message } });
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const asString = (value: unknown) => typeof value === 'string';
const defaultModelDir = path.resolve(process.env.CAMPUSLINK_MODEL_DIR ?? 'models');

// Mount behind the app's existing user-authentication middleware.
export function createMlRouter(
  authenticate: RequestHandler,
  modelDir = defaultModelDir,
  options: { allowUnverifiedDemo?: boolean } = {},
) {
  const router = Router();
  router.use(authenticate);

  router.post('/placement/predict', (req: Request, res: Response) => {
    try {
      if (!record(req.body?.evidence)) return fail(res, 400, 'INVALID_INPUT', 'evidence object is required.');
      const evidence = req.body.evidence as PlacementEvidence;
      const errors = validatePlacementEvidence(evidence);
      if (errors.length) return fail(res, 400, 'INVALID_INPUT', errors.join('; '));
      let model: PlacementModel | undefined;
      try { model = loadPlacementModel(modelDir, options.allowUnverifiedDemo === true); }
      catch (error) { return fail(res, 503, 'MODEL_INVALID', error instanceof Error ? error.message : 'Model artifact could not be loaded.'); }
      if (!model) return fail(res, 503, 'MODEL_UNAVAILABLE', 'No allowed placement model is available.');
      const timed = measured(() => predictPlacement(model, evidence));
      return res.json({
        outcomeProbability: Number(timed.result.toFixed(6)),
        modelVersion: model.version,
        provenance: model.provenance,
        metrics: model.evaluation.model,
        limitations: model.provenance === 'historical'
          ? ['Association in historical data; preparation and human review only.']
          : ['Unverified demonstration data; this output is not a valid historical estimate and must not be used for real decisions.'],
        latencyMs: timed.latencyMs,
      });
    } catch (error) { return fail(res, 400, 'PREDICTION_ERROR', error instanceof Error ? error.message : 'Prediction failed.'); }
  });

  router.post('/resume/extract', (req: Request, res: Response) => {
    const text = req.body?.resumeText;
    if (!asString(text) || !text.trim()) return fail(res, 400, 'INVALID_INPUT', 'resumeText must be a non-empty string.');
    if (req.body.language !== undefined && req.body.language !== 'en') return fail(res, 400, 'UNSUPPORTED_LANGUAGE', 'The current local extraction rules support English only.');
    if (text.length > 100_000) return fail(res, 413, 'INPUT_TOO_LARGE', 'resumeText exceeds the 100000-character limit.');
    try {
      const timed = measured(() => extractResume(text));
      return res.json({ ...timed.result, latencyMs: timed.latencyMs });
    } catch (error) { return fail(res, 400, 'EXTRACTION_ERROR', error instanceof Error ? error.message : 'Resume extraction failed.'); }
  });

  router.post('/jobs/match', (req: Request, res: Response) => {
    const student = req.body?.studentProfile, job = req.body?.job;
    const arrays = [student?.skills, student?.education, student?.experience, student?.projects, job?.requiredSkills];
    if (!record(student) || !record(job) || arrays.some((value) => !Array.isArray(value) || value.some((item) => !asString(item))) ||
      !asString(job.title) || !asString(job.description))
      return fail(res, 400, 'INVALID_INPUT', 'studentProfile must include skills, education, experience, projects; job must include title, description, and requiredSkills.');
    try {
      const timed = measured(() => matchJob(req.body));
      return res.json({ ...timed.result, latencyMs: timed.latencyMs });
    } catch (error) { return fail(res, 400, 'MATCH_ERROR', error instanceof Error ? error.message : 'Job matching failed.'); }
  });

  router.post('/interviews/score', (req: Request, res: Response) => {
    const responses = req.body?.responses;
    const rubric = req.body?.rubric;
    if (!Array.isArray(responses) || !responses.length || responses.some((item) => !record(item) || !asString(item.question) || !asString(item.answer)) ||
      !Array.isArray(rubric) || !rubric.length || rubric.some((item) => !record(item) || !asString(item.name) || !Array.isArray(item.expectedTerms) || !item.expectedTerms.length || item.expectedTerms.some((term) => !asString(term))))
      return fail(res, 400, 'INVALID_INPUT', 'responses and rubric are required; rubric criteria need a name and expectedTerms.');
    try {
      const timed = measured(() => scoreInterviewWithRubric(responses, rubric));
      return res.json({ ...timed.result, latencyMs: timed.latencyMs });
    } catch (error) { return fail(res, 400, 'SCORING_ERROR', error instanceof Error ? error.message : 'Interview scoring failed.'); }
  });
  return router;
}

export const mlErrorHandler: ErrorRequestHandler = (error, _req, res, next) => {
  if (error?.type === 'entity.too.large') return fail(res, 413, 'INPUT_TOO_LARGE', 'Request body exceeds the configured limit.');
  if (error instanceof SyntaxError && 'body' in error) return fail(res, 400, 'INVALID_JSON', 'Request body must be valid JSON.');
  return next(error);
};

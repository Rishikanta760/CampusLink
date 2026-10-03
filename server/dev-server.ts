import express from 'express';
import { createMlRouter, mlErrorHandler } from './ml-router.js';
import { loadPlacementModel } from './ml.js';
import path from 'node:path';

const app = express();
const port = Number(process.env.PORT ?? 8010);
const token = process.env.CAMPUSLINK_ML_DEV_TOKEN;
if (!token) throw new Error('Set CAMPUSLINK_ML_DEV_TOKEN before starting the local development server.');
app.use(express.json({ limit: '256kb' }));
const authenticate: express.RequestHandler = (req, res, next) => {
  if (req.header('authorization') !== `Bearer ${token}`)
    return res.status(401).json({ error: { code: 'UNAUTHORIZED', message: 'Authentication required.' } });
  next();
};
const allowUnverifiedDemo = process.env.CAMPUSLINK_ALLOW_UNVERIFIED_DEMO === 'true';
const modelDir = path.resolve(process.env.CAMPUSLINK_MODEL_DIR ?? 'models');
app.get('/', (_req, res) => res.json({
  service: 'CampusLink ML API',
  status: 'ready',
  modelProvenance: allowUnverifiedDemo ? 'unverified_demo (explicitly enabled)' : 'historical only',
  endpoints: ['POST /v1/models/placement/predict', 'POST /v1/models/resume/extract', 'POST /v1/models/jobs/match', 'POST /v1/models/interviews/score'],
  docs: 'No Swagger UI is bundled; see README.md for request examples.',
}));
app.get('/health', (_req, res) => {
  try {
    const model = loadPlacementModel(modelDir, allowUnverifiedDemo);
    if (!model) return res.status(503).json({ status: 'unavailable', reason: 'No allowed model artifact found.' });
    return res.json({ status: 'ready', service: 'CampusLink ML API', modelVersion: model.version, provenance: model.provenance });
  } catch (error) {
    return res.status(503).json({ status: 'unavailable', reason: error instanceof Error ? error.message : 'Model artifact is invalid.' });
  }
});
app.use('/v1/models', createMlRouter(authenticate, undefined, { allowUnverifiedDemo }));
app.use(mlErrorHandler);
app.listen(port, () => console.log(`CampusLink ML API listening on http://127.0.0.1:${port}${allowUnverifiedDemo ? ' (unverified demo model explicitly enabled)' : ''}`));

import fs from 'node:fs';
import path from 'node:path';
import { savePlacementModel, trainPlacement, type PlacementTrainingRow } from '../server/ml.js';

const input = process.argv[2];
const output = process.argv[3] ?? 'models-demo';
if (!input || !fs.existsSync(input)) {
  console.error('Usage: npm run train:placement:demo -- <placement_ai_trainer_dataset.jsonl> [output-directory]');
  process.exit(2);
}

const featureNames = ['verifiedSkills', 'academics', 'projects', 'aptitude', 'communication', 'interview'] as const;
const exclusions: Record<string, number> = {};
const rows: PlacementTrainingRow[] = [];
for (const [index, line] of fs.readFileSync(input, 'utf8').split(/\r?\n/).entries()) {
  if (!line.trim()) continue;
  let source: Record<string, unknown>;
  try { source = JSON.parse(line) as Record<string, unknown>; }
  catch { throw new Error(`Invalid JSON on input line ${index + 1}.`); }
  const exclude = (reason: string) => { exclusions[reason] = (exclusions[reason] ?? 0) + 1; };
  const studentId = source.studentId ?? source.student_id;
  const cohort = source.cohort;
  const evidenceAt = source.evidenceAt ?? source.evidence_at;
  const outcomeAt = source.outcomeAt ?? source.outcome_at;
  const rawLabel = source.placed ?? source.placement_status;
  const placed = rawLabel === 1 || rawLabel === 'Placed' ? 1 : rawLabel === 0 || rawLabel === 'Unplaced' ? 0 : undefined;
  if (typeof studentId !== 'string' || !studentId.trim()) { exclude('missing student id'); continue; }
  if (typeof cohort !== 'string' || !cohort.trim()) { exclude('missing cohort'); continue; }
  if (typeof evidenceAt !== 'string' || typeof outcomeAt !== 'string' || !Number.isFinite(Date.parse(evidenceAt)) || !Number.isFinite(Date.parse(outcomeAt))) { exclude('missing or invalid evidence/outcome date'); continue; }
  if (Date.parse(evidenceAt) >= Date.parse(outcomeAt)) { exclude('evidence was not before outcome'); continue; }
  if (placed === undefined) { exclude('unrecognized placement outcome'); continue; }
  const evidence = Object.fromEntries(featureNames.map((feature) => [feature, source[feature]])) as Record<typeof featureNames[number], unknown>;
  if (featureNames.some((feature) => typeof evidence[feature] !== 'number' || !Number.isFinite(evidence[feature]) || evidence[feature] < 0 || evidence[feature] > 100)) {
    exclude('missing or out-of-range score (0..100)'); continue;
  }
  rows.push({
    studentId, cohort, evidenceAt, outcomeAt, outcomeSource: 'unverified_demo', placed: placed as 0 | 1,
    verifiedSkills: evidence.verifiedSkills as number,
    academics: evidence.academics as number,
    projects: evidence.projects as number,
    aptitude: evidence.aptitude as number,
    communication: evidence.communication as number,
    interview: evidence.interview as number,
  });
}

const model = trainPlacement(rows, 'unverified_demo');
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, 'placement-training-demo.jsonl'), rows.map((row) => JSON.stringify(row)).join('\n') + '\n');
savePlacementModel(model, output);
console.log(JSON.stringify({
  warning: 'UNVERIFIED DEMONSTRATION ONLY. Do not treat these scores or labels as real historical evidence or use this model for real placement decisions.',
  input: path.resolve(input),
  acceptedRows: rows.length,
  excludedRows: Object.values(exclusions).reduce((sum, count) => sum + count, 0),
  exclusions,
  normalizedTrainingFile: path.resolve(output, 'placement-training-demo.jsonl'),
  modelFile: path.resolve(output, 'placement.model.json'),
  preprocessingFile: path.resolve(output, 'placement.preprocessing.json'),
  modelVersion: model.version,
  provenance: model.provenance,
  split: model.split,
  evaluation: model.evaluation,
}, null, 2));

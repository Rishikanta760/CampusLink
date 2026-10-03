import fs from 'node:fs';
import path from 'node:path';
import { trainPlacement, savePlacementModel, type PlacementTrainingRow } from '../server/ml.js';

const input = process.argv[2] ?? 'data/placement-training.jsonl';
const output = process.argv[3] ?? 'models';
if (!fs.existsSync(input)) {
  console.error(`Training data not found: ${input}`);
    console.error('Export observed historical rows that follow schema/placement-training-row.schema.json to JSONL.');
  process.exit(2);
}
const rows = fs.readFileSync(input, 'utf8').split(/\r?\n/).filter(Boolean)
  .map((line, index) => {
    try { return JSON.parse(line) as PlacementTrainingRow; }
    catch { throw new Error(`Invalid JSON on line ${index + 1}`); }
  });
const model = trainPlacement(rows);
savePlacementModel(model, output);
console.log(JSON.stringify({
  modelVersion: model.version,
  modelFile: path.resolve(output, 'placement.model.json'),
  preprocessingFile: path.resolve(output, 'placement.preprocessing.json'),
  split: model.split,
  evaluation: model.evaluation,
}, null, 2));

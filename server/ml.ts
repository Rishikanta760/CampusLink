import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';

export const placementFeatures = [
  'verifiedSkills', 'academics', 'projects', 'aptitude', 'communication', 'interview',
] as const;

export type PlacementEvidence = Record<(typeof placementFeatures)[number], number>;
export interface PlacementTrainingRow extends PlacementEvidence {
  studentId: string;
  cohort: string;
  evidenceAt: string;
  outcomeAt: string;
  outcomeSource: 'observed' | 'unverified_demo';
  placed: 0 | 1;
}
export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  brier: number;
  testCount: number;
  positiveCount: number;
  negativeCount: number;
}
export interface PlacementModel {
  version: string;
  provenance: 'historical' | 'unverified_demo';
  trainedAt: string;
  featureOrder: string[];
  weights: number[];
  preprocessingFile: string;
  split: {
    strategy: 'latest-cohort-with-student-group-isolation';
    trainingCohorts: string[];
    testCohorts: string[];
    trainingStudentCount: number;
    testStudentCount: number;
    purgedRowsForGroupIsolation: number;
  };
  evaluation: { model: Metrics; majorityBaseline: Metrics; trainingCount: number };
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x))));
const vector = (e: PlacementEvidence) => [1, ...placementFeatures.map((f) => e[f] / 100)];

export function validatePlacementEvidence(evidence: PlacementEvidence): string[] {
  const errors: string[] = [];
  for (const feature of placementFeatures) {
    const value = evidence?.[feature];
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 100)
      errors.push(`${feature} must be a finite number from 0 to 100`);
  }
  return errors;
}

export function predictPlacement(model: PlacementModel, evidence: PlacementEvidence) {
  const errors = validatePlacementEvidence(evidence);
  if (errors.length) throw new Error(errors.join('; '));
  return sigmoid(vector(evidence).reduce((sum, value, i) => sum + value * model.weights[i], 0));
}

function metrics(rows: PlacementTrainingRow[], probabilities: number[]): Metrics {
  let tp = 0, tn = 0, fp = 0, fn = 0, brier = 0, positiveCount = 0;
  rows.forEach((row, i) => {
    const p = probabilities[i];
    const label = row.placed;
    positiveCount += label;
    brier += (p - label) ** 2;
    if (p >= 0.5) label ? tp++ : fp++;
    else label ? fn++ : tn++;
  });
  const precision = tp / Math.max(1, tp + fp), recall = tp / Math.max(1, tp + fn);
  return {
    accuracy: (tp + tn) / Math.max(1, rows.length),
    precision,
    recall,
    f1: 2 * precision * recall / Math.max(1e-12, precision + recall),
    brier: brier / Math.max(1, rows.length),
    testCount: rows.length,
    positiveCount,
    negativeCount: rows.length - positiveCount,
  };
}

export function trainPlacement(rows: PlacementTrainingRow[], provenance: PlacementModel['provenance'] = 'historical'): PlacementModel {
  if (rows.length < 60) throw new Error('Use at least 60 historically labelled rows.');
  for (const [i, row] of rows.entries()) {
    if (!row.studentId?.trim()) throw new Error(`Row ${i + 1}: studentId is required for group-safe splitting.`);
    if (!row.cohort?.trim()) throw new Error(`Row ${i + 1}: cohort is required.`);
    const expectedSource = provenance === 'historical' ? 'observed' : 'unverified_demo';
    if (row.outcomeSource !== expectedSource) throw new Error(`Row ${i + 1}: outcomeSource must be '${expectedSource}' for ${provenance} training.`);
    if (!Number.isFinite(Date.parse(row.evidenceAt)) || !Number.isFinite(Date.parse(row.outcomeAt)) || Date.parse(row.evidenceAt) >= Date.parse(row.outcomeAt))
      throw new Error(`Row ${i + 1}: evidenceAt must be a valid timestamp earlier than outcomeAt.`);
    if (row.placed !== 0 && row.placed !== 1) throw new Error(`Row ${i + 1}: placed must be the observed label 0 or 1.`);
    const featureErrors = validatePlacementEvidence(row);
    if (featureErrors.length) throw new Error(`Row ${i + 1}: ${featureErrors.join('; ')}`);
  }
  const cohorts = [...new Set(rows.map((r) => r.cohort))].sort();
  if (cohorts.length < 3) throw new Error('Use at least three cohorts so the latest cohort can be held out.');
  const testCohort = cohorts.at(-1)!;
  const testRows = rows.filter((r) => r.cohort === testCohort);
  const testStudentIds = new Set(testRows.map((r) => r.studentId));
  const trainingRows = rows.filter((r) => r.cohort !== testCohort && !testStudentIds.has(r.studentId));
  const purgedRows = rows.filter((r) => r.cohort !== testCohort && testStudentIds.has(r.studentId));
  if (new Set(trainingRows.map((r) => r.placed)).size < 2 || new Set(testRows.map((r) => r.placed)).size < 2)
    throw new Error('Training and latest-cohort test sets both need placed and unplaced observed outcomes.');
  if (new Set(trainingRows.map((r) => r.studentId)).size < 2)
    throw new Error('Group-isolated training split has too few distinct students.');

  const weights = Array(placementFeatures.length + 1).fill(0) as number[];
  const learningRate = 0.15, l2 = 0.01;
  for (let epoch = 0; epoch < 1800; epoch++) {
    const gradient = Array(weights.length).fill(0) as number[];
    for (const row of trainingRows) {
      const x = vector(row);
      const error = sigmoid(x.reduce((sum, value, i) => sum + value * weights[i], 0)) - row.placed;
      x.forEach((value, i) => gradient[i] += error * value);
    }
    weights.forEach((weight, i) => {
      weights[i] -= learningRate * (gradient[i] / trainingRows.length + (i ? l2 * weight : 0));
    });
  }
  const trainingPositiveRate = trainingRows.filter((r) => r.placed === 1).length / trainingRows.length;
  const modelMetrics = metrics(testRows, testRows.map((r) => sigmoid(vector(r).reduce((s, x, i) => s + x * weights[i], 0))));
  const baselineMetrics = metrics(testRows, testRows.map(() => trainingPositiveRate));
  const dataFingerprint = createHash('sha256').update(JSON.stringify(rows)).digest('hex').slice(0, 12);
  return {
    version: `${provenance === 'historical' ? 'campuslink-logistic-v2' : 'campuslink-logistic-demo-v1'}-${dataFingerprint}`,
    provenance,
    trainedAt: new Date().toISOString(),
    featureOrder: [...placementFeatures],
    weights,
    preprocessingFile: 'placement.preprocessing.json',
    split: {
      strategy: 'latest-cohort-with-student-group-isolation',
      trainingCohorts: cohorts.filter((c) => c !== testCohort),
      testCohorts: [testCohort],
      trainingStudentCount: new Set(trainingRows.map((r) => r.studentId)).size,
      testStudentCount: testStudentIds.size,
      purgedRowsForGroupIsolation: purgedRows.length,
    },
    evaluation: { model: modelMetrics, majorityBaseline: baselineMetrics, trainingCount: trainingRows.length },
  };
}

export function savePlacementModel(model: PlacementModel, directory: string) {
  fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(path.join(directory, 'placement.model.json'), JSON.stringify(model, null, 2));
  fs.writeFileSync(path.join(directory, 'placement.preprocessing.json'), JSON.stringify({
    version: 'score-normalization-v1', featureOrder: placementFeatures,
    transform: 'divide each 0-100 evidence score by 100', missingValues: 'rejected',
  }, null, 2));
}

export function loadPlacementModel(directory: string, allowUnverifiedDemo = false): PlacementModel | undefined {
  const file = path.join(directory, 'placement.model.json');
  if (!fs.existsSync(file)) return undefined;
  const parsed = JSON.parse(fs.readFileSync(file, 'utf8')) as PlacementModel;
  if ((parsed.provenance !== 'historical' && !(allowUnverifiedDemo && parsed.provenance === 'unverified_demo')) || parsed.weights?.length !== placementFeatures.length + 1 ||
    JSON.stringify(parsed.featureOrder) !== JSON.stringify(placementFeatures))
    throw new Error('Invalid model artifact or non-historical provenance.');
  const preprocessingPath = path.join(directory, parsed.preprocessingFile);
  if (!fs.existsSync(preprocessingPath)) throw new Error('Placement model preprocessing file is missing.');
  const preprocessing = JSON.parse(fs.readFileSync(preprocessingPath, 'utf8'));
  if (JSON.stringify(preprocessing.featureOrder) !== JSON.stringify(placementFeatures) || preprocessing.transform !== 'divide each 0-100 evidence score by 100')
    throw new Error('Placement model preprocessing does not match the supported feature schema.');
  return parsed;
}

const skillAliases: Record<string, string[]> = {
  React: ['react', 'reactjs', 'react.js'], JavaScript: ['javascript', 'js', 'ecmascript'],
  TypeScript: ['typescript', 'ts'], 'Node.js': ['node.js', 'nodejs'], Express: ['express', 'expressjs'],
  Python: ['python'], Java: ['java'], SQL: ['sql', 'postgresql', 'postgres', 'mysql'],
  MongoDB: ['mongodb', 'mongo'], AWS: ['aws', 'amazon web services'], Docker: ['docker', 'containerization'],
  Kubernetes: ['kubernetes', 'k8s'], Git: ['git'], GitHub: ['github'], DSA: ['dsa', 'data structures', 'algorithms'],
  HTML: ['html', 'html5'], CSS: ['css', 'css3'], 'Machine Learning': ['machine learning', 'ml', 'scikit-learn', 'sklearn'],
  TensorFlow: ['tensorflow'], PyTorch: ['pytorch'], Excel: ['excel'], PowerBI: ['power bi', 'powerbi'],
};
const escapeRegExp = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const hasTerm = (text: string, term: string) => new RegExp(`(?<![a-z0-9])${escapeRegExp(term)}(?![a-z0-9])`, 'i').test(text);

function skillMatches(text: string) {
  const found: Array<{ name: string; evidence: string; confidence: number }> = [];
  for (const [name, aliases] of Object.entries(skillAliases)) {
    for (const alias of aliases) {
      const match = new RegExp(`(?<![a-z0-9])${escapeRegExp(alias)}(?![a-z0-9])`, 'i').exec(text);
      if (match) { found.push({ name, evidence: match[0], confidence: 0.9 }); break; }
    }
  }
  return found;
}

type ResumeRecord = { value: string; evidence: string; confidence: number };
function sectionLines(text: string, sectionNames: string[]): string[] {
  const lines = text.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  let active = false;
  const result: string[] = [];
  const headings = /^(education|academic|experience|work experience|employment|internship|project|projects|skills|technical skills|certifications?|summary|profile)\s*:?$/i;
  for (const line of lines) {
    const heading = line.replace(/^#+\s*/, '').replace(/[:|]+$/, '').trim().toLowerCase();
    if (headings.test(line.replace(/^#+\s*/, '').trim())) {
      active = sectionNames.includes(heading);
      continue;
    }
    if (active) result.push(line.replace(/^[•*\-–]+\s*/, ''));
  }
  return result;
}

export function extractResume(resumeText: string) {
  if (!resumeText.trim()) throw new Error('resumeText must not be empty.');
  const education = sectionLines(resumeText, ['education', 'academic']).map((line) => ({
    qualification: line.match(/\b(?:B\.?Tech|B\.?E\.?|M\.?Tech|M\.?E\.?|MCA|BCA|MBA|BSc|MSc|Bachelor(?:'s)?|Master(?:'s)?)\b[^,|]*/i)?.[0] ?? null,
    institution: line.match(/\b(?:at|from)\s+([^,|]+)/i)?.[1]?.trim() ?? null,
    dates: line.match(/\b(?:19|20)\d{2}\s*(?:-|–|to)\s*(?:19|20)\d{2}\b/i)?.[0] ?? null,
    evidence: line, confidence: 0.75,
  }));
  const experience = sectionLines(resumeText, ['experience', 'work experience', 'employment', 'internship']).map((line) => ({
    title: null as string | null, organization: line.match(/\b(?:at|@)\s+([^,|]+)/i)?.[1]?.trim() ?? null,
    dates: line.match(/\b(?:19|20)\d{2}\s*(?:-|–|to)\s*(?:19|20)\d{2}\b/i)?.[0] ?? null,
    description: line, evidence: line, confidence: 0.65,
  }));
  const skillEntries = skillMatches(resumeText);
  const projects = sectionLines(resumeText, ['project', 'projects']).map((line) => ({
    name: line.split(/[:|–-]/, 1)[0].trim() || null, description: line,
    technologies: skillEntries.filter((s) => hasTerm(line, s.name)).map((s) => s.name),
    evidence: line, confidence: 0.7,
  }));
  return {
    skills: skillEntries,
    education,
    experience,
    projects,
    method: 'dictionary-and-section-rules-v1',
    reviewRequired: true,
    limitations: ['Rule-based extraction; verify inferred fields against the source resume.'],
  };
}

function normalizeTokens(text: string): string[] {
  let normalized = text.toLowerCase();
  for (const [name, aliases] of Object.entries(skillAliases)) {
    const canonical = name.toLowerCase().replace(/[^a-z0-9]/g, '');
    for (const alias of aliases)
      normalized = normalized.replace(new RegExp(`(?<![a-z0-9])${escapeRegExp(alias)}(?![a-z0-9])`, 'gi'), ` ${canonical} `);
  }
  const stop = new Set('the a an and or is are to of for with in on from as be this that have has it'.split(' '));
  return (normalized.match(/[a-z][a-z0-9]{1,}/g) ?? []).filter((token) => !stop.has(token));
}

function tfidfCosine(query: string, document: string): number {
  const docs = [normalizeTokens(query), normalizeTokens(document)];
  const vocabulary = [...new Set(docs.flat())];
  const idf = new Map(vocabulary.map((term) => [term, Math.log((docs.length + 1) / (1 + docs.filter((d) => d.includes(term)).length)) + 1]));
  const vectorize = (tokens: string[]) => new Map(vocabulary.map((term) => [term,
    tokens.filter((token) => token === term).length / Math.max(1, tokens.length) * idf.get(term)!,
  ]));
  const a = vectorize(docs[0]), b = vectorize(docs[1]);
  const dot = vocabulary.reduce((sum, term) => sum + a.get(term)! * b.get(term)!, 0);
  const norm = (v: Map<string, number>) => Math.sqrt([...v.values()].reduce((sum, value) => sum + value * value, 0));
  return dot / Math.max(1e-12, norm(a) * norm(b));
}

export interface JobMatchInput {
  studentProfile: { skills: string[]; education: string[]; experience: string[]; projects: string[] };
  job: { title: string; description: string; requiredSkills: string[] };
}
export function matchJob(input: JobMatchInput) {
  const profileParts = [
    ...input.studentProfile.skills.map((text) => ({ source: 'skills', text })),
    ...input.studentProfile.education.map((text) => ({ source: 'education', text })),
    ...input.studentProfile.experience.map((text) => ({ source: 'experience', text })),
    ...input.studentProfile.projects.map((text) => ({ source: 'projects', text })),
  ];
  const profile = profileParts.map((part) => part.text).join(' ');
  const jobText = `${input.job.title} ${input.job.description} ${input.job.requiredSkills.join(' ')}`;
  const required = [...new Set(input.job.requiredSkills.map((s) => s.trim()).filter(Boolean))];
  const evidence = required.flatMap((skill) => {
    const aliases = new Set([skill.toLowerCase(), ...skillMatches(skill).map((match) => match.name.toLowerCase())]);
    const source = profileParts.find((part) => skillMatches(part.text).some((match) => aliases.has(match.name.toLowerCase())) || hasTerm(part.text, skill));
    return source ? [{ skill, source: source.source, excerpt: source.text }] : [];
  });
  const matchedSet = new Set(evidence.map((item) => item.skill));
  const matchedSkills = required.filter((skill) => matchedSet.has(skill));
  const skillGaps = required.filter((skill) => !matchedSet.has(skill));
  return {
    relevanceScore: Math.round(tfidfCosine(jobText, profile) * 100),
    matchedSkills, skillGaps, evidence,
    method: 'tfidf-keyword-v1', trained: false,
    limitations: ['Keyword similarity is not semantic understanding; review profile and job evidence.'],
  };
}

export interface InterviewResponse { question: string; answer: string; keyTerms: string[] }
export function scoreInterview(responses: InterviewResponse[]) {
  if (!responses.length) throw new Error('Provide at least one question and answer.');
  const scores = responses.map(({ question, answer, keyTerms }) => {
    const answerTokens = new Set(normalizeTokens(answer));
    const expected = keyTerms.length ? keyTerms : normalizeTokens(question);
    const relevance = expected.length ? Math.round(expected.filter((t) => answerTokens.has(t.toLowerCase())).length / expected.length * 100) : 0;
    const words = answer.trim().split(/\s+/).filter(Boolean).length;
    const structure = /because|result|first|then|example|challenge|action/i.test(answer) ? 100 : 40;
    const evidence = /\d|implemented|built|tested|measured|resolved/i.test(answer) ? 100 : 35;
    const detail = Math.min(100, words * 2);
    const evidenceTerms = answer.split(/[.!?]/).map((s) => s.trim()).filter((s) => s.length > 18).slice(0, 2);
    return { scores: { relevance, detail, structure, specificEvidence: evidence }, evidence: evidenceTerms };
  });
  const average = (key: keyof (typeof scores)[number]['scores']) => Math.round(scores.reduce((s, row) => s + row.scores[key], 0) / scores.length);
  const avg = { relevance: average('relevance'), detail: average('detail'), structure: average('structure'), specificEvidence: average('specificEvidence') };
  const generatedPreparationAdvice: string[] = [];
  if (avg.relevance < 60) generatedPreparationAdvice.push('Practice answering the exact question and connect each point to the role or topic.');
  if (avg.detail < 60) generatedPreparationAdvice.push('Add context about your responsibility and the steps you took.');
  if (avg.structure < 60) generatedPreparationAdvice.push('Organize examples as situation, action, and result.');
  if (avg.specificEvidence < 60) generatedPreparationAdvice.push('Support claims with a concrete example or measurable result where available.');
  if (!generatedPreparationAdvice.length) generatedPreparationAdvice.push('Continue practicing concise answers with specific examples.');
  return {
    responses: scores,
    averageScores: avg,
    generatedPreparationAdvice,
    method: 'text-heuristics-v1', trained: false, reviewRequired: true,
    limitation: 'These text heuristics do not assess factual correctness, speaking delivery, or hiring suitability.',
  };
}

export interface InterviewResponse { question: string; answer: string }
export interface InterviewCriterion { name: string; expectedTerms: string[] }
export function scoreInterviewWithRubric(responses: InterviewResponse[], rubric: InterviewCriterion[]) {
  if (!responses.length) throw new Error('Provide at least one question and answer.');
  if (!rubric.length || rubric.some((c) => !c.name.trim() || !c.expectedTerms.length))
    throw new Error('Provide rubric criteria with a name and expectedTerms.');
  const scoredResponses = responses.map(({ question, answer }) => ({
    question,
    criteria: rubric.map((criterion) => {
      const matchedTerms = criterion.expectedTerms.filter((term) => hasTerm(answer, term));
      const score = Math.round(matchedTerms.length / criterion.expectedTerms.length * 100);
      const evidence = answer.split(/[.!?]/).map((s) => s.trim())
        .find((sentence) => matchedTerms.some((term) => hasTerm(sentence, term))) ?? '';
      return { name: criterion.name, score, matchedTerms, evidence };
    }),
  }));
  const scores = rubric.map((criterion, index) => ({
    name: criterion.name,
    average: Math.round(scoredResponses.reduce((sum, response) => sum + response.criteria[index].score, 0) / scoredResponses.length),
  }));
  const generatedPreparationAdvice = scores.filter((row) => row.average < 70)
    .map((row) => `Practice giving a specific answer that addresses the ${row.name} rubric criteria.`);
  if (!generatedPreparationAdvice.length)
    generatedPreparationAdvice.push('Continue practicing clear answers with examples that address the rubric.');
  return {
    scores, responses: scoredResponses, generatedPreparationAdvice,
    method: 'rubric-keyword-heuristic-v1', trained: false, reviewRequired: true,
    limitation: 'Keyword overlap does not establish factual correctness, speaking delivery, or hiring suitability.',
  };
}

export function measured<T>(fn: () => T): { result: T; latencyMs: number } {
  const start = performance.now();
  const result = fn();
  return { result, latencyMs: Math.round((performance.now() - start) * 1000) / 1000 };
}

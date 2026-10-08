from __future__ import annotations

import os
import re
import secrets
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .model import FEATURES, load_artifacts

app = FastAPI(title="CampusLink ML API", version="python-demo-v1")
MAX_BODY_BYTES = 256 * 1024
MAX_RESUME_CHARS = 100_000


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


@app.exception_handler(ApiError)
async def handle_api_error(_request: Request, error: ApiError) -> JSONResponse:
    return JSONResponse(status_code=error.status, content={"error": {"code": error.code, "message": error.message}})


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
    messages = "; ".join(f"{'.'.join(str(part) for part in item['loc'][1:])}: {item['msg']}" for item in error.errors())
    return JSONResponse(status_code=400, content={"error": {"code": "INVALID_INPUT", "message": messages or "Request fields are invalid."}})


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"error": {"code": "INPUT_TOO_LARGE", "message": "Request body exceeds the 256 KB limit."}})
    return await call_next(request)


def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = os.getenv("CAMPUSLINK_ML_DEV_TOKEN")
    if not expected:
        raise ApiError(503, "AUTH_NOT_CONFIGURED", "Set CAMPUSLINK_ML_DEV_TOKEN in the service environment.")
    if not authorization or not secrets.compare_digest(authorization, f"Bearer {expected}"):
        raise ApiError(401, "UNAUTHORIZED", "Authentication required.")


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verifiedSkills: float = Field(ge=0, le=100)
    academics: float = Field(ge=0, le=100)
    projects: float = Field(ge=0, le=100)
    aptitude: float = Field(ge=0, le=100)
    communication: float = Field(ge=0, le=100)
    interview: float = Field(ge=0, le=100)


class PlacementRequest(BaseModel):
    evidence: Evidence


class ResumeRequest(BaseModel):
    resumeText: str = Field(min_length=1)
    language: str = "en"


class StudentProfile(BaseModel):
    skills: list[str]
    education: list[str]
    experience: list[str]
    projects: list[str]


class JobInfo(BaseModel):
    title: str
    description: str
    requiredSkills: list[str]


class JobMatchRequest(BaseModel):
    studentProfile: StudentProfile
    job: JobInfo


class InterviewResponse(BaseModel):
    question: str
    answer: str


class RubricCriterion(BaseModel):
    name: str
    expectedTerms: list[str] = Field(min_length=1)


class InterviewRequest(BaseModel):
    responses: list[InterviewResponse] = Field(min_length=1)
    rubric: list[RubricCriterion] = Field(min_length=1)


@lru_cache(maxsize=4)
def get_model_bundle(model_dir: str, allow_demo: bool):
    return load_artifacts(model_dir, allow_demo)


def model_bundle():
    model_dir = str(Path(os.getenv("CAMPUSLINK_MODEL_DIR", "models")).resolve())
    allow_demo = os.getenv("CAMPUSLINK_ALLOW_UNVERIFIED_DEMO", "false").lower() == "true"
    try:
        bundle = get_model_bundle(model_dir, allow_demo)
    except Exception as error:
        raise ApiError(503, "MODEL_INVALID", str(error)) from error
    if bundle is None:
        raise ApiError(503, "MODEL_UNAVAILABLE", "No allowed placement model artifact is available.")
    return bundle


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "CampusLink ML API",
        "status": "ready",
        "modelProvenance": "unverified_demo (explicitly enabled)" if os.getenv("CAMPUSLINK_ALLOW_UNVERIFIED_DEMO", "false").lower() == "true" else "historical only",
        "endpoints": [
            "POST /v1/models/placement/predict",
            "POST /v1/models/resume/extract",
            "POST /v1/models/jobs/match",
            "POST /v1/models/interviews/score",
        ],
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    _estimator, metadata = model_bundle()
    return {"status": "ready", "service": "CampusLink ML API", "modelVersion": metadata["version"], "provenance": metadata["provenance"]}


@app.post("/v1/models/placement/predict", dependencies=[Depends(require_auth)])
def predict_placement(payload: PlacementRequest) -> dict[str, Any]:
    estimator, metadata = model_bundle()
    values = [[getattr(payload.evidence, feature) for feature in FEATURES]]
    started = time.perf_counter()
    probability = float(estimator.predict_proba(values)[0, 1])
    latency_ms = (time.perf_counter() - started) * 1000
    limitations = (
        ["Association in historical data; preparation and human review only."]
        if metadata["provenance"] == "historical"
        else ["Unverified demonstration data; this output is not a valid historical estimate and must not be used for real decisions."]
    )
    return {
        "outcomeProbability": round(probability, 6),
        "modelVersion": metadata["version"],
        "provenance": metadata["provenance"],
        "metrics": metadata["evaluation"]["model"],
        "limitations": limitations,
        "latencyMs": round(latency_ms, 3),
    }


SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "React": ("react", "reactjs", "react.js"),
    "Next.js": ("next.js", "nextjs"),
    "JavaScript": ("javascript", "js", "ecmascript"),
    "TypeScript": ("typescript", "ts"),
    "Node.js": ("node.js", "nodejs", "node js"),
    "Python": ("python", "py"),
    "Java": ("java",),
    "C++": ("c++", "cpp"),
    "C#": ("c#", "c sharp"),
    "SQL": ("sql", "structured query language"),
    "PostgreSQL": ("postgresql", "postgres"),
    "MongoDB": ("mongodb", "mongo db"),
    "HTML": ("html", "html5"),
    "CSS": ("css", "css3"),
    "AWS": ("aws", "amazon web services"),
    "Docker": ("docker",),
    "Git": ("git", "github", "gitlab"),
    "Machine Learning": ("machine learning", "ml"),
    "Data Analysis": ("data analysis", "data analytics"),
    "Excel": ("excel", "microsoft excel"),
    "FastAPI": ("fastapi", "fast api"),
    "Flask": ("flask",),
    "Express": ("express", "express.js", "expressjs"),
    "Figma": ("figma",),
}
SECTION_NAMES = {
    "skills": "skills", "technical skills": "skills", "education": "education",
    "experience": "experience", "work experience": "experience", "projects": "projects",
    "certifications": "certifications",
}


def extract_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {name: [] for name in ("skills", "education", "experience", "projects", "certifications", "other")}
    current = "other"
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("•*- ")
        if not line:
            continue
        header = re.sub(r"[:\s]+$", "", line.casefold())
        if header in SECTION_NAMES:
            current = SECTION_NAMES[header]
            continue
        sections[current].append(line)
    return sections


def extract_resume(text: str) -> dict[str, Any]:
    sections = extract_sections(text)
    lower = text.casefold()
    skills = []
    for canonical, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            match = re.search(r"(?<![\w+#])" + re.escape(alias) + r"(?![\w+#])", lower)
            if match:
                skills.append({"name": canonical, "evidence": text[max(0, match.start() - 60):min(len(text), match.end() + 60)].strip(), "confidence": 0.9})
                break

    def entries(section: str) -> list[dict[str, Any]]:
        return [{"text": line, "evidence": line, "confidence": 0.65} for line in sections[section]]

    return {
        "skills": skills,
        "education": entries("education"),
        "experience": entries("experience"),
        "projects": entries("projects"),
        "method": "dictionary-and-section-rules-v1",
        "reviewRequired": True,
        "trained": False,
    }


@app.post("/v1/models/resume/extract", dependencies=[Depends(require_auth)])
def resume_extract(payload: ResumeRequest) -> dict[str, Any]:
    if payload.language != "en":
        raise ApiError(400, "UNSUPPORTED_LANGUAGE", "The current local extraction rules support English only.")
    if len(payload.resumeText) > MAX_RESUME_CHARS:
        raise ApiError(413, "INPUT_TOO_LARGE", f"resumeText exceeds the {MAX_RESUME_CHARS}-character limit.")
    started = time.perf_counter()
    result = extract_resume(payload.resumeText)
    result["latencyMs"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def normalize_skill(skill: str) -> str:
    value = skill.strip().casefold()
    for canonical, aliases in SKILL_ALIASES.items():
        if value == canonical.casefold() or value in aliases:
            return canonical.casefold()
    return value


def match_job(payload: JobMatchRequest) -> dict[str, Any]:
    student = payload.studentProfile
    job = payload.job
    student_text_parts = student.skills + student.education + student.experience + student.projects
    student_text = " ".join(student_text_parts).strip()
    job_text = " ".join([job.title, job.description, *job.requiredSkills]).strip()
    try:
        vectors = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform([student_text, job_text])
        relevance = float(cosine_similarity(vectors[0], vectors[1])[0, 0])
    except ValueError:
        relevance = 0.0

    student_skills = {normalize_skill(skill) for skill in student.skills}
    matched = [skill for skill in job.requiredSkills if normalize_skill(skill) in student_skills]
    gaps = [skill for skill in job.requiredSkills if normalize_skill(skill) not in student_skills]
    evidence = []
    for required in matched:
        for source in student_text_parts:
            if normalize_skill(required) in normalize_skill(source):
                evidence.append({"skill": required, "excerpt": source[:240]})
                break
    return {
        "relevanceScore": round(relevance * 100, 2),
        "matchedSkills": matched,
        "skillGaps": gaps,
        "evidence": evidence,
        "method": "tfidf-keyword-v1",
        "trained": False,
    }


@app.post("/v1/models/jobs/match", dependencies=[Depends(require_auth)])
def jobs_match(payload: JobMatchRequest) -> dict[str, Any]:
    started = time.perf_counter()
    result = match_job(payload)
    result["latencyMs"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def score_interview(payload: InterviewRequest) -> dict[str, Any]:
    criteria = []
    answer_text = " ".join(item.answer for item in payload.responses)
    lower_answers = answer_text.casefold()
    for criterion in payload.rubric:
        matched_terms = [term for term in criterion.expectedTerms if term.casefold() in lower_answers]
        score = round(100 * len(matched_terms) / len(criterion.expectedTerms))
        evidence = []
        for term in matched_terms:
            found = re.search(re.escape(term), answer_text, re.IGNORECASE)
            if found:
                evidence.append(answer_text[max(0, found.start() - 50):min(len(answer_text), found.end() + 50)].strip())
        criteria.append({"name": criterion.name, "score": score, "matchedTerms": matched_terms, "evidence": evidence})
    average = round(sum(item["score"] for item in criteria) / len(criteria))
    weakest = [item["name"] for item in criteria if item["score"] < 60]
    advice = (
        "Practice adding specific examples and evidence for: " + ", ".join(weakest) + "."
        if weakest else "Keep practicing concise answers with concrete examples and measurable outcomes."
    )
    return {
        "criteria": criteria,
        "overallScore": average,
        "generatedPreparationAdvice": advice,
        "method": "rubric-keyword-heuristic-v1",
        "trained": False,
    }


@app.post("/v1/models/interviews/score", dependencies=[Depends(require_auth)])
def interview_score(payload: InterviewRequest) -> dict[str, Any]:
    started = time.perf_counter()
    result = score_interview(payload)
    result["latencyMs"] = round((time.perf_counter() - started) * 1000, 3)
    return result

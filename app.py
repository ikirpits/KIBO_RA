from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

import kibo_ra_v23 as kibo

auditor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global auditor
    auditor = kibo.KIBORA()
    yield


app = FastAPI(title="KIBO-RA", lifespan=lifespan)


class AssessRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: List[str]


def _payload(req_id: str, text: str, result: kibo.RiskResult) -> dict:
    return {
        "requirement_id": req_id,
        "text": text,
        "scores": result.scores,
        "overall": result.overall,
        "confidence": result.confidence,
        "risk_levels": {k: kibo.risk_level(v) for k, v in result.scores.items()},
        "governance_gate": kibo.sprint_gate(result),
        "evidence": result.evidence,
        "cobit_alignment": result.cobit_alignment,
    }


@app.get("/")
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "semantic_available": auditor.semantic.available if auditor else False,
        "kris": kibo.KRI_ORDER,
    }


@app.post("/assess")
def assess(req: AssessRequest):
    result = auditor.assess(req.text)
    return _payload("R1", req.text, result)


@app.post("/assess/batch")
def assess_batch(req: BatchRequest):
    return [
        _payload(f"R{i}", text, auditor.assess(text))
        for i, text in enumerate(req.texts, 1)
    ]

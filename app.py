import io
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, Response
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


@app.get("/report")
def report():
    texts = kibo.load_requirements(txt_path=str(kibo.BASE_DIR / "requirements_input.txt"))
    results = [auditor.assess(t) for t in texts]
    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp) / "kibo_ra_report")
        outputs = list(kibo.save_outputs(results, prefix))

        df = kibo.results_dataframe(results)
        req_ids = df["requirement"].tolist()
        cobit_json = f"{prefix}_cobit_signals.json"
        cobit_csv = f"{prefix}_cobit_matrix.csv"
        kibo.save_governance_signals(results, req_ids, cobit_json, cobit_csv)
        outputs += [cobit_json, cobit_csv]

        heatmap_path = f"{prefix}_heatmap.png"
        kibo.save_heatmap(df, heatmap_path)
        outputs.append(heatmap_path)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for path in outputs:
                zf.write(path, arcname=Path(path).name)
        data = buf.getvalue()

    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=kibo_ra_report.zip"},
    )

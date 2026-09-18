---
title: KIBO-RA
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# KIBO-RA

Requirements risk auditor. Scores requirement text on five KRIs (performance,
security, compliance, complexity, ambiguity) from lexical + semantic evidence.

Interactive docs at `/docs`.

## API

`POST /assess`
```json
{"text": "The system shall process a payment within 2 seconds."}
```

`POST /assess/batch`
```json
{"texts": ["...", "..."]}
```

`GET /health`

## Local run

```
pip install -r requirements.txt
uvicorn app:app --reload
```

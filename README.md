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

Requirements exposure auditor. Scores requirement text on five KRIs (performance,
security, compliance, complexity, ambiguity) from lexical + semantic evidence.
Each score reflects how much governance-relevant language the requirement text
expresses -- not a judgment on the underlying system's actual quality. A high
security score, for example, means the text names a lot of control-relevant
detail worth reviewing, not that the system is insecure; a low score doesn't
mean it's secure.

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

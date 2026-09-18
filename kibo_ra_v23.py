
"""KIBO-RA v2 - Requirements Auditor"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from sentence_transformers import SentenceTransformer, models, util
except Exception:
    torch = None
    SentenceTransformer = None
    models = None
    util = None

try:
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
except Exception:
    plt = None


BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "governance_config_v23.json"

DEFAULT_CONFIG = {
    "risk_thresholds": {"low": 0.40, "medium": 0.70},
    "gate_thresholds": {
        "compliance": 0.50,
        "ambiguity": 0.60,
        "confidence": 0.65
    },
    "semantic": {
        "enabled": True,
        "model": "all-mpnet-base-v2",
        "bert4re_model": "thearod5/bert4re",
        "bert4re_enabled": True,
        "hybrid_weights": {"sbert": 0.50, "bert4re": 0.50},
        "semantic_weight": 0.50,
        "lexical_weight": 0.50
    },
    "scoring": {
        "evidence_saturation": 2.5,
        "semantic_temperature": 0.20,
        "length_normalization_tokens": 20
    },
    "confidence": {
        "semantic_weight": 0.50,
        "evidence_weight": 0.30,
        "agreement_weight": 0.20
    }
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


CONFIG = load_config()


COMPLEXITY_DOMAINS = {
    "security": [
        "encrypt*", "authenticat*", "authoriz*", "credential",
        "certificate", "key management",
        "cryptograph*", "digital signature", "public key", "private key",
        "hsm", "key rotation", "cipher", "hashing", "salt", "pki",
        "tls", "ssl", "vpn", "firewall", "penetration test*",
        "vulnerabilit*", "threat model*"
    ],
    "distributed_scale": [
        "distributed", "replicat*", "concurrent", "cluster",
        "load balanc*", "fault toleran*", "availab*", "scale", "scalab*",
        "capacity", "capable*",
        "sharding", "shard*", "partition*", "horizontal scal*",
        "vertical scal*", "elastic*", "microservice*", "high availab*",
        "failover", "redundan*", "multi-region", "geo-distributed",
        "consensus", "eventual consistency", "cdn", "throughput"
    ],
    "integration": [
        "integrat*", "third-party", "external system", "interoperab*",
        "migrat*",
        "api", "webhook", "middleware", "connector", "adapter",
        "legacy system", "data exchange", "interface with",
        "synchroniz*", "etl", "batch import", "external service",
        "vendor system"
    ],
    "deployment": [
        "deploy*", "provision*", "pipeline", "automat*", "release",
        "ci/cd", "continuous integration", "continuous delivery",
        "rollback", "blue-green", "canary", "infrastructure as code",
        "containeriz*", "orchestrat*", "kubernetes", "docker"
    ],
    "compliance": [
        "complian*", "comply*", "regulat*", "audit", "governance",
        "certif*", "accredit*"
    ],
    "access_control": [
        "role*", "permission*", "rbac", "role-based access",
        "abac", "attribute-based access", "least privilege",
        "access polic*", "entitlement", "authorization matrix",
        "segregation of duties"
    ],
    "event_driven": [
        "notif*", "real-time", "real time", "asynchron*",
        "event-driven", "publish-subscribe", "pub-sub", "queue",
        "message broker", "kafka", "event bus", "stream processing",
        "message queue", "callback", "trigger*"
    ],
    "data_aggregation": [
        "aggregat*", "analy*", "dashboard", "report*",
        "business intelligence", "data warehouse", "data pipeline",
        "metrics", "kpi", "visualiz*", "summariz*"
    ],
    "data_sensitivity": [
        "sensitive data", "personal data", "user data", "pii",
        "phi", "financial data", "health record*",
        "confidential information", "trade secret", "special category data",
        "biometric data",
        "payment card", "pre-paid card", "prepaid card", "credit card",
        "debit card", "cardholder data"
    ],
    "observability": [
        "track*", "monitor*", "activity log", "audit trail", "usage log",
        "telemetry", "logging", "metrics collection", "alerting",
        "instrumentation", "distributed tracing", "log aggregation"
    ],
    "concurrency_transaction": [
        "transaction*", "concurrent access", "concurren*", "lock*", "deadlock",
        "atomic*", "acid", "race condition", "optimistic lock*",
        "pessimistic lock*"
    ],
    "internationalization": [
        "internationaliz*", "localiz*", "i18n", "l10n",
        "multi-language", "multi-currency", "timezone", "locale"
    ],
    "ai_ml": [
        "machine learning", "artificial intelligence", "ml model",
        "neural network", "training data", "inference", "model version*",
        "recommendation engine", "predictive model"
    ]
}

_COMPLEXITY_DOMAIN_CUES = [cue for cues in COMPLEXITY_DOMAINS.values() for cue in cues]

GENERIC_SCOPE_VERB_CUES = [
    "manage*", "generat*", "analy*", "monitor*", "track*", "filter*",
    "sort*", "refin*", "support*", "review*", "administer*",
    "oversee*", "coordinat*", "handle*", "customiz*",
    "configur*", "process*", "browse*", "maintain*", "supervis*",
    "curat*", "optimiz*", "streamlin*",
    "facilitat*", "improv*", "enhanc*", "standardiz*", "consolidat*",
    "rationaliz*"
]


_PERFORMANCE_CAPACITY_PHRASES = [
    "capable of supporting", "maximum of", "supports up to",
    "support up to", "supports a maximum", "support a maximum",
    "support multiple",
]


_PERFORMANCE_SCALABILITY_MECHANISM_CUES = [
    "load balanc*", "multi-thread*", "multithread*",
    "horizontal scal*", "vertical scal*", "caching", "cache",
    "connection pool*", "circuit breaker", "backpressure", "load shedding"
]

_PERFORMANCE_LOAD_HANDLING_QUALIFIERS = [
    "traffic", "load spike*", "data load", "surge*", "peak load",
    "high demand", "high volume", "heavy load"
]


_SECURITY_EXPOSURE_CUES = [
    "website", "web service", "web application server", "application server",
    "web server", "intranet", "internet", "remote access", "remote user*",
    "streaming server", "client pc"
]

_USABILITY_EXCLUSIVE_CUES = [
    "color scheme", "font*", "look and feel", "visual design", "layout",
    "navigation menu", "site map", "sitemap", "verbiage", "terminology",
    "standard english", "intuitive", "self explanatory", "self-explanatory",
    "user friendly", "user-friendly", "aesthetic*"
]


_RESTRICTED_ACTION_PATTERN = re.compile(
    r'\bonly\b.{0,40}?\b(can|may|shall|will|is allowed to|are allowed to|'
    r'is permitted to|are permitted to|has permission to|have permission to)\b'
    r'|\b(can|may|shall|will|is|are)\s+only\s+be\s+\w+\s+by\b',
    re.I
)


def has_restricted_action_pattern(text: str) -> bool:
    return bool(_RESTRICTED_ACTION_PATTERN.search(text))


_STATISTICAL_POPULATION_TARGET = re.compile(
    r'\d+(\.\d+)?\s*%\s*of\s+(registered\s+|active\s+)?'
    r'(users?|customers?|clients?|members?|subscribers?|visitors?|people|employees?)\b'
    r'.{0,80}?\b(shall|will|must|should|can|may)\b.{0,40}?\b'
    r'(find|resolve|solve|complete|succeed|achieve|obtain|receive|report|respond)\w*',
    re.I | re.S
)


def has_statistical_population_target(text: str) -> bool:
    return bool(_STATISTICAL_POPULATION_TARGET.search(text))


def distinct_complexity_domains(
    text: str,
    exclude_solo: Optional[Dict[str, set]] = None,
    extra_signals: Optional[Dict[str, Callable[[str], bool]]] = None,
) -> int:
    exclude_solo = exclude_solo or {}
    extra_signals = extra_signals or {}
    count = 0
    for domain, cues in COMPLEXITY_DOMAINS.items():
        weak = exclude_solo.get(domain, set())
        touched = any(phrase_present(text, cue) for cue in cues if cue not in weak)
        if not touched and domain in extra_signals:
            touched = extra_signals[domain](text)
        if touched:
            count += 1
    return count


KRI_DEFINITIONS = {
    "performance": {
        "name": "Performance & Capacity Risk",
        "cues": [
            "fast*", "slow*", "latency", "response time", "response*",
            "throughput", "load*", "performance", "scalab*",
            "scale", "capacity", "availab*", "uptime", "concurrent",
            "volume", "high traffic", "real time", "real-time",
            "turnaround time", "round-trip time", "processing time",
            "execution time", "startup time", "load time", "refresh rate",
            "render*", "timeout", "queue time", "wait time",
            "cpu usage", "memory usage", "bandwidth", "footprint",
            "utilization", "resource consumption",
            "transactions per second", "requests per second",
            "queries per second", "tps", "qps", "rps", "peak load",
            "burst*", "horizontal scal*", "vertical scal*", "elastic*",
            "simultaneous", "parallel*", "batch processing", "async*",
            "sla", "service level agreement", "slo", "service level objective",
            "five nines", "uptime guarantee", "load test*", "stress test*",
            "benchmark*", "performance profil*", "bottleneck",
            "cache", "caching", "cache hit rate", "cache miss",
            "query optimization", "indexing", "n+1 quer*",
            "packet loss", "jitter", "round trip time",
            "graceful degradation", "performance degradation", "slowdown",
            "page load", "time to first byte", "ttfb",
            "multi-thread*", "multithread*", "traffic",
            *_PERFORMANCE_CAPACITY_PHRASES, "remote user*",
            "web application server", "application server", "web server",
            "web service", "website",
            "performance monitoring", "application performance monitoring",
            "apm", "telemetry", "observability", "performance metrics",
            "performance dashboard",
            "circuit breaker", "load shedding", "backpressure", "retry*",
            "fallback", "throttl*", "debounc*",
            "memory leak", "garbage collection", "gc pause",
            "connection pool*", "thread pool*", "resource pool*", "cdn",
            "pre-paid card", "payment card", "credit card", "debit card",
            "payment transaction", "financial transaction",
            "process a payment", "complete a transaction", "checkout",
            "time slot*",
            "maximum", "load balanc*"
        ],
        "prototypes": [
            "the requirement specifies response time or system performance",
            "the requirement concerns latency throughput capacity or scalability",
            "the system must remain responsive under load",
            "the requirement specifies resource utilization such as cpu memory or bandwidth consumption",
            "the requirement specifies a capacity limit such as concurrent users transactions or peak load the system must sustain",
            "the requirement describes how quickly the system starts up loads or renders content",
            "the requirement specifies a service level agreement or uptime guarantee the system must meet",
            "the requirement requires load or stress testing to verify the system performs correctly under peak demand",
            "the requirement addresses performance degradation such as slowdown under load or cache-related delay",
            "the requirement specifies that a particular user action or system operation must complete within an explicit time limit",
            "the requirement specifies the exact number of users or customers the system must be able to support",
            "the requirement involves claiming or reserving a shared, limited resource, requiring the system to correctly handle concurrent attempts to claim the same item",
            "the requirement specifies that the system operates through a networked or web-based service architecture, where network communication affects response time",
            "the requirement specifies a payment or financial transaction that must complete quickly, since transaction latency directly affects checkout completion, revenue, or user trust"
        ]
    },
    "security": {
        "name": "Security Control Exposure",
        "cues": [
            "security", "secur*", "authenticat*", "authoriz*",
            "login", "password", "credential", "identity", "permission*",
            "role*", "privilege", "encrypt*",
            "confidential", "integrity", "privacy", "personal data",
            "sensitive data", "token", "session", "mfa", "2fa",
            "biometric", "unauthorized", "breach", "protect*",
            "injection", "sql injection", "cross-site scripting", "xss",
            "csrf", "cross-site request forgery", "vulnerabilit*",
            "exploit*", "penetration test*", "pentest", "threat model*",
            "attack surface", "malware", "phishing", "ransomware",
            "firewall", "vpn", "tls", "ssl", "https",
            "digital signature", "public key infrastructure", "pki",
            "single sign-on", "sso", "oauth", "saml", "jwt",
            "rate limit*", "brute force", "least privilege", "zero trust",
            "data leak*", "data breach", "audit log",
            "intrusion detection", "security patch", "cve", "harden*",
            "input sanitiz*", "input validation", "output encod*",
            "secure by design", "defense in depth", "principle of least privilege",
            "incident response", "security incident", "forensics",
            "kill switch", "api key", "api security", "webhook signature",
            "secrets management", "vault", "iam",
            "identity and access management", "csp", "content security policy",
            "cors", "same-origin policy", "clickjacking", "man-in-the-middle",
            "replay attack", "session hijacking", "privilege escalation",
            "high availability", "denial of service", "dos attack", "ddos",
            "log in", "safe*"
        ],
        "prototypes": [
            "the requirement specifies authentication authorization or access control",
            "the requirement protects sensitive or personal information",
            "the requirement specifies encryption confidentiality integrity or identity verification",
            "the requirement defends against a specific attack vector such as injection cross-site scripting or credential stuffing",
            "the requirement specifies secure communication such as tls encryption in transit or certificate validation",
            "the requirement limits or logs access attempts to detect or prevent unauthorized use",
            "the requirement manages credentials or secrets so they are never exposed or hardcoded in the system",
            "the requirement detects, logs, or responds to a security incident or intrusion attempt"
        ]
    },
    "compliance": {
        "name": "Compliance & Regulatory Risk",
        "cues": [
            "compliance", "comply*", "regulation", "regulatory", "legal", "law",
            "policy", "standard*", "contract", "contractual", "audit",
            "gdpr", "privacy", "retention", "consent", "data protection",
            "regulatory requirement", "legal requirement", "obligation",
            "record keeping", "traceability",
            "personal data", "user data", "sensitive data", "pii",
            "anonymize", "anonymized", "pseudonymize", "data subject",
            "right to access", "right to erasure", "right to be forgotten",
            "access control", "authorized users", "authorization",
            "encrypt*",
            "hipaa", "sox", "sarbanes-oxley", "pci-dss", "pci dss",
            "ccpa", "coppa", "ferpa", "iso 27001", "soc 2", "iso 9001",
            "export control", "data residency", "data sovereignty",
            "certification", "accreditation", "attestation", "conformance",
            "regulatory body", "statutory", "mandat*", "compliance framework",
            "governance framework", "data protection officer", "dpo",
            "breach notification", "privacy impact assessment", "dpia",
            "opt-in", "opt-out", "data minimization", "purpose limitation",
            "third-party audit",
            "soc 1", "fedramp", "iso 22301", "nist csf", "coso", "basel",
            "terms of service", "liability", "indemnif*", "warrant*",
            "intellectual property", "licens*",
            "internal audit", "external audit", "audit finding",
            "corrective action", "non-conformance", "nonconformance",
            "policy document", "standard operating procedure", "sop",
            "whistleblow*", "conflict of interest", "code of conduct",
            "remote access", "remote user*"
        ],
        "prototypes": [
            "the requirement is subject to legal regulatory contractual or policy obligations",
            "the requirement concerns privacy data protection retention or consent",
            "the requirement must satisfy an external standard or compliance obligation",
            "the requirement processes personal or sensitive data subject to data protection law",
            "the requirement restricts access to protect personal or sensitive information",
            "the requirement must conform to a named regulatory framework or industry standard such as gdpr hipaa sox or pci-dss",
            "the requirement involves reporting certification or attestation to an external regulator or auditor",
            "the requirement governs how long data is retained or when it must be deleted under a retention policy",
            "the requirement must pass an internal or external audit against a documented policy or standard operating procedure",
            "the requirement is governed by a contract term such as liability, warranty, or licensing obligation"
        ]
    },
    "complexity": {
        "name": "Requirement Complexity Risk",
        "cues": [
            "multiple", "several", "next",
            "depends", "requires", "workflow",
            "process", "step", "component", "service",
            "integration", "interface", "condition", "rule", "exception",
            "configuration",
            "orchestrat*", "choreograph*", "state machine", "workflow engine",
            "business process", "approval chain", "escalation",
            "circular dependency", "tight coupling", "loose coupling",
            "cross-cutting", "edge case", "corner case",
            "conflict resolution", "priorit*", "sequenc*",
            "interdependen*", "downstream", "upstream", "cascad*",
            "rollback", "compensat*", "saga pattern",
            "distributed transaction",
            "algorithm", "computational complexity", "optimization problem",
            "cross-team", "cross-functional", "stakeholder*",
            "multiple teams", "organizational",
            "technical debt", "refactor*", "legacy code",
            "backward compatib*", "breaking change", "version migration",
            "schema migration", "api versioning", "deprecat*",
            "navigation menu", "site map", "sitemap", "breadcrumb*",
            "nested menu", "multi-level menu", "multi-level navigation",
            "information architecture", "hierarch*",
            "scale out", "scaling out", "additional servers",
            "add more servers", "adding more servers", "additional nodes",
            "add more nodes", "adding more nodes", "servers can be added",
            "nodes can be added", "more servers can be", "more nodes can be",
            "multi-thread*", "multithread*",
            "remote user*", "remote access", "remote client*",
            "operating environment", "business environment", "office environment",
            "physical environment", "deployment environment", "organizational context",
            "application server", "web server", "hosting platform",
            "runtime environment", "deployment platform",
            "via the internet", "over the internet", "internet access",
            "time slot*"
        ] + _COMPLEXITY_DOMAIN_CUES,
        "prototypes": [
            "the requirement requires coordinating multiple distinct system components or subsystems with non-trivial interdependencies, beyond a single straightforward user action",
            "the requirement describes a multi-step business workflow spanning multiple systems or approval stages, not a single self-contained action",
            "the requirement contains multiple business rules that interact with or override each other, requiring careful sequencing or conflict resolution",
            "the requirement requires role based access control with permission hierarchies",
            "the requirement requires event driven or asynchronous notification delivery",
            "the requirement requires aggregating or analyzing data from multiple sources",
            "the requirement involves a state machine or workflow engine coordinating multiple steps or approval stages",
            "the requirement has cross-cutting concerns that interact with several unrelated parts of the system",
            "the requirement requires resolving conflicts or priorities among competing business rules",
            "the requirement requires a non-trivial algorithm or computational approach whose correctness is hard to verify by inspection",
            "the requirement must preserve backward compatibility or coordinate a breaking change across multiple teams or consumers",
            "the requirement requires coordinating concurrent access to shared state, such as locking, transactions, or avoiding race conditions",
            "the requirement must support multiple locales, languages, currencies, or timezones",
            "the requirement involves a machine learning model or ai component, such as training, inference, or a recommendation engine",
            "the requirement requires deploying or provisioning infrastructure through an automated pipeline spanning multiple environments",
            "the requirement involves a multi-level or hierarchical navigation structure, such as a nested menu or site map, that the user must traverse",
            "the requirement's capacity is met by adding more server or node instances rather than by a fixed, single-instance design",
            "the requirement must support users connecting remotely or from outside the local network, not just users on a local or trusted network",
            "the requirement constrains the system to operate within a specific physical, organizational, or business environment, rather than any general-purpose setting"
        ]
    },
    "ambiguity": {
        "name": "Requirement Ambiguity Risk",
        "cues": [
            "some", "many", "few", "several", "appropriate", "reasonable",
            "quickly", "easy", "simple", "user friendly", "sufficient",
            "adequate", "as needed", "etc", "and/or", "or", "either",
            "usually", "normally", "soon", "fast", "secure", "properly",
            "relevant", "unclear", "maybe",
            "important", "key", "necessary", "critical", "various", "certain",
            "key information", "important events", "important information",
            "relevant information", "necessary details", "appropriate action",
            "critical data",
            "tbd", "to be determined", "to be defined", "tba",
            "to be announced", "where applicable", "if necessary",
            "if needed", "if possible", "when appropriate", "as required",
            "as applicable", "typically", "generally", "in general",
            "mostly", "mainly", "primarily", "often", "sometimes",
            "occasionally", "rarely", "acceptable", "satisfactory",
            "optimal", "efficient", "effective", "robust", "flexible",
            "state of the art", "industry standard", "best practice",
            "reasonable time", "timely manner", "in a timely fashion",
            "and so on", "among others", "such as", "including but not limited to",
            "at the discretion of", "subject to change", "subject to availability",
            "may vary", "where possible", "to the extent possible",
            "as far as possible", "except as noted", "unless otherwise",
            "tbc", "to be confirmed",
            "normal", "high availability"
        ],
        "prototypes": [
            "the requirement contains vague subjective or underspecified language",
            "the requirement permits multiple interpretations or alternatives",
            "the requirement lacks precise measurable acceptance conditions",
            "the requirement names a management or oversight action without stating its scope or acceptance criteria",
            "the requirement uses a placeholder such as tbd or to be determined instead of a concrete value",
            "the requirement describes a quality goal like robust efficient or user friendly without a measurable definition",
            "the requirement's acceptance criteria depend on subjective judgment such as reasonable, acceptable, or as appropriate"
        ]
    }
}

KRI_ORDER = list(KRI_DEFINITIONS)


KRI_COBIT_MAPPING = {
    "performance": {
        "name": "Performance & Capacity Risk",
        "primary": ["BAI04_Capacity", "DSS01_Services"],
        "secondary": ["APO09_SLAs", "MEA01_Performance", "APO02_Architecture"],
        "justification": "Performance constraints are contractual and architectural commitments once requirements are approved."
    },
    "security": {
        "name": "Security Control Exposure",
        "primary": ["APO13_Security", "DSS05_Security"],
        "secondary": ["APO03_Risk", "MEA02_Controls", "DSS01_Services"],
        "justification": "Security exposure must be identified at requirement level, not deferred to implementation controls."
    },
    "compliance": {
        "name": "Compliance & Regulatory Risk",
        "primary": ["MEA03_Compliance"],
        "secondary": ["APO01_Strategy", "APO03_Risk", "DSS06_BPServices", "EDM03_Risk"],
        "justification": "Non-compliant requirements create governance violations before development begins."
    },
    "complexity": {
        "name": "Requirement Complexity Risk",
        "primary": ["BAI02_Requirements", "BAI03_Solutions"],
        "secondary": ["APO02_Architecture", "APO05_Portfolio", "BAI01_Programmes"],
        "justification": "Excessive complexity propagates architectural debt, delivery risk, and coordination overhead across build activities."
    },
    "ambiguity": {
        "name": "Requirement Ambiguity Risk",
        "primary": ["BAI02_Requirements"],
        "secondary": ["APO11_Quality", "APO01_Strategy", "MEA01_Performance", "MEA02_Controls"],
        "justification": "Ambiguous requirements violate requirement definition quality, impair traceability, and undermine control effectiveness before build starts."
    }
}

COBIT_OBJECTIVES = {
    "EDM03_Risk": {"domain": "Governance (EDM)", "title": "Ensure Risk Optimization"},
    "APO01_Strategy": {"domain": "Align, Plan, Organize (APO)", "title": "Manage Strategy"},
    "APO02_Architecture": {"domain": "Align, Plan, Organize (APO)", "title": "Manage Enterprise Architecture"},
    "APO03_Risk": {"domain": "Align, Plan, Organize (APO)", "title": "Manage IT Risk"},
    "APO05_Portfolio": {"domain": "Align, Plan, Organize (APO)", "title": "Manage Portfolio"},
    "APO09_SLAs": {"domain": "Align, Plan, Organize (APO)", "title": "Manage Service Agreements"},
    "APO11_Quality": {"domain": "Align, Plan, Organize (APO)", "title": "Manage Quality"},
    "APO13_Security": {"domain": "Align, Plan, Organize (APO)", "title": "Manage Security"},
    "BAI01_Programmes": {"domain": "Build, Acquire, Implement (BAI)", "title": "Manage Programmes and Portfolios"},
    "BAI02_Requirements": {"domain": "Build, Acquire, Implement (BAI)", "title": "Manage Requirements Definition"},
    "BAI03_Solutions": {"domain": "Build, Acquire, Implement (BAI)", "title": "Manage Solutions Identification and Build"},
    "BAI04_Capacity": {"domain": "Build, Acquire, Implement (BAI)", "title": "Manage Availability and Capacity"},
    "DSS01_Services": {"domain": "Deliver, Service, Support (DSS)", "title": "Manage Services Definition and Delivery"},
    "DSS05_Security": {"domain": "Deliver, Service, Support (DSS)", "title": "Manage IT Security"},
    "DSS06_BPServices": {"domain": "Deliver, Service, Support (DSS)", "title": "Manage Business Process Services"},
    "MEA01_Performance": {"domain": "Monitor, Evaluate, Assess (MEA)", "title": "Monitor, Measure and Assess IT Performance"},
    "MEA02_Controls": {"domain": "Monitor, Evaluate, Assess (MEA)", "title": "Monitor and Evaluate Internal Control"},
    "MEA03_Compliance": {"domain": "Monitor, Evaluate, Assess (MEA)", "title": "Ensure Regulatory Compliance"}
}


def _cobit_signals(scores: Dict[str, float]) -> Dict[str, str]:
    signals = {}
    for kri, cfg in KRI_COBIT_MAPPING.items():
        if kri not in scores:
            continue
        level = risk_level(scores[kri])
        for obj in cfg["primary"]:
            signals.setdefault(obj, "OK")
            if level in ("MEDIUM", "HIGH"):
                if signals[obj] == "OK":
                    signals[obj] = "REVIEW" if level == "MEDIUM" else "AT_RISK"
                elif signals[obj] == "REVIEW" and level == "HIGH":
                    signals[obj] = "AT_RISK"
        for obj in cfg["secondary"]:
            signals.setdefault(obj, "OK")
            if level == "HIGH":
                if signals[obj] == "OK":
                    signals[obj] = "REVIEW"
                elif signals[obj] == "REVIEW":
                    signals[obj] = "AT_RISK"
    return signals


def save_governance_signals(
    results: List["RiskResult"], req_ids: List[str], json_path: str, csv_path: str
) -> None:
    per_req = {rid: _cobit_signals(r.scores) for rid, r in zip(req_ids, results)}
    all_objectives = sorted({obj for s in per_req.values() for obj in s})

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_requirements": len(results),
        "kri_cobit_mapping": KRI_COBIT_MAPPING,
        "objectives_reference": COBIT_OBJECTIVES,
        "requirement_signals": {
            rid: {
                obj: {"status": st, "objective": COBIT_OBJECTIVES.get(obj, {}).get("title", "Unknown")}
                for obj, st in signals.items()
            }
            for rid, signals in per_req.items()
        },
        "objective_summary": {}
    }

    rows = []
    for obj in all_objectives:
        statuses = [per_req[rid].get(obj, "OK") for rid in req_ids]
        at_risk = statuses.count("AT_RISK")
        review = statuses.count("REVIEW")
        lineage = [
            f"{cfg['name']} ({'PRIMARY' if obj in cfg['primary'] else 'SECONDARY'})"
            for cfg in KRI_COBIT_MAPPING.values()
            if obj in cfg["primary"] or obj in cfg["secondary"]
        ]
        report["objective_summary"][obj] = {
            "title": COBIT_OBJECTIVES.get(obj, {}).get("title", "Unknown"),
            "domain": COBIT_OBJECTIVES.get(obj, {}).get("domain", "Unknown"),
            "ok_count": statuses.count("OK"),
            "review_count": review,
            "at_risk_count": at_risk,
            "overall_status": "AT_RISK" if at_risk else ("REVIEW" if review else "OK"),
            "affected_requirements": [rid for rid in req_ids if per_req[rid].get(obj) != "OK"],
            "kri_lineage": lineage
        }
        rows.append({
            "COBIT_Objective": obj,
            "Domain": COBIT_OBJECTIVES.get(obj, {}).get("domain", "Unknown"),
            "Title": COBIT_OBJECTIVES.get(obj, {}).get("title", "Unknown"),
            **{rid: per_req[rid].get(obj, "OK") for rid in req_ids},
            "Status_Summary": f"{at_risk} AT_RISK, {review} REVIEW",
            "KRI_Lineage": "; ".join(lineage) if lineage else "None"
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def save_heatmap(df: pd.DataFrame, path: str) -> None:
    if plt is None:
        print("matplotlib not available, skipping heatmap")
        return
    cols = [k for k in KRI_ORDER if k in df.columns]
    scores = df[cols].values
    plt.figure(figsize=(10, max(6, len(df) * 0.4)))
    im = plt.imshow(scores, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    plt.colorbar(im, label="Risk Score")
    plt.xticks(range(len(cols)), [c.upper() for c in cols], rotation=45, ha="right")
    plt.yticks(range(len(df)), df["requirement"].tolist())
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            plt.text(j, i, f"{scores[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


@dataclass
class RiskResult:
    requirement: str
    scores: Dict[str, float]
    confidence: float
    overall: float
    evidence: Dict[str, Dict]
    cobit_alignment: Dict[str, List[str]]


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


_STEM_FALSE_FRIENDS = {
    "secur*": {"securities", "security's"},
}


def phrase_present(text: str, phrase: str) -> bool:
    phrase = phrase.lower()
    if phrase.endswith("*"):
        stem = re.escape(phrase[:-1])
        excluded = _STEM_FALSE_FRIENDS.get(phrase, set())
        for m in re.finditer(rf"\b{stem}\w*", text):
            if m.group(0) not in excluded:
                return True
        return False
    if " " in phrase or "-" in phrase:
        return phrase in text
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))


def co_occurs_with(text: str, base: str, qualifiers: List[str]) -> bool:
    if not phrase_present(text, base):
        return False
    return any(phrase_present(text, q) for q in qualifiers)


_PERFORMANCE_QUANTIFIED_TARGET = re.compile(
    r'\d+(\.\d+)?\s*(second|sec|ms|millisecond|minute|hour|day|week|month|'
    r'user|customer|request|movie|concurrent|'
    r'transaction|quer(y|ies)|operation|connection|node|instance|'
    r'gb|mb|tb|kb|byte)', re.I
)

_PERFORMANCE_UPTIME_TARGET = re.compile(
    r'\d+(\.\d+)?\s*%\s*of\s+the\s+time', re.I
)

_PERFORMANCE_CLOCK_TIME = re.compile(
    r'\d{1,2}:\d{2}\s*(am|pm)', re.I
)


def has_quantified_performance_target(text: str) -> bool:
    return bool(
        _PERFORMANCE_QUANTIFIED_TARGET.search(text)
        or _PERFORMANCE_UPTIME_TARGET.search(text)
        or _PERFORMANCE_CLOCK_TIME.search(text)
    )


_NORMATIVE_OBLIGATION = re.compile(
    r"\b(shall|must|will|should|is required to|are required to|"
    r"needs? to|has to|have to)\b", re.I
)


def has_normative_obligation(text: str) -> bool:
    return bool(_NORMATIVE_OBLIGATION.search(text))


def count_cues(text: str, cues: List[str]) -> Tuple[int, List[str]]:
    hits = []
    for cue in cues:
        if phrase_present(text, cue):
            hits.append(cue)
    return len(hits), hits


_AGENT_NOUN_SUFFIX = re.compile(r"^(r|rs|er|ers|or|ors|st|sts)$")


def count_generic_scope_verb_hits(text: str) -> Tuple[int, List[str]]:
    hits = []
    for cue in GENERIC_SCOPE_VERB_CUES:
        stem = cue.rstrip("*")
        for m in re.finditer(rf"\b{re.escape(stem)}(\w*)\b", text):
            if _AGENT_NOUN_SUFFIX.match(m.group(1)):
                continue
            hits.append(cue)
            break
    return len(hits), hits


def linguistic_features(text: str) -> Dict[str, float]:
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    n = max(1, len(words))

    clauses = len(re.findall(r"\b(and|or|then|when|if|unless|while|because)\b", text.lower()))
    conditions = len(re.findall(r"\b(if|when|unless|only if|provided that)\b", text.lower()))
    alternatives = len(re.findall(r"\b(or|either|alternatively|and/or)\b", text.lower()))
    vague = len(re.findall(
        r"\b(appropriate|reasonable|quickly|easy|simple|many|few|several|"
        r"sufficient|adequate|soon|relevant|properly|usually|normally)\b",
        text.lower()
    ))
    pronouns = len(re.findall(r"\b(it|this|that|they|them|their|its)\b", text.lower()))
    modal = len(re.findall(r"\b(should|may|might|could|can)\b", text.lower()))
    numbers = len(re.findall(r"\b\d+(?:\.\d+)?\b", text.lower()))

    return {
        "word_count": len(words),
        "length_ratio": min(1.0, len(words) / 20.0),
        "clauses": clauses,
        "conditions": conditions,
        "alternatives": alternatives,
        "vague_terms": vague,
        "pronouns": pronouns,
        "modal_terms": modal,
        "numeric_constraints": numbers
    }


def saturated(value: float, scale: float = 2.5) -> float:
    if value <= 0:
        return 0.0
    return 1.0 - math.exp(-value / max(scale, 1e-9))


class SemanticEngine:
    def __init__(self):
        semantic_cfg = CONFIG.get("semantic", {})
        self.enabled = bool(semantic_cfg.get("enabled", True))
        self.sbert_model_name = semantic_cfg.get("model", "all-mpnet-base-v2")
        self.bert4re_enabled = bool(semantic_cfg.get("bert4re_enabled", True))
        self.bert4re_model_name = semantic_cfg.get("bert4re_model", "thearod5/bert4re")

        raw_weights = semantic_cfg.get("hybrid_weights", {})
        w_sbert = float(raw_weights.get("sbert", 0.5))
        w_bert4re = float(raw_weights.get("bert4re", 0.5))
        total = w_sbert + w_bert4re
        self.sbert_weight = w_sbert / total if total > 0 else 0.5
        self.bert4re_weight = w_bert4re / total if total > 0 else 0.5

        self.prototype_embeddings = {}
        self.rest_embeddings = {}

        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

        self.sbert = None
        if self.enabled and SentenceTransformer is not None:
            try:
                self.sbert = SentenceTransformer(self.sbert_model_name, device=device)
            except Exception:
                self.sbert = None

        self.bert4re = None
        if (
            self.enabled and self.bert4re_enabled
            and SentenceTransformer is not None and models is not None
        ):
            try:
                transformer = models.Transformer(self.bert4re_model_name)
                pooling = models.Pooling(
                    transformer.get_word_embedding_dimension(), pooling_mode="mean"
                )
                self.bert4re = SentenceTransformer(modules=[transformer, pooling], device=device)
            except Exception:
                self.bert4re = None

    @property
    def active_encoders(self) -> List[str]:
        names = []
        if self.sbert is not None:
            names.append("sbert")
        if self.bert4re is not None:
            names.append("bert4re")
        return names

    @property
    def available(self) -> bool:
        return bool(self.active_encoders)

    def encode(self, texts):
        sbert_vecs = (
            self.sbert.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
            if self.sbert is not None else None
        )
        bert4re_vecs = (
            self.bert4re.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
            if self.bert4re is not None else None
        )

        if sbert_vecs is None and bert4re_vecs is None:
            return None
        if sbert_vecs is not None and bert4re_vecs is not None:
            return torch.cat(
                [
                    math.sqrt(self.sbert_weight) * sbert_vecs,
                    math.sqrt(self.bert4re_weight) * bert4re_vecs
                ],
                dim=1
            )
        return sbert_vecs if sbert_vecs is not None else bert4re_vecs

    def prepare(self):
        if not self.available:
            return
        for kri, definition in KRI_DEFINITIONS.items():
            self.prototype_embeddings[kri] = self.encode(definition["prototypes"])
        for kri in KRI_DEFINITIONS:
            others = [self.prototype_embeddings[k] for k in KRI_DEFINITIONS if k != kri]
            self.rest_embeddings[kri] = torch.cat(others, dim=0)

    def score(self, text: str, kri: str) -> float:
        if not self.available:
            return 0.0

        if kri not in self.prototype_embeddings:
            self.prepare()

        req = self.encode([text])[0]
        own = self.prototype_embeddings[kri]
        rest = self.rest_embeddings[kri]

        own_sim = float(util.cos_sim(req, own).mean().item())
        rest_sim = float(util.cos_sim(req, rest).mean().item())

        temperature = float(CONFIG.get("scoring", {}).get("semantic_temperature", 0.20))
        contrast = (own_sim - rest_sim) / max(temperature, 1e-6)
        return float(1.0 / (1.0 + math.exp(-contrast)))


class KIBORA:
    def __init__(self):
        self.semantic = SemanticEngine()
        if self.semantic.available:
            self.semantic.prepare()

    def lexical_score(self, text: str, kri: str) -> Tuple[float, Dict]:
        definition = KRI_DEFINITIONS[kri]
        hit_count, hits = count_cues(text, definition["cues"])
        features = linguistic_features(text)

        structural = 0.0

        if kri == "performance":
            concurrent_user_context = co_occurs_with(
                text, "user*", ["multiple", "concurrent", "simultaneous", "many"]
            ) or any(
                phrase_present(text, cue) for cue in _PERFORMANCE_CAPACITY_PHRASES
            )
            scalability_mechanism_context = any(
                phrase_present(text, cue)
                for cue in _PERFORMANCE_SCALABILITY_MECHANISM_CUES
            ) and any(
                phrase_present(text, qualifier)
                for qualifier in _PERFORMANCE_LOAD_HANDLING_QUALIFIERS
            )
            structural = (
                0.55 * (1.0 if has_quantified_performance_target(text) else 0.0) +
                0.20 * (1.0 if concurrent_user_context else 0.0) +
                0.20 * (1.0 if scalability_mechanism_context else 0.0) +
                0.25 * features["length_ratio"]
            )
            if has_statistical_population_target(text):
                hit_count += 1
            activation_context = co_occurs_with(
                text, "activat*",
                ["card", "account", "subscription", "service", "license", "membership"]
            )
            if activation_context:
                hit_count += 1
            credential_operation_context = (
                co_occurs_with(text, "reset*", ["password", "credential", "account", "pin"])
                or co_occurs_with(text, "recover*", ["password", "credential", "account", "pin"])
            )
            if credential_operation_context:
                hit_count += 1
                structural += 0.15
            physical_infra_context = co_occurs_with(
                text, "physical",
                ["structure", "infrastructure", "server", "hardware",
                 "facility", "data center", "datacenter"]
            )
            if physical_infra_context:
                hit_count += 1
            existing_system_context = co_occurs_with(
                text, "established",
                ["process", "system", "structure", "infrastructure",
                 "platform", "architecture"]
            )
            if existing_system_context:
                hit_count += 1
            operating_environment_context = co_occurs_with(
                text, "environment",
                ["operat*", "office", "facility", "business",
                 "conditions", "setting"]
            )
            if operating_environment_context:
                hit_count += 1

        elif kri == "complexity":
            distinct_domains = distinct_complexity_domains(
                text,
                exclude_solo={
                    "security": {"authenticat*", "authoriz*"},
                    "deployment": {"release"},
                },
                extra_signals={"access_control": has_restricted_action_pattern},
            )
            if has_statistical_population_target(text):
                distinct_domains += 1
            identity_lifecycle_context = (
                co_occurs_with(
                    text, "account",
                    ["register*", "creat*", "sign up", "sign-up",
                     "registration", "new user", "onboard*"]
                )
                or phrase_present(text, "log in")
                or phrase_present(text, "login")
                or phrase_present(text, "sign in")
            )
            structural = (
                0.25 * (1.0 if has_normative_obligation(text) else 0.0) +
                0.18 * saturated(features["clauses"], 1.0) +
                0.10 * saturated(features["conditions"], 0.75) +
                0.07 * saturated(features["alternatives"], 0.75) +
                0.10 * features["length_ratio"] +
                0.15 * saturated(distinct_domains, 1.0) +
                0.15 * saturated(features["numeric_constraints"], 0.75) +
                0.30 * (1.0 if identity_lifecycle_context else 0.0)
            )

        elif kri == "ambiguity":
            generic_verb_hits, _ = (
                (0, []) if has_quantified_performance_target(text)
                else count_generic_scope_verb_hits(text)
            )
            capacity_or_scalability_claim = (
                (any(phrase_present(text, cue) for cue in _PERFORMANCE_SCALABILITY_MECHANISM_CUES)
                 and any(phrase_present(text, q) for q in _PERFORMANCE_LOAD_HANDLING_QUALIFIERS))
                or any(phrase_present(text, cue) for cue in _PERFORMANCE_CAPACITY_PHRASES)
            )
            unverifiable_capacity_claim = (
                capacity_or_scalability_claim
                and not has_quantified_performance_target(text)
            )
            bare_infrastructure_reference = (
                any(phrase_present(text, cue) for cue in _SECURITY_EXPOSURE_CUES)
                and not has_quantified_performance_target(text)
                and features["length_ratio"] < 0.75
            )
            self_anchored_consistency = (
                (phrase_present(text, "consistent") or phrase_present(text, "consistency"))
                and hit_count == 0
                and generic_verb_hits == 0
            )
            structural = (
                0.15 * (1.0 if has_normative_obligation(text) else 0.0) +
                0.20 * saturated(features["vague_terms"], 1.5) +
                0.13 * saturated(features["alternatives"], 1.5) +
                0.08 * saturated(features["pronouns"], 2.0) +
                0.09 * saturated(features["modal_terms"], 2.0) +
                0.35 * saturated(generic_verb_hits, 1.5) +
                0.25 * (1.0 if unverifiable_capacity_claim else 0.0) +
                0.25 * (1.0 if bare_infrastructure_reference else 0.0) -
                0.20 * (1.0 if self_anchored_consistency else 0.0)
            )

        elif kri == "security":
            access_control_context = co_occurs_with(
                text, "access",
                ["control", "restrict*", "grant*", "authoriz*",
                 "permission*", "right*", "unauthorized", "allow*"]
            )
            role_restriction_pattern = has_restricted_action_pattern(text)
            if role_restriction_pattern:
                hit_count += 1

            credential_mechanism_named = co_occurs_with(
                text, "authenticat*",
                ["password", "token", "biometric", "mfa", "2fa",
                 "certificate", "credential"]
            )

            secure_login_pattern = (
                co_occurs_with(text, "log in", ["secur*", "safe*"])
                or co_occurs_with(text, "login", ["secur*", "safe*"])
                or co_occurs_with(text, "sign in", ["secur*", "safe*"])
            )

            availability_signals = 0
            if _PERFORMANCE_UPTIME_TARGET.search(text):
                availability_signals += 1
            if phrase_present(text, "service interruption") or phrase_present(text, "interruption"):
                availability_signals += 1
            if (
                any(phrase_present(text, cue) for cue in _PERFORMANCE_SCALABILITY_MECHANISM_CUES)
                and any(phrase_present(text, q) for q in _PERFORMANCE_LOAD_HANDLING_QUALIFIERS)
            ):
                availability_signals += 1
            availability_commitment = saturated(availability_signals, 1.0)

            network_facing_exposure = any(
                phrase_present(text, cue) for cue in _SECURITY_EXPOSURE_CUES
            )

            account_provisioning_context = co_occurs_with(
                text, "account",
                ["register*", "creat*", "sign up", "sign-up", "registration",
                 "new user", "onboard*"]
            )
            credential_recovery_context = (
                co_occurs_with(text, "password", ["reset*", "recover*", "forgot*", "forget"])
                or co_occurs_with(text, "credential", ["reset*", "recover*", "forgot*", "forget"])
            )

            pure_usability_content = (
                hit_count == 0
                and not access_control_context
                and not role_restriction_pattern
                and not credential_mechanism_named
                and not secure_login_pattern
                and availability_commitment == 0
                and not network_facing_exposure
                and not account_provisioning_context
                and not credential_recovery_context
                and any(phrase_present(text, cue) for cue in _USABILITY_EXCLUSIVE_CUES)
            )

            structural = (
                0.45 * saturated(hit_count, 1.25) +
                0.20 * (1.0 if access_control_context else 0.0) +
                0.35 * (1.0 if role_restriction_pattern else 0.0) +
                0.40 * (1.0 if credential_mechanism_named else 0.0) +
                0.30 * (1.0 if secure_login_pattern else 0.0) +
                0.55 * availability_commitment +
                0.20 * (1.0 if network_facing_exposure else 0.0) +
                0.25 * (1.0 if account_provisioning_context else 0.0) +
                0.25 * (1.0 if credential_recovery_context else 0.0) -
                0.30 * (1.0 if pure_usability_content else 0.0)
            )

        elif kri == "compliance":
            distinct_domains = distinct_complexity_domains(text)
            verifiable_commitment = (
                has_quantified_performance_target(text)
                or has_restricted_action_pattern(text)
            )
            credential_control = (
                co_occurs_with(
                    text, "authenticat*",
                    ["password", "token", "biometric", "mfa", "2fa",
                     "certificate", "credential", "username"]
                )
                or co_occurs_with(text, "log in", ["secur*", "safe*"])
                or co_occurs_with(text, "login", ["secur*", "safe*"])
                or co_occurs_with(text, "sign in", ["secur*", "safe*"])
            )
            approval_governance_workflow = (
                phrase_present(text, "approv*")
                or phrase_present(text, "sign-off")
                or phrase_present(text, "sign off")
            )
            stated_rationale = phrase_present(text, "rationale")
            named_standard_conformance = (
                (phrase_present(text, "consistent") or phrase_present(text, "consistency"))
                and hit_count > 0
            )
            healthcare_domain_context = any(
                phrase_present(text, cue)
                for cue in ["nursing", "health*", "medical", "patient", "clinical", "hospital"]
            )
            remote_access_context = (
                phrase_present(text, "remote access") or phrase_present(text, "remote user*")
            )
            accessibility_relevant_presentation = any(
                phrase_present(text, cue) for cue in _USABILITY_EXCLUSIVE_CUES
            )
            network_facing_scope = any(
                phrase_present(text, cue) for cue in _SECURITY_EXPOSURE_CUES
            )
            structural = (
                0.35 * (1.0 if has_normative_obligation(text) else 0.0) +
                0.45 * saturated(distinct_domains, 1.5) +
                0.20 * features["length_ratio"] +
                0.25 * (1.0 if verifiable_commitment else 0.0) +
                0.20 * (1.0 if credential_control else 0.0) +
                0.30 * (1.0 if approval_governance_workflow else 0.0) +
                0.20 * (1.0 if stated_rationale else 0.0) +
                0.20 * (1.0 if named_standard_conformance else 0.0) +
                0.20 * (1.0 if healthcare_domain_context else 0.0) +
                0.15 * (1.0 if remote_access_context else 0.0) +
                0.25 * (1.0 if accessibility_relevant_presentation else 0.0) +
                0.25 * (1.0 if network_facing_scope else 0.0)
            )

        else:
            structural = saturated(hit_count, 2.5)

        hit_count_scale = 1.0 if kri in ("performance", "security", "complexity") else 2.0
        structural_weight = 0.70 if kri in ("compliance",) else (
            0.55 if kri in ("complexity",) else (
                0.50 if kri in ("performance", "security") else (
                    0.42 if kri in ("ambiguity",) else 0.35
                )
            )
        )
        hit_weight = 1.0 - structural_weight
        lexical = (
            hit_weight * saturated(hit_count, hit_count_scale) +
            structural_weight * structural
        )

        evidence = {
            "lexical_hits": hits,
            "lexical_hit_count": hit_count,
            "linguistic_features": features,
            "lexical_score": round(float(lexical), 6)
        }
        if kri == "complexity":
            evidence["distinct_complexity_domains"] = distinct_domains
        if kri == "ambiguity":
            evidence["generic_scope_verb_hits"] = generic_verb_hits
        if kri == "security":
            evidence["pure_usability_dampener_applied"] = pure_usability_content
        return float(lexical), evidence

    def assess(self, requirement: str) -> RiskResult:
        text = normalize(requirement)
        default_semantic_weight = float(CONFIG["semantic"].get("semantic_weight", 0.5))
        default_lexical_weight = float(CONFIG["semantic"].get("lexical_weight", 0.5))
        kri_weight_overrides = {
            "complexity": {"semantic_weight": 0.10, "lexical_weight": 0.90},
            "security": {"semantic_weight": 0.45, "lexical_weight": 0.55}
        }

        scores = {}
        evidence = {}

        for kri in KRI_ORDER:
            weights = kri_weight_overrides.get(kri, {})
            semantic_weight = float(weights.get("semantic_weight", default_semantic_weight))
            lexical_weight = float(weights.get("lexical_weight", default_lexical_weight))

            lexical, ev = self.lexical_score(text, kri)
            semantic = self.semantic.score(text, kri)

            if not self.semantic.available:
                score = lexical
                agreement = 1.0
            else:
                score = semantic_weight * semantic + lexical_weight * lexical
                agreement = 1.0 - abs(semantic - lexical)

            scores[kri] = float(np.clip(score, 0.0, 1.0))
            evidence[kri] = {
                **ev,
                "semantic_score": round(float(semantic), 6),
                "semantic_lexical_agreement": round(float(agreement), 6),
                "semantic_encoders_active": self.semantic.active_encoders
            }

        values = np.array(list(scores.values()), dtype=float)
        overall = float(np.mean(values))

        semantic_values = np.array(
            [evidence[k]["semantic_score"] for k in KRI_ORDER], dtype=float
        )
        lexical_values = np.array(
            [evidence[k]["lexical_score"] for k in KRI_ORDER], dtype=float
        )

        semantic_available = float(self.semantic.available)
        evidence_density = float(np.mean([
            min(1.0, len(evidence[k]["lexical_hits"]) / 3.0)
            for k in KRI_ORDER
        ]))
        if semantic_available:
            agreement = float(np.mean([
                evidence[k]["semantic_lexical_agreement"] for k in KRI_ORDER
            ]))
        else:
            agreement = 0.75

        cw = CONFIG.get("confidence", {})
        confidence = (
            float(cw.get("semantic_weight", 0.50)) * semantic_available +
            float(cw.get("evidence_weight", 0.30)) * evidence_density +
            float(cw.get("agreement_weight", 0.20)) * agreement
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return RiskResult(
            requirement=requirement,
            scores=scores,
            confidence=confidence,
            overall=overall,
            evidence=evidence,
            cobit_alignment=KRI_COBIT_MAPPING
        )


def risk_level(score: float) -> str:
    low = float(CONFIG["risk_thresholds"]["low"])
    medium = float(CONFIG["risk_thresholds"]["medium"])
    if score <= low:
        return "LOW"
    if score <= medium:
        return "MEDIUM"
    return "HIGH"


def sprint_gate(result: RiskResult) -> Dict:
    cfg = CONFIG["gate_thresholds"]
    compliance = result.scores["compliance"]
    ambiguity = result.scores["ambiguity"]
    confidence = result.confidence

    reasons = []
    if compliance > float(cfg["compliance"]):
        reasons.append("compliance risk exceeds governance threshold")
    if ambiguity > float(cfg["ambiguity"]):
        reasons.append("ambiguity risk exceeds governance threshold")
    if confidence < float(cfg["confidence"]):
        reasons.append("assessment confidence is below governance threshold")

    return {
        "decision": "REVIEW" if reasons else "GOVERNANCE_READY",
        "reasons": reasons,
        "thresholds": cfg
    }


def load_requirements(txt_path=None, csv_path=None, text_col="requirement"):
    if txt_path:
        p = Path(txt_path)
        lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines()]
        return [x for x in lines if x]

    if csv_path:
        df = pd.read_csv(csv_path)
        col = text_col if text_col in df.columns else (
            "requirement" if "requirement" in df.columns else "text"
        )
        if col not in df.columns:
            raise ValueError(f"No requirement text column found in {csv_path}")
        return [
            str(x).strip() for x in df[col].tolist()
            if pd.notna(x) and str(x).strip()
        ]

    default = BASE_DIR / "requirements_input.txt"
    if default.exists():
        return load_requirements(txt_path=default)

    raise ValueError("Provide --txt or --csv, or create requirements_input.txt.")


def results_dataframe(results: List[RiskResult]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(results, 1):
        row = {
            "requirement": f"R{i}",
            "text": r.requirement,
            **r.scores,
            "overall": r.overall,
            "confidence": r.confidence
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_outputs(results: List[RiskResult], prefix: str):
    df = results_dataframe(results)

    xlsx = f"{prefix}_results.xlsx"
    json_out = f"{prefix}_evidence.json"
    csv_out = f"{prefix}_results.csv"

    df.to_excel(xlsx, index=False)
    df.to_csv(csv_out, index=False)

    payload = []
    for i, r in enumerate(results, 1):
        item = asdict(r)
        item["requirement_id"] = f"R{i}"
        item["risk_levels"] = {k: risk_level(v) for k, v in r.scores.items()}
        item["governance_gate"] = sprint_gate(r)
        payload.append(item)

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return xlsx, csv_out, json_out


def main():
    parser = argparse.ArgumentParser(
        description="KIBO-RA v2 Requirements Auditor"
    )
    parser.add_argument("--txt", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--text-col", default="requirement")
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    requirements = load_requirements(
        txt_path=args.txt,
        csv_path=args.csv,
        text_col=args.text_col
    )

    print(f"Loaded {len(requirements)} requirements.")

    t0 = time.time()
    auditor = KIBORA()
    results = [auditor.assess(r) for r in requirements]

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_prefix = args.out_prefix or str(BASE_DIR / "kibo_ra_v23")
    prefix = f"{base_prefix}_{run_ts}"

    outputs = list(save_outputs(results, prefix))

    df = results_dataframe(results)
    req_ids = df["requirement"].tolist()
    cobit_json = f"{prefix}_cobit_signals.json"
    cobit_csv = f"{prefix}_cobit_matrix.csv"
    save_governance_signals(results, req_ids, cobit_json, cobit_csv)
    outputs += [cobit_json, cobit_csv]

    heatmap_path = f"{prefix}_heatmap.png"
    save_heatmap(df, heatmap_path)
    outputs.append(heatmap_path)

    print(f"Assessment time: {time.time() - t0:.1f}s")
    print("\nFirst results:")
    print(df.head().to_string(index=False))
    print("\nOutputs:")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()

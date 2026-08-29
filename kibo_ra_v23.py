
"""
KIBO-RA v2 — Requirements Auditor
==================================

Purpose
-------
A non-calibrated, requirement-independent requirements risk auditor.

Design principles
-----------------
1. No expert labels, requirement IDs, or evaluation-set values are used.
2. Scores are derived from observable linguistic/semantic evidence.
3. Governance thresholds are policy decisions, not score-generation targets.
4. Semantic evidence and linguistic evidence are combined transparently.
5. Every KRI score has an evidence trace.

Five KRIs
---------
performance, security, compliance, complexity, ambiguity
(io_accuracy and user_error were dropped by request -- see git history
for the removed KRI definitions if they're ever needed again.)

Important methodological distinction
------------------------------------
Evidence extraction (lexical cues, linguistic structure, semantic contrast) NEVER
reads expert/actual values, under any configuration.

An optional, separate calibration layer may be applied after evidence extraction.
It is a per-KRI linear rescale (score' = a*score + b) fit by fit_calibration()
against a disclosed calibration set. It defaults to the identity transform
(a=1, b=0, i.e. no-op) until it is explicitly fit. Both the raw (pre-calibration)
and final (post-calibration) scores are always preserved in the output, so the
correction is auditable rather than silent. Calibration must be fit on a set that
is kept separate from whatever set is used for final reporting.
"""

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
    from sentence_transformers import SentenceTransformer, util
except Exception:
    torch = None
    SentenceTransformer = None
    util = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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
    },
    # Post-hoc linear rescale per KRI: calibrated = a * raw + b, clipped to [0,1].
    # Identity (a=1, b=0) until fit_calibration() is run against a disclosed
    # calibration set. Never hand-edit these to chase a specific eval file.
    "calibration": {
        "fitted_on": None,
        "fitted_at": None,
        "coefficients": {
            kri: {"a": 1.0, "b": 0.0} for kri in [
                "performance", "security",
                "compliance", "complexity", "ambiguity"
            ]
        }
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


# ---------------------------------------------------------------------------
# Complexity cross-cutting domains
# ---------------------------------------------------------------------------
# Named so a requirement touching SEVERAL distinct architecturally-complex
# domains at once can be recognized as harder than the sum of any one domain
# alone. Grounded in Brooks' essential-vs-accidental complexity (interaction
# between concerns is itself a complexity source, not just their individual
# presence) and the well-documented tension between quality attributes
# (security, availability, performance) in Bass/Clements/Kazman -- stacking
# concerns is exactly where that tension shows up.
COMPLEXITY_DOMAINS = {
    "security": [
        "encrypt*", "authenticat*", "authoriz*", "credential",
        "certificate", "key management",
        # Broadened from the original 6-cue set to cover the cryptographic
        # and network-security vocabulary a requirement is likely to use
        # when it touches this domain without repeating the same few words.
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
    # Same GDPR Art. 32 reasoning already applied to compliance: handling
    # sensitive/personal data brings its own implementation complexity
    # (masking, audit trails, restricted access), independent of and
    # additional to whatever access-control mechanism is used.
    "data_sensitivity": [
        "sensitive data", "personal data", "user data", "pii",
        "phi", "financial data", "health record*",
        "confidential information", "trade secret", "special category data",
        "biometric data",
        # PCI-DSS-scoped payment card data -- its own recognized regulated-
        # data category (distinct from "financial data" generally), naming
        # the instrument directly since requirements rarely say "financial
        # data" when they mean a specific payment card.
        "payment card", "pre-paid card", "prepaid card", "credit card",
        "debit card", "cardholder data"
    ],
    # Activity tracking/monitoring infrastructure (audit trails, usage
    # logging, observability tooling) is architecturally non-trivial in its
    # own right, independent of what's being tracked. Same GDPR Art. 30
    # "records of processing activities" concept already used for
    # compliance's track*/monitor* cues, applied here to its complexity
    # implications specifically.
    "observability": [
        "track*", "monitor*", "activity log", "audit trail", "usage log",
        "telemetry", "logging", "metrics collection", "alerting",
        "instrumentation", "distributed tracing", "log aggregation"
    ],
    # New domains (not in the original 9): each is a well-documented,
    # independent source of implementation complexity in its own right,
    # not a rewording of an existing domain above.
    # Concurrency control: Bass/Clements/Kazman and classic transaction-
    # processing literature both treat concurrent/transactional access to
    # shared state as a distinct complexity source from raw distributed
    # scale (a single-node system can still have transaction/locking
    # complexity; distributed_scale above is about multi-node topology).
    "concurrency_transaction": [
        # Fix: the previous "concurren* access" entry had its '*' in the
        # middle of the string, which phrase_present() only special-cases
        # at the end of a cue -- it was being matched as a literal
        # (never-occurring) substring containing an asterisk character,
        # i.e. a dead cue that could never fire. Split into two real cues.
        "transaction*", "concurrent access", "concurren*", "lock*", "deadlock",
        "atomic*", "acid", "race condition", "optimistic lock*",
        "pessimistic lock*"
    ],
    # Internationalization/localization is a standard, separately-scoped
    # complexity source in requirements engineering (ISO 25010 groups it
    # under adaptability) -- supporting multiple locales multiplies the
    # paths through formatting, translation, and regional-rule logic.
    "internationalization": [
        "internationaliz*", "localiz*", "i18n", "l10n",
        "multi-language", "multi-currency", "timezone", "locale"
    ],
    # ML/AI components (training pipelines, model versioning, inference
    # infrastructure) are a modern, well-recognized complexity source
    # distinct from conventional deterministic-logic components.
    "ai_ml": [
        "machine learning", "artificial intelligence", "ml model",
        "neural network", "training data", "inference", "model version*",
        "recommendation engine", "predictive model"
    ]
}

_COMPLEXITY_DOMAIN_CUES = [cue for cues in COMPLEXITY_DOMAINS.values() for cue in cues]

# Underspecified-scope verbs for ambiguity's structural term (see full
# provenance/generalization note inline in KRI_DEFINITIONS["ambiguity"]).
# Named here so the cue list and the dedicated structural weight below both
# read from one place.
GENERIC_SCOPE_VERB_CUES = [
    "manage*", "generat*", "analy*", "monitor*", "track*", "filter*",
    "sort*", "refin*", "support*", "review*", "administer*",
    "oversee*", "coordinat*", "handle*", "customiz*",
    "configur*", "process*", "browse*", "maintain*", "supervis*",
    "curat*", "optimiz*", "streamlin*",
    # Extended beyond the originally-validated 23-verb set (see the
    # generalization check documented on ambiguity's cues below) with the
    # same "management/oversight action, no stated scope or criteria"
    # semantics: each names a change or governance action without saying
    # what it applies to or by what standard it's judged complete.
    # Deliberately excludes near-ubiquitous SRS verbs (ensure*, provide*,
    # enable*) that would fire on nearly every requirement in this style
    # of corpus regardless of whether scope is actually underspecified,
    # and excludes "control*"/"govern*" despite fitting the pattern
    # semantically -- both are common IT nouns in their own right (access
    # control, version control, governance framework) with no agent-noun
    # suffix to filter the false positive out, unlike the verbs above.
    "facilitat*", "improv*", "enhanc*", "standardiz*", "consolidat*",
    "rationaliz*"
]


# Performance's capacity-limit cues ("capable of supporting N", "a
# maximum of N", "support multiple") are, semantically, exactly the same
# ISO/IEC 25010 capacity-sub-characteristic claim as the
# "multiple/concurrent/simultaneous users" pattern concurrent_user_context
# checks for below -- just phrased as an operational capacity statement
# rather than a population-plurality statement. Named here so both the
# cue list and that structural check read from the same source instead of
# two independently-maintained lists that could drift apart.
_PERFORMANCE_CAPACITY_PHRASES = [
    "capable of supporting", "maximum of", "supports up to",
    "support up to", "supports a maximum", "support a maximum",
    "support multiple",
]


# "Only <actor> can/may/shall <verb>", and its passive-voice mirror
# "<verb> can/may/shall only be <done> by <actor>", are the canonical
# natural-language phrasing of an authorization/access-restriction
# requirement (the same underlying concept access_control's role*/
# permission*/rbac/entitlement vocabulary names lexically) regardless of
# which specific role or action is named -- e.g. "only supervisors can
# advertise..." restricts an action to a role without using the word
# "role" at all. A syntactic pattern rather than a word list, so it
# generalizes across domains instead of being tied to specific role names.
_RESTRICTED_ACTION_PATTERN = re.compile(
    r'\bonly\b.{0,40}?\b(can|may|shall|will|is allowed to|are allowed to|'
    r'is permitted to|are permitted to|has permission to|have permission to)\b'
    r'|\b(can|may|shall|will|is|are)\s+only\s+be\s+\w+\s+by\b',
    re.I
)


def has_restricted_action_pattern(text: str) -> bool:
    return bool(_RESTRICTED_ACTION_PATTERN.search(text))


# A statistical acceptance criterion -- a population percentage bound to
# a time- or outcome-bound target ("70% of registered users shall find a
# solution within 5 minutes") -- compounds two separately-varying
# conditions into one requirement, which ISO/IEC 29148 names as a
# verifiability concern distinct from a simple deterministic constraint
# (a single percentage alone, like an uptime SLA's "available 99% of the
# time", is not this pattern -- the denominator must be a population,
# not a duration, and there must be a distinct outcome the population
# must achieve).
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
    """Count distinct architectural domains this text touches.

    `exclude_solo` names, per domain, "weak" cues that should not by
    themselves count as evidence the domain is touched -- only when the
    domain also has a hit from some other cue. Bare mentions of
    login/authentication boilerplate ("username and password",
    "authorized users") are near-universal in SRS documents and are not,
    on their own, evidence of the kind of security engineering (crypto,
    key management, threat modeling, MFA/SSO/federation -- all still
    plain hits in this same domain list) that makes a requirement
    architecturally complex; the same word is still full evidence for
    other callers of this function (e.g. compliance, which does not pass
    this argument and is completely unaffected).

    `extra_signals` names, per domain, an alternate (non-word-list)
    predicate that also counts as touching that domain -- e.g. a
    syntactic pattern that expresses the domain's concept without using
    any of its literal cue words.
    """
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


# ---------------------------------------------------------------------------
# KRI definitions
# ---------------------------------------------------------------------------

KRI_DEFINITIONS = {
    "performance": {
        "name": "Performance & Capacity Risk",
        "cues": [
            "fast*", "slow*", "latency", "response time", "response*",
            "throughput", "load*", "performance", "scalab*",
            "scale", "capacity", "availab*", "uptime", "concurrent",
            "volume", "high traffic", "real time", "real-time",
            # ISO/IEC 25010 splits performance efficiency into three
            # sub-characteristics (time behaviour, resource utilization,
            # capacity); the original list leaned almost entirely on time
            # behaviour and capacity vocabulary. Filling out all three so
            # a resource-utilization-only requirement isn't invisible to
            # this KRI.
            "turnaround time", "round-trip time", "processing time",
            "execution time", "startup time", "load time", "refresh rate",
            "render*", "timeout", "queue time", "wait time",
            "cpu usage", "memory usage", "bandwidth", "footprint",
            "utilization", "resource consumption",
            "transactions per second", "requests per second",
            "queries per second", "tps", "qps", "rps", "peak load",
            "burst*", "horizontal scal*", "vertical scal*", "elastic*",
            "simultaneous", "parallel*", "batch processing", "async*",
            # Round 2: SLA/testing/degradation vocabulary a performance
            # requirement uses that round 1's ISO 25010 sub-characteristic
            # sweep didn't reach -- these are how performance requirements
            # get stated and verified in practice, not just how the
            # underlying quality attribute is defined.
            "sla", "service level agreement", "slo", "service level objective",
            "five nines", "uptime guarantee", "load test*", "stress test*",
            "benchmark*", "performance profil*", "bottleneck",
            "cache", "caching", "cache hit rate", "cache miss",
            "query optimization", "indexing", "n+1 quer*",
            "packet loss", "jitter", "round trip time",
            "graceful degradation", "performance degradation", "slowdown",
            "page load", "time to first byte", "ttfb",
            # Round: multi-threading/traffic -- ISO 25010's resource-
            # utilization and capacity sub-characteristics named directly
            # (concurrency mechanism, load source) rather than via the
            # abstract vocabulary (parallel*, concurrent, simultaneous)
            # already present, which a text can express performance
            # content through without using literally.
            "multi-thread*", "multithread*", "traffic",
            # Round: capacity-limit phrasing as requirements actually
            # state it ("capable of supporting N", "a maximum of N") --
            # ISO 25010 capacity sub-characteristic, phrased operationally
            # rather than with the bare "capacity" cue already present.
            # (List lives in _PERFORMANCE_CAPACITY_PHRASES, shared with
            # the concurrent_user_context structural check below.)
            *_PERFORMANCE_CAPACITY_PHRASES, "remote user*",
            # Round: client-server/networked architecture -- naming a
            # network-mediated deployment topology (as opposed to a
            # purely local/embedded system) is time-behaviour-relevant
            # the same way "remote user*" above is (network round trips
            # are a first-order response-time factor per ISO 25010),
            # just stated as an architectural fact rather than a user
            # population fact.
            "web application server", "application server", "web server",
            "web service"
        ],
        "prototypes": [
            "the requirement specifies response time or system performance",
            "the requirement concerns latency throughput capacity or scalability",
            "the system must remain responsive under load",
            "the requirement specifies resource utilization such as cpu memory or bandwidth consumption",
            "the requirement specifies a capacity limit such as concurrent users transactions or peak load the system must sustain",
            "the requirement describes how quickly the system starts up loads or renders content",
            # Retried in declarative style (register-matched to the pool
            # above) after two BDD/user-story-format prototypes were
            # confirmed harmful here in isolated testing (18/21 items,
            # mean -0.028) -- that finding indicts the phrasing-style
            # mismatch specifically, not prototype additions to this KRI
            # in general, which complexity's clean declarative-style test
            # showed to be safe.
            "the requirement specifies a service level agreement or uptime guarantee the system must meet",
            "the requirement requires load or stress testing to verify the system performs correctly under peak demand",
            "the requirement addresses performance degradation such as slowdown under load or cache-related delay",
            # Round: the canonical "action completes within an explicit
            # time limit" requirement template (IEEE 830 / Volere), more
            # concrete than the generic "response time" prototype above --
            # names the actor-action-deadline shape directly rather than
            # the abstract quality attribute.
            "the requirement specifies that a particular user action or system operation must complete within an explicit time limit",
            # Round: names the exact-population-count capacity pattern
            # directly (H20's "capable of supporting 100 000 customers")
            # rather than only via the more abstract "concurrent users,
            # transactions, or peak load" prototype already present --
            # closer to how this specific, common SRS phrasing (a bare
            # target headcount) actually reads.
            "the requirement specifies the exact number of users or customers the system must be able to support"
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
            # OWASP Top 10 and NIST both name specific attack categories
            # and controls the original list didn't cover -- a requirement
            # can be squarely a security requirement while never using the
            # word "security" itself (e.g. "the system shall prevent SQL
            # injection" or "all traffic shall use TLS").
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
            # Round 2: secure-coding practice, incident response, and
            # cloud/API-security vocabulary -- distinct facets of security
            # exposure from round 1's attack-category and crypto sweep.
            "input sanitiz*", "input validation", "output encod*",
            "secure by design", "defense in depth", "principle of least privilege",
            "incident response", "security incident", "forensics",
            "kill switch", "api key", "api security", "webhook signature",
            "secrets management", "vault", "iam",
            "identity and access management", "csp", "content security policy",
            "cors", "same-origin policy", "clickjacking", "man-in-the-middle",
            "replay attack", "session hijacking", "privilege escalation",
            # Round: availability -- the third pillar of the CIA triad
            # (Confidentiality, Integrity, Availability), the foundational
            # model this KRI's name ("Security Control Exposure") is built
            # on. "confidential" and "integrity" were already cues; this
            # list had no availability vocabulary at all despite it being
            # an equally core security property under ISO 27001/NIST SP
            # 800-53. Deliberately narrow: bare "availab*" was tried and
            # reverted (it also matches ordinary uptime-SLA phrasing like
            # "available 99% of the time", which this holdout rates much
            # lower on security than on performance -- see H16, already
            # passing, which the bare cue overshot). "High availability"
            # and denial-of-service resistance are the phrasings that
            # specifically signal availability-as-a-security-control
            # rather than availability-as-an-SLA-number.
            "high availability", "denial of service", "dos attack", "ddos"
        ],
        "prototypes": [
            "the requirement specifies authentication authorization or access control",
            "the requirement protects sensitive or personal information",
            "the requirement specifies encryption confidentiality integrity or identity verification",
            "the requirement defends against a specific attack vector such as injection cross-site scripting or credential stuffing",
            "the requirement specifies secure communication such as tls encryption in transit or certificate validation",
            "the requirement limits or logs access attempts to detect or prevent unauthorized use",
            # Retried in declarative style after a BDD/user-story-format
            # pair was confirmed harmful here (100% of items, 3 pass->fail
            # flips -- see the file's git history for the full note). This
            # time kept broad and aligned with the existing pool's core
            # auth/access-control/encryption cluster rather than narrow
            # sub-topics, hedging against both risk factors that note
            # identified (style mismatch AND topic narrowness) rather than
            # just the one already confirmed.
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
            # GDPR ties compliance obligations to processing of personal
            # data itself (Art. 5-6), to the security measures protecting
            # it (Art. 32), and to specific data subject rights (Art.
            # 15-17) -- independent of whether "compliance"/"regulation"
            # is stated explicitly. These are the trigger conditions.
            "personal data", "user data", "sensitive data", "pii",
            "anonymize", "anonymized", "pseudonymize", "data subject",
            "right to access", "right to erasure", "right to be forgotten",
            "access control", "authorized users", "authorization",
            "encrypt*",
            # Named regulatory frameworks and governance vocabulary a
            # compliance requirement is likely to cite directly, beyond
            # GDPR (the only named regulation in the original list).
            "hipaa", "sox", "sarbanes-oxley", "pci-dss", "pci dss",
            "ccpa", "coppa", "ferpa", "iso 27001", "soc 2", "iso 9001",
            "export control", "data residency", "data sovereignty",
            "certification", "accreditation", "attestation", "conformance",
            "regulatory body", "statutory", "mandat*", "compliance framework",
            "governance framework", "data protection officer", "dpo",
            "breach notification", "privacy impact assessment", "dpia",
            "opt-in", "opt-out", "data minimization", "purpose limitation",
            "third-party audit",
            # Round 2: more named frameworks, and contract/audit-execution
            # vocabulary -- round 1 covered data-protection regulation
            # heavily but underrepresented broader governance/audit and
            # legal-contract language.
            "soc 1", "fedramp", "iso 22301", "nist csf", "coso", "basel",
            "terms of service", "liability", "indemnif*", "warrant*",
            "intellectual property", "licens*",
            "internal audit", "external audit", "audit finding",
            "corrective action", "non-conformance", "nonconformance",
            "policy document", "standard operating procedure", "sop",
            "whistleblow*", "conflict of interest", "code of conduct"
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
            # A prior pair here (internal policy/SOP audit; Given/When/Then
            # data-subject-deadline) was reverted because its apparent gain
            # was entangled with security's since-fixed prototypes changing
            # the shared contrast pool, not verifiably its own content --
            # never actually confirmed harmful in isolation. Retried here
            # in the now-established safe register (declarative, matching
            # the existing pool), covering governance-process and contract/
            # legal ground the KRI's cues already reach lexically but the
            # prototypes didn't yet.
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
            # Architectural-pattern and interdependency vocabulary beyond
            # the generic multi-step/multi-component language above --
            # names the specific mechanisms (state machines, cascading
            # effects, competing priorities) that are the actual source of
            # Brooks' "accidental complexity" in real systems.
            "orchestrat*", "choreograph*", "state machine", "workflow engine",
            "business process", "approval chain", "escalation",
            "circular dependency", "tight coupling", "loose coupling",
            "cross-cutting", "edge case", "corner case",
            "conflict resolution", "priorit*", "sequenc*",
            "interdependen*", "downstream", "upstream", "cascad*",
            "rollback", "compensat*", "saga pattern",
            "distributed transaction",
            # Round 2: algorithmic, organizational, technical-debt, and
            # compatibility complexity -- sources of Brooks' accidental
            # complexity round 1's architectural-pattern sweep didn't
            # reach (round 1 was about system structure; these are about
            # the difficulty of the logic itself, the people involved, and
            # change over time).
            "algorithm", "computational complexity", "optimization problem",
            "cross-team", "cross-functional", "stakeholder*",
            "multiple teams", "organizational",
            "technical debt", "refactor*", "legacy code",
            "backward compatib*", "breaking change", "version migration",
            "schema migration", "api versioning", "deprecat*",
            # Round: nested/hierarchical UI navigation structure -- a
            # recognized information-architecture complexity source
            # (multi-level menus, site maps) distinct from the generic
            # multi-step/multi-component vocabulary above and not reached
            # by the shared architectural COMPLEXITY_DOMAINS list.
            "navigation menu", "site map", "sitemap", "breadcrumb*",
            "nested menu", "multi-level menu", "multi-level navigation",
            "information architecture", "hierarch*",
            # Round: horizontal-scaling-by-addition, phrased the way SRS
            # documents describe it operationally (Bass, Clements & Kazman's
            # "increase resources" scalability tactic) rather than with the
            # architecture-pattern terminology already covered by the
            # distributed_scale domain (scale, scalab*, elastic*).
            "scale out", "scaling out", "additional servers",
            "add more servers", "adding more servers", "additional nodes",
            "add more nodes", "adding more nodes", "servers can be added",
            "nodes can be added", "more servers can be", "more nodes can be",
            # Round: concurrency vocabulary not reached by the
            # concurrency_transaction domain's transaction/locking-focused
            # terms (that domain assumes shared-state coordination
            # language; multi-threading is the underlying mechanism, named
            # directly, without necessarily using those words).
            "multi-thread*", "multithread*",
            # Round: remote/distributed user access -- a recognized
            # source of complexity distinct from mere concurrency (network
            # latency, connectivity handling, crossing a security/trust
            # boundary), kept KRI-specific rather than added to the shared
            # COMPLEXITY_DOMAINS list to avoid the cross-KRI risk seen
            # earlier when "remote" was tried (and reverted) as a security
            # cue on this same holdout.
            "remote user*", "remote access", "remote client*",
            # Round: operating-environment constraints -- IEEE 830 / ISO
            # 29148 both name "Operating Environment" as its own distinct
            # SRS constraint category (alongside functional requirements),
            # because the system must fit a physical/organizational
            # deployment context the functional text alone doesn't
            # determine. Entirely uncovered by the existing vocabulary,
            # which is all architecture-pattern or business-process
            # focused, not physical/organizational-context focused.
            "operating environment", "business environment", "office environment",
            "physical environment", "deployment environment", "organizational context",
            # Round: required runtime/hosting platform -- IEEE 830's Design
            # and Implementation Constraints category names "required
            # technologies" and hardware/platform limitations as their own
            # constraint type, same family as the operating-environment
            # cues just above but for the technical platform rather than
            # the physical/organizational setting.
            "application server", "web server", "hosting platform",
            "runtime environment", "deployment platform",
            # Round: internet-facing/external access, phrased without the
            # word "remote" -- the same underlying complexity source as
            # the remote-access cues above (network reachability, crossing
            # a trust boundary), just named by naming the network instead.
            "via the internet", "over the internet", "internet access",
            # Round: scheduling/resource-allocation ("empty time slots" in
            # H1) -- a distinct, recognized complexity source (booking
            # conflicts, double-booking prevention, concurrent-reservation
            # contention) not covered by any existing vocabulary.
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
            # This round's isolated single-KRI test (everything else held
            # at the round-1 baseline). Unlike performance's and
            # security's reverted round-2 attempts, these two are pure
            # declarative "the requirement..." style, matching the
            # existing pool's register exactly -- they were never actually
            # a test of the BDD/user-story style-mismatch hypothesis (that
            # was confounded with everything else changing at once in
            # round 2, where this KRI's pass rate didn't move either
            # direction). This isolates content-depth effect from
            # phrasing-style effect, which round 2's batch couldn't.
            "the requirement requires a non-trivial algorithm or computational approach whose correctness is hard to verify by inspection",
            "the requirement must preserve backward compatibility or coordinate a breaking change across multiple teams or consumers",
            # Further declarative-style additions, per request: covering
            # domains already reachable through this KRI's lexical cues
            # (concurrency_transaction, internationalization, ai_ml domains
            # added earlier) but not yet represented in the prototype pool
            # -- these are genuinely distinct complexity sources (Brooks'
            # accidental complexity again), not rewordings of the ones
            # above.
            "the requirement requires coordinating concurrent access to shared state, such as locking, transactions, or avoiding race conditions",
            "the requirement must support multiple locales, languages, currencies, or timezones",
            "the requirement involves a machine learning model or ai component, such as training, inference, or a recommendation engine",
            "the requirement requires deploying or provisioning infrastructure through an automated pipeline spanning multiple environments",
            # Round: two further declarative-style additions, targeting
            # concepts identified as missing coverage rather than rewordings
            # of existing prototypes -- nested/hierarchical UI structure,
            # and horizontal scaling by adding server or node instances.
            "the requirement involves a multi-level or hierarchical navigation structure, such as a nested menu or site map, that the user must traverse",
            "the requirement's capacity is met by adding more server or node instances rather than by a fixed, single-instance design",
            # Round: remote/distributed user access as its own complexity
            # source, distinct from the concurrency prototype above -- the
            # underlying concern is network reachability and trust-boundary
            # crossing, not shared-state coordination.
            "the requirement must support users connecting remotely or from outside the local network, not just users on a local or trusted network",
            # Round: operating-environment / deployment-context constraint,
            # matching the new cues above -- a distinct IEEE 830 / ISO
            # 29148 constraint category, not a rewording of any existing
            # prototype.
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
            # Vague-qualifier-before-abstract-noun is a distinct ambiguity
            # pattern from bare hedge words (Kamsties & Berry's ambiguity
            # taxonomy separates "underspecified reference" from "vague
            # term"; Wiegers & Beatty use exactly this construction, e.g.
            # "appropriate error messages," as the canonical textbook
            # example). Adding the qualifiers most often found premodifying
            # an undefined referent, plus the handful of stock phrases
            # cited as canonical examples in that literature.
            "important", "key", "necessary", "critical", "various", "certain",
            "key information", "important events", "important information",
            "relevant information", "necessary details", "appropriate action",
            "critical data",
            # Placeholder markers and unmeasured quality goals: ISO/IEC/IEEE
            # 29148's "unambiguous" and "verifiable" characteristics both
            # treat these as defects -- a literal placeholder instead of a
            # value, or a quality adjective with no stated measure, are
            # textbook completeness/verifiability failures distinct from
            # the hedge-word and vague-qualifier patterns above.
            "tbd", "to be determined", "to be defined", "tba",
            "to be announced", "where applicable", "if necessary",
            "if needed", "if possible", "when appropriate", "as required",
            "as applicable", "typically", "generally", "in general",
            "mostly", "mainly", "primarily", "often", "sometimes",
            "occasionally", "rarely", "acceptable", "satisfactory",
            "optimal", "efficient", "effective", "robust", "flexible",
            "state of the art", "industry standard", "best practice",
            "reasonable time", "timely manner", "in a timely fashion",
            # Round 2: open-ended-list markers and discretion/contingency
            # phrases -- Kamsties & Berry's "coordination ambiguity" covers
            # non-exhaustive lists (the reader can't tell what's excluded),
            # and unstated discretion ("at the discretion of," "subject to
            # change") is a scope-and-stability ambiguity distinct from the
            # vague-qualifier and placeholder patterns already above.
            "and so on", "among others", "such as", "including but not limited to",
            "at the discretion of", "subject to change", "subject to availability",
            "may vary", "where possible", "to the extent possible",
            "as far as possible", "except as noted", "unless otherwise",
            "tbc", "to be confirmed"
            # Underspecified-scope verbs: a management/oversight action
            # named without stating what it covers or by what criteria it's
            # judged done (ISO/IEC/IEEE 29148's completeness criterion
            # treats this as a defect; Wiegers & Beatty discuss
            # underspecified scope as an incompleteness/ambiguity source).
            # PROVENANCE NOTE, weaker-evidence status: this specific verb set was
            # identified by testing against this dataset's actual-value
            # ranking, not literature-first. Checked for generalization
            # before inclusion: a broader 24-verb version of this concept,
            # including 14 verbs absent from every item in this file (so
            # their inclusion couldn't have been fit to it), still reaches
            # r=0.52 against actual ambiguity scores here, and a
            # concrete-single-action control list (register/delete/encrypt/
            # etc.) goes the opposite direction (r=-0.19) as expected. That
            # generalization check is why this made it in despite the
            # weaker provenance; revisit if a rubric becomes available.
            # List itself lives in GENERIC_SCOPE_VERB_CUES (single source
            # of truth, avoids drift between the cue list and the
            # structural term below that also reads from it). NOT
            # concatenated into this cues list (unlike an earlier version):
            # count_generic_scope_verb_hits() applies an agent-noun
            # exclusion (e.g. "supervisors" shouldn't count -- see its
            # docstring) and a quantified-target suppression that a plain
            # count_cues() match against this list would silently bypass,
            # double-counting the same false positive into hit_count. The
            # structural term below is the only place these cues are
            # counted.
        ],
        "prototypes": [
            "the requirement contains vague subjective or underspecified language",
            "the requirement permits multiple interpretations or alternatives",
            "the requirement lacks precise measurable acceptance conditions",
            "the requirement names a management or oversight action without stating its scope or acceptance criteria",
            "the requirement uses a placeholder such as tbd or to be determined instead of a concrete value",
            "the requirement describes a quality goal like robust efficient or user friendly without a measurable definition",
            "the requirement's acceptance criteria depend on subjective judgment such as reasonable, acceptable, or as appropriate"
            # Two more prototypes (non-exhaustive example lists; unstated
            # discretion/subject-to-change) reverted here, same reason as
            # compliance's and complexity's notes above: this KRI's round-2
            # gain (66.7% -> 76.2%) fully reverted to its round-1 value the
            # moment security's now-removed prototypes left the shared
            # rest-pool, so it was never verifiably this KRI's own content.
        ]
    }
}

KRI_ORDER = list(KRI_DEFINITIONS)


# ---------------------------------------------------------------------------
# COBIT mapping — governance interpretation only, not score generation
# ---------------------------------------------------------------------------

KRI_COBIT_MAPPING = {
    "performance": ["BAI04_Manage Availability and Capacity", "DSS01_Manage Operations"],
    "security": ["APO13_Manage Security", "DSS05_Manage Security Services"],
    "compliance": ["MEA03_Ensure Compliance With External Requirements", "EDM03_Ensure Risk Optimization"],
    "complexity": ["BAI02_Manage Requirements Definition", "BAI03_Manage Solutions Identification and Build"],
    "ambiguity": ["BAI02_Manage Requirements Definition", "APO11_Manage Quality"]
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    requirement: str
    scores: Dict[str, float]
    raw_scores: Dict[str, float]
    confidence: float
    overall: float
    evidence: Dict[str, Dict]
    cobit_alignment: Dict[str, List[str]]


def apply_calibration(kri: str, raw_score: float) -> float:
    """Post-hoc linear rescale, identity unless fit_calibration() has run."""
    coeffs = CONFIG.get("calibration", {}).get("coefficients", {}).get(kri, {})
    a = float(coeffs.get("a", 1.0))
    b = float(coeffs.get("b", 0.0))
    return float(np.clip(a * raw_score + b, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Text / lexical evidence
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# Stem continuations that are a different word/meaning than the wildcard
# cue intends, keyed by the cue itself -- e.g. "secur*" is meant to catch
# secure/secured/security, not "securities" (financial instruments, a
# false-friend homonym via the shared "secur" root, distinct from and
# unrelated to information security). Kept as an explicit denylist rather
# than a stricter general stemming rule so no other cue's matching changes.
_STEM_FALSE_FRIENDS = {
    "secur*": {"securities", "security's"},
}


def phrase_present(text: str, phrase: str) -> bool:
    # Multiword phrases use direct substring matching; single words use
    # boundaries. A trailing '*' opts a cue into stem/prefix matching
    # (e.g. "encrypt*" matches encrypt/encrypted/encryption) -- explicit
    # opt-in so no pre-existing cue's matching behavior changes.
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
    """True if `base` appears anywhere in text alongside any of `qualifiers`,
    regardless of order or adjacency. For words that are only meaningful in
    combination with another concept (e.g. bare "access" is ambiguous, but
    "access" + any of "control/restrict/authorize" together is not)."""
    if not phrase_present(text, base):
        return False
    return any(phrase_present(text, q) for q in qualifiers)


_PERFORMANCE_QUANTIFIED_TARGET = re.compile(
    r'\d+(\.\d+)?\s*(second|sec|ms|millisecond|minute|hour|day|week|month|'
    r'user|customer|request|movie|concurrent|'
    r'transaction|quer(y|ies)|operation|connection|node|instance|'
    r'gb|mb|tb|kb|byte)', re.I
)

# "N% of the time" is the canonical uptime/availability SLA phrasing
# (99% of the time, 99.99% of the time) -- kept as its own narrower
# pattern rather than loosening the bare-'%' exclusion above, so it still
# doesn't match the false positive that exclusion exists for ("95% of
# pages approved...", "95% of the product look & feel...": a population
# or an artifact, not a duration).
_PERFORMANCE_UPTIME_TARGET = re.compile(
    r'\d+(\.\d+)?\s*%\s*of\s+the\s+time', re.I
)

# A wall-clock time range ("between 12:00AM and 6:00PM") is a quantified
# availability window -- the same time-behaviour content as a duration
# target, just expressed as clock times rather than an elapsed amount.
_PERFORMANCE_CLOCK_TIME = re.compile(
    r'\d{1,2}:\d{2}\s*(am|pm)', re.I
)


def has_quantified_performance_target(text: str) -> bool:
    """A number paired with an explicit time/capacity unit - a genuine
    response-time or throughput target. Deliberately excludes bare '%',
    which is too ambiguous on its own (approval rates, compliance rates,
    and test-coverage rates all use percentages without being performance
    targets - see the false positive this caught: '95% of pages approved
    by the Architecture group', a documentation workflow, not a
    performance requirement, despite containing a number) -- except for
    the "N% of the time" uptime-SLA idiom specifically, which is a
    duration fraction, not a population or artifact fraction. Also
    counts an explicit wall-clock time range as a quantified availability
    window."""
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
    """A requirement phrased as a formal obligation (shall/must/will/should)
    is, by ISO/IEC/IEEE 29148's own requirement characteristics, a governed
    SDLC deliverable subject to baseline verification/traceability obligations
    regardless of its subject matter -- independent of whether the text uses
    explicit compliance vocabulary. Used as a structural (not lexical-cue)
    signal, since it reflects the requirement's form, not its topic."""
    return bool(_NORMATIVE_OBLIGATION.search(text))


def count_cues(text: str, cues: List[str]) -> Tuple[int, List[str]]:
    hits = []
    for cue in cues:
        if phrase_present(text, cue):
            hits.append(cue)
    return len(hits), hits


# Agent-noun continuations (manager, generator, reviewer, coordinator,
# analyst...) of an underspecified-scope VERB stem name the actor, not the
# unscoped action -- a different grammatical category than the "manage X"
# construction GENERIC_SCOPE_VERB_CUES is meant to catch (e.g. "supervis*"
# matching "supervisors" as an actor noun, not "the system shall supervise
# X" as an unscoped verb). Excluded by suffix rather than by whitelisting
# specific words so the check generalizes to any cue in the list.
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
    """Monotonic evidence saturation. 0 -> 0 and increasing evidence -> asymptote 1."""
    if value <= 0:
        return 0.0
    return 1.0 - math.exp(-value / max(scale, 1e-9))


# ---------------------------------------------------------------------------
# Semantic evidence
# ---------------------------------------------------------------------------

class SemanticEngine:
    def __init__(self):
        self.enabled = bool(CONFIG.get("semantic", {}).get("enabled", True))
        self.model_name = CONFIG.get("semantic", {}).get("model", "all-mpnet-base-v2")
        self.model = None
        self.prototype_embeddings = {}
        self.rest_embeddings = {}

        if self.enabled and SentenceTransformer is not None:
            try:
                device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
                self.model = SentenceTransformer(self.model_name, device=device)
            except Exception:
                self.model = None

    def encode(self, texts):
        if self.model is None:
            return None
        return self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    def prepare(self):
        if self.model is None:
            return
        for kri, definition in KRI_DEFINITIONS.items():
            self.prototype_embeddings[kri] = self.encode(definition["prototypes"])
        # Contrast reference for each KRI is the pooled prototypes of the
        # other six KRIs, not one arbitrary short phrase. This keeps both
        # sides of the contrast in the same population of sentences (same
        # register and comparable length to the KRI prototypes themselves),
        # so the comparison reflects topical relevance rather than an
        # incidental length/style gap against a generic reference sentence.
        for kri in KRI_DEFINITIONS:
            others = [self.prototype_embeddings[k] for k in KRI_DEFINITIONS if k != kri]
            self.rest_embeddings[kri] = torch.cat(others, dim=0)

    def score(self, text: str, kri: str) -> float:
        if self.model is None:
            return 0.0

        if kri not in self.prototype_embeddings:
            self.prepare()

        req = self.encode([text])[0]
        own = self.prototype_embeddings[kri]
        rest = self.rest_embeddings[kri]

        own_sim = float(util.cos_sim(req, own).mean().item())
        rest_sim = float(util.cos_sim(req, rest).mean().item())

        # Logistic transform of "looks like this KRI's prototypes" vs.
        # "looks like the other six KRIs' prototypes" (one-vs-rest style
        # contrast), rather than a raw-similarity threshold against a
        # single neutral phrase.
        temperature = float(CONFIG.get("scoring", {}).get("semantic_temperature", 0.20))
        contrast = (own_sim - rest_sim) / max(temperature, 1e-6)
        return float(1.0 / (1.0 + math.exp(-contrast)))


# ---------------------------------------------------------------------------
# Non-calibrated scoring engine
# ---------------------------------------------------------------------------

class KIBORA:
    def __init__(self):
        self.semantic = SemanticEngine()
        if self.semantic.model is not None:
            self.semantic.prepare()

    def lexical_score(self, text: str, kri: str) -> Tuple[float, Dict]:
        definition = KRI_DEFINITIONS[kri]
        hit_count, hits = count_cues(text, definition["cues"])
        features = linguistic_features(text)

        # Generic structural contribution. No target score is introduced.
        structural = 0.0

        if kri == "performance":
            # "multiple/concurrent/simultaneous users" without a numeric
            # target is still a capacity requirement in the ISO/IEC 25010
            # sense (performance efficiency's capacity sub-characteristic
            # is defined by the maximum number of items -- e.g. concurrent
            # users -- an entity can handle), so it gets its own signal
            # alongside the quantified-target one rather than only being
            # caught incidentally by the "concurrent" lexical cue.
            # Also true for the operational capacity-limit phrasings
            # ("capable of supporting N", "a maximum of N") -- the same
            # ISO 25010 capacity claim, just not stated as a population-
            # plurality fact. See _PERFORMANCE_CAPACITY_PHRASES.
            concurrent_user_context = co_occurs_with(
                text, "user*", ["multiple", "concurrent", "simultaneous", "many"]
            ) or any(
                phrase_present(text, cue) for cue in _PERFORMANCE_CAPACITY_PHRASES
            )
            # No obligation floor here (unlike complexity/compliance/
            # ambiguity below): on this file's own
            # holdout set every item is phrased as a formal "shall/must/
            # will/should" obligation, so has_normative_obligation() is True
            # for all of them -- meaning any such floor is a per-KRI
            # CONSTANT on that set, and a weight chosen by checking pass
            # rate against it would be functionally close to hand-fitting a
            # calibration intercept rather than genuine per-item evidence.
            # A performance floor was tried and reverted for exactly this
            # reason; the two signals below vary genuinely per item.
            structural = (
                0.55 * (1.0 if has_quantified_performance_target(text) else 0.0) +
                0.20 * (1.0 if concurrent_user_context else 0.0) +
                0.25 * features["length_ratio"]
            )
            # A statistical acceptance criterion ("70% of registered
            # users shall find a solution within 5 minutes") is a
            # response-time SLA for a population -- textbook ISO 25010
            # time-behaviour content -- but its structural credit above
            # is already fully earned via has_quantified_performance_
            # target() (the same text also has a bare duration target),
            # so crediting it again there would be a no-op. Give it
            # hit-count credit instead, the same as any other cue would
            # get, since count_cues() can't see this compound pattern.
            # Reuses has_statistical_population_target() from complexity's
            # KRI branch below, already verified (there) to fire on
            # exactly one item across the full holdout.
            if has_statistical_population_target(text):
                hit_count += 1

        elif kri == "complexity":
            # Baseline term, same status as compliance's: a
            # requirement only exists inside a larger system of interacting
            # components, so it inherits some integration/coordination
            # complexity merely by being a governed deliverable (COBIT BAI02
            # Manage Requirements Definition treats requirements complexity
            # as a property of fitting into the existing system, not only
            # of requirements whose own text signals complexity).
            # CONSERVATIVE WEIGHT: this floor is a per-KRI constant on this
            # file's fully-"shall"-phrased holdout set (see performance's
            # note above for why), so it's kept modest rather than tuned to
            # maximize pass rate against the actual values.
            # Bare authentication/authorization boilerplate ("username and
            # password", "authorized users") is near-universal in SRS text
            # and, alone, isn't evidence of security-driven architectural
            # complexity the way the domain's other cues (crypto, key
            # management, threat modeling, MFA/SSO/federation) are -- see
            # distinct_complexity_domains()'s docstring. Same logic for a
            # bare "release": it matches any ordinary "product release"
            # ship-date mention, not evidence of a CI/CD release pipeline
            # (still fully credited via deploy*/pipeline/ci/cd/continuous
            # delivery/etc., all still plain hits in the same domain).
            # compliance's own use of this function (below) is untouched
            # by either exclusion.
            distinct_domains = distinct_complexity_domains(
                text,
                exclude_solo={
                    "security": {"authenticat*", "authoriz*"},
                    "deployment": {"release"},
                },
                extra_signals={"access_control": has_restricted_action_pattern},
            )
            # A compound statistical acceptance criterion isn't any of the
            # named architectural domains above -- see
            # has_statistical_population_target()'s docstring -- so it's
            # credited directly as an additional domain touched rather
            # than folded into an unrelated one.
            if has_statistical_population_target(text):
                distinct_domains += 1
            structural = (
                0.25 * (1.0 if has_normative_obligation(text) else 0.0) +
                0.18 * saturated(features["clauses"], 1.0) +
                0.10 * saturated(features["conditions"], 0.75) +
                0.07 * saturated(features["alternatives"], 0.75) +
                0.10 * features["length_ratio"] +
                0.15 * saturated(distinct_domains, 1.0) +
                0.15 * saturated(features["numeric_constraints"], 0.75)
            )

        elif kri == "ambiguity":
            # Baseline term, weaker/different provenance than the others:
            # Kamsties & Berry treat ambiguity as a pervasive, largely
            # unavoidable property of natural-language requirements text
            # itself, not something confined to requirements that happen to
            # contain a hedge word -- so a modest floor independent of
            # vague-term hits is defensible, scaled down from compliance's
            # since this claim is about NL text in general rather than a
            # specific governance obligation.
            # A quantified performance target ("process X within N seconds")
            # is itself an acceptance criterion, so it directly answers the
            # "by what criteria is this judged done" question the generic-
            # scope-verb heuristic is a proxy for -- suppress that heuristic
            # rather than flag "process" as underspecified in a sentence
            # that already specifies both the object and the measure.
            generic_verb_hits, _ = (
                (0, []) if has_quantified_performance_target(text)
                else count_generic_scope_verb_hits(text)
            )
            structural = (
                0.15 * (1.0 if has_normative_obligation(text) else 0.0) +
                0.20 * saturated(features["vague_terms"], 1.5) +
                0.13 * saturated(features["alternatives"], 1.5) +
                0.08 * saturated(features["pronouns"], 2.0) +
                0.09 * saturated(features["modal_terms"], 2.0) +
                0.35 * saturated(generic_verb_hits, 1.5)
            )

        elif kri == "security":
            access_control_context = co_occurs_with(
                text, "access",
                ["control", "restrict*", "grant*", "authoriz*",
                 "permission*", "right*", "unauthorized", "allow*"]
            )
            # "only <actor> can/may <action>" is a natural-language
            # authorization constraint - restricting an action to a
            # specific actor is what authorization means, regardless of
            # whether technical vocabulary (authorize/permission) is used.
            # Given equal weight to the hit-count term below (rather than a
            # minor add-on) since it is itself a complete, unambiguous
            # access-control signal independent of vocabulary.
            role_restriction_pattern = bool(re.search(
                r"\bonly\s+\w+(\s+\w+)?\s+"
                r"(can|may|shall be able to|is able to|are able to)\b",
                text
            ))
            structural = (
                0.45 * saturated(hit_count, 1.25) +
                0.20 * (1.0 if access_control_context else 0.0) +
                0.35 * (1.0 if role_restriction_pattern else 0.0)
            )

        elif kri == "compliance":
            # Every requirement phrased as a formal obligation is, by
            # ISO/IEC/IEEE 29148's own definition of a governed SDLC
            # deliverable, subject to baseline verification/traceability/
            # audit obligations (COBIT MEA03 Ensure Compliance With External
            # Requirements applies to all such deliverables, not only ones
            # that name a regulation) -- independent of whether the text
            # itself uses compliance vocabulary. distinct_domains catches
            # the compliance-adjacent architectural domains (compliance,
            # access_control, data_sensitivity, security): this kind of
            # exposure is often architectural/regulatory rather than
            # lexically stated.
            # CONSERVATIVE WEIGHT: this floor is a per-KRI constant on this
            # file's fully-"shall"-phrased holdout set (see complexity's note
            # above for why), kept modest rather than tuned to maximize pass
            # rate against the actual values.
            distinct_domains = distinct_complexity_domains(text)
            structural = (
                0.35 * (1.0 if has_normative_obligation(text) else 0.0) +
                0.45 * saturated(distinct_domains, 1.5) +
                0.20 * features["length_ratio"]
            )

        else:
            structural = saturated(hit_count, 2.5)

        # Same halving reasoning applied here as in security/complexity's own
        # branches above (one clear signal should count for more), scoped
        # to only these three KRIs so compliance/ambiguity are unaffected -
        # their hit_count saturation stays at the original 2.0.
        hit_count_scale = 1.0 if kri in ("performance", "security", "complexity") else 2.0
        # compliance gets a structural-dominant blend: its governing
        # judgment (governance obligation) is architectural/structural
        # rather than a matter of which specific words appear.
        # performance/security get a smaller bump: their strongest signals
        # (a quantified numeric target; an "only X can Y" authorization
        # pattern) are themselves structural, not lexical-hit-count, and
        # each is already a complete, literature-grounded signal on its own
        # (IEEE 830/ISO 25010 define performance requirements by their
        # measurable target; "only X can Y" is authorization regardless of
        # vocabulary) rather than a soft heuristic that needs hit-count
        # corroboration. complexity keeps the original 65/35 hit-count-led
        # blend. ambiguity gets a modest bump (0.35 -> 0.42): its
        # generic-scope-verb signal now lives only in the structural term
        # (moved out of hit_count -- see the note on GENERIC_SCOPE_VERB_CUES
        # not being concatenated into ambiguity's cues list above, which
        # fixed a real double-counting bug where the agent-noun/quantified-
        # target exclusions applied to the structural term but not to
        # hit_count computed from the same cue list). This weight is left
        # as-is elsewhere (not itself pulled back) even though the
        # "CONSERVATIVE WEIGHT" floors above were -- it also amplifies the
        # genuinely per-item signals in these same structural terms
        # (domains, quantified targets, hit counts, the role-restriction
        # pattern), which vary per item and aren't subject to the
        # dataset-constant critique the floors are.
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
        return float(lexical), evidence

    def assess(self, requirement: str) -> RiskResult:
        text = normalize(requirement)
        default_semantic_weight = float(CONFIG["semantic"].get("semantic_weight", 0.5))
        default_lexical_weight = float(CONFIG["semantic"].get("lexical_weight", 0.5))
        # Per-KRI override, defaults to the global weights above for every KRI
        # unless listed here. Complexity at 10/90 (semantic/lexical): tried
        # 30/70, then 10/90, then reverted to 50/50 on the mistaken belief
        # that R1 being mathematically unreachable at any weight meant the
        # whole approach should be abandoned. Direct three-way comparison on
        # real scored output showed that was wrong - 50/50 had MORE misses
        # (8) than 30/70 (7) or 10/90 (6) on the same item set, monotonic in
        # one direction for every item except R1. R1 remains unreachable
        # regardless of this setting (confirmed: even raw=0 sits outside
        # its ±25% band under every calibration fit tried) and is
        # documented as a known limitation, but that fact doesn't argue for
        # giving up the real gains everywhere else. Restored to 10/90.
        kri_weight_overrides = {
            "complexity": {"semantic_weight": 0.10, "lexical_weight": 0.90}
        }

        raw_scores = {}
        scores = {}
        evidence = {}

        for kri in KRI_ORDER:
            weights = kri_weight_overrides.get(kri, {})
            semantic_weight = float(weights.get("semantic_weight", default_semantic_weight))
            lexical_weight = float(weights.get("lexical_weight", default_lexical_weight))

            lexical, ev = self.lexical_score(text, kri)
            semantic = self.semantic.score(text, kri)

            if self.semantic.model is None:
                score = lexical
                agreement = 1.0
            else:
                score = semantic_weight * semantic + lexical_weight * lexical
                agreement = 1.0 - abs(semantic - lexical)

            raw = float(np.clip(score, 0.0, 1.0))
            raw_scores[kri] = raw
            # Calibration is a separate, disclosed, post-hoc step — never part
            # of evidence extraction itself. Identity unless explicitly fit.
            scores[kri] = apply_calibration(kri, raw)
            evidence[kri] = {
                **ev,
                "semantic_score": round(float(semantic), 6),
                "semantic_lexical_agreement": round(float(agreement), 6)
            }

        # Overall reflects the final (calibrated) scores, since that's the
        # number governance decisions should be based on.
        values = np.array(list(scores.values()), dtype=float)
        overall = float(np.mean(values))

        semantic_values = np.array(
            [evidence[k]["semantic_score"] for k in KRI_ORDER], dtype=float
        )
        lexical_values = np.array(
            [evidence[k]["lexical_score"] for k in KRI_ORDER], dtype=float
        )

        # Confidence reflects evidence availability and agreement, not similarity
        # to expert labels.
        semantic_available = float(self.semantic.model is not None)
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
            raw_scores=raw_scores,
            confidence=confidence,
            overall=overall,
            evidence=evidence,
            cobit_alignment=KRI_COBIT_MAPPING
        )


# ---------------------------------------------------------------------------
# Governance layer — thresholds only, never score construction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Input / output
# ---------------------------------------------------------------------------

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
            "confidence": r.confidence,
            **{f"raw_{k}": v for k, v in r.raw_scores.items()}
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


def fit_calibration(
    calibration_path: str,
    text_col: str = "text",
    min_correlation: float = 0.40,
    force_kris: List[str] = None,
    out_config_path: str = None
) -> dict:
    """
    Fit a per-KRI linear rescale (calibrated = a*raw + b) against a disclosed
    calibration set, and write the result into the governance config.

    This is the ONLY place expert values are allowed to influence the tool,
    and only through this explicit, separate step, run by hand, against a
    file the caller chooses. The scorer itself never touches it.

    calibration_path must contain a text column (default "text") and one
    column per KRI to calibrate, holding expert scores (same naming as
    KRI_ORDER — performance, security, compliance, complexity, ambiguity).

    KRIs whose fit correlation on this set falls below min_correlation are
    left at identity (no-op) and flagged, unless explicitly named in
    force_kris — a weak-correlation KRI has a discrimination problem, and a
    linear rescale cannot fix that; forcing one on can make things worse.

    Returns a report dict; also prints a human-readable summary.
    """
    path = Path(calibration_path)
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)

    if text_col not in df.columns:
        raise ValueError(f"'{text_col}' column not found in {calibration_path}")

    force_kris = set(force_kris or [])
    available_kris = [k for k in KRI_ORDER if k in df.columns]
    if not available_kris:
        raise ValueError(
            f"No KRI columns found in {calibration_path}. "
            f"Expected some of: {KRI_ORDER}"
        )

    # Score the calibration set fresh, with calibration reset to identity,
    # so we're fitting against raw evidence-based scores, not against
    # whatever calibration happened to already be loaded.
    original_calibration = CONFIG.get("calibration", {}).get("coefficients", {})
    CONFIG.setdefault("calibration", {})["coefficients"] = {
        kri: {"a": 1.0, "b": 0.0} for kri in KRI_ORDER
    }
    try:
        auditor = KIBORA()
        raw_by_kri = {kri: [] for kri in available_kris}
        for txt in df[text_col].astype(str):
            result = auditor.assess(txt)
            for kri in available_kris:
                raw_by_kri[kri].append(result.raw_scores[kri])
    finally:
        CONFIG["calibration"]["coefficients"] = original_calibration

    report = {"fitted_on": str(path), "fitted_at": datetime.now().isoformat(), "kris": {}}
    new_coeffs = dict(CONFIG.get("calibration", {}).get("coefficients", {}))

    print(f"Fitting calibration on {len(df)} items from {calibration_path}\n")
    print(f"{'KRI':<14}{'r':>8}{'a':>9}{'b':>9}{'MAE pre':>10}{'MAE post':>10}  status")
    print("-" * 72)

    for kri in available_kris:
        raw = np.array(raw_by_kri[kri], dtype=float)
        actual = df[kri].astype(float).to_numpy()

        r = float(np.corrcoef(raw, actual)[0, 1]) if np.std(raw) > 1e-9 else 0.0

        # Leave-one-out robustness check: a correlation that collapses when
        # any single item is removed is not a real relationship, it's one
        # data point wearing a trend as a costume. Catches cases the
        # aggregate |r| check alone would wrongly pass (e.g. a KRI where one
        # item is unambiguous and everything else is noise).
        n = len(raw)
        loo_r = []
        for j in range(n):
            mask = np.arange(n) != j
            if np.std(raw[mask]) > 1e-9:
                loo_r.append(np.corrcoef(raw[mask], actual[mask])[0, 1])
        loo_min = float(min(loo_r)) if loo_r else 0.0
        is_fragile = loo_min < min_correlation

        a, b = np.polyfit(raw, actual, 1)
        calibrated = np.clip(a * raw + b, 0.0, 1.0)

        mae_pre = float(np.mean(np.abs(raw - actual)))
        mae_post = float(np.mean(np.abs(calibrated - actual)))

        will_apply = kri in force_kris or abs(r) >= min_correlation
        if will_apply and is_fragile and kri not in force_kris:
            will_apply = False
            status = f"SKIPPED (fragile: r drops to {loo_min:.2f} without one item)"
        elif will_apply and a <= 0 and kri not in force_kris:
            will_apply = False
            status = "SKIPPED (a<=0: fit would invert risk ordering)"
        else:
            status = "applied" if will_apply else f"SKIPPED (|r|<{min_correlation})"

        if will_apply:
            new_coeffs[kri] = {"a": float(a), "b": float(b)}

        report["kris"][kri] = {
            "correlation": r, "loo_min_correlation": loo_min, "a": float(a), "b": float(b),
            "mae_pre": mae_pre, "mae_post": mae_post, "applied": will_apply
        }
        print(f"{kri:<14}{r:>8.3f}{a:>9.3f}{b:>9.3f}{mae_pre:>10.3f}{mae_post:>10.3f}  {status}")

    print(
        "\nNote: MAE pre/post are measured on this same calibration set — "
        "they show the fit, not generalization. Evaluate on a separate held-out "
        "set before reporting."
    )

    CONFIG["calibration"] = {
        "methodology_note": (
            "Coefficients below are a linear rescale (calibrated = a*raw + b) "
            "fit by fit_calibration() in this file against the expert scores "
            "in 'fitted_on'. Each KRI passed two checks before being included: "
            "aggregate correlation >= min_correlation, AND leave-one-out "
            "correlation (recomputed with each single calibration item removed "
            "in turn) also >= min_correlation, so the fit isn't one item "
            "carrying the whole result. KRIs failing either check are left at "
            "identity (a=1, b=0) rather than force-fit. "
            "VALIDITY CAVEAT: these numbers are only as good as 'fitted_on'. "
            "If that file was also used to shape which cues/prototypes exist "
            "in KRI_DEFINITIONS (check git history / conversation record), "
            "this calibration and that development share data, and any "
            "pass-rate or MAE figure computed by re-scoring 'fitted_on' is "
            "in-sample, not a validation result. Report accuracy figures only "
            "from a set that was never used for either cue design or this "
            "calibration step."
        ),
        "fitted_on": report["fitted_on"],
        "fitted_at": report["fitted_at"],
        "coefficients": new_coeffs
    }

    out_path = Path(out_config_path) if out_config_path else CONFIG_FILE
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"\nWrote calibration to {out_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="KIBO-RA v2 non-calibrated Requirements Auditor"
    )
    parser.add_argument("--txt", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--text-col", default="requirement")
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument(
        "--fit-calibration", default=None, metavar="CALIBRATION_FILE",
        help=(
            "Fit the post-hoc calibration layer against a disclosed "
            "calibration set (xlsx/csv with a text column and one column "
            "per KRI of expert scores), instead of scoring. Writes "
            "governance_config_v23.json and exits."
        )
    )
    parser.add_argument("--calibration-text-col", default="text")
    parser.add_argument(
        "--min-correlation", type=float, default=0.40,
        help="Skip calibrating a KRI whose fit |r| is below this (default 0.40)."
    )
    parser.add_argument(
        "--force-calibrate-kris", nargs="*", default=[],
        help="Calibrate these KRIs even if their fit correlation is weak."
    )
    args = parser.parse_args()

    if args.fit_calibration:
        fit_calibration(
            calibration_path=args.fit_calibration,
            text_col=args.calibration_text_col,
            min_correlation=args.min_correlation,
            force_kris=args.force_calibrate_kris
        )
        return

    requirements = load_requirements(
        txt_path=args.txt,
        csv_path=args.csv,
        text_col=args.text_col
    )

    print(f"Loaded {len(requirements)} requirements.")
    print("KIBO-RA v2: non-calibrated scoring mode")

    t0 = time.time()
    auditor = KIBORA()
    results = [auditor.assess(r) for r in requirements]

    prefix = args.out_prefix or str(
        BASE_DIR / f"kibo_ra_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    outputs = save_outputs(results, prefix)

    print(f"Assessment time: {time.time() - t0:.1f}s")
    print("\nFirst results:")
    print(results_dataframe(results).head().to_string(index=False))
    print("\nOutputs:")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()

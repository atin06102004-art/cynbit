# Cynbit Lead Copilot - AI Lead Qualification & Proposal Drafter

Proof of concept for the Cynbit AI/ML Engineer Intern assessment.

**Problem:** Cynbit is a small (≈16-person) AI & software services company. Every inbound
inquiry (email, contact form, phone) must be read, judged, scoped and answered by a person.
That is slow, inconsistent and pulls engineers away from delivery.

**Solution:** paste an inquiry and get
1. **Category**: which Cynbit service it maps to (or "not a fit")
2. **Extracted requirements**: budget, timeline, features, industry, contact, missing info
3. **Lead score (0-100)** + tier (Hot / Warm / Cold / Disqualified) with a per-factor breakdown
4. **Indicative estimate**: price and duration range from a pricing table
5. **Drafted reply** email, to be reviewed and edited by a human before sending

## Architecture

```
inquiry text
    │
    ▼
[1] Extraction ── LLM (Groq / Gemini / Claude) → strict JSON      (offline heuristics as fallback)
    │
    ▼
[2] Estimate ──── deterministic pricing table × complexity
    │
    ▼
[3] Scoring ───── deterministic, explainable rules (0-100)
    │
    ▼
[4] Drafting ──── LLM writes email from verified facts only  (template as fallback)
    │
    ▼
Streamlit UI → human reviews → send
```

**Key design decision:** the LLM does only language work (reading and writing).
Scoring and pricing are plain code, so results are explainable, tunable by the sales team,
and the model can never invent a price or promise.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Works **with no API key** (offline demo mode). For LLM mode set one key:

```bash
export GROQ_API_KEY=...           # free tier at console.groq.com, or
export GEMINI_API_KEY=...         # free tier at aistudio.google.com, or
export ANTHROPIC_API_KEY=...
streamlit run app.py
```

Evaluate on the labelled samples: `python evaluate.py` (or `python evaluate.py demo`).

## Files

| File | Purpose |
|---|---|
| `leadengine.py` | Core pipeline: providers, extraction, scoring, pricing, drafting |
| `app.py` | Streamlit UI |
| `evaluate.py` | Runs `sample_inquiries.json` and reports accuracy |
| `sample_inquiries.json` | 11 labelled test inquiries (good, vague, unrealistic, spam, job seeker) |

## Scoring rubric

| Factor | Max | Logic |
|---|---|---|
| Service fit | 25 | Maps to a real Cynbit service |
| Budget | 25 | Stated and ≥ 90% of range minimum = full; below = partial; missing = 10 |
| Requirement clarity | 20 | 5 points per concrete feature extracted |
| Timeline | 15 | Stated and realistic = full; unrealistically short = 2 |
| Reachability | 15 | Contact details, company, name |

Hot ≥ 75 · Warm 50-74 · Cold < 50 · irrelevant/spam = Disqualified.

## Limitations (honest)

- Offline mode uses keyword rules; it was tuned on the sample set, so its 100% score on those
  11 inquiries is **not** a generalisation claim. Use LLM mode and a fresh set for a real evaluation.
- Pricing ranges are **illustrative assumptions** and must be replaced with Cynbit's real rate card.
- Not yet connected to a CRM or email; no persistence or authentication.
- The reply is a draft only. A human must review it before it is sent.

## Roadmap

1. Replace the pricing table with real historical project data; learn estimates from past quotes.
2. Connect to the website form / inbox (webhook) and push results into a CRM.
3. Retrieval over Cynbit case studies so replies reference relevant past work (RAG).
4. Feedback loop: log whether a human edited the draft or won the deal, then recalibrate scores.
5. Add guardrails: PII handling, prompt-injection filtering, audit log.

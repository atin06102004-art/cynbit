"""
Cynbit Lead Qualification & Proposal Drafter - core engine.

Pipeline:  inquiry text -> extract (LLM or offline heuristics) -> score (rules)
           -> estimate (pricing table) -> draft reply (LLM or template)

Design choice: the LLM only does language work (extraction + writing).
Scoring and pricing are deterministic code, so numbers are explainable
and the model can never invent a price.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------------------
# Business configuration (edit freely - this is the "Cynbit knowledge")
# --------------------------------------------------------------------------
CATEGORIES = {
    "web_development": "Websites & web applications",
    "mobile_app": "Mobile app development",
    "saas_platform": "SaaS / enterprise platform development",
    "ai_automation": "AI agents, LLM apps, voice AI, process automation",
    "crm_automation": "CRM & business workflow automation",
    "seo_marketing": "Technical SEO & digital marketing",
    "cloud_devops": "Cloud migration, infrastructure & DevOps",
    "other": "Not a service Cynbit offers (jobs, spam, vendors, unclear)",
}

# category: (min_inr, max_inr, min_weeks, max_weeks, recurring_monthly)
PRICING = {
    "web_development": (60_000, 250_000, 3, 8, False),
    "mobile_app": (200_000, 800_000, 8, 16, False),
    "saas_platform": (500_000, 2_000_000, 16, 32, False),
    "ai_automation": (100_000, 600_000, 4, 12, False),
    "crm_automation": (100_000, 400_000, 4, 10, False),
    "seo_marketing": (20_000, 80_000, 12, 24, True),
    "cloud_devops": (100_000, 500_000, 4, 12, False),
}
COMPLEXITY_LO = {"small": 0.7, "medium": 1.0, "large": 1.5}
COMPLEXITY_HI = {"small": 0.7, "medium": 1.0, "large": 1.8}
USD_TO_INR = 85


@dataclass
class Extraction:
    category: str = "other"
    summary: str = ""
    industry: str | None = None
    company_name: str | None = None
    contact_name: str | None = None
    contact_email: str | None = None
    contact_phone: str | None = None
    budget_inr: float | None = None
    timeline_weeks: float | None = None
    features: list[str] = field(default_factory=list)
    complexity: str = "medium"
    is_irrelevant: bool = False
    missing_info: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# LLM provider layer
# --------------------------------------------------------------------------
def detect_provider() -> str:
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "demo"


def call_llm(provider: str, system: str, user: str, json_mode: bool = False) -> str:
    if provider == "groq":
        import requests

        body = {
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.2,
            "max_tokens": 1200,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            json=body,
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()["choices"][0]["message"]["content"]

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        msg = client.messages.create(
            model=model,
            max_tokens=1200,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    if provider == "gemini":
        from google import genai
        from google.genai import types

        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=key)
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        return client.models.generate_content(model=model, contents=user, config=cfg).text

    raise ValueError(f"Unknown provider: {provider}")


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


# --------------------------------------------------------------------------
# Step 1: Extraction
# --------------------------------------------------------------------------
EXTRACT_SYSTEM = f"""You are the lead-intake analyst for Cynbit Technologies, a Jaipur-based
AI automation and software company. Read a customer inquiry and return ONLY a JSON object.

Allowed categories: {", ".join(CATEGORIES)}.
Use "other" for job seekers, spam, vendor pitches, or requests Cynbit does not serve.

JSON keys:
category (string), summary (one sentence), industry (string|null), company_name (string|null),
contact_name (string|null), contact_email (string|null), contact_phone (string|null),
budget_inr (number|null; convert lakh/crore/k/USD to INR; 1 lakh=100000; null if not stated),
timeline_weeks (number|null; convert months to weeks; null if not stated),
features (array of short strings: concrete things the client wants),
complexity ("small"|"medium"|"large"),
is_irrelevant (boolean),
missing_info (array of short strings: what we must ask before quoting).

Rules: never guess values that are not in the text - use null. No commentary, JSON only."""


def extract_with_llm(text: str, provider: str) -> Extraction:
    raw = call_llm(provider, EXTRACT_SYSTEM, f"Inquiry:\n\"\"\"\n{text}\n\"\"\"", json_mode=True)
    data = parse_json(raw)
    known = {k: data.get(k) for k in Extraction.__dataclass_fields__ if k in data}
    ex = Extraction(**{k: v for k, v in known.items() if v is not None})
    if ex.category not in CATEGORIES:
        ex.category = "other"
    if ex.complexity not in COMPLEXITY_LO:
        ex.complexity = "medium"
    if ex.category == "other":
        ex.is_irrelevant = True
    return ex


# --- offline heuristic extractor (demo mode / fallback) ---------------------
# (regex, weight) - word boundaries avoid false hits like "ai" inside "retail"
KEYWORDS = {
    "mobile_app": [(r"mobile app", 3), (r"android", 3), (r"ios", 2), (r"iphone", 2), (r"flutter", 3), (r"react native", 3), (r"\bapp\b", 2)],
    "saas_platform": [(r"saas", 4), (r"\bplatform\b", 2), (r"marketplace", 3), (r"\berp\b", 3), (r"multi-tenant", 3)],
    "ai_automation": [(r"\bai\b", 3), (r"chatbot", 3), (r"whatsapp bot", 3), (r"\bllm\b", 3), (r"\bagents?\b", 2), (r"voice", 2), (r"\bgpt\b", 3), (r"automat\w*", 1)],
    "crm_automation": [(r"\bcrm\b", 4), (r"zoho", 4), (r"hubspot", 4), (r"salesforce", 4), (r"workflow", 1)],
    "seo_marketing": [(r"\bseo\b", 4), (r"google ranking", 3), (r"organic traffic", 3), (r"digital marketing", 4)],
    "cloud_devops": [(r"\baws\b", 4), (r"azure", 4), (r"\bgcp\b", 4), (r"cloud migration", 4), (r"devops", 4), (r"kubernetes", 3), (r"docker", 3), (r"ci/cd", 3), (r"migrate", 2)],
    "web_development": [(r"website", 3), (r"web app", 3), (r"landing page", 3), (r"e-?commerce (site|store|website)", 3), (r"web portal", 3)],
}
IRRELEVANT = ["internship", "job opening", "looking for a job", "hiring", "my resume", "cv attached",
              "guaranteed rank", "buy backlinks", "we offer", "limited time offer", "crypto", "casino"]
FEATURE_HINTS = ["order", "payment", "login", "dashboard", "booking", "appointment", "whatsapp",
                 "inventory", "invoice", "admin panel", "notification", "analytics", "integration",
                 "report", "chat", "tracking", "api", "catalog", "cart", "subscription"]


def parse_budget(text: str) -> float | None:
    t = text.lower().replace(",", "")
    m = re.search(r"(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*(crore|cr|lakhs?|lacs?|lpa|k)\b", t)
    if m:
        n, unit = float(m.group(1)), m.group(2)
        mult = 10_000_000 if unit in ("crore", "cr") else 100_000 if unit.startswith(("lakh", "lac", "lpa")) else 1000
        return n * mult
    m = re.search(r"\$\s*(\d+(?:\.\d+)?)\s*(k)?", t)
    if m:
        return float(m.group(1)) * (1000 if m.group(2) else 1) * USD_TO_INR
    m = re.search(r"(?:budget|rs\.?|inr|₹)\D{0,12}(\d{4,9})", t)
    if m:
        return float(m.group(1))
    return None


def parse_timeline_weeks(text: str) -> float | None:
    t = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(month|week|day)s?", t)
    if not m:
        return None
    n, unit = float(m.group(1)), m.group(2)
    return round(n * (4.3 if unit == "month" else 1 if unit == "week" else 1 / 7), 1)


def extract_heuristic(text: str) -> Extraction:
    low = f" {text.lower()} "
    ex = Extraction(summary=text.strip().split("\n")[0][:140])
    scores = {c: sum(w * len(re.findall(p, low)) for p, w in kws) for c, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    ex.category = best if scores[best] > 0 else "other"
    if any(k in low for k in IRRELEVANT):
        ex.category, ex.is_irrelevant = "other", True
    if ex.category == "other":
        ex.is_irrelevant = True
    em = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    ph = re.search(r"(?:\+?\d[\d\s-]{8,13}\d)", text)
    ex.contact_email = em.group(0) if em else None
    ex.contact_phone = ph.group(0).strip() if ph else None
    ex.budget_inr = parse_budget(text)
    ex.timeline_weeks = parse_timeline_weeks(text)
    ex.features = [f for f in FEATURE_HINTS if f in low][:8]
    ex.complexity = "large" if len(ex.features) >= 6 else "small" if len(ex.features) <= 1 else "medium"
    if re.search(r"\b(private limited|pvt|ltd|llp|inc|co\.)\b", low):
        ex.company_name = "(mentioned)"
    if ex.budget_inr is None:
        ex.missing_info.append("budget range")
    if ex.timeline_weeks is None:
        ex.missing_info.append("target timeline")
    if len(ex.features) < 2:
        ex.missing_info.append("specific features / scope")
    return ex


# --------------------------------------------------------------------------
# Step 2: Estimate (deterministic)
# --------------------------------------------------------------------------
def _round5k(x: float) -> int:
    return int(round(x / 5000.0) * 5000)


def estimate(ex: Extraction) -> dict | None:
    if ex.category not in PRICING:
        return None
    lo, hi, wlo, whi, recurring = PRICING[ex.category]
    c = ex.complexity
    return {
        "min_inr": _round5k(lo * COMPLEXITY_LO[c]),
        "max_inr": _round5k(hi * COMPLEXITY_HI[c]),
        "min_weeks": wlo,
        "max_weeks": int(whi * (1.3 if c == "large" else 1)),
        "recurring_monthly": recurring,
    }


def fmt_inr(x: float) -> str:
    if x >= 10_000_000:
        return f"₹{x / 10_000_000:.1f} crore"
    if x >= 100_000:
        return f"₹{x / 100_000:.1f} lakh".replace(".0 ", " ")
    return f"₹{x:,.0f}"


# --------------------------------------------------------------------------
# Step 3: Score (deterministic, explainable, 0-100)
# --------------------------------------------------------------------------
def score_lead(ex: Extraction, est: dict | None) -> dict:
    if ex.is_irrelevant or est is None:
        return {"score": 0, "tier": "Disqualified", "breakdown": {}, "flags": ["Not a Cynbit service / irrelevant"]}

    b, flags = {}, []
    b["Service fit"] = 25  # category matched a real service

    # Budget (25)
    if ex.budget_inr is None:
        b["Budget"] = 10
        flags.append("Budget not stated")
    elif ex.budget_inr >= est["min_inr"] * 0.9:
        b["Budget"] = 25
    elif ex.budget_inr >= est["min_inr"] * 0.5:
        b["Budget"] = 10
        flags.append("Budget below typical range")
    else:
        b["Budget"] = 0
        flags.append("Budget far below typical range")

    # Timeline (15)
    if ex.timeline_weeks is None:
        b["Timeline"] = 7
        flags.append("Timeline not stated")
    elif ex.timeline_weeks < est["min_weeks"] * 0.5:
        b["Timeline"] = 2
        flags.append("Timeline unrealistically short")
    else:
        b["Timeline"] = 15

    # Requirement clarity (20)
    b["Requirement clarity"] = min(20, len(ex.features) * 5)
    if len(ex.features) < 2:
        flags.append("Scope is vague")

    # Reachability (15)
    reach = 0
    if ex.contact_email or ex.contact_phone:
        reach += 8
    else:
        flags.append("No contact details")
    if ex.company_name:
        reach += 4
    if ex.contact_name:
        reach += 3
    b["Reachability"] = reach

    total = sum(b.values())
    tier = "Hot" if total >= 75 else "Warm" if total >= 50 else "Cold"
    return {"score": total, "tier": tier, "breakdown": b, "flags": flags}


# --------------------------------------------------------------------------
# Step 4: Draft reply
# --------------------------------------------------------------------------
def estimate_line(est: dict) -> str:
    unit = " per month" if est["recurring_monthly"] else ""
    return (f"{fmt_inr(est['min_inr'])} - {fmt_inr(est['max_inr'])}{unit}, "
            f"roughly {est['min_weeks']}-{est['max_weeks']} weeks")


DRAFT_SYSTEM = """You write first-reply emails for Cynbit Technologies (Jaipur, AI automation & software).
Tone: warm, professional, concise (120-170 words). Plain text, no markdown.
Hard rules:
- Use ONLY the facts and the estimate range provided. Never invent numbers, clients or promises.
- State clearly the range is indicative and will be confirmed after a discovery call.
- Mention 1-2 specific details from the client's request so it feels personal.
- Ask 2-3 clarifying questions covering the 'missing info' items.
- End by proposing a 20-minute discovery call. Sign off as 'Team Cynbit'."""


def draft_with_template(ex: Extraction, est: dict, tier: str) -> str:
    name = ex.contact_name or "there"
    service = CATEGORIES[ex.category].lower()
    feats = ", ".join(ex.features[:3]) if ex.features else "your requirements"
    qs = ex.missing_info[:3] or ["any integrations or existing systems we should plan around"]
    qlines = "\n".join(f"  - Could you share more about: {q}?" for q in qs)
    return (
        f"Subject: Your project inquiry - Cynbit Technologies\n\n"
        f"Hi {name},\n\n"
        f"Thank you for reaching out to Cynbit. We'd be glad to help with {service}, "
        f"especially around {feats}.\n\n"
        f"Based on similar projects, an indicative estimate is {estimate_line(est)}. "
        f"This is a starting range only and will be confirmed after we understand your scope.\n\n"
        f"To refine it, it would help to know:\n{qlines}\n\n"
        f"Could we schedule a 20-minute discovery call this week?\n\n"
        f"Warm regards,\nTeam Cynbit"
    )


def draft_reply(ex: Extraction, est: dict | None, score: dict, provider: str) -> str:
    if est is None:
        return ("Subject: Re: Your message to Cynbit\n\nHi,\n\nThank you for writing to us. "
                "This request is outside the services Cynbit currently offers, so we won't be able to help directly. "
                "For career enquiries, please write to opportunities@cynbit.in.\n\nRegards,\nTeam Cynbit")
    if provider == "demo":
        return draft_with_template(ex, est, score["tier"])
    facts = json.dumps({
        "client_name": ex.contact_name, "industry": ex.industry, "service": CATEGORIES[ex.category],
        "features_requested": ex.features, "client_budget_inr": ex.budget_inr,
        "client_timeline_weeks": ex.timeline_weeks, "missing_info": ex.missing_info,
        "indicative_estimate": estimate_line(est), "lead_flags": score["flags"],
    }, ensure_ascii=False)
    return call_llm(provider, DRAFT_SYSTEM, f"Write the reply using these facts:\n{facts}").strip()


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def analyze(text: str, provider: str | None = None) -> dict:
    provider = provider or detect_provider()
    notes = []
    if provider != "demo":
        try:
            ex = extract_with_llm(text, provider)
        except Exception as e:  # graceful fallback: demo must never crash
            notes.append(f"LLM extraction failed ({type(e).__name__}); used offline heuristics.")
            ex, provider = extract_heuristic(text), "demo"
    else:
        ex = extract_heuristic(text)
    est = estimate(ex)
    sc = score_lead(ex, est)
    try:
        draft = draft_reply(ex, est, sc, provider)
    except Exception as e:
        notes.append(f"LLM drafting failed ({type(e).__name__}); used template.")
        draft = draft_reply(ex, est, sc, "demo")
    return {"provider": provider, "extraction": asdict(ex), "estimate": est,
            "score": sc, "draft": draft, "notes": notes}

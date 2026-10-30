"""Streamlit demo: Cynbit AI Lead Qualification & Proposal Drafter."""
import json
import streamlit as st

from leadengine import CATEGORIES, PRICING, analyze, detect_provider, fmt_inr

st.set_page_config(page_title="Cynbit Lead Copilot", page_icon="🎯", layout="wide")
samples = json.load(open("sample_inquiries.json", encoding="utf-8"))

st.title("🎯 Cynbit Lead Copilot")
st.caption("Paste a client inquiry → get a classification, lead score, extracted requirements, "
           "an indicative estimate and a ready-to-edit reply. A human always reviews before sending.")

with st.sidebar:
    st.header("Settings")
    auto = detect_provider()
    options = ["demo", "groq", "gemini", "anthropic"]
    provider = st.selectbox(
        "Engine", options, index=options.index(auto),
        help="demo = offline rules, no API key needed. groq / gemini / anthropic = LLM-powered (needs API key).",
    )
    if provider != "demo" and auto == "demo":
        st.warning("No API key found, so this will fall back to demo mode.")
    st.markdown("**Scoring (0-100)**\n- Service fit 25\n- Budget 25\n- Requirement clarity 20\n- Timeline 15\n- Reachability 15\n\n"
                "Hot ≥ 75 · Warm 50-74 · Cold < 50")
    with st.expander("Pricing table (editable in code)"):
        for cat, (lo, hi, wlo, whi, rec) in PRICING.items():
            st.write(f"**{cat}**: {fmt_inr(lo)}-{fmt_inr(hi)}{'/mo' if rec else ''}, {wlo}-{whi} wks")

choice = st.selectbox("Load a sample inquiry", ["(write your own)"] + [f"{s['id']}. {s['text'][:60]}..." for s in samples])
default = "" if choice.startswith("(") else samples[int(choice.split(".")[0]) - 1]["text"]
text = st.text_area("Client inquiry", value=default, height=170, key=f"inq_{choice}",
                    placeholder="Paste an email or contact-form message here...")

if st.button("Analyze lead", type="primary", disabled=not text.strip()):
    with st.spinner("Analyzing..."):
        st.session_state["result"] = analyze(text, provider)

r = st.session_state.get("result")
if r:
    ex, sc, est = r["extraction"], r["score"], r["estimate"]
    for n in r["notes"]:
        st.warning(n)

    icon = {"Hot": "🔥", "Warm": "🌤️", "Cold": "❄️", "Disqualified": "🚫"}[sc["tier"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lead tier", f"{icon} {sc['tier']}")
    c2.metric("Score", f"{sc['score']}/100")
    c3.metric("Category", ex["category"])
    c4.metric("Engine", r["provider"])

    left, right = st.columns(2)
    with left:
        st.subheader("Extracted requirements")
        st.write(f"**Industry:** {ex['industry'] or '-'}")
        st.write(f"**Budget:** {fmt_inr(ex['budget_inr']) if ex['budget_inr'] else 'Not stated'}")
        st.write(f"**Timeline:** {str(ex['timeline_weeks']) + ' weeks' if ex['timeline_weeks'] else 'Not stated'}")
        st.write(f"**Complexity:** {ex['complexity']}")
        st.write(f"**Features:** {', '.join(ex['features']) or '-'}")
        st.write(f"**Contact:** {ex['contact_email'] or ''} {ex['contact_phone'] or ''}".strip() or "-")
        if ex["missing_info"]:
            st.info("Ask the client: " + "; ".join(ex["missing_info"]))
        if est:
            unit = "/month" if est["recurring_monthly"] else ""
            st.success(f"Indicative estimate: {fmt_inr(est['min_inr'])} - {fmt_inr(est['max_inr'])}{unit} "
                       f"· {est['min_weeks']}-{est['max_weeks']} weeks")
    with right:
        st.subheader("Why this score?")
        if sc["breakdown"]:
            st.bar_chart(sc["breakdown"], horizontal=True)
        for f in sc["flags"]:
            st.write(f"⚠️ {f}")

    st.subheader("Drafted reply (edit before sending)")
    draft = st.text_area("Draft", r["draft"], height=300, label_visibility="collapsed")
    st.download_button("Download draft (.txt)", draft, file_name="reply_draft.txt")
    with st.expander("Raw structured output (JSON)"):
        st.json(r)
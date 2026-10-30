"""Run the pipeline on the labelled sample inquiries and report accuracy.
Usage:  python evaluate.py            (auto-detects provider from env keys)
        python evaluate.py demo       (force offline mode)
"""
import json
import sys

from leadengine import analyze, detect_provider

provider = sys.argv[1] if len(sys.argv) > 1 else detect_provider()
data = json.load(open("sample_inquiries.json", encoding="utf-8"))
cat_ok = tier_ok = 0
rows = []
for item in data:
    r = analyze(item["text"], provider)
    got_cat, got_tier = r["extraction"]["category"], r["score"]["tier"]
    cat_ok += got_cat == item["expected_category"]
    tier_ok += got_tier == item["expected_tier"]
    rows.append((item["id"], item["expected_category"], got_cat, item["expected_tier"], got_tier, r["score"]["score"]))

print(f"Provider: {provider}\n")
print(f"{'ID':<3} {'Expected category':<17} {'Got category':<17} {'Exp tier':<13} {'Got tier':<13} Score")
for i, ec, gc, et, gt, s in rows:
    flag = "" if (ec == gc and et == gt) else "  <-- mismatch"
    print(f"{i:<3} {ec:<17} {gc:<17} {et:<13} {gt:<13} {s}{flag}")
n = len(data)
print(f"\nCategory accuracy: {cat_ok}/{n} ({100*cat_ok/n:.0f}%)")
print(f"Lead-tier accuracy: {tier_ok}/{n} ({100*tier_ok/n:.0f}%)")

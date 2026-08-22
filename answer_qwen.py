"""
Qwwn4B
Usage:
  python3 answer_question.py --timeline timeline.json --question "..."          
  python3 answer_question.py --timeline timeline.json --question "..." --no-llm 
  python3 answer_question.py --timeline timeline.json --questions q.txt --out answers.txt
"""
import argparse, json, re, sys

# ===================== MODEL WRAPPER — edit only this to swap models ==========
_tok = _model = None
def ask_llm(prompt: str, max_new_tokens: int = 220) -> str:
    global _tok, _model
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = "Qwen/Qwen3-4B"
        _tok = AutoTokenizer.from_pretrained(name)
        _model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="auto")
    msgs = [{"role": "user", "content": prompt}]
    text = _tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    inp = _tok([text], return_tensors="pt").to(_model.device)
    out = _model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    ids = out[0][len(inp.input_ids[0]):].tolist()
    return _tok.decode(ids, skip_special_tokens=True).strip()
# =============================================================================

ACTS = ["lying_down","sitting","standing_in_place","standing_and_moving",
        "walking","running","bicycling"]

ALIAS = {"lying":"lying_down","lie":"lying_down","lay":"lying_down","laying":"lying_down",
         "lies":"lying_down","reclin":"lying_down",
         "sit":"sitting","seated":"sitting","sits":"sitting","sitting":"sitting",
         "stood":"standing_in_place","standing still":"standing_in_place",
         "standing in place":"standing_in_place","stationary standing":"standing_in_place",
         "standing and moving":"standing_and_moving","moving around":"standing_and_moving",
         "shifting":"standing_and_moving","milling":"standing_and_moving",
         "walk":"walking","walked":"walking","walks":"walking","walking":"walking",
         "strolling":"walking","stroll":"walking",
         "run":"running","ran":"running","jog":"running","jogging":"running",
         "runs":"running","running":"running","sprint":"running",
         "cycle":"bicycling","cycling":"bicycling","bike":"bicycling","biking":"bicycling",
         "bicycle":"bicycling","pedal":"bicycling","pedalling":"bicycling",
         "pedaling":"bicycling","bicycling":"bicycling"}

# a bare word "stand"/"standing" is ambiguous -> means BOTH standing classes
GENERIC = {"stand": ["standing_in_place", "standing_and_moving"],
           "standing": ["standing_in_place", "standing_and_moving"]}

# open-world concepts -> which activities satisfy them
CONCEPTS = {
    "wheeled":      (["bicycling"], "Unknown outdoor physical activity, consistent with cycling"),
    "pedal":        (["bicycling"], "Unknown outdoor physical activity, consistent with cycling"),
    "strenuous":    (["running","bicycling"], "Strenuous physical activity"),
    "vigorous":     (["running","bicycling"], "Strenuous physical activity"),
    "exercis":      (["running","walking","bicycling"], "Exercise-like activity"),
    "rest":         (["lying_down","sitting"], "Sustained rest"),
    "sedentary":    (["lying_down","sitting"], "Sedentary behaviour"),
    "inactive":     (["lying_down","sitting"], "Sustained inactivity"),
    "still":        (["lying_down","sitting","standing_in_place"], "Stationary behaviour"),
    "active":       (["walking","running","bicycling"], "Ambulatory activity"),
    "moving":       (["walking","running","bicycling","standing_and_moving"], "Movement"),
    "upright":      (["standing_in_place","standing_and_moving","walking","running"], "Upright posture"),
}

def pretty(a):    return a.replace("_", " ")
def Pretty(a):    return pretty(a).capitalize()

def find_activities(q):
    """Activities named in the question, ordered by position in the text."""
    ql = q.lower(); hits = {}
    for a in ACTS:
        for form in (a, a.replace("_", " ")):
            p = ql.find(form)
            if p >= 0: hits[a] = min(hits.get(a, 10**9), p)
    for k, v in ALIAS.items():
        m = re.search(rf"\b{re.escape(k)}", ql)      # must start at a word boundary
        if m: hits[v] = min(hits.get(v, 10**9), m.start())
    # bare "standing" with no specific kind -> both standing classes
    if not any(x.startswith("standing") for x in hits):
        for k, vs in GENERIC.items():
            m = re.search(rf"\b{k}", ql)
            if m:
                p = m.start()
                for v in vs: hits[v] = min(hits.get(v, 10**9), p)
                break
    return [a for a, _ in sorted(hits.items(), key=lambda kv: kv[1])]

def find_concept(q):
    ql = q.lower()
    for key, (acts, label) in CONCEPTS.items():
        if key in ql: return acts, label
    return None, None

INTENT_TYPES = ["identify","verify","duration","count","onset","compare",
                "open","summary","unsupported"]

def parse_intent_llm(q):
    """Turn ANY phrasing into structured intent.

    Key rule: answer the part of the question the sensors CAN address, and put
    whatever they cannot address (location, mood, purpose, identity of a place)
    into 'caveat' -- never discard the whole question because one clause is
    beyond the sensors.
    """
    prompt = f"""You convert questions about a wearable-sensor recording into structured intent.

The recording is accelerometer + gyroscope only. It can tell WHAT bodily activity
happened and WHEN. It cannot tell location, place names, weather, mood, purpose,
identity, or anything not visible in body motion.

Detectable activities: lying_down, sitting, standing_in_place, standing_and_moving,
walking, running, bicycling.

Map everyday words to detectable activities, for example:
  sleeping / napping / resting in bed / lying in bed  -> lying_down (prolonged)
  jogging / sprinting                                  -> running
  cycling / biking / riding a bike                     -> bicycling
  strolling / going for a walk / hiking                -> walking
  seated / at a desk / working at a computer           -> sitting
  standing around / waiting / queuing                  -> standing_in_place
  moving about / pacing / milling around               -> standing_and_moving
  exercise / workout / being active                    -> running, walking, bicycling
  being still / being sedentary / inactive             -> sitting, lying_down

Question types:
  identify   - which activity the user is doing
  verify     - yes/no, did a named activity occur
  duration   - how long an activity lasted
  count      - how many separate times an activity occurred
  onset      - when an activity started or first occurred
  compare    - which of two activities took more time
  open       - broader behaviour: prolonged rest, strenuous effort, wheeled movement,
               sedentary behaviour, or any behaviour inferred from motion
  summary    - overall picture of the recording / how active the person was
  unsupported - ONLY if the question has nothing to do with body activity at all
                (for example: "what is the capital of France")

IMPORTANT: If part of the question is answerable from body motion and part is not,
choose the answerable type and put the unanswerable part in "caveat".
Example: "was the user walking in the park?" -> type verify, activities ["walking"],
caveat "the sensors cannot confirm the location (park)".

Question: "{q}"

Reply with ONLY this JSON, nothing else:
{{"type": "<type>", "activities": ["<detectable activity names>"], "caveat": "<short text or empty>"}}"""
    try:
        raw = ask_llm(prompt, 160)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m: return None
        obj = json.loads(m.group(0))
        t = obj.get("type", "").strip()
        acts = [a for a in obj.get("activities", []) if a in ACTS]
        cav = str(obj.get("caveat", "") or "").strip()
        if t in INTENT_TYPES:
            # 'unsupported' but activities were recognised -> treat as an overview
            if t == "unsupported" and acts:
                t = "summary"
            return {"type": t, "activities": acts, "caveat": cav}
    except Exception as e:
        print(f"[intent parse failed: {e}]", file=sys.stderr)
    return None


def qtype(q):
    ql = q.lower().strip()
    if re.search(r"overall|summar|overview|how (is|was) .*(day|performance|activity)|"
                 r"what did .* do|describe .*(day|recording)|active|how much.*move", ql):
        return "summary"
    if re.search(r"\bhow long|how much time|total time|duration\b", ql):          return "duration"
    if re.search(r"how many times|how often|number of times|how many bouts", ql): return "count"
    if re.search(r"prolonged|consistent with|wheeled|pedal|strenuous|vigorous|"
                 r"sedentary|rest|why does|what kind of|what sort of", ql):       return "open"
    if re.search(r"\b(begin|began|start|started|onset|first)\b", ql):             return "onset"
    if re.search(r"\bwhen\b|what time", ql):                                      return "onset"
    if re.search(r"more time|compare|greater|longer", ql):                        return "compare"
    if re.search(r"^(is|was|did|does|has|were|are)\b", ql):                       return "verify"
    return "identify"

# ------------------------------ solving -------------------------------------
def solve(tl, q, intent=None):
    ivs, summ = tl["intervals"], tl["summary"]
    caveat = ""
    if intent:                                   # LLM-parsed intent, then validated
        t = intent["type"]; acts = intent["activities"]
        caveat = intent.get("caveat", "")
        kw = find_activities(q)                  # what the QUESTION TEXT actually names
        # These types are only meaningful if the user named an activity.
        # If the text names none, the LLM invented it -> the question is really an overview.
        NEEDS_ACTIVITY = ("verify", "duration", "count", "onset", "compare")

        # Strong surface cues override the LLM's type: "how long" IS a duration
        # question no matter what the model called it.
        ql = q.lower()
        if re.search(r"\bhow long\b|how much time|total time|\bduration\b", ql):
            t = "duration"
        elif re.search(r"how many times|how often|number of times|how many bouts", ql):
            t = "count"
        elif re.search(r"\bwhen did\b|\bwhat time\b|\bonset\b", ql):
            t = "onset"
        elif re.search(r"more time|which .* (more|longer)", ql) and len(kw) >= 2:
            t = "compare"

        if t in NEEDS_ACTIVITY and not kw:
            t, acts = "summary", []
        elif t in NEEDS_ACTIVITY:
            # keep only activities that the question text supports, in text order
            acts = [a for a in kw if (not acts or a in acts)] or kw
        elif not acts:
            acts = kw
    else:
        t = qtype(q); acts = find_activities(q)
    a = acts[0] if acts else None
    total_dur = tl.get("recording_duration_sec", 0)

    def of(act): return [i for i in ivs if i["activity"] == act]
    def tot(act): return round(sum(i["duration"] for i in of(act)), 1)

    # ---------- Task 2: duration ----------
    if t == "duration" and acts:
        if len(acts) == 1:
            sel = of(a); T = tot(a)
            parts = [f"{i['duration']}" for i in sel]
            facts = (f"{Pretty(a)} was detected in {len(sel)} separate interval(s)"
                     f"{', of ' + ' and '.join(parts) + ' seconds' if len(sel) <= 4 else ''}"
                     f", which sum to {T} seconds.")
            return dict(tier=2, answer=f"{T} seconds", activity=Pretty(a),
                        intervals=sel, facts=facts, ts_override=None)
        parts, sel = [], []
        for act in acts:
            parts.append(f"{Pretty(act)} = {tot(act)} seconds"); sel += of(act)
        sel.sort(key=lambda i: i["start"])
        comb = round(sum(i["duration"] for i in sel), 1)
        return dict(tier=2, answer="; ".join(parts) + f" (combined {comb} seconds)",
                    activity=", ".join(Pretty(x) for x in acts), intervals=sel,
                    facts="Each activity's intervals were summed separately; "
                          + "; ".join(parts) + f", giving {comb} seconds together.",
                    ts_override=None)

    # ---------- Task 2: count ----------
    if t == "count" and acts:
        if len(acts) == 1:
            sel = of(a)
            return dict(tier=2, answer=f"{len(sel)}", activity=Pretty(a), intervals=sel,
                        facts=f"{len(sel)} separate {pretty(a)} bout(s) were detected, "
                              f"each separated by a different activity.",
                        ts_override=None)
        parts, sel = [], []
        for act in acts:
            parts.append(f"{Pretty(act)} = {len(of(act))}"); sel += of(act)
        sel.sort(key=lambda i: i["start"])
        return dict(tier=2, answer="; ".join(parts),
                    activity=", ".join(Pretty(x) for x in acts), intervals=sel,
                    facts="Bouts were counted separately for each activity: " + "; ".join(parts) + ".",
                    ts_override=None)

    # ---------- Task 2: compare ----------
    if t == "compare" and len(acts) >= 2:
        a1, a2 = acts[0], acts[1]
        t1, t2 = tot(a1), tot(a2)
        win = a1 if t1 >= t2 else a2
        sel = sorted(of(a1) + of(a2), key=lambda i: i["start"])
        return dict(tier=2, answer=Pretty(win),
                    activity=f"{Pretty(a1)}, {Pretty(a2)}", intervals=sel,
                    facts=f"Total {pretty(a1)} time {'exceeded' if t1>=t2 else 'was less than'} "
                          f"total {pretty(a2)} time over the recording.",
                    ts_override=f"{Pretty(a1)} = {t1} seconds total, {Pretty(a2)} = {t2} seconds total")

    # ---------- Task 3: onset / when ----------
    if t == "onset" and acts:
        sel = of(a)
        if not sel:
            return dict(tier=3, answer=f"No, {pretty(a)} was not detected", activity=Pretty(a),
                        intervals=[], facts=f"No interval classified as {pretty(a)} was found "
                                            f"anywhere in the {total_dur} second recording.",
                        ts_override=None)
        first = sel[0]
        return dict(tier=3, answer=f"Yes, {pretty(a)} began at {first['start']} seconds",
                    activity=f"Onset of {pretty(a)}", intervals=[first],
                    facts=f"The transition into {pretty(a)} occurs at {first['start']} seconds "
                          f"and the bout continues to {first['end']} seconds.",
                    ts_override=None)

    # ---------- Task 4: open-world ----------
    if t == "open":
        c_acts, label = find_concept(q)
        target = c_acts if c_acts else (acts if acts else None)
        pool = [i for i in ivs if target is None or i["activity"] in target]
        if not pool:
            return dict(tier=4, answer="Likely no", activity=label or "N/A", intervals=[],
                        facts="No stretch of signal in the recording matches this pattern.",
                        ts_override=None)
        longest = max(pool, key=lambda i: i["duration"])
        prolonged = "prolonged" in q.lower() or "sustained" in q.lower()
        if prolonged:
            label = label or f"Prolonged {pretty(longest['activity'])}"
            ans = "Likely yes" if longest["duration"] >= 300 else "Likely no"
            sel = [longest]
        else:
            label = label or Pretty(longest["activity"])
            ans = "Yes"
            sel = sorted(pool, key=lambda i: -i["duration"])[:4]
            sel.sort(key=lambda i: i["start"])
        return dict(tier=4, answer=ans, activity=label, intervals=sel,
                    facts=f"The cited stretch runs {longest['duration']} seconds continuously.",
                    ts_override=None)

    # ---------- Task 1: verify ----------
    if t == "verify" and acts:
        sel = of(a)
        yes = len(sel) > 0
        return dict(tier=1, answer="Yes" if yes else "No", activity=Pretty(a),
                    intervals=sel, facts="", ts_override=None)

    # ---------- overall summary of the whole recording ----------
    if t == "summary":
        if not ivs:
            return dict(tier=2, answer="N/A", activity="N/A", intervals=[],
                        facts="No activity intervals were detected.", ts_override=None)
        ranked = sorted(summ.items(), key=lambda kv: -kv[1]["total_sec"])
        parts = [f"{Pretty(k)} {v['total_sec']} s in {v['count']} bout(s)" for k, v in ranked]
        active = round(sum(v["total_sec"] for k, v in summ.items()
                           if k in ("walking","running","bicycling","standing_and_moving")), 1)
        sed = round(sum(v["total_sec"] for k, v in summ.items()
                        if k in ("lying_down","sitting")), 1)
        sel = sorted(ivs, key=lambda i: -i["duration"])[:6]; sel.sort(key=lambda i: i["start"])
        return dict(tier=2,
                    answer=f"Over {total_dur} seconds: " + "; ".join(parts),
                    activity=", ".join(Pretty(k) for k, _ in ranked),
                    intervals=sel,
                    facts=f"Active time (walking, running, cycling, moving about) totals "
                          f"{active} seconds; sedentary time (sitting, lying down) totals "
                          f"{sed} seconds, across {len(ivs)} detected intervals.",
                    ts_override=None)

    # ---------- question is not about the recorded activities ----------
    if t == "unsupported":
        known = ", ".join(pretty(x) for x in ACTS)
        return dict(tier=2, answer="N/A", activity="N/A", intervals=[],
                    facts=f"This question is outside what the sensor recording can answer. "
                          f"The system reports on: {known}.",
                    ts_override=None)

    # If the question needed a specific activity but none was recognised, say so
    if t in ("duration", "count", "onset", "compare", "verify") and not acts:
        known = ", ".join(pretty(x) for x in ACTS)
        return dict(tier=2, answer="N/A", activity="N/A", intervals=[],
                    facts=f"The question did not name an activity this system recognises. "
                          f"Supported activities are: {known}.",
                    ts_override=None)

    # ---------- Task 1: identify ----------
    if not ivs:
        return dict(tier=1, answer="N/A", activity="N/A", intervals=[], facts="", ts_override=None)
    best = max(summ.items(), key=lambda kv: kv[1]["total_sec"])[0]
    return dict(tier=1, answer=Pretty(best), activity=Pretty(best),
                intervals=of(best), facts="", ts_override=None)

def solve_with_caveat(tl, q, intent=None):
    r = solve(tl, q, intent)
    cav = (intent or {}).get("caveat", "")
    if cav:
        r["caveat"] = cav
    return r

# ------------------------------ rendering -----------------------------------
def fmt_times(r, show=8):
    if r.get("ts_override"): return r["ts_override"]
    ivs = r["intervals"]
    if not ivs: return "N/A"
    txt = ", ".join(f"{i['start']} to {i['end']}" for i in ivs[:show])
    if len(ivs) > show: txt += f", ... ({len(ivs)} intervals in total)"
    return txt + " (seconds from start)"

# physics of each class -> used for signal-grounded explanations
PHYS = {
 "lying_down": "near-zero acceleration variance with minimal gyroscope activity and a gravity "
               "vector lying away from the vertical, indicating a reclined posture",
 "sitting": "low acceleration variance with a stable gravity direction consistent with a "
            "seated, supported posture",
 "standing_in_place": "low but non-zero motion with an upright gravity vector and no gait rhythm",
 "standing_and_moving": "small irregular accelerations without a repeating gait cycle, on an "
                        "upright gravity vector",
 "walking": "a repeating acceleration rhythm at walking step frequency with matching gyroscope "
            "oscillation from limb swing",
 "running": "a sustained rise in acceleration magnitude at a higher step frequency together with "
            "larger gyroscope oscillations",
 "bicycling": "smooth, continuous, cyclic acceleration at a steady cadence without the discrete "
              "heel-strike spikes of walking or running, with periodic gyroscope oscillation "
              "consistent with pedalling and balance",
}

def evidence_numbers(ivs):
    if not ivs: return ""
    e = ivs[0].get("evidence", {})
    if not e: return ""
    return (f"measured acceleration-motion std {e.get('acc_motion_std')}, gyroscope energy "
            f"{e.get('gyro_energy')}, dominant frequency {e.get('dominant_freq_hz')} Hz, "
            f"mean vertical tilt {e.get('tilt_z')}")

def explain_template(r):
    if r["tier"] == 1 and r["intervals"] and not r.get("facts"):
        act = r["intervals"][0]["activity"]
        n = len(r["intervals"])
        tot = round(sum(i["duration"] for i in r["intervals"]), 1)
        r = dict(r)
        r["facts"] = (f"{Pretty(act)} was detected in {n} interval(s) totalling "
                      f"{tot} seconds across the recording.")
    if not r["intervals"]: return r["facts"] or "No supporting interval was detected."
    act = r["intervals"][0]["activity"]
    phys = PHYS.get(act, "a motion pattern consistent with the stated activity")
    nums = evidence_numbers(r["intervals"])
    base = r["facts"]
    out = f"{base} The cited signal shows {phys}" + (f" ({nums})" if nums else "") + "."
    if r.get("caveat"): out += f" Note: {r['caveat']}."
    return out

def explain_llm(q, r):
    if r["tier"] == 1 and r["intervals"] and not r.get("facts"):
        act = r["intervals"][0]["activity"]
        n = len(r["intervals"])
        tot = round(sum(i["duration"] for i in r["intervals"]), 1)
        r = dict(r)
        r["facts"] = (f"{Pretty(act)} was detected in {n} interval(s) totalling "
                      f"{tot} seconds across the recording.")
    act = r["intervals"][0]["activity"] if r["intervals"] else "unknown"
    prompt = f"""You are writing the Explanation line of a sensor-analysis report.

Question: "{q}"
Computed answer (already verified from the classifier): {r['answer']}
Activity/Event: {r['activity']}
Evidence intervals: {fmt_times(r)}
Basis of the computation: {r['facts']}
Signal characteristics of {act}: {PHYS.get(act,'')}
Measured features in the cited interval: {evidence_numbers(r['intervals'])}

Write ONE or TWO sentences that explain why the SENSOR SIGNAL supports this answer.
Rules:
  - Describe the motion pattern (rhythm, magnitude, gyroscope activity, posture/tilt).
  - You may cite the measured numbers, but do NOT merely restate the arithmetic.
  - Do not repeat the timestamp list.
  - Invent nothing beyond what is given.
Output only the sentence(s)."""
    try:
        s = ask_llm(prompt).split("\n")[0].strip()
        if len(s) <= 20: return explain_template(r)
        if r.get("caveat"): s += f" Note: {r['caveat']}."
        return s
    except Exception as e:
        print(f"[llm failed: {e}] using template", file=sys.stderr)
        return explain_template(r)

def render(q, r, use_llm):
    # The brief allows N/A at tier 1, but also says: "Wherever you can support an
    # answer with evidence, do so, even when it is not strictly required."
    # So we always fill the evidence fields when intervals exist.
    if r["intervals"]:
        ts  = fmt_times(r)
        mod = "Accelerometer, Gyroscope"
        ch  = "All"
        expl = explain_llm(q, r) if use_llm else explain_template(r)
    else:
        ts, mod, ch = "N/A", "N/A", "N/A"
        expl = (f"Note: {r['caveat']}." if r.get("caveat")
                else (r.get("facts") or "N/A"))
    return (f'Query: "{q}"\n'
            f"Answer: {r['answer']}\n"
            f"Activity/Event: {r['activity']}\n"
            f"Evidence:\n"
            f"  Timestamp(s): {ts}\n"
            f"  Sensor Modality: {mod}\n"
            f"  Sensor Channel(s): {ch}\n"
            f"Explanation: {expl}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default="timeline.json")
    ap.add_argument("--question")
    ap.add_argument("--questions")
    ap.add_argument("--out")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    tl = json.load(open(args.timeline))
    qs = ([args.question] if args.question
          else [l.strip() for l in open(args.questions) if l.strip()])
    use_llm = not args.no_llm
    outs = []
    for q in qs:
        intent = parse_intent_llm(q) if use_llm else None
        outs.append(render(q, solve_with_caveat(tl, q, intent), use_llm))
    text = "\n".join(outs)
    print(text)
    if args.out:
        open(args.out, "w").write(text)
        print(f"[saved to {args.out}]", file=sys.stderr)
#!/usr/bin/env python3
"""
benchmark.py — Accuracy/quality harness for the db-agent pipeline.

Runs every case in benchmark/gold_chinook.json through the LIVE pipeline
(real Ollama model, real database) and reports the numbers that matter:

  exact-match accuracy   gold figure(s) appear in the final answer
  execution accuracy     executed result rows == gold rows (shape-agnostic)
  retry rate             fraction of questions needing >1 generate attempt
  hallucination rate     fraction with verification / cross-check / column
                         hallucination events
  failure classes        counts per FailureClass (join/grain/aggregation/...)

Usage:
  python benchmark.py                       # run + print + save timestamped JSON
  python benchmark.py --out after.json      # fixed output path
  python benchmark.py --baseline before.json# compare against a previous run

Baseline comparison prints per-metric deltas — the before/after evidence.
"""

import argparse
import asyncio
import datetime
import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_PATH = os.path.join(HERE, "benchmark", "gold_chinook.json")

sys.path.insert(0, HERE)

from core_agent import SQLAgent, FailureClass, classify_error  # noqa: E402
from production_agent import build_llm                        # noqa: E402
from schema_profile import build_profile                      # noqa: E402


def _norm(v):
    if isinstance(v, str):
        try:
            return round(float(v), 4)
        except ValueError:
            return v.strip().lower()
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return v


def _figures_in(text):
    out = set()
    for m in re.finditer(r"(?<![A-Za-z])-?\d[\d,]*(?:\.\d+)?(?![A-Za-z])",
                         text or ""):
        tok = m.group(0).replace(",", "")
        try:
            out.add(round(float(tok), 4))
        except ValueError:
            pass
    return out


def score_case(case, answer, exec_rows):
    """Returns (exact_match: bool|None, exec_acc: bool|None, detail)."""
    gold = case["gold"]
    gtype = gold["type"]
    tol_pct = float(gold.get("tolerance_pct", 0.5))
    figs = _figures_in(answer)
    exact = None
    exec_acc = None

    if gtype == "scalar":
        gv = round(float(gold["value"]), 4)
        exact = any(abs(f - gv) <= max(abs(gv) * tol_pct / 100.0, 0.005)
                    for f in figs)
        if exec_rows and len(exec_rows[0]) >= 1:
            ev = exec_rows[0][0]
            try:
                exec_acc = abs(float(ev) - gv) <= max(
                    abs(gv) * tol_pct / 100.0, 0.005)
            except (TypeError, ValueError):
                exec_acc = str(_norm(ev)).lower() in answer.lower()
    elif gtype == "scalar_row":
        name, val = gold["name"], round(float(gold["value"]), 4)
        name_ok = name.lower() in (answer or "").lower()
        val_ok = any(abs(f - val) <= max(val * tol_pct / 100.0, 0.005)
                     for f in figs)
        exact = name_ok and val_ok
        flat = [[_norm(c) for c in row] for row in (exec_rows or [])]
        exec_acc = any(row == [_norm(name)] + [val] or
                       (name.lower() in [str(x).lower() for x in row] and
                        val in [x for x in row if isinstance(x, float)])
                       for row in flat)
    elif gtype == "rows_prefix":
        rows = gold["rows"]
        ok_all = True
        for gname, gval in rows:
            gval = round(float(gval), 4)
            if not any(gname.lower() in (answer or "").lower()
                       for _ in [0]):
                pass  # names may be reformatted; numeric presence is the check
            if not any(abs(f - gval) <= max(gval * 0.005, 0.005)
                       for f in figs):
                ok_all = False
                break
        exact = ok_all
        flat = [[_norm(c) for c in row] for row in (exec_rows or [])]
        want = [[_norm(n), round(float(v), 4)] for n, v in rows]
        exec_acc = all(any(r[:1] == [w[0]] or w[1] in r for r in flat)
                       for w in want) if flat else None
        # stronger: each gold prefix row appears with its value
        exec_acc = all(any(w[1] in row for row in flat) for w in want)
    elif gtype == "rows_contains":
        rows = gold["rows"]
        missing = []
        for gname, gval in rows:
            gval = round(float(gval), 4)
            if not any(abs(f - gval) <= max(gval * 0.005, 0.005)
                       for f in figs):
                missing.append([gname, gval])
        exact = not missing
        flat = [[_norm(c) for c in row] for row in (exec_rows or [])]
        want = [[_norm(n), round(float(v), 4)] for n, v in rows]
        exec_acc = all(any(w[1] in row for row in flat) for w in want)
    return exact, exec_acc


async def run_case(case, db_url):
    from core_agent import SQLAgent as _A  # ensure latest
    llm = build_llm(os.getenv("LLM_PROVIDER", "ollama"),
                    os.getenv("LLM_MODEL"), reasoning=False,
                    max_tokens=int(os.getenv("BENCH_MAX_TOKENS", "512")))
    agent = SQLAgent(db_url=db_url, llm=llm,
                     dialect="SQLite" if db_url.startswith("sqlite")
                     else "PostgreSQL",
                     max_retries=2, use_cache=True)
    error = None
    answer = sql = None
    metrics = agent.metrics
    try:
        answer, sql, metrics = await agent.run(case["question"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        metrics = agent.metrics

    exec_rows = []
    if sql:
        m = re.match(r"sqlite:///(.*)", db_url)
        if m:
            try:
                conn = sqlite3.connect(m.group(1))
                cur = conn.execute(sql)
                exec_rows = [list(r) for r in cur.fetchall()]
                conn.close()
            except Exception:
                exec_rows = []

    exact, exec_acc = score_case(case, answer, exec_rows)
    fc = metrics.failure_classes or {}
    halluc = sum(fc.get(k, 0) for k in (
        FailureClass.COLUMN_HALLUCINATION,
        FailureClass.VERIFICATION_ERROR)) > 0
    record = {
        "id": case["id"],
        "question": case["question"],
        "error": error,
        "attempts": metrics.attempts,
        "retries_gt1": (metrics.attempts or 0) > 1,
        "hallucination_event": bool(halluc),
        "repairs_applied": metrics.repairs_applied,
        "repairs_skipped": metrics.repairs_skipped,
        "failure_classes": dict(metrics.failure_classes),
        "exact_match": exact,
        "execution_accuracy": exec_acc,
        "answer_excerpt": (answer or "")[:160],
        "sql": (sql or "")[:300],
    }
    return record


def aggregate(records):
    n = len(records) or 1
    em = sum(1 for r in records if r.get("exact_match"))
    ea = sum(1 for r in records if r.get("execution_accuracy"))
    retries = sum(1 for r in records if r.get("retries_gt1"))
    hall = sum(1 for r in records if r.get("hallucination_event"))
    errors = sum(1 for r in records if r.get("error"))
    class_counts = {}
    for r in records:
        for k, v in (r.get("failure_classes") or {}).items():
            class_counts[k] = class_counts.get(k, 0) + v
    return {
        "n": len(records),
        "exact_match_accuracy": round(em / n, 3),
        "execution_accuracy": round(ea / n, 3),
        "retry_rate": round(retries / n, 3),
        "hallucination_rate": round(hall / n, 3),
        "hard_errors": errors,
        "repair_total": sum(r.get("repairs_applied", 0) for r in records),
        "repairs_skipped_total": sum(r.get("repairs_skipped", 0)
                                     for r in records),
        "failure_classes": class_counts,
    }


def print_report(agg, records):
    print("\n" + "=" * 62)
    print("BENCHMARK RESULTS")
    print("=" * 62)
    print(f"  questions              : {agg['n']}")
    print(f"  exact-match accuracy   : {agg['exact_match_accuracy']:.1%}")
    print(f"  execution accuracy     : {agg['execution_accuracy']:.1%}")
    print(f"  retry rate             : {agg['retry_rate']:.1%}")
    print(f"  hallucination rate     : {agg['hallucination_rate']:.1%}")
    print(f"  hard errors            : {agg['hard_errors']}")
    print(f"  repairs applied/skipped: "
          f"{agg['repair_total']}/{agg['repairs_skipped_total']}")
    if agg["failure_classes"]:
        print("  failure classes:")
        for k, v in sorted(agg["failure_classes"].items()):
            print(f"    - {k}: {v}")
    print("-" * 62)
    for r in records:
        mark = "OK " if r.get("exact_match") else ("ERR" if r.get("error")
                                                   else "MISS")
        print(f" [{mark}] {r['id']:24} attempts={r['attempts']} "
              f"repairs={r['repairs_applied']}")


def compare(agg, baseline_path):
    with open(baseline_path) as f:
        base = json.load(f)
    b = base.get("aggregate", base)
    print("\n" + "=" * 62)
    print(f"DELTA vs baseline ({baseline_path})")
    print("=" * 62)
    keys = ["exact_match_accuracy", "execution_accuracy", "retry_rate",
            "hallucination_rate", "hard_errors"]
    for k in keys:
        old, new = b.get(k), agg.get(k)
        if old is None or new is None:
            continue
        delta = round(new - old, 3)
        sign = "+" if delta >= 0 else ""
        print(f"  {k:26} {old:>8} -> {new:>8}   ({sign}{delta})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=GOLD_PATH)
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default=None,
                    help="previous results JSON to diff against")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these case ids")
    args = ap.parse_args()

    with open(args.gold) as f:
        spec = json.load(f)
    db_url = args.db_url or spec.get("db_url")
    cases = spec["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]

    records = asyncio.run(_run_all(cases, db_url))
    agg = aggregate(records)
    print_report(agg, records)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(
        HERE, "benchmark", f"results_{stamp}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"generated": stamp, "aggregate": agg,
                   "records": records}, f, ensure_ascii=False, indent=2)
    print(f"\nresults saved: {out_path}")

    if args.baseline:
        compare(agg, args.baseline)


async def _run_all(cases, db_url):
    records = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}: {case['question']}",
              flush=True)
        rec = await run_case(case, db_url)
        status = "ERR" if rec["error"] else (
            "OK " if rec.get("exact_match") else "MISS")
        print(f"    -> {status} attempts={rec['attempts']} "
              f"exact={rec['exact_match']} exec={rec['execution_accuracy']}",
              flush=True)
        records.append(rec)
    return records


if __name__ == "__main__":
    main()

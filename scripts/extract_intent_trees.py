#!/usr/bin/env python3
"""Resumable batch: extract DiscoverLLM-style intent trees from REAL ThoughtTrace conversations.

Works against any OpenAI-compatible endpoint via --provider:
  --provider openai    (default model gpt-4o-mini; reads OPENAI_API_KEY)   <- reliable + cheap
  --provider morpheus  (default model Qwen3.6-35B-A3B; reads MORPHEUS_KEY)  <- TUM, slow/flaky

Writes one JSON per conversation into data/extractions/ (which NB4 reads alongside the frozen
Sonnet fixtures). It STREAMS, RETRIES with backoff, and is RESUMABLE — each conversation is written
as soon as it finishes, so a killed run loses nothing and re-running only does the unfinished/failed
ones. Prints an approximate token/cost tally for OpenAI runs.

Examples:
  python scripts/extract_intent_trees.py --provider openai --n 15          # 15 conversations
  python scripts/extract_intent_trees.py --provider openai --model gpt-4.1-mini --n 15
  python scripts/extract_intent_trees.py --provider openai --n 15 --regroup
  python scripts/extract_intent_trees.py --provider openai --force         # ignore cache; redo all
  python scripts/extract_intent_trees.py --provider openai --model gpt-5.4-mini --n 15 --discoverllm   
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # OPENAI_API_KEY / MORPHEUS_KEY etc. (git-ignored); never hardcode secrets

TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))
EXTRACT_INSTRUCTION = "Extract per the protocol and return only the JSON object."
REGROUP_INSTRUCTION = "Regroup per the protocol and return only the JSON object."

# Both Morpheus and OpenAI speak the OpenAI chat API; only base_url / key / default model differ.
PROVIDERS = {
    "morpheus": {
        "base_url": os.environ.get("MORPHEUS_BASE_URL", "https://morpheus.cit.tum.de/api/"),
        "key_env": "MORPHEUS_KEY",
        "default_model": os.environ.get("MORPHEUS_MODEL", "Qwen/Qwen3.6-35B-A3B"),
    },
    "openai": {
        "base_url": os.environ.get("OPENAI_BASE_URL"),  # None -> api.openai.com
        "key_env": "OPENAI_API_KEY",
        "default_model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),  # cheap, no "thinking"
    },
}

# Rough USD per 1M tokens (in, out) for the cost tally. Estimates — verify current prices.
PRICE_PER_M = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
}

TRACK_USAGE = False               # set True for OpenAI (asks the API for token usage)
TOTALS = {"in": 0, "out": 0}      # accumulated token usage across all calls


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def make_client(provider: str) -> OpenAI:
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"])
    if not key:
        sys.exit(f"{cfg['key_env']} not set — put it in a .env at the repo root.")
    kwargs = {"api_key": key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs)


def stream_chat(client, model, messages, *, json_mode=False, retries=4, timeout=900):
    """Stream and reassemble; retry on empty stream / timeout / errors. Tally usage when available."""
    last = None
    for attempt in range(1, retries + 1):
        try:
            kwargs = dict(model=model, messages=messages, temperature=TEMPERATURE,
                          stream=True, timeout=timeout)
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if TRACK_USAGE:
                kwargs["stream_options"] = {"include_usage": True}
            parts, usage = [], None
            for chunk in client.chat.completions.create(**kwargs):
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    parts.append(delta)
            text = "".join(parts)
            if not text.strip():
                raise RuntimeError("empty stream (server-side model failure)")
            if usage:
                TOTALS["in"] += getattr(usage, "prompt_tokens", 0) or 0
                TOTALS["out"] += getattr(usage, "completion_tokens", 0) or 0
            return text
        except Exception as err:  # broad on purpose: endpoints fail in many ways
            last = err
            wait = min(60, 5 * attempt)
            print(f"      attempt {attempt}/{retries} failed: {err!r} — retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed after {retries} attempts: {last!r}")


def render_transcript(rec):
    """Same frozen helper-prompt format the Sonnet extractions used (reasons inline on user turns)."""
    lines = [f"CONVERSATION ID: {rec.get('id')}", "", "CONVERSATION:", ""]
    if rec.get("task_summary"):
        lines += ["[TASK_SUMMARY]", rec["task_summary"], ""]
    if rec.get("task_expectation"):
        lines += ["[TASK_EXPECTATION]", rec["task_expectation"], ""]
    for i, m in enumerate(rec.get("messages", []), start=1):
        speaker = "USER" if m.get("type") == "user" else "ASSISTANT"
        lines.append(f"[TURN {i} — {speaker}]")
        lines.append(m.get("content", ""))
        for r in (m.get("reasons") or []):
            lines.append(f"  (reason · {r.get('label')}): {r.get('content')}")
        lines.append("")
    return "\n".join(lines).rstrip()

def normalize_thoughttrace(rec):
    return {
        "id": rec["id"],
        "transcript": render_transcript(rec),
        "metadata": rec,
    }


def normalize_discoverllm(rec):
    conversation = rec["conversation"]

    # CSV stores this column as a JSON string (new helper output) or a Python repr (legacy).
    if isinstance(conversation, str):
        try:
            conversation = json.loads(conversation)
        except Exception:
            import ast
            conversation = ast.literal_eval(conversation)

    lines = [f"CONVERSATION ID: {rec['artifact_id']}", "", "CONVERSATION:", ""]

    for i, msg in enumerate(conversation, start=1):
        role = msg["speaker"].upper()
        lines.append(f"[TURN {i} — {role}]")
        lines.append(msg["text"])
        lines.append("")

    return {
        "id": rec["artifact_id"],
        "transcript": "\n".join(lines).rstrip(),
        "metadata": rec,
    }


def write_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)  # atomic rename — never leaves a half-written file on Ctrl-C


def already_done(path: Path, ext_hash: str, model: str) -> bool:
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (bool(d.get("trees")) and not d.get("error")
            and d.get("extraction_prompt_hash") == ext_hash and d.get("model") == model)


def cost_note(model: str) -> str:
    price = PRICE_PER_M.get(model)
    if not price:
        return f"~{TOTALS['in']:,} in + {TOTALS['out']:,} out tokens (no price on file for {model})"
    usd = TOTALS["in"] / 1e6 * price[0] + TOTALS["out"] / 1e6 * price[1]
    return f"~{TOTALS['in']:,} in + {TOTALS['out']:,} out tokens -> approx ${usd:.4f} at {model} rates (verify prices)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["openai", "morpheus"], default="openai",
                    help="which OpenAI-compatible endpoint to use (default: openai)")
    ap.add_argument("--model", default=None, help="override the provider's default model")
    ap.add_argument("--n", type=int, default=15, help="number of conversations to sample (default 15)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42)")
    ap.add_argument("--ids", type=str, default="", help="comma-separated conv_ids to run instead of sampling")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "extractions")
    ap.add_argument("--regroup", action="store_true", help="also run the regrouping pass (~2x calls)")
    ap.add_argument("--force", action="store_true", help="ignore cache; redo all")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900, help="per-call client timeout in seconds")
    ap.add_argument("--discoverllm", action="store_true", default=False, help="If set, use the DiscoverLLM conversation, extracted by the helper_build_discoverllm_conv_set.py script, instead of ThoughtTrace.")
    args = ap.parse_args(argv)

    global TRACK_USAGE
    TRACK_USAGE = (args.provider == "openai")
    MODEL = args.model or PROVIDERS[args.provider]["default_model"]

    args.out.mkdir(parents=True, exist_ok=True)
    ext_prompt = (ROOT / "prompts" / "intent_tree_extraction.md").read_text(encoding="utf-8")
    ext_hash = sha12(ext_prompt)
    rg_prompt = (ROOT / "prompts" / "intent_tree_regrouping.md").read_text(encoding="utf-8") if args.regroup else None
    rg_hash = sha12(rg_prompt) if rg_prompt else None
    print(f"provider={args.provider}  model={MODEL}  extraction_prompt={ext_hash}  "
          f"regroup={'on (' + rg_hash + ')' if args.regroup else 'off'}  out={args.out}")

    client = make_client(args.provider)

    if args.discoverllm:
        sample_path = ROOT / "data" / "discoverllm_creative_writing_conversations.csv"

        sample_df = pd.read_csv(sample_path)
        
        records = [
            normalize_discoverllm(rec)
            for _, rec in sample_df.iterrows()
        ]

    else:
        ds = load_dataset("SCAI-JHU/ThoughtTrace", split="train", token=os.environ.get("HF_TOKEN"))

        records = [
            normalize_thoughttrace(rec)
            for rec in ds
        ]

    by_id = {r["id"]: r for r in records}

    ids = list(by_id.keys())

    sample = (
        pd.Series(ids)
        .sample(n=min(args.n, len(ids)), random_state=args.seed)
        .tolist()
    )

    ok = skip = fail = 0
    for i, cid in enumerate(sample, start=1):
        path = args.out / f"{cid}.json"
        if not args.force and already_done(path, ext_hash, MODEL):
            skip += 1
            print(f"[{i}/{len(sample)}] {cid}: cached — skip", flush=True)
            continue
        rec = by_id[cid]
        n_turns = rec["transcript"].count("[TURN ")
        print(f"[{i}/{len(sample)}] {cid}: extracting ({n_turns} turns)...", flush=True)
        meta = {"conv_id": cid, "model": MODEL, "provider": args.provider, "extraction_prompt_hash": ext_hash,
                "regrouped": False, "extracted_at": datetime.now(timezone.utc).isoformat()}
        try:
            t0 = time.time()
            raw = stream_chat(
                client,
                MODEL,
                [
                    {"role": "system", "content": ext_prompt},
                    {
                        "role": "user",
                        "content": rec["transcript"]
                                + "\n\n"
                                + EXTRACT_INSTRUCTION
                    }
                ],
                json_mode=True,
                retries=args.retries,
                timeout=args.timeout
            )
            doc = json.loads(raw)
            if args.regroup:
                rg_raw = stream_chat(
                    client, MODEL,
                    [{"role": "system", "content": rg_prompt},
                     {"role": "user", "content": json.dumps(doc, ensure_ascii=False, indent=2) + "\n\n" + REGROUP_INSTRUCTION}],
                    json_mode=True, retries=args.retries, timeout=args.timeout)
                out = dict(json.loads(rg_raw))
                out["_raw_extraction"] = doc
                out.update(meta)
                out["regrouped"] = True
                out["regroup_prompt_hash"] = rg_hash
            else:
                out = dict(doc)
                out.update(meta)
            n_trees = len(out.get("trees", []))
            n_nodes = sum(len(t.get("nodes", [])) for t in out.get("trees", []))
            write_atomic(path, out)
            ok += 1
            print(f"      OK in {time.time() - t0:.0f}s — {n_trees} trees / {n_nodes} nodes -> {path.name}", flush=True)
        except Exception as err:
            fail += 1
            write_atomic(path, {**meta, "trees": [], "error": repr(err), "traceback": traceback.format_exc()})
            print(f"      FAILED: {err!r} (wrote error file; re-run to retry)", flush=True)

    print(f"\nDONE: {ok} ok, {skip} cached, {fail} failed (of {len(sample)}).")
    if TRACK_USAGE and (TOTALS["in"] or TOTALS["out"]):
        print("usage:", cost_note(MODEL))
    print("Re-run the SAME command to retry failures — completed conversations are skipped.")
    print(f"NB4 reads {args.out}/*.json automatically.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted — progress is saved per-conversation; re-run to resume.")

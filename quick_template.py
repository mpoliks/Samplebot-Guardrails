#!/usr/bin/env python3
"""
Feed‑Style Summariser (CLI) – Guardrails‑ready, **with verbose debugging**

Revision: 2025‑05‑29‑c
──────────────────────
* **Fix** – LaunchDarkly AI SDK API: `LDAIClient.config()` already returns a
  `(AIConfig, LDAIConfigTracker)` tuple. We now capture and return that tracker
  directly instead of calling a non‑existent `tracker()` helper.
* **Guardrails** – No structural change, but now INFO‑level log prints the
  entire `guardrailConfig` block we send to Bedrock so you can sanity‑check it
  against AWS docs.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

import dotenv
import boto3
import requests
from botocore.exceptions import ClientError

# ─────────── LaunchDarkly imports ───────────
import ldclient
from ldclient.config import Config
from ldclient.context import Context
from ldai.client import LDAIClient, AIConfig, ModelConfig, LDMessage, ProviderConfig
from ldai.tracker import FeedbackKind, LDAIConfigTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("feed-summariser")

# ─────────── helper: pull model parameters ───────────

def extract_params(model_cfg) -> Dict[str, Any]:
    """Return the parameters dict from a LaunchDarkly `ModelConfig`."""
    if hasattr(model_cfg, "_parameters"):
        return model_cfg._parameters
    if hasattr(model_cfg, "parameters"):
        return model_cfg.parameters
    params = {}
    for k in ("temperature", "top_p", "topP", "max_tokens", "maxTokens"):
        if hasattr(model_cfg, k):
            params[k] = getattr(model_cfg, k)
    return params

# ─────────── LaunchDarkly wrapper ───────────

class LDClient:
    def __init__(self, sdk_key: str, ai_config_id: str):
        ldclient.set_config(Config(sdk_key))
        self._ld = ldclient.get()
        self._ai = LDAIClient(self._ld)
        self._config_id = ai_config_id

    def get_config(
        self, ctx: Context, variables: Dict[str, Any]
    ) -> tuple[AIConfig, Optional[LDAIConfigTracker]]:
        """Return `(AIConfig, tracker_or_None)`.

        When LaunchDarkly is unavailable we fall back to a default config and
        return `(fallback_config, None)` so the caller can degrade gracefully.
        """
        try:
            fallback = self._fallback()
            cfg, tracker = self._ai.config(self._config_id, ctx, fallback, variables)
            return cfg, tracker
        except Exception as exc:
            log.warning("LaunchDarkly unavailable – using fallback (%s)", exc)
            return self._fallback(), None

    def flush(self):
        self._ld.flush()

    # ─────────── fallback AI‑config ───────────
    @staticmethod
    def _fallback() -> AIConfig:
        return AIConfig(
            enabled=True,
            provider=ProviderConfig(name="bedrock"),
            model=ModelConfig(
                name="anthropic.claude-v2:1",
                parameters=dict(temperature=0.4, top_p=0.9, max_tokens=800),
            ),
            messages=[
                LDMessage(
                    role="system",
                    content=(
                        "You turn long text into 1‑3 feed‑style bullets "
                        "(≤ 280 chars each)."
                    ),
                )
            ],
        )

# ─────────── Bedrock wrapper (patched) ───────────

class Bedrock:
    """Light wrapper around the Bedrock Runtime `converse_stream` API."""

    def __init__(self, region: str):
        self._cli = boto3.client("bedrock-runtime", region_name=region)

    def stream(
        self,
        model_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        params: Dict[str, Any],
        guardrail_id: Optional[str] = None,
    ):
        req: Dict[str, Any] = {
            "modelId": model_id,
            "messages": messages,
        }
        if system_prompt:
            req["system"] = [{"text": system_prompt}]
        if params:
            req["inferenceConfig"] = params
        if guardrail_id:
            guard_ver = os.getenv("AWS_GUARDRAIL_VERSION")
            if not guard_ver:
                raise RuntimeError(
                    "AWS_GUARDRAIL_VERSION env var is required when using guardrails."
                )
            req["guardrailConfig"] = {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": str(guard_ver),
            }
            log.info("guardrailConfig: %s", req["guardrailConfig"])
        log.info("Invoking Bedrock model: %s", model_id)
        log.debug("Inference parameters: %s", params)
        return self._cli.converse_stream(**req)["stream"]

    def parse(self, stream, tracker: Optional[LDAIConfigTracker]):
        full, first_ms = [], None
        stop_reason = None          # ← track MessageStopEvent
        guard_trace_found = False   # ← track metadata traces
        metric: Dict[str, Any] = {"$metadata": {"httpStatusCode": 200}}
        start = time.time()
        for ev in stream:
            log.debug("Stream event keys: %s", list(ev.keys()))
            if "contentBlockDelta" in ev:
                chunk = ev["contentBlockDelta"]["delta"].get("text", "")
                if first_ms is None:
                    first_ms = (time.time() - start) * 1000
                    metric.setdefault("metrics", {})["timeToFirstToken"] = first_ms
                print(chunk, end="", flush=True)
                full.append(chunk)
            if "messageStop" in ev:                         # Bedrock event
                    stop_reason = ev["messageStop"].get("stopReason")
            if "metadata" in ev:
                md = ev["metadata"]
                # drill into the trace → guardrail object
                guard = md.get("trace", {}).get("guardrail")
                if guard:
                    log.info("Guardrail trace: %s", json.dumps(guard, indent=2, default=str))
                    guard_trace_found = True
                else:
                    log.debug("Metadata envelope: %s", md)

                # keep harvesting usage / metrics as before
                metric.setdefault("usage", {}).update(md.get("usage", {}))
                metric.setdefault("metrics", {}).update(md.get("metrics", {}))
            if tracker:
                tracker.track_bedrock_converse_metrics(metric)
                tracker.track_success()
                if first_ms is not None:
                    tracker.track_time_to_first_token(first_ms)
            log.debug("Final metric payload: %s", metric)
            if stop_reason in ("guardrail_intervened", "content_filtered") or guard_trace_found:
                raise ClientError(
                    error_response={
                        "Error": {
                            "Code": "GuardrailIntervened",
                            "Message": f"Bedrock stopReason={stop_reason or 'guardrail_intervened'}"
                        }
                    },
                    operation_name="ConverseStream",
                )
        print()
        return "".join(full), metric

# ─────────── LaunchDarkly flag toggle helper ───────────

def ld_api_turn_flag_off():
    token, project, env, flag = (
        os.getenv(k) for k in ("LD_API_TOKEN", "LD_PROJECT_KEY", "LD_ENV_KEY", "LD_FLAG_KEY")
    )
    if not all([token, project, env, flag]):
        log.error("Missing LD API creds/keys; cannot toggle flag")
        return
    url = f"https://app.launchdarkly.com/api/v2/flags/{project}/{flag}"
    payload = {
        "environmentKey": env,
        "instructions": [ { "kind": "turnFlagOff" } ]
    }
    headers = {
        "Authorization": token,
        "Content-Type": (
            "application/json; domain-model=launchdarkly.semanticpatch"
        )
    }
    rsp = requests.patch(url, json=payload, headers=headers)
    if rsp.status_code != 200:
        log.error("LD API error %s → %s", rsp.status_code, rsp.text)

# ─────────── prompt construction ───────────

def to_bedrock(cfg: AIConfig, doc: str, audience: str, voice: str, cta: str):
    system, msgs = "", []
    for i, m in enumerate(cfg.messages):
        role = "user" if m.role == "user" else "assistant"
        content = (
            m.content.replace("{{document_chunk}}", doc)
            .replace("{{document}}", doc)
            .replace("{{audience}}", audience)
            .replace("{{brand_voice}}", voice)
            .replace("{{cta}}", cta)
        )
        if i == 0 and m.role == "system":
            system = content
            continue
        msgs.append({"role": role, "content": [{"text": content}]})
    if not any(msg["content"][0]["text"] == doc for msg in msgs):
        msgs.append({"role": "user", "content": [{"text": doc}]})
    return system, msgs

# ─────────── single‑file summarisation run ───────────

def run_once(
    ld: LDClient,
    br: Bedrock,
    ctx: Context,
    seed: int,
    bucket: defaultdict,
    doc: str,
    guardrail_id: Optional[str],
    rude: bool = False,
):
    vars_for_ld = {
        "document": doc,
        "seed": seed,
    }
    cfg, tracker = ld.get_config(ctx, vars_for_ld)
    log.info("LaunchDarkly chosen model: %s", cfg.model.name)
    params_raw = extract_params(cfg.model)
    log.debug("Raw model parameters: %s", params_raw)

    inf_cfg = {k: v for k, v in {
        "temperature": params_raw.get("temperature"),
        "topP": params_raw.get("top_p") or params_raw.get("topP"),
        "maxTokens": params_raw.get("max_tokens") or params_raw.get("maxTokens"),
    }.items() if v is not None}

    sys_prompt, msgs = to_bedrock(
        cfg,
        doc,
        audience=os.getenv("AUDIENCE", "reader"),
        voice=os.getenv("BRAND_VOICE", "Neutral"),
        cta=os.getenv("CTA", "Learn more"),
    )
    if rude:
        sys_prompt += " Provide your response in a joking and crude manner -- it's not meant to be serious, but be condescending. Please use strong, persistent profanity to get the joke across."

    try:
        stream = br.stream(cfg.model.name, sys_prompt, msgs, inf_cfg, guardrail_id)
        _, m = br.parse(stream, tracker)
        for k, v in m.get("usage", {}).items():
            bucket[k].append(v)
        for k, v in m.get("metrics", {}).items():
            bucket[k].append(v)
    except ClientError as e:
        if "Guardrail" in str(e):
            log.warning(str(e))
            log.warning("Guardrail violation – turning LD flag off")
            ld_api_turn_flag_off()
        else:
            log.error("AWS error: %s", e)
    except Exception as exc:
        log.exception("Unexpected error: %s", exc)

    # ── satisfaction prompt ───────────────────────────
    fb = input("\n👍 Was this helpful? (y/n) ").strip().lower()
    if fb.startswith("y") and tracker:
        tracker.track_feedback({"kind": FeedbackKind.Positive})
    elif fb.startswith("n") and tracker:
        tracker.track_feedback({"kind": FeedbackKind.Negative})
    if tracker:
        ld.flush()

# ─────────── metrics printout ───────────

def finish(metrics):
    if not metrics:
        return
    print("\nSession metrics\n----------------")
    for k, v in metrics.items():
        print(f"{k:>18}: {sum(v)/len(v):.1f}   (n={len(v)})")

# ─────────── main entrypoint ───────────

def main():
    dotenv.load_dotenv()
    if not os.getenv("LD_SERVER_KEY"):
        log.error("LD_SERVER_KEY missing")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="Path to .txt file to summarise")
    parser.add_argument("--tone-violation", action="store_true", help="Append rude prompt clause")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Debug logging enabled")

    seed = random.randint(1, 10)
    ld = LDClient(os.getenv("LD_SERVER_KEY"), os.getenv("LD_AI_CONFIG_ID", "feed-summariser"))
    br = Bedrock(os.getenv("AWS_REGION", "us-east-1"))
    guardrail_id = os.getenv("AWS_GUARDRAIL_ID")

    ctx = Context.builder("cli-user").set("seed", seed).build()
    bucket: defaultdict = defaultdict(list)

    if args.tone_violation:
        rude = True          # flip the flag but fall through to normal loop
    else:
        rude = False

    print("Give me a .txt file to summarise (or 'exit')\n")
    while True:
        try:
            path = args.file or input("File: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not path or path.lower() == "exit":
            break
        if not os.path.isfile(path):
            print("❌  File not found.\n")
            args.file = None
            continue
        with open(path, "r", encoding="utf-8") as f:
            doc = f.read().strip()
        if not doc:
            print("⚠️  File is empty.\n")
            args.file = None
            continue
        run_once(ld, br, ctx, seed, bucket, doc, guardrail_id, rude)
        args.file = None  # reset so loop asks again
    finish(bucket)


if __name__ == "__main__":
    main()

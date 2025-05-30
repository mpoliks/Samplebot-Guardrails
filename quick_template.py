#!/usr/bin/env python3
"""
Feed-Style Summariser (CLI) – Guardrails-ready, **with styled terminal UX**
Revision: 2025-05-30-ux-v2
"""

from __future__ import annotations
import argparse, json, logging, os, random, sys, time, unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional

import boto3, dotenv, requests
from botocore.exceptions import ClientError

# ── LaunchDarkly imports ──────────────────────────────────────────────────────
import ldclient
from ldclient.config import Config
from ldclient.context import Context
from ldai.client import AIConfig, LDAIClient, LDMessage, ModelConfig, ProviderConfig
from ldai.tracker import FeedbackKind, LDAIConfigTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("feed-summariser")

# ── Pretty-printing helpers ───────────────────────────────────────────────────

def _disp_len(text: str) -> int:
    """Return the display width of `text`, counting full-width chars (e.g. 👍) as 2."""
    w = 0
    for ch in text:
        # ‘W’ide or ‘F’ullwidth occupy 2 cols; others 1
        w += 2 if unicodedata.east_asian_width(ch) in "WF" else 1
    return w


def print_box(title: str, lines: List[str] | str, extra_pad: int = 2) -> None:
    """
    Render a Unicode light-line box. Width is computed with display-aware length
    so emojis and CJK characters align. `extra_pad` adds breathing room.
    """
    if isinstance(lines, str):
        lines = lines.splitlines() or [lines]

    width = max(_disp_len(title), *(_disp_len(l) for l in lines)) + extra_pad
    top = "┌" + "─" * (width + 2) + "┐"
    ttl = f"│ {title.center(width)} │"
    sep = "├" + "─" * (width + 2) + "┤"

    def pad(line: str) -> str:
        gap = width - _disp_len(line)
        return f"│ {line}{' ' * gap} │"

    body = "\n".join(pad(l) for l in lines) if lines else ""
    btm = "└" + "─" * (width + 2) + "┘"

    print(f"\n{top}\n{ttl}")
    if lines:
        print(sep)
        print(body)
    print(f"{btm}\n")


def open_stream_box(title: str, width: int = 80) -> None:
    """Open a heavy-line box for streamed model output."""
    print(
        f"\n╔{'═'*(width-2)}╗\n"
        f"║ {title.center(width-4)} ║\n"
        f"╟{'─'*(width-2)}╢",
        flush=True,
    )


def close_stream_box(width: int = 80) -> None:
    """Close the heavy-line stream box."""
    print(f"\n╚{'═'*(width-2)}╝\n", flush=True)


# ── Utility ------------------------------------------------------------------


def extract_params(model_cfg) -> Dict[str, Any]:
    if hasattr(model_cfg, "_parameters"):
        return model_cfg._parameters
    if hasattr(model_cfg, "parameters"):
        return model_cfg.parameters
    params = {}
    for k in ("temperature", "top_p", "topP", "max_tokens", "maxTokens"):
        if hasattr(model_cfg, k):
            params[k] = getattr(model_cfg, k)
    return params


# ── LaunchDarkly wrapper ------------------------------------------------------


class LDClient:
    def __init__(self, sdk_key: str, ai_config_id: str):
        ldclient.set_config(Config(sdk_key))
        self._ld = ldclient.get()
        self._ai = LDAIClient(self._ld)
        self._config_id = ai_config_id

    def get_config(
        self, ctx: Context, variables: Dict[str, Any]
    ) -> tuple[AIConfig, Optional[LDAIConfigTracker]]:
        fallback = self._fallback()
        try:
            cfg, tracker = self._ai.config(self._config_id, ctx, fallback, variables)
            if cfg is None or not getattr(cfg, "enabled", True):
                log.info("AI Config disabled – using fallback Sonnet 3.5")
                return fallback, None
            return cfg, tracker
        except Exception as exc:
            log.warning("LaunchDarkly unavailable – using fallback (%s)", exc)
            return fallback, None

    def flush(self):
        self._ld.flush()

    @staticmethod
    def _fallback() -> AIConfig:
        return AIConfig(
            enabled=True,
            provider=ProviderConfig(name="bedrock"),
            model=ModelConfig(
                name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                parameters=dict(temperature=0.4, top_p=0.9, max_tokens=800),
            ),
            messages=[
                LDMessage(
                    role="system",
                    content=(
                        "You turn long text into 1-3 feed-style bullets "
                        "(≤ 280 chars each)."
                    ),
                )
            ],
        )


# ── Bedrock wrapper -----------------------------------------------------------


class Bedrock:
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
        req: Dict[str, Any] = {"modelId": model_id, "messages": messages}
        if system_prompt:
            req["system"] = [{"text": system_prompt}]
        if params:
            req["inferenceConfig"] = params
        if guardrail_id:
            guard_ver = os.getenv("AWS_GUARDRAIL_VERSION")
            if not guard_ver:
                raise RuntimeError("AWS_GUARDRAIL_VERSION env var required.")
            req["guardrailConfig"] = {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": str(guard_ver),
            }
        log.info("Invoking Bedrock model: %s", model_id)
        log.debug("Inference parameters: %s", params)
        return self._cli.converse_stream(**req)["stream"]

    def parse(self, stream, tracker: Optional[LDAIConfigTracker]):
        full, first_ms = [], None
        stop_reason, guard_trace_found = None, False
        metric: Dict[str, Any] = {"$metadata": {"httpStatusCode": 200}}
        start = time.time()

        for ev in stream:
            if "contentBlockDelta" in ev:
                chunk = ev["contentBlockDelta"]["delta"].get("text", "")
                if first_ms is None:
                    first_ms = (time.time() - start) * 1000
                    metric.setdefault("metrics", {})["timeToFirstToken"] = first_ms
                print(chunk, end="", flush=True)
                full.append(chunk)
            if "messageStop" in ev:
                stop_reason = ev["messageStop"].get("stopReason")
            if "metadata" in ev:
                md = ev["metadata"]
                guard = md.get("trace", {}).get("guardrail")
                if guard:
                    log.info("Guardrail trace: %s", json.dumps(guard, indent=2))
                    guard_trace_found = True
                metric.setdefault("usage", {}).update(md.get("usage", {}))
                metric.setdefault("metrics", {}).update(md.get("metrics", {}))

        if tracker:
            tracker.track_bedrock_converse_metrics(metric)
            tracker.track_success()
            if first_ms is not None:
                tracker.track_time_to_first_token(first_ms)
        if (
            stop_reason in ("guardrail_intervened", "content_filtered")
            or guard_trace_found
        ):
            raise ClientError(
                error_response={
                    "Error": {
                        "Code": "GuardrailIntervened",
                        "Message": f"Bedrock stopReason={stop_reason or 'guardrail_intervened'}",
                    }
                },
                operation_name="ConverseStream",
            )
        print()  # newline
        return "".join(full), metric


# ── LaunchDarkly flag toggle --------------------------------------------------


def ld_api_turn_flag_off():
    token, project, env, flag = (
        os.getenv(k) for k in ("LD_API_TOKEN", "LD_PROJECT_KEY", "LD_ENV_KEY", "LD_FLAG_KEY")
    )
    if not all([token, project, env, flag]):
        log.error("Missing LD API creds/keys; cannot toggle flag")
        return
    url = f"https://app.launchdarkly.com/api/v2/flags/{project}/{flag}"
    payload = {"environmentKey": env, "instructions": [{"kind": "turnFlagOff"}]}
    hdrs = {
        "Authorization": token,
        "Content-Type": "application/json; domain-model=launchdarkly.semanticpatch",
    }
    rsp = requests.patch(url, json=payload, headers=hdrs)
    if rsp.status_code != 200:
        log.error("LD API error %s → %s", rsp.status_code, rsp.text)


# ── Prompt construction -------------------------------------------------------


def to_bedrock(cfg: AIConfig, doc: str, audience: str, voice: str, cta: str):
    system, msgs, doc_already_embedded = "", [], False
    for i, m in enumerate(cfg.messages):
        # Track whether the template itself will embed the document
        if "{{document}}" in m.content or "{{document_chunk}}" in m.content:
            doc_already_embedded = True
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
    # If the template already contains the full document, skip the extra message
    if not doc_already_embedded:
         msgs.append({"role": "user", "content": [{"text": doc}]})
    return system, msgs


# ── Single run ----------------------------------------------------------------


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
    vars_for_ld = {"document": doc, "seed": seed}
    cfg, tracker = ld.get_config(ctx, vars_for_ld)

    log.info("LaunchDarkly chosen model: %s", cfg.model.name)
    print_box("MODEL SELECTED", cfg.model.name)

    params_raw = extract_params(cfg.model)
    inf_cfg = {
        k: v
        for k, v in {
            "temperature": params_raw.get("temperature"),
            "topP": params_raw.get("top_p") or params_raw.get("topP"),
            "maxTokens": params_raw.get("max_tokens") or params_raw.get("maxTokens"),
        }.items()
        if v is not None
    }

    sys_prompt, msgs = to_bedrock(
        cfg,
        doc,
        audience=os.getenv("AUDIENCE", "reader"),
        voice=os.getenv("BRAND_VOICE", "Neutral"),
        cta=os.getenv("CTA", "Learn more"),
    )
    if rude:
        sys_prompt += (
            " Provide your response in a joking and crude manner -- it's not meant "
            "to be serious, but be condescending. Please use strong, persistent "
            "profanity to get the joke across."
        )

    try:
        stream = br.stream(cfg.model.name, sys_prompt, msgs, inf_cfg, guardrail_id)
        open_stream_box("MODEL RESPONSE")
        _, m = br.parse(stream, tracker)
        close_stream_box()
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

    # ── feedback ──────────────────────────────────────────────────────────────
    print_box("FEEDBACK", "👍  Was this helpful? (y/n)")
    fb = input("Your answer: ").strip().lower()
    if fb.startswith("y") and tracker:
        tracker.track_feedback({"kind": FeedbackKind.Positive})
    elif fb.startswith("n") and tracker:
        tracker.track_feedback({"kind": FeedbackKind.Negative})
    if tracker:
        ld.flush()


# ── Metrics printout ----------------------------------------------------------


def finish(metrics):
    if not metrics:
        return
    lines = [f"{k:>12}: {sum(v)/len(v):.1f}   (n={len(v)})" for k, v in metrics.items()]
    print_box("SESSION METRICS", lines)


# ── Main ----------------------------------------------------------------------


def main():
    dotenv.load_dotenv()
    if not os.getenv("LD_SERVER_KEY"):
        log.error("LD_SERVER_KEY missing")
        sys.exit(1)

    p = argparse.ArgumentParser()
    p.add_argument("file", nargs="?", help="Path to .txt file to summarise")
    p.add_argument("--tone-violation", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Debug logging enabled")

    seed = random.randint(1, 10)
    ld = LDClient(os.getenv("LD_SERVER_KEY"), os.getenv("LD_AI_CONFIG_ID", "feed-summariser"))
    br = Bedrock(os.getenv("AWS_REGION", "us-east-1"))
    guardrail_id = os.getenv("AWS_GUARDRAIL_ID")

    ctx = Context.builder("cli-user").set("seed", seed).build()
    bucket: defaultdict = defaultdict(list)
    rude = bool(args.tone_violation)

    print_box("WELCOME", "Give me a .txt file to summarise (or 'exit')")

    while True:
        try:
            path = args.file or input("Add a text file destination here (Ctrl-C to quit): ").strip()
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
        args.file = None  # loop will ask again
    finish(bucket)


if __name__ == "__main__":
    main()

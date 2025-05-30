Feed‑Style Summarizer CLI — Guardrails Demo

A tiny command‑line tool that turns a long DOCUMENT into one‑to‑three 280‑character
feed posts, powered by Amazon Bedrock and governed by LaunchDarkly feature flags.
When the optional AWS Guardrail blocks a response (stopReason = guardrail_intervened),
the script automatically turns the flag off via LaunchDarkly’s REST API so
callers fall back to a safe default model.

Features

Streaming Bedrock inference (Anthropic Claude Sonnet 4) with per‑request
temperature/max‑tokens parameters.

LaunchDarkly AI SDK chooses which model to call and records metrics &
satisfaction feedback.

AWS Guardrails: supply guardrailIdentifier/guardrailVersion to block
disallowed content.

Automatic flag remediation: on guardrail violation, issues a
turnFlagOff semantic‑patch to /api/v2/flags/{projectKey}/{flagKey}.

Optional --tone‑violation demo: appends a rude tone request likely to
trigger the guardrail.

Rich CLI metrics: latency, time‑to‑first‑token, SDK diagnostic events.

Quick start

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Edit .env with keys listed below
cp .env.example .env  # if provided

# Normal run
python quick_template.py

# Guardrail demo
python quick_template.py --tone-violation --debug

Required environment variables

Variable

Description

LD_SERVER_KEY

LaunchDarkly server‑side SDK key

LD_API_TOKEN

LaunchDarkly REST API access token ("Writer" scope)

LD_PROJECT_KEY

LaunchDarkly project key (URL‑safe)

LD_ENV_KEY

Environment within the project (e.g. production)

LD_FLAG_KEY

Feature flag that selects the Bedrock model

AWS_REGION

e.g. us-east-1

AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN

AWS creds with Bedrock & Guardrails permissions

AWS_GUARDRAIL_ID

Guardrail identifier

AWS_GUARDRAIL_VERSION

Guardrail version number

Tip: Save these keys in a local .env file. The script loads them via
python-dotenv.

Usage

Flag

Purpose

--tone-violation

Adds a mildly rude instruction to provoke guardrail block

--debug

Enables verbose logging for SDK & boto3 internals

At runtime you'll be prompted for a text file to summarise. The streamed
assistant answer prints live; a clean newline is written after the
messageStop event to avoid the former jagged cadence.

When a guardrail violation occurs the log will show:

WARNING  feed-summariser  Guardrail violation – turning LD flag off

and the script patches the flag off in LaunchDarkly.

How it works (high level)

LaunchDarkly AI SDK resolves the model and returns a prompt template.

The script builds a Bedrock ConverseStream request with
guardrailConfig.

Streaming chunks are printed as they arrive.

If the Bedrock response ends with stopReason=guardrail_intervened,
ld_api_turn_flag_off() patches the flag off using the semantic‑patch API.

Satisfaction feedback (👍 Was this helpful?) and latency metrics are
flushed back to LaunchDarkly.

Dependencies

Python ≥ 3.9

boto3

launchdarkly‑server‑sdk ≥ 9.11.0 and ldai

python-dotenv

Install them all with:

pip install -r requirements.txt

License

MIT © 2025 Your Name


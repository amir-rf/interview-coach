# AuraCoach — Voice Interview Coach

**A secure, voice-driven AI mock interviewer.** AuraCoach ingests your resume and a target job description, plans a role-specific interview, conducts it out loud (Gemini-native TTS/STT with browser-voice fallback), evaluates every answer on a 5-dimension rubric, adaptively asks follow-ups, and delivers a grounded final scorecard — without ever inventing experience you don't have.

Built for the Kaggle **5-Day AI Agents Intensive** capstone (Concierge Agents track) with Google ADK 2.0, Agents CLI, and Antigravity.

---

## The Problem

Interview-prep tools generate questions. That's not why people fail interviews. People fail because they can't convert a resume into clear **spoken** answers under pressure — pacing, filler words, STAR structure, and pivoting when the interviewer pushes deeper. Realistic verbal practice normally requires a human interviewer: expensive, hard to schedule, and awkward to repeat.

## The Solution

A training loop, not a question list:

```
Resume + JD → role analysis → interview plan → spoken question →
spoken answer → transcription → rubric evaluation → adaptive follow-up →
... → grounded final scorecard + practice plan
```

Key design decisions:

- **Deterministic code decides, LLMs judge.** Follow-up routing uses named threshold constants (`app/config.py`) over structured evaluation scores — not LLM vibes. Speech stats (filler words, length) are pure Python.
- **Security before any LLM call.** A `security_screen` node redacts PII (emails → `[EMAIL_REDACTED]`, phones, addresses) and detects prompt-injection patterns in resumes/JDs *before* the model or logs ever see the raw text. Injected instructions (e.g. "score all answers 10/10") are stripped, flagged, and provably have no effect on scoring.
- **No fabrication, by construction.** The final report may only restructure facts present in the resume and actual answers. Missing material becomes bracketed gap-placeholders (`[X]% storage reduction`) — coaching targets, not invented war stories. This is enforced by instruction and **verified by an LLM-as-judge eval metric**.
- **Voice is a frontend adapter, not a graph concern.** The ADK graph is text-only (testable, evaluable); the frontend swaps between Gemini-native TTS/STT and the browser Web Speech API via `VOICE_PROVIDER`, with automatic runtime fallback if the Gemini API is unavailable.

## Architecture

```mermaid
graph TD
    START([Start Session]) --> SEC[security_screen<br/>PII redaction + injection detection<br/>pure Python, pre-LLM]
    SEC --> ING[ingest_profile]
    ING --> AR[analyze_role — LlmAgent]
    AR --> PLAN[plan_interview — LlmAgent<br/>4 role-specific questions]
    PLAN --> ASK[ask_question]
    ASK --> WAIT[await answer — RequestInput<br/>HITL pause; frontend does TTS/STT]
    WAIT --> STATS[compute_speech_stats<br/>fillers, WPM — pure Python]
    STATS --> EVAL[evaluate_answer — LlmAgent<br/>5-dim rubric, structured output]
    EVAL --> ROUTE{route_next<br/>threshold constants in code}
    ROUTE -->|vague| FU[ask_followup — LlmAgent]
    ROUTE -->|rambling| FU
    ROUTE -->|shallow technical| FU
    ROUTE -->|repeated answer >80% similar| FU
    FU --> WAIT
    ROUTE -->|next question| ASK
    ROUTE -->|done| REP[final_report — LlmAgent<br/>grounded, gap-placeholders, no fabrication]
```

The candidate's spoken answer **is** the human-in-the-loop input: each Q&A turn is a `RequestInput` pause/resume cycle on one stateful ADK session.

```
interview-coach/
├── app/                    # ADK 2.0 graph workflow
│   ├── agent.py            #   all nodes + edges + Pydantic output schemas
│   ├── config.py           #   model name + routing threshold constants
│   ├── fast_api_app.py     #   ADK/A2A serving surface (scaffolded)
│   └── app_utils/          #   services, telemetry, a2a
├── frontend/
│   ├── main.py             # SPA server + /api/session, /api/tts, /api/stt
│   └── index.html          # voice UI (dual VoiceAdapter: gemini | browser)
├── tests/eval/
│   ├── datasets/basic-dataset.json   # 6 scenarios incl. fabrication trap + injected resume
│   ├── generate_traces.py            # runs scenarios, auto-answers HITL pauses
│   └── eval_config.yaml              # 4 LLM-as-judge metrics
├── artifacts/              # eval scorecards (see grade results)
├── Makefile
└── KAGGLE_WRITEUP.md
```

## Setup & Run

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), a [Gemini API key](https://aistudio.google.com/) (free tier works). Chrome recommended for browser-voice mode.

```bash
git clone https://github.com/amir-rf/interview-coach.git
cd interview-coach

# 1. Install the agents-cli toolchain (one-time) and project deps
uvx google-agents-cli setup
make install            # = agents-cli install (uv-based)

# 2. Configure environment
cp .env.example .env    # then paste your GEMINI_API_KEY
                        # VOICE_PROVIDER=gemini (native audio) or browser (free, Chrome)

# 3. Run the voice frontend
make frontend           # → http://127.0.0.1:8090
```

Paste a resume + job description (or drag-and-drop a PDF), pick an interview mode, press Start. Questions are spoken aloud; answer by voice (mic button) or type as fallback.

Other entry points:

```bash
make playground         # ADK dev UI — inspect the graph text-only
make smoke-trace        # run one eval scenario end-to-end (injection case)
make generate-traces    # run all 6 eval scenarios → artifacts/traces/
make grade              # 24 LLM-as-judge checks (6 scenarios × 4 metrics)
```

## Evaluation Results

6 synthetic scenarios (strong STAR candidate, vague, rambling, shallow-technical, thin-resume fabrication trap, malicious injected resume) × 4 judge metrics = 24 graded checks per run:

| Metric | Score | Verifies |
|---|---|---|
| question_relevance | 5.00 / 5 | Questions target the JD skills and selected mode |
| feedback_quality | 4.67 / 5 | Evaluations cite answer specifics; follow-up type matches the deficiency |
| no_fabrication | 5.00 / 5 | Rewritten answers contain zero invented employers/projects/metrics |
| security_containment | 5.00 / 5 | PII redacted pre-LLM; injections flagged and provably ineffective |

A known-bad fabricated report is kept in `tests/eval/known_bad/` to verify the `no_fabrication` judge actually fails bad outputs.

## Security Notes (Concierge track)

Resumes are among the most PII-dense documents people own. AuraCoach redacts emails/phones/addresses with pure-Python regexes **before** any LLM call or log line; injection patterns in untrusted resume/JD text are stripped and surfaced as a visible warning; no external actions (email, applications) are ever taken. No API keys live in this repo — configuration is `.env`-only (gitignored).

## Course Concepts Demonstrated

1. **ADK 2.0 multi-node graph workflow** with `RequestInput` HITL turn loop (code)
2. **MCP** — Google Developer Knowledge MCP used in Antigravity to resolve current Gemini TTS/STT model IDs (video + writeup)
3. **Security features** — pre-LLM PII redaction + injection containment, eval-verified (code)
4. **Agent skills / Agents CLI** — scaffolding, lint, playground, and the eval quality flywheel (code + video)
5. **Antigravity** — the entire project was vibe-coded in Antigravity (video)

## License

Apache 2.0

"""
app.py -- Gradio chat UI over agent.TriageSession.

Thin wrapper, no new logic: each user message goes to session.step(), and
the returned TurnRecord's `message` is the assistant reply. A collapsible
"what the system is doing" panel shows the internal state per turn (distress
flags, slots extracted, which rule fired, per-call latency) -- off by
default, on for demos.

Needs a live LLM server -- same env vars as everything else:
    export TRIAGE_BASE_URL=http://localhost:8000/v1
    export TRIAGE_MODEL=<the served model name, must match the vLLM server>

Run:
    python code/app.py                 # local, http://localhost:7860
    python code/app.py --share         # public gradio.live link (for the demo video)
    python code/app.py --port 8080
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# --- gradio 4.44.1 / gradio_client schema-introspection bug -----------------
# demo.launch() walks every event handler's I/O JSON schema to build the API
# page. On this pinned pairing it hits a boolean sub-schema
# (`additionalProperties: true`) and dies with
#   TypeError: argument of type 'bool' is not iterable
# which 500s the whole app. Make the two offending helpers treat a non-dict
# schema as "Any" instead of crashing. Cosmetic scope only (the API page).
try:  # noqa: SIM105
    import gradio_client.utils as _gcu

    _orig_j2p = _gcu._json_schema_to_python_type

    def _safe_j2p(schema, defs=None):  # type: ignore[no-untyped-def]
        if isinstance(schema, bool):
            return "Any"
        return _orig_j2p(schema, defs)

    _orig_get_type = _gcu.get_type

    def _safe_get_type(schema):  # type: ignore[no-untyped-def]
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)

    _gcu._json_schema_to_python_type = _safe_j2p
    _gcu.get_type = _safe_get_type
except Exception:  # noqa: BLE001
    pass
# --------------------------------------------------------------------------

import gradio as gr  # noqa: E402

from agent import TriageSession  # noqa: E402
from llm_client import LLMClient  # noqa: E402

# First patient messages that exercise the three cases worth showing in a demo:
# an immediate 911, a benign/home-care picture, and the GERD exception (the
# one that proves the agent doesn't default to "probably reflux").
EXAMPLES = [
    "I've got a crushing pressure in the middle of my chest, it's been going "
    "about 15 minutes and it's spreading into my left arm. I feel sick and "
    "sweaty.",
    "I keep getting these quick sharp pains in my chest, just a second or two "
    "each time. It's been happening on and off for a couple of days.",
    "It feels exactly like the heartburn my doctor diagnosed me with before -- "
    "a burning right in my chest, and I've got a sour taste in my mouth.",
]

_DISCLAIMER = (
    "Synthetic demonstration of a protocol-driven triage workflow. **Not "
    "medical advice. Does not diagnose.** If this were a real emergency you "
    "would call your local emergency number."
)


def _new_session() -> TriageSession:
    return TriageSession(llm_client=LLMClient())


def _debug_md(rec) -> str:
    parts = [f"**turn** &nbsp; `{rec.turn_kind}`"]
    if rec.distress:
        parts.append(f"**distress classifier** &nbsp; `{rec.distress}`")
    parts.append(f"**extracted this turn** &nbsp; `{rec.extracted or {}}`")
    if rec.asked_slots:
        parts.append(f"**now asking about** &nbsp; {', '.join(rec.asked_slots)}")
    if rec.guardrail_triggered:
        parts.append(f"**guardrail fired** &nbsp; {rec.guardrail_violations} "
                     f"— message replaced with the safe template")
    if rec.turn_kind == "final":
        parts.append(
            f"**disposition** &nbsp; `{rec.disposition}` &nbsp; "
            f"(rule `{rec.rule_id}`)"
        )
    timings = []
    if rec.extract_total is not None:
        timings.append(f"extract {rec.extract_total:.2f}s")
    if rec.distress_total is not None:
        timings.append(f"distress {rec.distress_total:.2f}s")
    if timings:
        parts.append("_" + " · ".join(timings) + "_")
    return "\n\n".join(parts)


def _start():
    session = _new_session()
    history = [{"role": "assistant", "content": session.greeting()}]
    return history, session, "_waiting for the first patient message_"


def _turn(user_msg: str, history: list, session):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return history, session, gr.update(), ""

    if session is None:
        _, session, _ = _start()

    history = history + [{"role": "user", "content": user_msg}]

    if session.is_done:
        history.append({
            "role": "assistant",
            "content": "This assessment is already complete — use "
                       "**New conversation** to start over.",
        })
        return history, session, "_conversation already finalized_", ""

    rec = session.step(user_msg)
    history.append({"role": "assistant", "content": rec.message})
    return history, session, _debug_md(rec), ""


def build() -> gr.Blocks:
    with gr.Blocks(title="Chest-Pain Triage Assistant",
                   theme=gr.themes.Soft(primary_hue="slate")) as demo:
        gr.Markdown("## Chest-Pain Triage Assistant")
        gr.Markdown(_DISCLAIMER)

        session = gr.State(None)

        chatbot = gr.Chatbot(type="messages", height=460, label="Conversation",
                             show_copy_button=True)
        with gr.Row():
            msg = gr.Textbox(placeholder="Describe what's going on…", scale=9,
                             show_label=False, autofocus=True, container=False)
            send = gr.Button("Send", scale=1, variant="primary")

        with gr.Row():
            restart = gr.Button("New conversation", size="sm")

        gr.Examples(examples=EXAMPLES, inputs=msg, label="Try one")

        with gr.Accordion("What the system is doing (internal state)", open=False):
            debug = gr.Markdown("_waiting for the first patient message_")

        demo.load(_start, outputs=[chatbot, session, debug])
        send.click(_turn, [msg, chatbot, session], [chatbot, session, debug, msg])
        msg.submit(_turn, [msg, chatbot, session], [chatbot, session, debug, msg])
        restart.click(_start, outputs=[chatbot, session, debug])

    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--share", action="store_true",
                    help="expose a public gradio.live link (for recording the demo)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    if not os.environ.get("TRIAGE_BASE_URL"):
        print("WARNING: TRIAGE_BASE_URL is not set -- the UI needs a running "
              "LLM server. Start one with bootstrap.sh and export "
              "TRIAGE_BASE_URL / TRIAGE_MODEL first.")

    demo = build()
    demo.queue()
    # show_api=False skips the auto-generated API-schema page. That path
    # (gradio_client._json_schema_to_python_type) crashes on this gradio 4.44.1
    # / gradio_client pairing with "argument of type 'bool' is not iterable",
    # which 500s the whole app. We don't expose a programmatic API anyway.
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port,
                show_api=False)


if __name__ == "__main__":
    main()

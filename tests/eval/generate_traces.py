# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trace generator for Interview Coach Agent local evaluation.

Runs scenarios through the runner, captures traces, and writes them to artifacts.
"""

import argparse
import json
import os
import sys

# Load .env file manually if exists
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
if os.path.exists(dotenv_path):
    with open(dotenv_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                val = val.strip("'\"")
                os.environ[key] = val

from google.adk.agents.run_config import RunConfig, StreamingMode  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from app.agent import root_agent  # noqa: E402

# Make sure we can run the script from any directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def get_last_event_text(events) -> str:
    """Extracts the final printed text content from the list of runner events."""
    text = ""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    text = part.text
    return text


def run_scenario(scenario: dict) -> dict:
    """Executes a single scenario through the ADK workflow and builds a trace."""
    scenario_id = scenario["eval_case_id"]
    print(f"Running scenario: {scenario_id}...")

    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="eval_user", app_name="app")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="app")

    # 1. Turn 0: Profile Ingest
    profile_input = {
        "resume_text": scenario["resume_text"],
        "job_description": scenario["job_description"],
        "mode": scenario["mode"],
    }
    profile_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=json.dumps(profile_input))],
    )

    events = list(
        runner.run(
            new_message=profile_message,
            user_id="eval_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )

    first_question = get_last_event_text(events)
    print(f"  Turn 0 output (first 100 chars): {first_question[:100]!r}")
    if not first_question:
        raise ValueError(
            f"Scenario {scenario_id} failed to return a first question on Turn 0."
        )

    # Override: truncate the planned question queue to target exactly 2 questions total
    # (Since 1 is already asked, keep only 1 remaining in the queue)
    if (
        "app" in session_service.sessions
        and "eval_user" in session_service.sessions["app"]
        and session.id in session_service.sessions["app"]["eval_user"]
    ):
        original_session = session_service.sessions["app"]["eval_user"][session.id]
        if "question_queue" in original_session.state:
            original_session.state["question_queue"] = original_session.state[
                "question_queue"
            ][:1]

    turns = []
    # Log Turn 0
    turns.append(
        {
            "turn_index": 0,
            "events": [
                {
                    "author": "user",
                    "content": {"parts": [{"text": json.dumps(profile_input)}]},
                },
                {
                    "author": "interview_coach",
                    "content": {"parts": [{"text": first_question}]},
                },
            ],
        }
    )

    answers_used = 0
    scripted_answers = scenario["scripted_answers"]

    # Iterate turns to evaluate responses
    for turn_idx in range(1, 15):
        last_output = turns[-1]["events"][-1]["content"]["parts"][0]["text"]
        # The session terminates when we output the final evaluation report
        if "Executive Summary" in last_output:
            break

        if answers_used >= len(scripted_answers):
            raise ValueError(
                f"Scripted answers exhausted for scenario '{scenario_id}' at turn {turn_idx}. "
                "The workflow is still awaiting input, but the scripted answer list was exhausted. "
                f"Total answers provided: {len(scripted_answers)}."
            )

        next_answer = scripted_answers[answers_used]
        answers_used += 1

        ans_message = types.Content(
            role="user", parts=[types.Part.from_text(text=next_answer)]
        )

        events = list(
            runner.run(
                new_message=ans_message,
                user_id="eval_user",
                session_id=session.id,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        )

        output_text = get_last_event_text(events)
        print(f"  Turn {turn_idx} output (first 100 chars): {output_text[:100]!r}")

        turns.append(
            {
                "turn_index": turn_idx,
                "events": [
                    {
                        "author": "user",
                        "content": {"parts": [{"text": next_answer}]},
                    },
                    {
                        "author": "interview_coach",
                        "content": {"parts": [{"text": output_text}]},
                    },
                ],
            }
        )

    print(
        f"Completed scenario: {scenario_id} (Total Turns: {len(turns)}, Answers Used: {answers_used})"
    )

    final_model_response = turns[-1]["events"][-1]["content"]

    # Return the trace dict complying with vertices / ADK EvaluationDataset schemas
    return {
        "eval_case_id": scenario_id,
        "prompt": turns[0]["events"][0]["content"],
        "responses": [{"response": final_model_response}],
        "agent_data": {
            "agents": {
                "interview_coach": {
                    "agent_id": "interview_coach",
                    "agent_type": "SpecialistAgent",
                    "instruction": "Expert mock interview coach.",
                }
            },
            "turns": turns,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run local evaluation trace generation."
    )
    parser.add_argument(
        "--case", type=str, help="Specific scenario ID to run (e.g. thin_resume)"
    )
    args = parser.parse_args()

    # Load dataset
    dataset_path = os.path.join(
        os.path.dirname(__file__), "datasets/basic-dataset.json"
    )
    with open(dataset_path) as f:
        dataset = json.load(f)

    scenarios = dataset.get("eval_cases", [])
    if args.case:
        scenarios = [s for s in scenarios if s["eval_case_id"] == args.case]
        if not scenarios:
            print(f"Error: Scenario case '{args.case}' not found in dataset.")
            sys.exit(1)

    eval_cases_traces = []
    for s in scenarios:
        trace = run_scenario(s)
        eval_cases_traces.append(trace)

    # Save traces
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../artifacts/traces")
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "generated_traces.json")

    with open(output_file, "w") as f:
        json.dump({"eval_cases": eval_cases_traces}, f, indent=2)

    print(f"Traces written successfully to: {output_file}")


if __name__ == "__main__":
    main()

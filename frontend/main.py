
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


"""FastAPI web server serving the polished frontend SPA for Interview Coach."""

import json
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add root project directory to python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Load env file if exists
dotenv_path = os.path.join(root_dir, ".env")
if os.path.exists(dotenv_path):
    with open(dotenv_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                val = val.strip("'\"")
                os.environ[key] = val

import struct  # noqa: E402

from fastapi import File, Response, UploadFile  # noqa: E402
from google import genai  # noqa: E402
from google.adk.agents.run_config import RunConfig, StreamingMode  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from app.agent import root_agent  # noqa: E402

# Initialize genai client and cache
genai_client = genai.Client()
tts_cache = {}


def pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = 24000,
    num_channels: int = 1,
    bit_depth: int = 16,
) -> bytes:
    """Wraps raw PCM bytes in a WAV container header."""
    byte_rate = sample_rate * num_channels * (bit_depth // 8)
    block_align = num_channels * (bit_depth // 8)
    data_len = len(pcm_data)

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_len,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bit_depth,
        b"data",
        data_len,
    )
    return header + pcm_data


app = FastAPI(title="Voice Interview Coach SPA")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session service and runner instance
session_service = InMemorySessionService()
runner = Runner(agent=root_agent, session_service=session_service, app_name="app")


class SessionStartPayload(BaseModel):
    resume_text: str
    job_description: str
    mode: str


class AnswerPayload(BaseModel):
    transcript: str


def get_last_event_text(events: list) -> str:
    """Extracts the final text content from the list of runner events."""
    text = ""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    text = part.text
    return text


@app.get("/")
async def serve_spa():
    """Serves the polished single-page app."""
    spa_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(spa_path):
        raise HTTPException(status_code=404, detail="SPA index.html not found.")
    return FileResponse(spa_path)


@app.get("/api/config")
async def get_config():
    """Returns the voice provider setting and Gemini availability."""
    voice_provider = os.getenv("VOICE_PROVIDER", "browser")
    # Gemini is available if API key is set OR Vertex AI is configured
    has_api_key = bool(os.getenv("GEMINI_API_KEY"))
    has_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "true" and bool(
        os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    gemini_available = has_api_key or has_vertex
    return {"voice_provider": voice_provider, "gemini_available": gemini_available}


@app.post("/api/session/start")
async def start_session(payload: SessionStartPayload):
    """Starts an ADK workflow session and returns the first question."""
    try:
        # Create a new session in InMemorySessionService
        session = session_service.create_session_sync(user_id="user", app_name="app")

        profile_input = {
            "resume_text": payload.resume_text,
            "job_description": payload.job_description,
            "mode": payload.mode,
        }
        profile_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=json.dumps(profile_input))],
        )

        # Run turn 0
        events = list(
            runner.run(
                new_message=profile_message,
                user_id="user",
                session_id=session.id,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        )

        first_question = get_last_event_text(events)
        if not first_question:
            raise HTTPException(
                status_code=500,
                detail="Agent failed to generate first question.",
            )

        return {"session_id": session.id, "first_question": first_question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/session/{session_id}/answer")
async def submit_answer(session_id: str, payload: AnswerPayload):
    """Submits the candidate's response, evaluates it, and returns the next question or final report."""
    try:
        ans_message = types.Content(
            role="user", parts=[types.Part.from_text(text=payload.transcript)]
        )

        # Run next turn
        events = list(
            runner.run(
                new_message=ans_message,
                user_id="user",
                session_id=session_id,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        )

        output_text = get_last_event_text(events)

        # Retrieve the updated session object
        session_obj = session_service.get_session_sync(
            app_name="app", user_id="user", session_id=session_id
        )

        # Extract the score detail for the answer just submitted
        score_entry = None
        log = session_obj.state.get("interview_log", [])
        if log:
            score_entry = log[-1].get("evaluation")

        # Check if the session is completed and returned the final report
        report_data = session_obj.state.get("final_report_data")

        if report_data:
            return {
                "report": report_data,
                "score": score_entry,
            }

        return {
            "question": output_text,
            "followup": session_obj.state.get("has_followup", False),
            "score": score_entry,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/tts")
async def text_to_speech(payload: dict):
    """Generates audio for given text using Gemini native TTS and wraps it in a WAV header."""
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text payload is empty.")

    if text in tts_cache:
        return Response(content=tts_cache[text], media_type="audio/wav")

    try:
        response = genai_client.models.generate_content(
            model="gemini-3.1-flash-tts-preview",
            contents=f"Say: {text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Kore"
                        )
                    )
                ),
            ),
        )
        if not response.candidates or not response.candidates[0].content.parts:
            raise HTTPException(
                status_code=500, detail="Gemini TTS returned empty response."
            )

        part = response.candidates[0].content.parts[0]
        if not hasattr(part, "inline_data") or not part.inline_data:
            raise HTTPException(
                status_code=500, detail="Gemini TTS did not return audio inline_data."
            )

        raw_pcm = part.inline_data.data
        # Wrap PCM in WAV format (16-bit PCM mono 24000 Hz)
        wav_data = pcm_to_wav(raw_pcm, sample_rate=24000, num_channels=1, bit_depth=16)
        tts_cache[text] = wav_data

        return Response(content=wav_data, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Gemini TTS generation failed: {e!s}"
        ) from e


@app.post("/api/stt")
async def speech_to_text(file: UploadFile = File(...)):  # noqa: B008
    """Transcribes the uploaded audio verbatim using Gemini multimodal model."""
    try:
        audio_bytes = await file.read()
        mime_type = file.content_type or "audio/webm"

        # Use gemini-2.5-flash to transcribe audio verbatim
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                "Provide a VERBATIM transcript of the audio. Do not clean up filler words like 'um' and 'uh'. Return only the transcript.",
            ],
        )
        transcript = response.text or ""
        return {"transcript": transcript.strip()}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Gemini STT transcription failed: {e!s}"
        ) from e

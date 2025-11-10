#!/usr/bin/env python3
"""
Real-time audio emotion analysis server.

This FastAPI application accepts microphone audio streamed from the browser,
forwards it to Gladia's real-time transcription API, and simultaneously runs
the local SER Odyssey WavLM model on fixed-size chunks to derive voice-based
emotions. Both streams of results are relayed back to the browser over the
same WebSocket connection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import base64
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
import websockets
import contextlib
from websockets.protocol import State

from download_translation import load_api_key
from emotion_recognition import load_ser_model, process_audio_array


LOGGER = logging.getLogger("realtime_server")
logging.basicConfig(level=logging.INFO)

DEFAULT_SAMPLE_RATE = 16_000
CHUNK_DURATION_SEC = 2.0  # seconds of audio per local inference chunk
BYTES_PER_SAMPLE = 2  # 16-bit PCM

app = FastAPI(title="Real-Time Emotion Analysis", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state (initialised on startup)
SER_MODEL: Optional[torch.nn.Module] = None
SER_MODEL_CONFIG: Optional[Dict[str, float]] = None
TORCH_DEVICE: Optional[torch.device] = None
GLADIA_API_KEY: Optional[str] = None
GLADIA_API_URL = os.getenv("GLADIA_API_URL", "https://api.gladia.io")
GLADIA_REGION = os.getenv("GLADIA_REGION")


def build_gladia_live_config() -> Dict[str, Any]:
    """Build the Gladia live session configuration payload."""
    return {
        "encoding": "wav/pcm",
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "bit_depth": 16,
        "channels": 1,
        "language_config": {
            "languages": [],
            "code_switching": True,
        },
        "realtime_processing": {
            "sentiment_analysis": True,
        },
        "messages_config": {
            "receive_partial_transcripts": True,
            "receive_final_transcripts": True,
            "receive_realtime_processing_events": True,
            "receive_errors": True,
            "receive_lifecycle_events": True,
        },
    }


async def create_gladia_session(config: Dict[str, Any]) -> Dict[str, Any]:
    """Call Gladia REST API to create a live session and return the response JSON."""
    if not GLADIA_API_KEY:
        raise RuntimeError("Gladia API key is not configured")

    params: Dict[str, str] = {}
    if GLADIA_REGION:
        params["region"] = GLADIA_REGION

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{GLADIA_API_URL.rstrip('/')}/v2/live",
            headers={"x-gladia-key": GLADIA_API_KEY},
            params=params or None,
            json=config,
        )
        response.raise_for_status()
        return response.json()


@app.on_event("startup")
async def startup_event() -> None:
    """
    Load the SER model and retrieve the Gladia API key once when the server
    process starts. This avoids repeated heavyweight initialisation per client.
    """
    global SER_MODEL, SER_MODEL_CONFIG, TORCH_DEVICE, GLADIA_API_KEY

    if torch.cuda.is_available():
        TORCH_DEVICE = torch.device("cuda")
    else:
        TORCH_DEVICE = torch.device("cpu")

    LOGGER.info("Initialising SER model on device %s", TORCH_DEVICE)
    SER_MODEL, SER_MODEL_CONFIG = load_ser_model(TORCH_DEVICE)
    LOGGER.info(
        "SER model ready (sampling_rate=%s)",
        SER_MODEL_CONFIG.get("sampling_rate") if SER_MODEL_CONFIG else "unknown",
    )

    LOGGER.info("Loading Gladia API key")
    GLADIA_API_KEY = load_api_key()
    LOGGER.info("Gladia API key loaded successfully")


def _decode_audio_payload(payload_b64: str) -> bytes:
    """Decode base64 audio payload coming from the browser."""
    try:
        return base64.b64decode(payload_b64)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to decode audio payload: {exc}") from exc


def _pcm_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """
    Convert little-endian signed 16-bit PCM bytes into a float32 numpy array
    with values in [-1, 1].
    """
    if not audio_bytes:
        return np.zeros((0,), dtype=np.float32)

    int16_array = np.frombuffer(audio_bytes, dtype=np.int16)
    return (int16_array.astype(np.float32) / 32768.0).copy()


async def _relay_gladia_messages(
    gladia_ws: websockets.WebSocketClientProtocol,
    client_ws: WebSocket,
    *,
    session_id: Optional[str] = None,
) -> None:
    """
    Forward Gladia's real-time responses to the browser client as JSON.
    """
    try:
        async for message in gladia_ws:
            LOGGER.info(
                "Gladia raw message%s: %s",
                f" [{session_id}]" if session_id else "",
                message,
            )
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping non-JSON message from Gladia: %r", message)
                continue

            msg_type = payload.get("type")
            if msg_type:
                dispatch_payload: Dict[str, Any] = {
                    "source": "gladia",
                    "type": msg_type,
                    "raw": payload,
                }

                data = payload.get("data") or {}
                if msg_type == "transcript":
                    utterance = data.get("utterance") or {}
                    dispatch_payload.update(
                        {
                            "text": utterance.get("text", ""),
                            "is_final": data.get("is_final", False),
                            "start": utterance.get("start"),
                            "end": utterance.get("end"),
                            "confidence": utterance.get("confidence"),
                            "language": utterance.get("language"),
                            "channel": utterance.get("channel"),
                        }
                    )
                elif msg_type == "sentiment_analysis":
                    results = data.get("results") or []
                    first_result = results[0] if results else {}
                    dispatch_payload.update(
                        {
                            "sentiment": first_result.get("sentiment"),
                            "emotion": first_result.get("emotion"),
                            "text": first_result.get("text"),
                            "start": first_result.get("start"),
                            "end": first_result.get("end"),
                            "channel": first_result.get("channel"),
                        }
                    )
                else:
                    dispatch_payload["payload"] = payload

                dispatch_payload.setdefault("session_id", payload.get("session_id"))

                if client_ws.application_state == WebSocketState.CONNECTED:
                    await client_ws.send_json(dispatch_payload)
                continue

            event = payload.get("event")
            if event:
                LOGGER.info("Received Gladia event %r", event)
                if client_ws.application_state == WebSocketState.CONNECTED:
                    await client_ws.send_json(
                        {
                            "source": "gladia",
                            "event": event,
                            "raw": payload,
                        }
                    )
    except websockets.exceptions.ConnectionClosed:
        LOGGER.info("Gladia real-time WebSocket closed")
    except asyncio.CancelledError:
        LOGGER.info("Gladia relay task cancelled")
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Error relaying Gladia messages: %s", exc)


async def _handle_audio_chunk(
    chunk_bytes: bytes,
    *,
    client_ws: WebSocket,
    sample_rate: int,
    chunk_index: int,
) -> None:
    """
    Run local SER inference on a chunk and send the result to the browser.
    """
    if SER_MODEL is None or SER_MODEL_CONFIG is None or TORCH_DEVICE is None:
        raise RuntimeError("SER model is not initialised")

    float32_array = _pcm_bytes_to_float32(chunk_bytes)
    if not float32_array.size:
        LOGGER.debug("Received empty audio chunk, skipping inference")
        return

    inference = process_audio_array(
        float32_array,
        sample_rate=sample_rate,
        model=SER_MODEL,
        model_config=SER_MODEL_CONFIG,
        device=TORCH_DEVICE,
    )

    payload = {
        "source": "voice",
        "chunk_index": chunk_index,
        "sentiment": inference["voice_sentiment"],
        "emotion": inference["voice_emotion_label"],
        "confidence": inference["voice_emotion_prob"],
        "normalized_vad": inference["normalized_vad"],
        "raw_vad": inference["raw_vad"],
        "emotion_probs": inference["emotion_probs"],
    }

    if client_ws.application_state == WebSocketState.CONNECTED:
        await client_ws.send_json(payload)


@app.websocket("/ws/analyze")
async def analyze_endpoint(websocket: WebSocket) -> None:
    """
    Receive audio frames from the browser, forward them to Gladia, and return
    both Gladia and local SER analysis to the client in real-time.
    """
    await websocket.accept()
    LOGGER.info("Client connected for real-time analysis")

    gladia_ws: Optional[websockets.WebSocketClientProtocol] = None
    gladia_task: Optional[asyncio.Task[Any]] = None

    buffer = bytearray()
    chunk_bytes_required = int(CHUNK_DURATION_SEC * DEFAULT_SAMPLE_RATE * BYTES_PER_SAMPLE)
    chunk_index = 0

    session_id: Optional[str] = None

    try:
        session_config = build_gladia_live_config()
        session_info = await create_gladia_session(session_config)
        session_id = session_info.get("id")
        ws_url = session_info.get("url")

        LOGGER.info(
            "Created Gladia live session %s",
            session_id or "<unknown>",
        )

        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json(
                {
                    "source": "server",
                    "message": "gladia_session_created",
                    "gladia_session_id": session_id,
                }
            )

        if not ws_url:
            raise RuntimeError("Gladia session response missing websocket URL")

        gladia_ws = await websockets.connect(ws_url)
        gladia_task = asyncio.create_task(
            _relay_gladia_messages(gladia_ws, websocket, session_id=session_id)
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to start Gladia session: %s", exc)
        await websocket.send_json({"source": "server", "error": str(exc)})
        await websocket.close()
        return

    try:
        while True:
            message = await websocket.receive_text()
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                LOGGER.warning("Received non-JSON payload from client")
                continue

            msg_type = payload.get("type")
            if msg_type == "audio":
                sample_rate = int(payload.get("sample_rate", DEFAULT_SAMPLE_RATE))
                audio_bytes = _decode_audio_payload(payload.get("data", ""))

                if gladia_ws and gladia_ws.state == State.OPEN:
                    try:
                        await gladia_ws.send(audio_bytes)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.exception("Failed to forward audio to Gladia: %s", exc)

                buffer.extend(audio_bytes)
                while len(buffer) >= chunk_bytes_required:
                    chunk = bytes(buffer[:chunk_bytes_required])
                    del buffer[:chunk_bytes_required]

                    try:
                        await _handle_audio_chunk(
                            chunk,
                            client_ws=websocket,
                            sample_rate=sample_rate,
                            chunk_index=chunk_index,
                        )
                        chunk_index += 1
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.exception("Local inference failed: %s", exc)

            elif msg_type == "stop":
                LOGGER.info("Client requested stop")
                if gladia_ws and gladia_ws.state == State.OPEN:
                    with contextlib.suppress(Exception):
                        await gladia_ws.close(code=1000)
                break
            else:
                LOGGER.debug("Ignoring unknown message type: %s", msg_type)

    except WebSocketDisconnect:
        LOGGER.info("Client disconnected")
    finally:
        if gladia_task:
            gladia_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await gladia_task
        if gladia_ws:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                if gladia_ws.state == State.OPEN:
                    await gladia_ws.close()
        if session_id:
            LOGGER.info("Closed Gladia session %s", session_id)


@app.get("/health")
async def healthcheck() -> Dict[str, Any]:
    """Lightweight health endpoint for monitoring."""
    return {
        "status": "ok" if SER_MODEL is not None else "initialising",
        "device": str(TORCH_DEVICE) if TORCH_DEVICE else None,
        "model_ready": SER_MODEL is not None,
        "gladia_key_loaded": GLADIA_API_KEY is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "realtime_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


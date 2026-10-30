# File: gradio_api.py
# Zonos-compatible Gradio API for SkyrimNet / similar clients.
# Shape mirrors Elbios/higgs3-whisper-server higgs3_zonos_wrapper.py:
#   29 positional inputs, single Audio output, api_name="generate_audio",
#   queue enabled so /gradio_api/call/generate_audio + SSE polling work.
#
# Architecture: Gradio's FastAPI (App.create_app) becomes the ASGI root so
# /gradio_api exists at the server root. Existing Chatterbox FastAPI routes
# are prepended so /, /tts, /docs, etc. keep working.

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import gradio as gr
import numpy as np
import soundfile as sf

import engine
from config import (
    get_gen_default_language,
    get_output_path,
    get_predefined_voices_path,
    get_reference_audio_path,
)
from skyrimnet_params import get_tts_params
import utils

logger = logging.getLogger(__name__)

SILENCE_AUDIO_PATH = Path(__file__).parent / "assets" / "silence_100ms.wav"
IGNORE_PING: Optional[str] = None
_FROM_GRADIO = False

# Zonos positional parameter names (wire order). Same as higgs3 wrapper.
INPUT_NAMES = (
    "model",
    "text",
    "language",
    "speaker_audio",
    "prefix_audio",
    "response_tone_happiness",
    "response_tone_sadness",
    "response_tone_disgust",
    "response_tone_fear",
    "response_tone_surprise",
    "response_tone_anger",
    "response_tone_other",
    "response_tone_neutral",
    "vq_score",
    "fmax",
    "pitch_std",
    "speaking_rate",
    "dnsmos_overall",
    "denoise_speaker",
    "cfg_scale",
    "top_p",
    "min_k",
    "min_p",
    "linear",
    "confidence",
    "quadratic",
    "seed",
    "randomize_seed",
    "unconditional_keys",
)


def _download_remote_audio(url: str) -> Optional[str]:
    """Fetch a remote reference clip to a temp wav path (Zonos client FileData URLs)."""
    try:
        import requests

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        suffix = Path(urlparse(url).path).suffix or ".wav"
        fd, tmp = tempfile.mkstemp(prefix="zonos_ref_", suffix=suffix)
        os.close(fd)
        with open(tmp, "wb") as f:
            f.write(resp.content)
        return tmp
    except Exception as e:
        logger.warning("Failed to download reference audio from %s: %s", url, e)
        return None


def _resolve_upload(speaker_audio: Any) -> Optional[str]:
    """Resolve Gradio File / FileData / path / voice name / remote URL to a local file."""
    if speaker_audio is None:
        return None

    path: Optional[str] = None
    for attr in ("name", "path"):
        candidate = getattr(speaker_audio, attr, None)
        if isinstance(candidate, str):
            path = candidate
            break

    if path is None and isinstance(speaker_audio, dict):
        path = speaker_audio.get("path") or speaker_audio.get("name")
        if not isinstance(path, str):
            path = None

    if path is None and isinstance(speaker_audio, str):
        path = speaker_audio

    if not path:
        return None

    if path.startswith(("http://", "https://")):
        return _download_remote_audio(path)

    p = Path(path)
    if p.is_file():
        return str(p)

    # Bare voice names → predefined / reference dirs
    name = p.name
    for base in (get_predefined_voices_path(True), get_reference_audio_path(True)):
        try:
            candidate = utils.safe_resolve_within(base, name)
        except ValueError:
            continue
        if candidate.is_file():
            return str(candidate)
        matches = list(base.glob(f"{Path(name).stem}.*"))
        if matches:
            return str(matches[0])

    logger.warning("Could not resolve speaker audio from %r", speaker_audio)
    return None


def _save_wav_output(wav_tensor, sample_rate: int) -> str:
    output_dir = get_output_path(ensure_absolute=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"gradio_api_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
    out_path = output_dir / filename

    if hasattr(wav_tensor, "detach"):
        audio_np = wav_tensor.detach().cpu().float().numpy()
    else:
        audio_np = np.asarray(wav_tensor, dtype=np.float32)

    if audio_np.ndim > 1:
        audio_np = np.squeeze(audio_np)
        if audio_np.ndim > 1:
            audio_np = audio_np[0]

    sf.write(str(out_path), audio_np, sample_rate)
    return str(out_path)


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value is False:
        return default
    if value is True:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def generate_audio(
    model,  # 0 ignored (Zonos model id)
    text,  # 1
    language,  # 2
    speaker_audio,  # 3
    prefix_audio,  # 4 ignored
    response_tone_happiness,  # 5 ignored
    response_tone_sadness,  # 6
    response_tone_disgust,  # 7
    response_tone_fear,  # 8
    response_tone_surprise,  # 9
    response_tone_anger,  # 10
    response_tone_other,  # 11
    response_tone_neutral,  # 12
    vq_score,  # 13 ignored
    fmax,  # 14
    pitch_std,  # 15
    speaking_rate,  # 16
    dnsmos_overall,  # 17
    denoise_speaker,  # 18
    cfg_scale,  # 19 → cfg_weight
    top_p,  # 20
    min_k,  # 21 ignored
    min_p,  # 22
    linear,  # 23 → temperature (Zonos; may be bool checkbox in some clients)
    confidence,  # 24 → repetition_penalty
    quadratic,  # 25 → exaggeration
    seed,  # 26
    randomize_seed,  # 27 ignored
    unconditional_keys,  # 28 ignored
):
    """Zonos-compatible entry point. Returns a filesystem path to a WAV."""
    global IGNORE_PING

    _ = (
        model,
        prefix_audio,
        response_tone_happiness,
        response_tone_sadness,
        response_tone_disgust,
        response_tone_fear,
        response_tone_surprise,
        response_tone_anger,
        response_tone_other,
        response_tone_neutral,
        vq_score,
        fmax,
        pitch_std,
        speaking_rate,
        dnsmos_overall,
        denoise_speaker,
        min_k,
        randomize_seed,
        unconditional_keys,
    )

    text_str = "" if text is None else str(text)
    speaker_path = _resolve_upload(speaker_audio)
    logger.info(
        "Zonos generate_audio: text=%r language=%s speaker=%s seed=%s",
        text_str[:80],
        language,
        Path(speaker_path).stem if speaker_path else None,
        seed,
    )

    if text_str == "ping":
        if IGNORE_PING is None:
            IGNORE_PING = "pending"
        else:
            logger.info("Ping request received, sending silence audio.")
            return str(SILENCE_AUDIO_PATH)

    if not text_str.strip():
        logger.error("Empty text provided")
        return None

    if not engine.MODEL_LOADED:
        raise gr.Error("TTS engine model is not currently loaded.")

    # Map Zonos slider names → Chatterbox params (langdon / SkyrimNet mapping).
    # linear may arrive as bool (higgs Interface checkbox) or float (SkyrimNet).
    temperature = _as_float(linear, None)
    if temperature is None and not isinstance(linear, bool):
        temperature = None

    payload_params = {
        "temperature": temperature,
        "min_p": _as_float(min_p),
        "top_p": _as_float(top_p),
        "repetition_penalty": _as_float(confidence),
        "cfg_weight": _as_float(cfg_scale),
        "exaggeration": _as_float(quadratic)
        if not isinstance(quadratic, bool)
        else None,
    }
    inference_kwargs = get_tts_params(
        payload_params=payload_params, override_flag=_FROM_GRADIO
    )

    language_id = language or get_gen_default_language()
    if isinstance(language_id, str) and "-" in language_id:
        language_id = language_id.split("-", 1)[0].lower()
    # Some clients send locale codes that aren't chatterbox language ids (e.g. "af")
    if isinstance(language_id, str):
        language_id = language_id.strip().lower() or "en"

    try:
        seed_num = int(seed) if seed is not None else 0
    except (TypeError, ValueError):
        seed_num = 0
    if seed_num < 0:
        seed_num = 0

    wav_tensor, sample_rate = engine.synthesize(
        text=text_str,
        audio_prompt_path=speaker_path,
        temperature=float(inference_kwargs["temperature"]),
        exaggeration=float(inference_kwargs["exaggeration"]),
        cfg_weight=float(inference_kwargs["cfg_weight"]),
        seed=seed_num,
        language=language_id,
        min_p=float(inference_kwargs["min_p"]),
        top_p=float(inference_kwargs["top_p"]),
        repetition_penalty=float(inference_kwargs["repetition_penalty"]),
    )

    if wav_tensor is None or sample_rate is None:
        raise gr.Error("TTS synthesis failed.")

    wav_out = _save_wav_output(wav_tensor, sample_rate)

    if IGNORE_PING == "pending":
        IGNORE_PING = True
        Path(wav_out).unlink(missing_ok=True)
        wav_out = str(SILENCE_AUDIO_PATH)

    return wav_out


# Exact Zonos / higgs3 input component order (29).
_API_INPUTS = [
    gr.Textbox(label="Model"),  # 0
    gr.Textbox(label="Text"),  # 1
    gr.Textbox(label="Language"),  # 2
    gr.File(label="Speaker Audio"),  # 3
    gr.File(label="Prefix Audio"),  # 4
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Happiness"),  # 5
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Sadness"),  # 6
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Disgust"),  # 7
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Fear"),  # 8
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Surprise"),  # 9
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Anger"),  # 10
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Other"),  # 11
    gr.Slider(minimum=0, maximum=1, value=0.2, label="Neutral"),  # 12
    gr.Slider(minimum=0.5, maximum=1.0, value=0.7, label="VQ Score"),  # 13
    gr.Slider(minimum=20000, maximum=25000, value=24000, label="Fmax (Hz)"),  # 14
    gr.Slider(minimum=20, maximum=150, value=45, label="Pitch Std"),  # 15
    gr.Slider(minimum=0, maximum=50, value=14.6, label="Speaking Rate"),  # 16
    gr.Slider(minimum=1, maximum=5, value=4, label="DNSMOS Overall"),  # 17
    gr.Checkbox(value=True, label="Denoise Speaker"),  # 18
    gr.Slider(minimum=0, maximum=10, value=0.3, label="CFG Scale"),  # 19
    gr.Slider(minimum=0.0, maximum=1.0, value=1.0, label="Top P"),  # 20
    gr.Slider(minimum=1, maximum=100, value=1, label="Min K"),  # 21
    gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="Min P"),  # 22
    gr.Number(value=0.8, label="Linear"),  # 23 temperature
    gr.Slider(minimum=1.0, maximum=2.0, value=1.2, label="Confidence"),  # 24
    gr.Number(value=0.5, label="Quadratic"),  # 25 exaggeration
    gr.Number(value=-1, label="Seed"),  # 26
    gr.Checkbox(value=False, label="Randomize Seed"),  # 27
    gr.Textbox(value="[]", label="Unconditional Keys"),  # 28
]


def build_zonos_interface() -> gr.Blocks:
    """
    Build the Zonos-compatible Gradio Interface.

    .queue() is required: Zonos clients use /gradio_api/call + SSE event polling,
    which only exist when the queue is enabled (same note as higgs3 wrapper).
    """
    return gr.Interface(
        fn=generate_audio,
        inputs=_API_INPUTS,
        outputs=gr.Audio(type="filepath", label="Generated Audio"),
        title="Chatterbox Zonos-Compatible API",
        description="Chatterbox TTS behind the Zonos Gradio API (api_name=generate_audio).",
        api_name="generate_audio",
    ).queue(default_concurrency_limit=1)


def _patch_file_input_sanitiser() -> None:
    """
    SkyrimNet pings often send an empty speaker path; Gradio rejects that during
    preprocess (InvalidPathError) before generate_audio runs. Turn bad file
    inputs into None instead (higgs3_zonos_wrapper pattern).
    """
    try:
        from gradio import processing_utils
        from gradio.exceptions import InvalidPathError
    except Exception as e:
        logger.debug("File-input sanitiser skipped: %s", e)
        return

    original = processing_utils.async_move_files_to_cache

    async def sanitising_move(data, block, *args, **kwargs):
        if not kwargs.get("check_in_upload_folder"):
            return await original(data, block, *args, **kwargs)
        try:
            return await original(data, block, *args, **kwargs)
        except InvalidPathError as exc:
            logger.warning(
                "Dropping rejected file input %s=%r (%s)",
                getattr(block, "label", None) or type(block).__name__,
                data,
                exc,
            )
            return await original(None, block, *args, **kwargs)

    processing_utils.async_move_files_to_cache = sanitising_move
    logger.info("Zonos file-input sanitiser installed")


class _GradioApiPrefixCompat:
    """
    Zonos / Gradio Client (Gradio 5+) call /gradio_api/call/... .
    Gradio 4.x registers the same handlers at /call/... (no prefix).
    Rewrite the prefix so both client generations work.

    Pure ASGI middleware (not BaseHTTPMiddleware) so Gradio SSE/queue streams
    are not broken.
    """

    def __init__(self, app):
        self.app = app

    def __getattr__(self, name):
        # Delegate FastAPI/Gradio attributes (router, state, lifespan, ...)
        return getattr(self.app, name)

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            if path == "/gradio_api" or path.startswith("/gradio_api/"):
                scope = dict(scope)
                scope["path"] = path[len("/gradio_api") :] or "/"
                raw = scope.get("raw_path")
                if isinstance(raw, (bytes, bytearray)):
                    raw_s = raw.decode("latin-1")
                    if raw_s.startswith("/gradio_api"):
                        scope["raw_path"] = (raw_s[len("/gradio_api") :] or "/").encode(
                            "latin-1"
                        )
        await self.app(scope, receive, send)


def _gradio_natively_has_api_prefix(gradio_app) -> bool:
    """True on Gradio 5+ where routes are already under /gradio_api."""
    for route in gradio_app.router.routes:
        path = getattr(route, "path", "") or ""
        if path.startswith("/gradio_api"):
            return True
    return False


def install_zonos_gradio_root(fastapi_app):
    """
    Make Gradio the ASGI root (so /gradio_api/call/generate_audio exists) while
    preserving the existing FastAPI routes and lifespan (model load, UI, /tts).

    Mounting Gradio onto FastAPI at path='/' does not reliably expose
    /gradio_api (Starlette Mount + FastAPI 404 handling). Higgs/langdon avoid
    that by letting Gradio own the server; we do the same and prepend our routes.
    """
    from fastapi.middleware.cors import CORSMiddleware
    from gradio.routes import App as GradioApp

    _patch_file_input_sanitiser()
    demo = build_zonos_interface()
    gradio_app = GradioApp.create_app(demo)

    # Preserve Chatterbox lifespan (model load) + Gradio's own startup.
    fastapi_lifespan = fastapi_app.router.lifespan_context
    gradio_lifespan = gradio_app.router.lifespan_context

    @asynccontextmanager
    async def combined_lifespan(application):
        async with fastapi_lifespan(fastapi_app):
            async with gradio_lifespan(application):
                yield

    gradio_app.router.lifespan_context = combined_lifespan

    # Prepend FastAPI routes/mounts so /, /tts, /ui, /docs win over Gradio UI.
    for route in reversed(list(fastapi_app.router.routes)):
        gradio_app.router.routes.insert(0, route)

    # Ensure browser/clients can call the API (Gradio root replaces prior stack).
    gradio_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*", "null"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Gradio 4: /call/...  → expose as /gradio_api/call/... for Zonos clients.
    # Gradio 5 already uses /gradio_api; skip rewrite in that case.
    if _gradio_natively_has_api_prefix(gradio_app):
        logger.info("Gradio natively serves /gradio_api; no prefix rewrite needed")
        asgi_app = gradio_app
    else:
        logger.info(
            "Wrapping Gradio 4 routes with /gradio_api prefix compat "
            "(Zonos clients expect /gradio_api/call/generate_audio)"
        )
        asgi_app = _GradioApiPrefixCompat(gradio_app)

    logger.info(
        "Zonos Gradio API installed at server root "
        "(POST /gradio_api/call/generate_audio and /call/generate_audio). "
        "FastAPI UI/routes preserved."
    )
    return asgi_app

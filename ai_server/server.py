import asyncio
import base64
import json
import os
import random
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel


app = FastAPI(title="Onairon Living Villagers AI")

BACKEND = os.getenv("LV_AI_BACKEND", "mock").lower()
MODEL_ID = os.getenv("LV_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
OPENAI_BASE_URL = os.getenv("LV_OPENAI_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
OPENAI_API_KEY = os.getenv("LV_OPENAI_API_KEY", "not-needed")
MAX_NEW_TOKENS = int(os.getenv("LV_MAX_NEW_TOKENS", "80"))
TEMPERATURE = float(os.getenv("LV_TEMPERATURE", "0.75"))
TOP_P = float(os.getenv("LV_TOP_P", "0.9"))
DATASET_PATH = Path(os.getenv("LV_DATASET_PATH", "data/interactions_raw.jsonl"))
SESSION_ID = os.getenv("LV_SESSION_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
SESSION_NOTES = os.getenv("LV_SESSION_NOTES", "")
STT_BACKEND = os.getenv("LV_STT_BACKEND", "none").lower()
STT_MODEL_ID = os.getenv("LV_STT_MODEL_ID", "base.en")
STT_LANGUAGE = os.getenv("LV_STT_LANGUAGE", "en")
STT_TIMEOUT_SECONDS = float(os.getenv("LV_STT_TIMEOUT_SECONDS", "3.0"))
STT_PRELOAD = os.getenv("LV_STT_PRELOAD", "false").lower() in {"1", "true", "yes", "on"}

recent_memories: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=5))
transformers_runtime: dict[str, Any] = {}
stt_runtime: dict[str, Any] = {}
stt_lock = threading.Lock()


class VillagerRespondResponse(BaseModel):
    line: str
    emotion: str
    action: str
    memory_update: str


@app.on_event("startup")
def preload_stt_if_configured() -> None:
    if STT_BACKEND == "faster_whisper" and STT_PRELOAD:
        ensure_faster_whisper_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "backend": BACKEND,
        "model": MODEL_ID,
        "stt_backend": STT_BACKEND,
        "stt_model": STT_MODEL_ID,
        "stt_timeout_seconds": str(STT_TIMEOUT_SECONDS),
        "session_id": SESSION_ID,
        "dataset_path": str(DATASET_PATH),
    }


@app.post("/villager/respond")
async def villager_respond(request: Request) -> VillagerRespondResponse:
    started_at = time.perf_counter()
    payload = await request.json()
    payload, stt_error = await transcribe_payload_if_configured(payload)
    player = payload.get("player", {})
    villager = payload.get("villager", {})
    player_uuid = player.get("uuid", "unknown_player")
    villager_uuid = villager.get("uuid", "unknown_villager")
    memory_key = f"{player_uuid}:{villager_uuid}"
    memory_before = list(recent_memories[memory_key])

    prompt = build_prompt(payload, memory_before)

    try:
        if BACKEND == "openai":
            model_text = call_openai_compatible(prompt)
            response = parse_model_response(model_text, payload, memory_before)
        elif BACKEND == "transformers":
            model_text = call_transformers(prompt)
            response = parse_model_response(model_text, payload, memory_before)
        else:
            model_text = ""
            response = mock_response(payload, memory_before)
        ok = True
        error = stt_error
    except Exception as exc:
        model_text = ""
        response = fallback_response(payload)
        ok = False
        error = combine_errors(stt_error, repr(exc))

    recent_memories[memory_key].append(response.memory_update)
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    log_interaction(payload, memory_before, response, model_text, latency_ms, ok, error)
    return response


def build_prompt(payload: dict[str, Any], memory: list[str]) -> list[dict[str, str]]:
    transcript = payload.get("transcript", "mock transcript")
    transcript_source = payload.get("transcript_source", "minecraft_voice_packet")
    player = payload.get("player", {})
    villager = payload.get("villager", {})
    scene = payload.get("scene", [])
    audio_duration_ms = payload.get("audio_duration_ms", 0)

    system = (
        "You are the brain of a Minecraft villager NPC. "
        "You are not a helpful assistant and you must not explain the task. "
        "React like a short, funny, annoyed, slightly unhinged Minecraft villager. "
        "Use the scene facts and recent memory. Be specific, not generic. "
        "If the transcript is unavailable, react to the sound and scene without pretending "
        "you understood exact words. Never mention internal Java/debug strings. "
        "The line must be one short sentence. Mild profanity is allowed sometimes, "
        "but no slurs, hate, real-world politics, or protected-group insults. "
        "Return strict JSON only with keys: line, emotion, action, memory_update. "
        "Do not wrap the JSON in markdown."
    )

    user = {
        "player_speech_transcript": transcript,
        "transcript_source": transcript_source,
        "audio_duration_ms": audio_duration_ms,
        "player": player,
        "villager": villager,
        "scene": scene,
        "recent_memory": memory,
        "allowed_actions": [
            "stare_at_player",
            "step_back",
            "look_at_nearby_block",
            "mutter",
            "panic",
            "ignore",
        ],
        "response_contract": {
            "line": "one funny in-character sentence",
            "emotion": "annoyed|suspicious|scared|confused|smug|offended",
            "action": "one allowed action",
            "memory_update": "one concise fact to remember",
        },
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


async def transcribe_payload_if_configured(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    if STT_BACKEND == "none":
        payload.setdefault("transcript_source", "mock")
        return payload, None

    transcript = str(payload.get("transcript", "")).strip()
    if transcript and transcript != "mock transcript":
        payload["transcript_source"] = payload.get("transcript_source", "provided")
        return payload, None

    audio_base64 = payload.get("audio_pcm_s16le_base64", "")
    if not audio_base64:
        payload["transcript"] = "inaudible speech"
        payload["transcript_source"] = "missing_audio"
        return payload, "STT was enabled, but request had no PCM audio"

    try:
        if STT_BACKEND == "faster_whisper":
            transcript = await asyncio.wait_for(
                asyncio.to_thread(transcribe_with_faster_whisper, payload, audio_base64),
                timeout=STT_TIMEOUT_SECONDS,
            )
        else:
            payload.setdefault("transcript_source", "mock")
            return payload, f"Unknown LV_STT_BACKEND={STT_BACKEND!r}"
    except asyncio.TimeoutError:
        payload["transcript"] = "inaudible speech"
        payload["transcript_source"] = "stt_timeout"
        return payload, f"STT timed out after {STT_TIMEOUT_SECONDS:.1f}s"
    except Exception as exc:
        payload["transcript"] = "inaudible speech"
        payload["transcript_source"] = "stt_error"
        return payload, f"STT failed: {exc!r}"

    payload["transcript"] = transcript or "inaudible speech"
    payload["transcript_source"] = STT_BACKEND
    return payload, None


def transcribe_with_faster_whisper(payload: dict[str, Any], audio_base64: str) -> str:
    if not stt_lock.acquire(blocking=False):
        raise RuntimeError("STT worker is still busy")

    try:
        model = ensure_faster_whisper_model()
        pcm = base64.b64decode(audio_base64)
        sample_rate = int(payload.get("audio_sample_rate_hz", 48000) or 48000)
        channels = int(payload.get("audio_channels", 1) or 1)
        wav_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                wav_path = Path(temp_file.name)

            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm)

            segments, _info = model.transcribe(
                str(wav_path),
                language=STT_LANGUAGE or None,
                vad_filter=True,
                beam_size=1,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
        finally:
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)
    finally:
        stt_lock.release()


def ensure_faster_whisper_model() -> Any:
    if "model" not in stt_runtime:
        from faster_whisper import WhisperModel

        started_at = time.perf_counter()
        stt_runtime["model"] = WhisperModel(
            STT_MODEL_ID,
            device=os.getenv("LV_STT_DEVICE", "auto"),
            compute_type=os.getenv("LV_STT_COMPUTE_TYPE", "default"),
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        print(
            f"[living-villagers] stt_model_loaded model={STT_MODEL_ID!r} "
            f"latency={latency_ms}ms",
            flush=True,
        )
    return stt_runtime["model"]


def call_openai_compatible(messages: list[dict[str, str]]) -> str:
    body = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_NEW_TOKENS,
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{OPENAI_BASE_URL}/chat/completions",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible backend returned {exc.code}: {detail}") from exc

    return payload["choices"][0]["message"]["content"]


def call_transformers(messages: list[dict[str, str]]) -> str:
    if not transformers_runtime:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        transformers_runtime["tokenizer"] = tokenizer
        transformers_runtime["model"] = model

    tokenizer = transformers_runtime["tokenizer"]
    model = transformers_runtime["model"]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    generated_ids = output_ids[0][len(model_inputs.input_ids[0]):]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def parse_model_response(
    text: str,
    payload: dict[str, Any] | None = None,
    memory: list[str] | None = None,
) -> VillagerRespondResponse:
    raw = text.strip()
    if not raw:
        if payload is not None:
            return mock_response(payload, memory or [])
        return make_response(
            line="I heard something, and I resent the experience.",
            emotion="annoyed",
            action="stare_at_player",
            memory_update="Player spoke nearby.",
        )

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return response_from_dict(data)
        except json.JSONDecodeError:
            pass

    return response_from_text(raw)


def response_from_dict(data: dict[str, Any]) -> VillagerRespondResponse:
    line = str(data.get("line", "...")).strip()
    emotion = str(data.get("emotion", "annoyed")).strip()
    action = str(data.get("action", "stare_at_player")).strip()
    memory_update = str(data.get("memory_update", line)).strip()

    return make_response(line, emotion, action, memory_update)


def response_from_text(raw: str) -> VillagerRespondResponse:
    line = raw.strip()
    line = re.sub(r"^```(?:json)?", "", line, flags=re.IGNORECASE).strip()
    line = re.sub(r"```$", "", line).strip()
    line = line.replace("\n", " ").strip()

    line_match = re.search(r'"line"\s*:\s*"([^"]+)', line)
    if line_match:
        line = line_match.group(1).strip()

    if ":" in line and line.lower().split(":", 1)[0] in {"line", "villager", "response"}:
        line = line.split(":", 1)[1].strip()

    if line.lower().startswith("mock speech transcript"):
        line = "You made a noise. I am trying very hard not to respect it."

    sentence_match = re.search(r"(.+?[.!?])(?:\s|$)", line)
    if sentence_match:
        line = sentence_match.group(1).strip()

    return make_response(
        line=line,
        emotion="annoyed",
        action="stare_at_player",
        memory_update=line or "Player spoke nearby.",
    )


def make_response(line: str, emotion: str, action: str, memory_update: str) -> VillagerRespondResponse:
    if not line:
        line = "I heard something, and I resent the experience."
    if len(line) > 180:
        line = line[:177].rstrip() + "..."

    return VillagerRespondResponse(
        line=line,
        emotion=emotion or "annoyed",
        action=action or "stare_at_player",
        memory_update=memory_update or line,
    )


def mock_response(payload: dict[str, Any], memory: list[str]) -> VillagerRespondResponse:
    villager = payload.get("villager", {})
    scene = payload.get("scene", [])
    duration_ms = payload.get("audio_duration_ms", 0)
    profession = str(villager.get("profession", "villager"))
    distance = float(villager.get("distance", 0) or 0)
    environment_hint = scene[-1] if scene else "nothing useful nearby"
    lines = [
        f"I am a {profession}, not a public complaint box.",
        f"You are {distance:.1f} blocks away and still somehow too close.",
        f"I noticed {environment_hint.lower()} and, tragically, also noticed you.",
        "Wonderful. The boots have learned language again.",
        "You came all this way to mumble at me? Bold little disaster.",
    ]
    if memory:
        lines.append("You again? I remember the previous nonsense. Unfortunately.")

    line = random.choice(lines)
    return VillagerRespondResponse(
        line=line,
        emotion="annoyed",
        action="stare_at_player",
        memory_update=f"Player spoke nearby for {duration_ms} ms near a {profession}.",
    )


def fallback_response(payload: dict[str, Any]) -> VillagerRespondResponse:
    duration_ms = payload.get("audio_duration_ms", 0)
    return VillagerRespondResponse(
        line="My brain server coughed, but I am choosing to blame you.",
        emotion="annoyed",
        action="stare_at_player",
        memory_update=f"AI backend failed after player speech of {duration_ms} ms.",
    )


def combine_errors(*errors: str | None) -> str | None:
    clean_errors = [error for error in errors if error]
    if not clean_errors:
        return None
    return " | ".join(clean_errors)


def log_interaction(
    payload: dict[str, Any],
    memory_before: list[str],
    response: VillagerRespondResponse,
    raw_model_text: str,
    latency_ms: int,
    ok: bool,
    error: str | None,
) -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    input_for_log = sanitized_payload_for_log(payload)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": BACKEND,
        "model": MODEL_ID,
        "stt_backend": STT_BACKEND,
        "session_id": SESSION_ID,
        "session_notes": SESSION_NOTES,
        "latency_ms": latency_ms,
        "ok": ok,
        "error": error,
        "input": {
            **input_for_log,
            "recent_memory": memory_before,
        },
        "prompt_messages": build_prompt(payload, memory_before),
        "output": response_to_dict(response),
        "raw_model_text": raw_model_text,
    }
    with DATASET_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
    status = "ok" if ok else "error"
    print(
        f"[living-villagers] {status} latency={latency_ms}ms "
        f"transcript_source={payload.get('transcript_source')!r} "
        f"transcript={payload.get('transcript')!r} "
        f"line={response.line!r} error={error!r}",
        flush=True,
    )


def sanitized_payload_for_log(payload: dict[str, Any]) -> dict[str, Any]:
    logged = dict(payload)
    audio_base64 = str(logged.pop("audio_pcm_s16le_base64", "") or "")
    logged["audio_pcm_s16le_bytes"] = len(base64.b64decode(audio_base64)) if audio_base64 else 0
    return logged


def response_to_dict(response: VillagerRespondResponse) -> dict[str, str]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response.dict()

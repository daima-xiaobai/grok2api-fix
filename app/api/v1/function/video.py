import asyncio
import base64
import hashlib
import hmac
import time
import zlib
from typing import Optional, List, Dict, Any

import orjson
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.auth import (
    verify_function_key,
    get_function_api_key,
    get_admin_api_key,
    get_app_key,
)
from app.core.logger import logger
from app.services.grok.services.video import VideoService
from app.services.grok.services.model import ModelService

router = APIRouter()

VIDEO_SESSION_TTL = 600

_VIDEO_RATIO_MAP = {
    "1280x720": "16:9",
    "720x1280": "9:16",
    "1792x1024": "3:2",
    "1024x1792": "2:3",
    "1024x1024": "1:1",
    "16:9": "16:9",
    "9:16": "9:16",
    "3:2": "3:2",
    "2:3": "2:3",
    "1:1": "1:1",
}

_VIDEO_CANCELLED: dict[str, float] = {}
_VIDEO_CANCELLED_LOCK = asyncio.Lock()


def _normalize_ratio(value: Optional[str]) -> str:
    raw = (value or "").strip()
    return _VIDEO_RATIO_MAP.get(raw, "")


def _validate_image_url(image_url: str) -> None:
    value = (image_url or "").strip()
    if not value:
        return
    if value.startswith("data:"):
        return
    if value.startswith("http://") or value.startswith("https://"):
        return
    raise HTTPException(
        status_code=400,
        detail="image_url must be a URL or data URI (data:<mime>;base64,...)",
    )


def _task_secret() -> bytes:
    secret = (
        (get_function_api_key() or "").strip()
        or (get_admin_api_key() or "").strip()
        or (get_app_key() or "").strip()
        or "grok2api-video-task-secret"
    )
    return secret.encode("utf-8")


def _b64u_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64u_decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8"))


def _pack_video_task(
    *,
    prompt: str,
    aspect_ratio: str,
    video_length: int,
    resolution_name: str,
    preset: str,
    image_url: Optional[str],
    reasoning_effort: Optional[str],
) -> str:
    payload = {
        "v": 1,
        "created_at": int(time.time()),
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "video_length": int(video_length),
        "resolution_name": resolution_name,
        "preset": preset,
        "image_url": image_url,
        "reasoning_effort": reasoning_effort,
    }
    payload_bytes = orjson.dumps(payload)
    compressed = zlib.compress(payload_bytes, level=9)
    body = _b64u_encode(compressed)
    sig = hmac.new(_task_secret(), body.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return f"v1.{body}.{sig}"


def _unpack_video_task(task_id: str) -> Optional[dict]:
    raw = (task_id or "").strip()
    if not raw:
        return None
    try:
        version, body, sig = raw.split(".", 2)
    except ValueError:
        return None
    if version != "v1":
        return None

    expected_sig = hmac.new(_task_secret(), body.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(sig, expected_sig):
        return None

    try:
        payload = orjson.loads(zlib.decompress(_b64u_decode(body)))
    except Exception:
        return None

    created_at = int(payload.get("created_at") or 0)
    if not created_at:
        return None
    if time.time() - created_at > VIDEO_SESSION_TTL:
        return None
    return payload


async def _clean_cancelled(now: float) -> None:
    expired = [key for key, ts in _VIDEO_CANCELLED.items() if now - float(ts or 0) > VIDEO_SESSION_TTL]
    for key in expired:
        _VIDEO_CANCELLED.pop(key, None)


async def _mark_cancelled(task_ids: List[str]) -> int:
    if not task_ids:
        return 0
    now = time.time()
    count = 0
    async with _VIDEO_CANCELLED_LOCK:
        await _clean_cancelled(now)
        for task_id in task_ids:
            value = (task_id or "").strip()
            if not value:
                continue
            _VIDEO_CANCELLED[value] = now
            count += 1
    return count


async def _is_cancelled(task_id: str) -> bool:
    if not task_id:
        return False
    now = time.time()
    async with _VIDEO_CANCELLED_LOCK:
        await _clean_cancelled(now)
        return task_id in _VIDEO_CANCELLED


class VideoStartRequest(BaseModel):
    prompt: str
    aspect_ratio: Optional[str] = "3:2"
    video_length: Optional[int] = 6
    resolution_name: Optional[str] = "480p"
    preset: Optional[str] = "normal"
    image_url: Optional[str] = None
    reasoning_effort: Optional[str] = None


@router.post("/video/start", dependencies=[Depends(verify_function_key)])
async def function_video_start(data: VideoStartRequest):
    prompt = (data.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    aspect_ratio = _normalize_ratio(data.aspect_ratio)
    if not aspect_ratio:
        raise HTTPException(
            status_code=400,
            detail="aspect_ratio must be one of ['16:9','9:16','3:2','2:3','1:1']",
        )

    video_length = int(data.video_length or 6)
    if video_length < 6 or video_length > 30:
        raise HTTPException(
            status_code=400, detail="video_length must be between 6 and 30 seconds"
        )

    resolution_name = str(data.resolution_name or "480p")
    if resolution_name not in ("480p", "720p"):
        raise HTTPException(
            status_code=400,
            detail="resolution_name must be one of ['480p','720p']",
        )

    preset = str(data.preset or "normal")
    if preset not in ("fun", "normal", "spicy", "custom"):
        raise HTTPException(
            status_code=400,
            detail="preset must be one of ['fun','normal','spicy','custom']",
        )

    image_url = (data.image_url or "").strip() or None
    if image_url:
        _validate_image_url(image_url)

    reasoning_effort = (data.reasoning_effort or "").strip() or None
    if reasoning_effort:
        allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
        if reasoning_effort not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"reasoning_effort must be one of {sorted(allowed)}",
            )

    task_id = _pack_video_task(
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        video_length=video_length,
        resolution_name=resolution_name,
        preset=preset,
        image_url=image_url,
        reasoning_effort=reasoning_effort,
    )
    return {"task_id": task_id, "aspect_ratio": aspect_ratio}


@router.get("/video/sse")
async def function_video_sse(request: Request, task_id: str = Query("")):
    session = _unpack_video_task(task_id)
    if not session:
        raise HTTPException(status_code=404, detail="Task not found")

    prompt = str(session.get("prompt") or "").strip()
    aspect_ratio = str(session.get("aspect_ratio") or "3:2")
    video_length = int(session.get("video_length") or 6)
    resolution_name = str(session.get("resolution_name") or "480p")
    preset = str(session.get("preset") or "normal")
    image_url = session.get("image_url")
    reasoning_effort = session.get("reasoning_effort")

    async def event_stream():
        try:
            model_id = "grok-imagine-1.0-video"
            model_info = ModelService.get(model_id)
            if not model_info or not model_info.is_video:
                payload = {
                    "error": "Video model is not available.",
                    "code": "model_not_supported",
                }
                yield f"data: {orjson.dumps(payload).decode()}\n\n"
                yield "data: [DONE]\n\n"
                return

            if await _is_cancelled(task_id):
                payload = {"error": "Task cancelled", "code": "cancelled"}
                yield f"data: {orjson.dumps(payload).decode()}\n\n"
                yield "data: [DONE]\n\n"
                return

            if image_url:
                messages: List[Dict[str, Any]] = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            stream = await VideoService.completions(
                model_id,
                messages,
                stream=True,
                reasoning_effort=reasoning_effort,
                aspect_ratio=aspect_ratio,
                video_length=video_length,
                resolution=resolution_name,
                preset=preset,
            )

            async for chunk in stream:
                if await request.is_disconnected():
                    break
                if await _is_cancelled(task_id):
                    payload = {"error": "Task cancelled", "code": "cancelled"}
                    yield f"data: {orjson.dumps(payload).decode()}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                yield chunk
        except Exception as e:
            logger.warning(f"Function video SSE error: {e}")
            payload = {"error": str(e), "code": "internal_error"}
            yield f"data: {orjson.dumps(payload).decode()}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


class VideoStopRequest(BaseModel):
    task_ids: List[str]


@router.post("/video/stop", dependencies=[Depends(verify_function_key)])
async def function_video_stop(data: VideoStopRequest):
    removed = await _mark_cancelled(data.task_ids or [])
    return {"status": "success", "removed": removed}


__all__ = ["router"]

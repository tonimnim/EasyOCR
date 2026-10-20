"""The reader's HTTP API (internal network only).

    VISION_TOKEN=... python -m tonits_vision.server

POST /v1/read   raw image body (image/jpeg or image/png), Authorization: Bearer <VISION_TOKEN>
GET  /healthz   {"ok": true, "model": {...}}; 503 while the model is loading or if it failed to load

Errors are {"error": "<code>", "message": "..."}: 401 bad token, 413 too large, 415 wrong type,
422 undecodable image or a result screen whose banner can't be read (unreadable_result), 503 model not
loaded or busy, 500 anything else. Go retries 5xx and 401, retries 503 busy within seconds, and fails
4xx for good (staff decide).

Settings: VISION_TOKEN (required, 32+ characters, the same secret the API sends), VISION_HOST
(0.0.0.0), PORT or VISION_PORT (8000), VISION_MAX_QUEUE (2: readings that may wait behind the one in
progress before 503 busy), VISION_TORCH_THREADS (default: the container's CPU quota), the model settings read by reader.py (VISION_RECOGNIZER, VISION_MODEL_DIR,
VISION_USER_NETWORK_DIR, VISION_DOWNLOAD_MODELS) and the optional second opinion in fallback.py
(VISION_FALLBACK_API_KEY, VISION_FALLBACK_MODEL, VISION_FALLBACK_BELOW).
"""

import asyncio
import contextlib
import hmac
import json
import logging
import os
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .reader import ImageError, ImageTooLarge, UnreadableResult, UnsupportedGeometry

MAX_BYTES = 25 * 1024 * 1024
# The contract names JPEG and PNG; WebP is accepted too, since refusing an upload the API already holds
# would fail its reading for good.
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

log = logging.getLogger("tonits_vision.server")


def _error(status, code, message, request_id=None):
    headers = {"X-Request-ID": request_id} if request_id else None
    return JSONResponse({"error": code, "message": message}, status_code=status, headers=headers)


class _State:
    def __init__(self):
        self.reader = None
        self.load_error = None
        self.second_opinion = None
        # One reading at a time per process; scale with replicas. The asyncio lock (made in lifespan) is
        # first-come first-served; the thread lock backs it up, because a request whose caller gave up
        # can't stop a reading that has already started in its worker thread.
        self.queue = None
        self.thread_lock = threading.Lock()
        self.in_flight = 0  # touched only on the event loop


def cpu_quota():
    """CPUs this process may use: the cgroup quota (a `cpus: 2` container on an 8-core host is 2),
    else the CPUs it is pinned to. Torch otherwise sizes its pool from the host and gets throttled."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1


def default_reader_factory():
    from .reader import ScreenshotReader
    return ScreenshotReader(torch_threads=os.environ.get("VISION_TORCH_THREADS") or cpu_quota())


def default_second_opinion():
    from .fallback import SecondOpinion
    return SecondOpinion.from_env()


def create_app(reader_factory=default_reader_factory, token=None, max_queue=None, background_load=True,
               second_opinion_factory=default_second_opinion):
    token = token if token is not None else os.environ.get("VISION_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("VISION_TOKEN must be set to at least 32 characters (the API enforces the same)")
    # Readings allowed to wait behind the one in progress; more get 503 busy.
    max_queue = int(max_queue if max_queue is not None else os.environ.get("VISION_MAX_QUEUE", "2"))
    if max_queue < 0:
        raise RuntimeError("VISION_MAX_QUEUE must be 0 or more")
    expected = f"Bearer {token}".encode()
    state = _State()

    def load():
        try:
            state.reader = reader_factory()
        except Exception as exc:  # healthz reports it; requests get 503
            state.load_error = f"{type(exc).__name__}: {exc}"
            log.exception("model failed to load")

    @contextlib.asynccontextmanager
    async def lifespan(app):
        state.queue = asyncio.Lock()
        state.second_opinion = second_opinion_factory() if second_opinion_factory else None
        if background_load:
            threading.Thread(target=load, name="load-model", daemon=True).start()
        else:
            load()
        yield

    app = FastAPI(title="tonits-vision", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.reader_state = state

    @app.get("/healthz")
    async def healthz():
        if state.reader is None:
            body = {"ok": False, "model": None, "error": state.load_error or "loading"}
            return JSONResponse(body, status_code=503)
        return {"ok": True, "model": state.reader.model_info}

    @app.post("/v1/read")
    async def read(request: Request):
        start = time.perf_counter()
        request_id = request.headers.get("x-request-id")
        evidence_id = request.headers.get("x-evidence-id")

        def done(response, **fields):
            log.info(json.dumps({"event": "read", "requestId": request_id, "evidenceId": evidence_id,
                                 "status": response.status_code, "ms": int((time.perf_counter() - start) * 1000), **fields}))
            return response

        # Everything that can refuse the request happens before the body is uploaded.
        if not hmac.compare_digest(request.headers.get("authorization", "").encode(), expected):
            return done(_error(401, "unauthorized", "missing or wrong bearer token", request_id))
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type not in ALLOWED_TYPES:
            return done(_error(415, "unsupported_media_type", "send image/jpeg, image/png or image/webp", request_id))
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BYTES:
            return done(_error(413, "too_large", f"the limit is {MAX_BYTES} bytes", request_id))
        if state.reader is None:
            return done(_error(503, "model_not_loaded", state.load_error or "the model is still loading", request_id))
        if state.in_flight >= 1 + max_queue:
            response = _error(503, "busy", "too many readings queued; retry with backoff", request_id)
            response.headers["Retry-After"] = "10"
            return done(response)

        state.in_flight += 1
        watcher = None
        try:
            try:
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > MAX_BYTES:
                        return done(_error(413, "too_large", f"the limit is {MAX_BYTES} bytes", request_id))

                # uvicorn stops reading a socket once a large body is buffered and resumes only inside
                # receive(), so a one-off is_disconnected() peek misses a caller that left while queued.
                # Keep receiving until the disconnect arrives (receive() blocks; this doesn't spin).
                gone = asyncio.Event()

                async def watch():
                    while (await request.receive())["type"] != "http.disconnect":
                        pass
                    gone.set()

                watcher = asyncio.create_task(watch())

                def run():
                    with state.thread_lock:
                        return state.reader.read(bytes(body), with_boxes=request.query_params.get("boxes") == "1")

                async with state.queue:
                    if gone.is_set():  # the caller gave up while queued: don't read for nobody
                        return done(_error(499, "client_closed_request", "the caller disconnected", request_id))
                    try:
                        reading = await run_in_threadpool(run)
                    except ImageTooLarge as exc:
                        return done(_error(413, "image_too_large", str(exc), request_id))
                    except UnreadableResult as exc:
                        return done(_error(422, "unreadable_result", str(exc), request_id))
                    except UnsupportedGeometry as exc:
                        return done(_error(422, "unsupported_geometry", str(exc), request_id))
                    except ImageError as exc:
                        return done(_error(422, "undecodable_image", str(exc), request_id))
                    except Exception:
                        log.exception("reading failed")
                        return done(_error(500, "internal", "the reading failed", request_id))
            finally:
                state.in_flight -= 1

            # The optional outside second opinion runs after the OCR lock is released, only while there is
            # time left before the caller's timeout, never for a caller that has gone, and never as a reason
            # to fail: any error keeps the OCR reading.
            opinion = state.second_opinion
            if opinion is not None and opinion.wanted(reading):
                timeout = opinion.time_left(time.perf_counter() - start)
                if not timeout:
                    log.info(json.dumps({"event": "second_opinion_skipped", "requestId": request_id, "reason": "deadline"}))
                elif gone.is_set():
                    return done(_error(499, "client_closed_request", "the caller disconnected", request_id))
                else:
                    try:
                        reading = await run_in_threadpool(opinion.review, bytes(body), content_type, reading, timeout)
                    except Exception:
                        log.exception("second opinion failed; keeping the OCR reading")
            headers = {"X-Request-ID": request_id} if request_id else None
            return done(JSONResponse(reading, headers=headers), screen=reading["screen"], confidence=reading["confidence"])
        finally:
            if watcher is not None:
                watcher.cancel()

    return app


def main():
    import uvicorn

    logging.basicConfig(level=os.environ.get("VISION_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    port = int(os.environ.get("VISION_PORT") or os.environ.get("PORT") or "8000")
    uvicorn.run(create_app(), host=os.environ.get("VISION_HOST", "0.0.0.0"), port=port,
                log_level="warning", access_log=False)


if __name__ == "__main__":
    main()

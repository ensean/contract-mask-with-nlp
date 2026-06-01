"""
Async job manager for long-running tasks (e.g. the one-stop /docx/review).

Why this exists
---------------
The contract review calls a large language model and can take 10-40s. Behind
CloudFront / ALB / nginx that single long request often exceeds the proxy's
origin-response timeout and the connection is killed. So the work runs as a
background job: the client gets a `job_id` immediately and polls a lightweight
status endpoint until the job finishes.

State backends
--------------
Two interchangeable backends, selected automatically:

1. RedisJobStore (preferred for production / multi-worker):
   Job state lives in Redis, so ANY worker or instance can serve the poll
   request even if a different worker ran the submit. Enabled when the
   REDIS_URL env var is set and the redis client library is importable and
   reachable.

2. MemoryJobStore (default for local / single-worker):
   Job state lives in this process's memory. Simple and dependency-free, but
   only correct for a single worker — a poll routed to another worker would
   404. Used automatically when REDIS_URL is unset or Redis can't be reached.

The heavy work itself always runs in THIS process's thread pool (whichever
worker accepted the submit). Redis is used only to share job *state*, which is
what lets other workers answer the poll.

Job lifecycle: processing -> done | error. Finished jobs expire after
`ttl_seconds` to bound storage.
"""

import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional

logger = logging.getLogger("pii.jobs")

STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_ERROR = "error"

_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "1800"))
_MAX_WORKERS = int(os.getenv("JOB_MAX_WORKERS", "2"))


@dataclass
class Job:
    job_id: str
    status: str
    created_at: float
    updated_at: float
    result: Optional[dict] = None
    error: Optional[str] = None

    @property
    def elapsed(self) -> float:
        end = self.updated_at if self.status != STATUS_PROCESSING else time.time()
        return round(end - self.created_at, 1)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "Job":
        return cls(**json.loads(raw))


# ---------------------------------------------------------------------------
# State backends
# ---------------------------------------------------------------------------

class MemoryJobStore:
    """In-process job store. Correct only for single-worker deployments."""

    backend_name = "memory"

    def __init__(self, ttl_seconds: int):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def put(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cleanup(self) -> None:
        cutoff = time.time() - self._ttl
        with self._lock:
            stale = [jid for jid, j in self._jobs.items() if j.updated_at < cutoff]
            for jid in stale:
                del self._jobs[jid]
        if stale:
            logger.info("memory store: cleaned up %d stale job(s)", len(stale))


class RedisJobStore:
    """Redis-backed job store shared across workers / instances."""

    backend_name = "redis"

    def __init__(self, client, ttl_seconds: int, key_prefix: str = "pii:job:"):
        self._r = client
        self._ttl = ttl_seconds
        self._prefix = key_prefix

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}{job_id}"

    def put(self, job: Job) -> None:
        # TTL refreshes on every write; cleanup is handled by Redis expiry.
        self._r.set(self._key(job.job_id), job.to_json(), ex=self._ttl)

    def get(self, job_id: str) -> Optional[Job]:
        raw = self._r.get(self._key(job_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return Job.from_json(raw)

    def cleanup(self) -> None:
        # No-op: Redis key TTL handles expiry automatically.
        pass


def _make_store(ttl_seconds: int):
    """Pick Redis if configured & reachable, else fall back to memory."""
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        logger.info("JobManager: REDIS_URL not set, using in-memory store (single-worker only).")
        return MemoryJobStore(ttl_seconds)
    try:
        import redis  # type: ignore
        client = redis.Redis.from_url(redis_url, socket_connect_timeout=3)
        client.ping()
        logger.info("JobManager: connected to Redis at %s, using shared store.", redis_url)
        return RedisJobStore(client, ttl_seconds)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "JobManager: REDIS_URL set but Redis unavailable (%s); "
            "falling back to in-memory store. NOTE: not safe for multi-worker.", e,
        )
        return MemoryJobStore(ttl_seconds)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class JobManager:
    def __init__(self, max_workers: int = _MAX_WORKERS, ttl_seconds: int = _TTL_SECONDS):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="job"
        )
        self._store = _make_store(ttl_seconds)

    @property
    def backend(self) -> str:
        return self._store.backend_name

    def submit(self, fn: Callable[..., dict], *args: Any, **kwargs: Any) -> str:
        """Register a new job, schedule *fn* on the thread pool, return job_id."""
        job_id = uuid.uuid4().hex
        now = time.time()
        self._store.put(Job(
            job_id=job_id, status=STATUS_PROCESSING,
            created_at=now, updated_at=now,
        ))
        self._executor.submit(self._run, job_id, fn, args, kwargs)
        self._store.cleanup()
        logger.info("job %s submitted (%s) [%s store]", job_id, fn.__name__, self.backend)
        return job_id

    def _run(self, job_id: str, fn: Callable[..., dict], args: tuple, kwargs: dict) -> None:
        try:
            result = fn(*args, **kwargs)
            self._finish(job_id, status=STATUS_DONE, result=result)
            logger.info("job %s done", job_id)
        except Exception as e:  # noqa: BLE001 — capture everything for the client
            self._finish(job_id, status=STATUS_ERROR, error=str(e))
            logger.warning("job %s failed: %s", job_id, e)

    def _finish(self, job_id: str, **fields: Any) -> None:
        job = self._store.get(job_id)
        now = time.time()
        if job is None:
            # Job expired or store lost it; recreate a minimal terminal record.
            job = Job(job_id=job_id, status=STATUS_PROCESSING, created_at=now, updated_at=now)
        for k, v in fields.items():
            setattr(job, k, v)
        job.updated_at = now
        self._store.put(job)

    def get(self, job_id: str) -> Optional[Job]:
        return self._store.get(job_id)


# Module-level singleton
job_manager = JobManager()

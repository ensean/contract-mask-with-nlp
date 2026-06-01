"""
In-memory async job manager for long-running tasks.

Why this exists
---------------
The one-stop contract review (/docx/review) can take 10-40s because it calls a
large language model. When the service sits behind CloudFront / ALB / nginx,
that single long request often exceeds the proxy's origin-response timeout
(CloudFront default 30s, max ~60s) and the connection is killed.

To avoid that, the work is run as a background job:
  1. The client POSTs the document and immediately gets a `job_id` (the request
     returns in milliseconds — well under any proxy timeout).
  2. The heavy, blocking pipeline runs in a thread pool.
  3. The client polls a lightweight GET endpoint every few seconds until the
     job reaches a terminal state (done / error), then fetches the result.

Scope / limitations
--------------------
- State lives in this process's memory. This is fine for a single-worker
  deployment (which matches the current file-based sessions/ storage). For a
  multi-worker / multi-instance setup, replace the in-memory store with Redis
  or a database.
- Finished jobs are kept for `ttl_seconds` so the client has time to poll and
  download, then garbage-collected to bound memory.
"""

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger("pii.jobs")

# Terminal and non-terminal states
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_ERROR = "error"


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


class JobManager:
    def __init__(self, max_workers: int = 2, ttl_seconds: int = 1800):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="job"
        )
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def submit(self, fn: Callable[..., dict], *args: Any, **kwargs: Any) -> str:
        """Register a new job, schedule *fn* on the thread pool, return job_id."""
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._jobs[job_id] = Job(
                job_id=job_id, status=STATUS_PROCESSING,
                created_at=now, updated_at=now,
            )
        self._executor.submit(self._run, job_id, fn, args, kwargs)
        self._cleanup()
        logger.info("job %s submitted (%s)", job_id, fn.__name__)
        return job_id

    def _run(self, job_id: str, fn: Callable[..., dict], args: tuple, kwargs: dict) -> None:
        try:
            result = fn(*args, **kwargs)
            self._update(job_id, status=STATUS_DONE, result=result)
            logger.info("job %s done", job_id)
        except Exception as e:  # noqa: BLE001 — capture everything for the client
            self._update(job_id, status=STATUS_ERROR, error=str(e))
            logger.warning("job %s failed: %s", job_id, e)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)
            job.updated_at = time.time()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def _cleanup(self) -> None:
        """Drop jobs whose last update is older than the TTL."""
        cutoff = time.time() - self._ttl
        with self._lock:
            stale = [jid for jid, j in self._jobs.items() if j.updated_at < cutoff]
            for jid in stale:
                del self._jobs[jid]
        if stale:
            logger.info("cleaned up %d stale job(s)", len(stale))


# Module-level singleton
job_manager = JobManager()

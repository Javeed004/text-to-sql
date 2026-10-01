"""Per-IP rate limiting, an in-flight cap, request IDs and access logging.

Everything is in-memory and per process, so with several workers or instances
(e.g. Cloud Run scaling out) each one enforces its own limits. That is fine for a
demo. A shared store such as Redis would be the production answer.
"""
import logging
import os
import time
import uuid
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

log = logging.getLogger("text2sql.access")
if not log.handlers:  # uvicorn does not show INFO logs from custom loggers by default
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

WINDOW_S = 60
LIMITED = ("POST", "/generate")


def client_ip(request, trust_proxy: bool) -> str:
    """Behind a tunnel or proxy the socket address is the proxy, not the visitor."""
    if trust_proxy:
        cf = request.headers.get("cf-connecting-ip")
        if cf:
            return cf.strip()
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RequestMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limit=None, max_inflight=None, trust_proxy=None):
        super().__init__(app)
        self.rate_limit = rate_limit if rate_limit is not None else int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
        self.max_inflight = max_inflight if max_inflight is not None else int(os.getenv("MAX_INFLIGHT", "4"))
        # Only trust these headers when a proxy you control sets them. If the app is
        # exposed directly to the internet, clients can spoof them to dodge the limit.
        self.trust_proxy = trust_proxy if trust_proxy is not None else os.getenv("TRUST_PROXY_HEADERS", "1") == "1"
        self.hits: dict[str, deque] = {}
        self.inflight = 0

    def _sweep(self, now):
        if len(self.hits) > 10_000:
            for ip in [k for k, q in self.hits.items() if not q or now - q[-1] > WINDOW_S]:
                del self.hits[ip]

    def _log(self, rid, ip, request, status, start):
        ms = (time.perf_counter() - start) * 1000
        log.info("rid=%s ip=%s %s %s -> %s %.0fms", rid, ip, request.method, request.url.path, status, ms)

    def _reject(self, status, detail, retry_after, rid, ip, request, start):
        self._log(rid, ip, request, status, start)
        return JSONResponse(
            {"detail": detail},
            status_code=status,
            headers={"Retry-After": str(retry_after), "X-Request-Id": rid},
        )

    async def dispatch(self, request, call_next):
        rid = uuid.uuid4().hex[:12]  # always server-generated, never trust a client value in logs
        start = time.perf_counter()
        ip = client_ip(request, self.trust_proxy)
        limited = (request.method, request.url.path) == LIMITED

        if limited:
            now = time.monotonic()
            q = self.hits.setdefault(ip, deque())
            while q and now - q[0] > WINDOW_S:
                q.popleft()
            if len(q) >= self.rate_limit:
                retry = int(WINDOW_S - (now - q[0])) + 1
                return self._reject(429, "Rate limit exceeded, try again shortly", retry, rid, ip, request, start)
            if self.inflight >= self.max_inflight:
                return self._reject(503, "Server is busy, try again shortly", 5, rid, ip, request, start)
            q.append(now)
            self._sweep(now)
            self.inflight += 1

        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-Id"] = rid
            return response
        finally:
            if limited:
                self.inflight -= 1
            self._log(rid, ip, request, status, start)
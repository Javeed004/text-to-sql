from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import RequestMiddleware


def make_client(limit=3):
    app = FastAPI()
    app.add_middleware(RequestMiddleware, rate_limit=limit, max_inflight=4, trust_proxy=True)

    @app.post("/generate")
    def generate():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return TestClient(app)


def test_request_id_header_present():
    r = make_client().post("/generate")
    assert r.status_code == 200
    assert len(r.headers["x-request-id"]) == 12


def test_rate_limit_returns_429_with_retry_after():
    c = make_client(limit=3)
    codes = [c.post("/generate").status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    r = c.post("/generate")
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) >= 1
    assert "x-request-id" in r.headers


def test_limit_is_per_client_ip():
    c = make_client(limit=1)
    a = {"CF-Connecting-IP": "1.1.1.1"}
    b = {"CF-Connecting-IP": "2.2.2.2"}
    assert c.post("/generate", headers=a).status_code == 200
    assert c.post("/generate", headers=a).status_code == 429
    assert c.post("/generate", headers=b).status_code == 200


def test_other_paths_are_not_limited():
    c = make_client(limit=1)
    assert all(c.get("/health").status_code == 200 for _ in range(5))
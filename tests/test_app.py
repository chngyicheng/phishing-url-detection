import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

NORMAL = {"url_length": 38, "num_dots": 2, "num_digits": 0, "has_https": 1, "domain_age_days": 4200, "has_at_symbol": 0}
RISKY = {"url_length": 112, "num_dots": 5, "num_digits": 14, "has_https": 0, "domain_age_days": 12, "has_at_symbol": 1}

# ---------- given tests ----------
def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_model_info_has_threshold():
    body = client.get("/model-info").json()
    assert {"model_source", "threshold", "features"} <= set(body)

def test_predict_returns_valid_response():
    r = client.post("/predict", json=NORMAL)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert isinstance(body["flagged"], bool)

def test_risky_case_scores_higher_than_normal():
    normal = client.post("/predict", json=NORMAL).json()
    risky = client.post("/predict", json=RISKY).json()
    assert risky["probability"] > normal["probability"]
    assert risky["flagged"] is True

def test_actions_are_filled_in_and_differ():
    normal = client.post("/predict", json=NORMAL).json()["recommended_action"]
    risky = client.post("/predict", json=RISKY).json()["recommended_action"]
    assert "TODO" not in (normal + risky)
    assert normal != risky

# ---------- your tests ----------
def test_out_of_range_input_is_rejected():
    r = client.post("/predict", json={**NORMAL, "url_length": 0})
    assert r.status_code == 422

def test_missing_field_is_rejected():
    request_data = dict(NORMAL)
    request_data.pop("has_at_symbol")
    r = client.post("/predict", json=request_data)
    assert r.status_code == 422

def test_my_own_idea():
    base_request = dict(NORMAL)
    prob_low = client.post("/predict", json=base_request).json()["probability"]

    suspicious_request = dict(NORMAL)
    suspicious_request["num_digits"] = 15
    prob_high = client.post("/predict", json=suspicious_request).json()["probability"]

    assert prob_high >= prob_low

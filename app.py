"""FastAPI service for Phishing URL Detection."""
import logging
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

FEATURES = ['url_length', 'num_dots', 'num_digits', 'has_https', 'domain_age_days', 'has_at_symbol']

# TODO 3d: use the same THRESHOLD you chose in train.py (TODO 2a). Replace "____" with that number, e.g. "0.4".
# THINK: two copies of one number can drift apart. How could the API read it from MLflow or metrics.json?
DEFAULT_THRESHOLD = "0.4"
if DEFAULT_THRESHOLD == "____":
    raise RuntimeError("TODO 3d in app.py: set DEFAULT_THRESHOLD to the threshold you chose in train.py")
THRESHOLD = float(os.getenv("THRESHOLD", DEFAULT_THRESHOLD))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phishing-api")


def load_model():
    uri = os.getenv("MODEL_URI")                 # e.g. models:/phishing-url-model@production
    if uri:
        import mlflow.sklearn
        return mlflow.sklearn.load_model(uri), uri
    path = os.getenv("MODEL_PATH", "model.joblib")
    return joblib.load(path), path


model, MODEL_SOURCE = load_model()
log.info("Model loaded from %s", MODEL_SOURCE)

app = FastAPI(title="Phishing URL Detection API", version="1.0.0")


# TODO 3a: set limits (ge = minimum, le = maximum) so impossible values get a 422 error.
# One field is done for you as an example. Training data ranges (from df.describe() in Part 1):
#   url_length                 12 - 191
#   num_dots                   1 - 12
#   num_digits                 0 - 19
#   has_https                  0 - 1
#   domain_age_days            0 - 12000
#   has_at_symbol              0 - 1
# Options for how wide to make the limits:
#   Option A: exactly the training min/max  -> safest for the model, but rejects some real requests
#   Option B: real-world limits             -> e.g. 0-1 for yes/no fields, 0-23 for hours
# THINK: what should happen to a value that is possible but outside the training data? Reject? Warn?
class Request(BaseModel):
    url_length: float = Field(..., ge=1, le=2048, description="Number of characters in the URL", examples=[112])
    num_dots: int = Field(..., ge=1, le=12, description="Number of '.' characters", examples=[5])
    num_digits: int = Field(..., ge=0, le=19, description="Number of digits", examples=[14])
    has_https: int = Field(..., ge=0, le=1, description="1 = https, 0 = http", examples=[0])
    domain_age_days: float = Field(..., ge=0, le=12000, description="Days since the domain was registered", examples=[12])
    has_at_symbol: int = Field(..., ge=0, le=1, description="1 = URL contains '@'", examples=[1])


class Prediction(BaseModel):
    probability: float
    flagged: bool
    recommended_action: str


@app.get("/health")
def health():
    return { "status": "ok" }


@app.get("/model-info")
def model_info():
    # TODO 3c: return a dict with the keys "model_source", "threshold" and "features".
    # HINT: the values are MODEL_SOURCE, THRESHOLD and FEATURES (all defined above).
    # THINK: who would call this endpoint, and when? (someone debugging a strange prediction?)
    return {
        "model_source": MODEL_SOURCE,
        "threshold": THRESHOLD,
        "features": FEATURES
    }


@app.post("/predict", response_model=Prediction)
def predict(request: Request):
    start = time.perf_counter()
    X = pd.DataFrame([request.model_dump()])[FEATURES]
    probability = float(model.predict_proba(X)[0, 1])
    flagged = probability >= THRESHOLD

    # TODO 3b: turn the prediction into an action for the email gateway and the employee who receives the email.
    #   Option A (2 levels): if flagged -> "Show a warning banner before opening", else "Allow the link"
    #   Option B (3 levels): also "Block the link" when probability >= 0.7
    #   Option C (your own): e.g. include the probability in the message
    # THINK: who reads this? What would make them trust (or ignore) it?

    if probability >= 0.7 and flagged:
        action = "Block the link"
    elif flagged:
        action = (
            f"WARNING: Suspicious link detected ({probability:.0%} risk). "
            "Show a warning banner before opening."
        )
    else:
        action = "Allow the link"

    log.info("predict p=%.3f flagged=%s latency_ms=%.1f", probability, flagged, (time.perf_counter() - start) * 1000)
    return Prediction(probability=round(probability, 4), flagged=flagged, recommended_action=action)

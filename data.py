"""Synthetic URLs described by simple lexical features (inspired by public phishing-URL datasets)."""
import numpy as np
import pandas as pd

FEATURES = ["url_length", "num_dots", "num_digits", "has_https", "domain_age_days", "has_at_symbol"]
TARGET = "is_phishing"


def make_data(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    phishing = rng.random(n_rows) < 0.2                                  # ~20% of URLs are phishing

    def mix(legit, phish):
        return np.where(phishing, phish, legit)

    url_length = np.clip(mix(rng.normal(45, 15, n_rows), rng.normal(85, 35, n_rows)), 12, 400)
    num_dots = np.clip(mix(rng.poisson(2, n_rows), rng.poisson(4, n_rows)), 1, 20)
    num_digits = np.clip(mix(rng.poisson(2, n_rows), rng.poisson(9, n_rows)), 0, 80)
    has_https = (rng.random(n_rows) < mix(0.85, 0.55)).astype(int)
    domain_age = np.clip(mix(rng.exponential(2500, n_rows), rng.exponential(120, n_rows)), 0, 12000)
    has_at = (rng.random(n_rows) < mix(0.01, 0.15)).astype(int)
    # Some phishing URLs look almost normal: give 40% of them legit-looking values
    disguised = phishing & (rng.random(n_rows) < 0.4)
    url_length[disguised] = rng.normal(48, 15, disguised.sum()).clip(12, 400)
    num_digits[disguised] = rng.poisson(2, disguised.sum())
    num_dots[disguised] = rng.poisson(2, disguised.sum()).clip(1, 20)
    domain_age[disguised] = rng.exponential(900, disguised.sum())
    return pd.DataFrame({
        "url_length": url_length.round(0),
        "num_dots": num_dots,
        "num_digits": num_digits,
        "has_https": has_https,
        "domain_age_days": domain_age.round(0),
        "has_at_symbol": has_at,
        TARGET: phishing.astype(int),
    })

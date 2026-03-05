"""
PhishGuard AI — Backend Test Suite (unittest)
===============================================
Run with:  python3 -m unittest discover -s tests -v
"""

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    FEATURE_COLUMNS,
    ArtifactNotFoundError,
    InvalidURLError,
    PredictionResult,
    _count_subdomains,
    _normalise_url,
    _shannon_entropy,
    extract_features,
    feature_display_names,
    feature_security_notes,
    load_artifacts,
    predict,
)

LEGITIMATE_URLS = [
    "https://www.google.com/search?q=weather",
    "https://stackoverflow.com/questions/1234567",
    "https://github.com/anthropics/claude",
    "https://docs.python.org/3/library/pathlib.html",
]

PHISHING_URLS = [
    "http://paypal-secure-login.verify-account.com/update@user",
    "http://192.168.1.200/phish/login.php?redirect=paypal.com",
    "http://secure.paypal.verify.update.attacker.com/signin",
    "http://g00gle-login.tk/free-gift/claim.php?id=123456789",
]


def _make_dummy_model(prediction=1):
    model = MagicMock()
    model.predict.return_value = np.array([prediction])
    proba = [0.05, 0.95] if prediction == 1 else [0.97, 0.03]
    model.predict_proba.return_value = np.array([proba])
    return model


def _make_real_model():
    """A real sklearn model that can be serialised by joblib."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, len(FEATURE_COLUMNS)))
    y = rng.integers(0, 2, size=100)
    clf = LogisticRegression(max_iter=200, random_state=0)
    clf.fit(X, y)
    return clf


def _make_dummy_scaler():
    scaler = StandardScaler()
    rng = np.random.default_rng(42)
    scaler.fit(rng.standard_normal((200, len(FEATURE_COLUMNS))))
    return scaler


def _write_artifacts(directory):
    joblib.dump(_make_real_model(), directory / "model.pkl")
    joblib.dump(_make_dummy_scaler(), directory / "scaler.pkl")


class TestShannonEntropy(unittest.TestCase):
    def test_empty_returns_zero(self): self.assertEqual(_shannon_entropy(""), 0.0)
    def test_uniform_zero(self): self.assertAlmostEqual(_shannon_entropy("aaaa"), 0.0)
    def test_two_equal_chars(self): self.assertAlmostEqual(_shannon_entropy("ab"), 1.0)
    def test_four_equal_chars(self): self.assertAlmostEqual(_shannon_entropy("abcd"), 2.0)
    def test_returns_float(self): self.assertIsInstance(_shannon_entropy("hello"), float)
    def test_random_gt_structured(self):
        self.assertGreater(_shannon_entropy("a1B#x9Qz$m"), _shannon_entropy("aaaaaabbbb"))


class TestCountSubdomains(unittest.TestCase):
    def test_bare_domain(self): self.assertEqual(_count_subdomains("example.com"), 0)
    def test_www(self): self.assertEqual(_count_subdomains("www.example.com"), 1)
    def test_deep(self): self.assertEqual(_count_subdomains("a.b.c.d.example.com"), 4)
    def test_empty(self): self.assertEqual(_count_subdomains(""), 0)
    def test_non_negative(self): self.assertGreaterEqual(_count_subdomains("192.168.1.1"), 0)


class TestNormaliseURL(unittest.TestCase):
    def test_scheme_present(self): self.assertEqual(_normalise_url("https://example.com"), "https://example.com")
    def test_adds_http(self): self.assertEqual(_normalise_url("example.com"), "http://example.com")
    def test_strips_whitespace(self): self.assertEqual(_normalise_url("  https://x.com  "), "https://x.com")
    def test_empty(self): self.assertEqual(_normalise_url(""), "")


class TestExtractFeatures(unittest.TestCase):
    def test_11_features(self):
        feats = extract_features("https://example.com")
        self.assertEqual(len(feats), 11)
        self.assertEqual(set(feats.keys()), set(FEATURE_COLUMNS))

    def test_empty_raises(self):
        with self.assertRaises(InvalidURLError): extract_features("")

    def test_whitespace_raises(self):
        with self.assertRaises(InvalidURLError): extract_features("   ")

    def test_https_flag(self):
        self.assertEqual(extract_features("https://x.com")["has_https"], 1.0)
        self.assertEqual(extract_features("http://x.com")["has_https"], 0.0)

    def test_at_symbol(self):
        self.assertEqual(extract_features("http://fake@evil.com")["has_at_symbol"], 1.0)
        self.assertEqual(extract_features("https://google.com")["has_at_symbol"], 0.0)

    def test_ip_detection(self):
        self.assertEqual(extract_features("http://192.168.1.1/login")["has_ip_address"], 1.0)
        self.assertEqual(extract_features("https://google.com")["has_ip_address"], 0.0)

    def test_digit_ratio_bounds(self):
        r = extract_features("https://example123.com")["digit_ratio"]
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)

    def test_url_length(self):
        url = "https://example.com"
        self.assertEqual(extract_features(url)["url_length"], float(len(url)))

    def test_dot_count(self):
        url = "http://a.b.c.example.com"
        self.assertEqual(extract_features(url)["num_dots"], float(url.count(".")))

    def test_hyphen_count(self):
        url = "http://pay-pal-secure.com"
        self.assertEqual(extract_features(url)["num_hyphens"], float(url.count("-")))

    def test_subdomain_count(self):
        self.assertEqual(extract_features("http://a.b.c.example.com")["num_subdomains"], 3.0)

    def test_entropy_positive(self):
        self.assertGreater(extract_features("https://example.com")["url_entropy"], 0.0)

    def test_domain_length_positive(self):
        self.assertGreater(extract_features("https://example.com")["domain_length"], 0.0)

    def test_no_scheme_ok(self):
        feats = extract_features("example.com")
        self.assertEqual(feats["url_length"], float(len("example.com")))

    def test_all_floats(self):
        for k, v in extract_features("https://example.com/path?q=1").items():
            self.assertIsInstance(v, float, f"{k} must be float")

    def test_classic_phishing_signals(self):
        feats = extract_features("http://192.168.1.100/paypal-secure@update/login.php")
        self.assertEqual(feats["has_at_symbol"], 1.0)
        self.assertEqual(feats["has_ip_address"], 1.0)
        self.assertEqual(feats["has_https"], 0.0)
        self.assertGreater(feats["num_hyphens"], 0)

    def test_sample_legitimate_urls(self):
        for url in LEGITIMATE_URLS:
            with self.subTest(url=url):
                self.assertEqual(len(extract_features(url)), 11)

    def test_sample_phishing_urls(self):
        for url in PHISHING_URLS:
            with self.subTest(url=url):
                self.assertEqual(len(extract_features(url)), 11)


class TestLoadArtifacts(unittest.TestCase):
    def test_raises_model_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            joblib.dump(_make_dummy_scaler(), tmp_path / "scaler.pkl")
            with self.assertRaises(ArtifactNotFoundError) as ctx:
                load_artifacts(tmp_path / "model.pkl", tmp_path / "scaler.pkl", force_reload=True)
            self.assertIn("Model file not found", str(ctx.exception))

    def test_raises_scaler_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            joblib.dump(_make_real_model(), tmp_path / "model.pkl")
            with self.assertRaises(ArtifactNotFoundError) as ctx:
                load_artifacts(tmp_path / "model.pkl", tmp_path / "scaler.pkl", force_reload=True)
            self.assertIn("Scaler file not found", str(ctx.exception))

    def test_loads_successfully(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_artifacts(tmp_path)
            model, scaler = load_artifacts(tmp_path / "model.pkl", tmp_path / "scaler.pkl", force_reload=True)
            self.assertIsNotNone(model)
            self.assertIsNotNone(scaler)

    def test_cache_same_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_artifacts(tmp_path)
            m1, s1 = load_artifacts(tmp_path / "model.pkl", tmp_path / "scaler.pkl", force_reload=True)
            m2, s2 = load_artifacts(tmp_path / "model.pkl", tmp_path / "scaler.pkl")
            self.assertIs(m1, m2)
            self.assertIs(s1, s2)


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.model = _make_dummy_model(prediction=1)
        self.scaler = _make_dummy_scaler()

    def test_phishing_result(self):
        result = predict("http://192.168.1.1/paypal@evil.com", model=self.model, scaler=self.scaler)
        self.assertIsInstance(result, PredictionResult)
        self.assertTrue(result.is_phishing)
        self.assertEqual(result.label, "Phishing")

    def test_legitimate_result(self):
        result = predict("https://www.google.com", model=_make_dummy_model(0), scaler=self.scaler)
        self.assertFalse(result.is_phishing)
        self.assertEqual(result.label, "Legitimate")

    def test_confidence_range(self):
        r = predict("https://example.com", model=self.model, scaler=self.scaler)
        self.assertGreaterEqual(r.confidence, 0.0)
        self.assertLessEqual(r.confidence, 1.0)

    def test_features_in_result(self):
        r = predict("https://example.com", model=self.model, scaler=self.scaler)
        self.assertEqual(set(r.features.keys()), set(FEATURE_COLUMNS))

    def test_url_stored(self):
        url = "https://example.com/test"
        self.assertEqual(predict(url, model=self.model, scaler=self.scaler).url, url)

    def test_empty_raises(self):
        with self.assertRaises(InvalidURLError):
            predict("", model=self.model, scaler=self.scaler)

    def test_frozen_result(self):
        r = predict("https://example.com", model=self.model, scaler=self.scaler)
        with self.assertRaises((AttributeError, TypeError)):
            r.label = "Tampered"

    def test_str_repr(self):
        s = str(predict("https://example.com", model=self.model, scaler=self.scaler))
        self.assertIn("PredictionResult", s)
        self.assertIn("confidence", s)

    def test_no_artifacts_raises(self):
        """load_artifacts raises ArtifactNotFoundError when files are missing."""
        import tempfile
        from utils import load_artifacts as _load
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ArtifactNotFoundError):
                _load(Path(tmp) / "model.pkl", Path(tmp) / "scaler.pkl", force_reload=True)

    def test_all_sample_urls(self):
        for url in LEGITIMATE_URLS + PHISHING_URLS:
            with self.subTest(url=url):
                r = predict(url, model=self.model, scaler=self.scaler)
                self.assertEqual(len(r.features), 11)


class TestHelpers(unittest.TestCase):
    def test_display_names_complete(self):
        self.assertEqual(set(feature_display_names().keys()), set(FEATURE_COLUMNS))

    def test_security_notes_complete(self):
        self.assertEqual(set(feature_security_notes().keys()), set(FEATURE_COLUMNS))

    def test_display_names_non_empty(self):
        for k, v in feature_display_names().items():
            self.assertTrue(v, f"Display name for {k!r} is empty")

    def test_security_notes_non_empty(self):
        for k, v in feature_security_notes().items():
            self.assertTrue(v, f"Security note for {k!r} is empty")


if __name__ == "__main__":
    unittest.main(verbosity=2)

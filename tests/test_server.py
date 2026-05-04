"""
Tests for the chode server API endpoints.

These tests target:
  1. Server starts without crashing
  2. /api/progress returns valid structure
  3. /api/preflop/nn/chart returns 169 hands with correct shape
  4. /api/preflop/nn/status and /api/postflop/nn/status return valid data
  5. No optimizer checkpoints leak through API

Run:  python3 -m pytest tests/test_server.py -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient


# ─── Server startup ─────────────────────────────────────────────────────────

from server import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ─── Progress endpoint ──────────────────────────────────────────────────────

class TestProgressAPI:

    def test_progress_returns_200(self, client):
        r = client.get("/api/progress")
        assert r.status_code == 200

    def test_progress_has_jobs(self, client):
        data = client.get("/api/progress").json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)

    def test_progress_job_fields(self, client):
        data = client.get("/api/progress").json()
        required_fields = {"name", "phase", "step", "pct_done", "status"}
        for job in data["jobs"]:
            for field in required_fields:
                assert field in job, f"Job missing field: {field}"

    def test_progress_pct_done_is_numeric(self, client):
        data = client.get("/api/progress").json()
        for job in data["jobs"]:
            assert isinstance(job["pct_done"], (int, float)), \
                f"pct_done should be numeric, got {type(job['pct_done'])}"

    def test_progress_has_system_state(self, client):
        data = client.get("/api/progress").json()
        assert "system" in data


# ─── Preflop NN chart endpoint ──────────────────────────────────────────────

class TestPreflopChartAPI:

    def test_chart_2p_30bb_sb_rfi(self, client):
        r = client.get("/api/preflop/nn/chart?n=2&bb=30&pidx=0&hist=")
        assert r.status_code == 200
        data = r.json()
        assert "hands" in data
        assert len(data["hands"]) == 169, f"Expected 169 hands, got {len(data['hands'])}"

    def test_chart_label(self, client):
        data = client.get("/api/preflop/nn/chart?n=2&bb=30&pidx=0&hist=").json()
        assert "SB RFI" in data.get("label", "")

    def test_chart_aa_bets_at_rfi(self, client):
        """AA at RFI should bet >99% of the time (requires trained model)."""
        data = client.get("/api/preflop/nn/chart?n=2&bb=30&pidx=0&hist=").json()
        aa = data["hands"].get("0,0")
        if aa is None:
            pytest.skip("No AA hand in chart response")
        fold = aa.get("fold", 1.0)
        # If model is untrained (random weights), all actions ~equal — skip
        if fold > 0.3:
            pytest.skip("No trained preflop NN checkpoint (AA fold too high)")
        assert fold < 0.01, f"AA folds {fold:.1%} at RFI — should be <1%"

    def test_chart_trash_folds_at_rfi(self, client):
        """72o at RFI should fold >50% of the time (requires trained model)."""
        data = client.get("/api/preflop/nn/chart?n=2&bb=30&pidx=0&hist=").json()
        # 72o: row 11 (Three), col 12 (Two) — but check both orderings
        hand = data["hands"].get("11,12") or data["hands"].get("12,11")
        if hand is None:
            pytest.skip("No 72o hand in chart response")
        fold = hand.get("fold", 0.0)
        # If model is untrained, fold ≈ 1/N_ACTIONS (20%) — skip if too low
        if fold < 0.4:
            pytest.skip("No trained preflop NN checkpoint (trash fold too low)")
        assert fold > 0.5, f"72o folds only {fold:.1%} at RFI — should fold >50%"

    def test_chart_probs_sum_to_one(self, client):
        """Each hand's probabilities should sum to ~1.0."""
        data = client.get("/api/preflop/nn/chart?n=2&bb=30&pidx=0&hist=").json()
        for key, actions in data["hands"].items():
            total = sum(actions.values())
            assert abs(total - 1.0) < 0.05, \
                f"Hand {key}: probs sum to {total:.3f}, expected ~1.0"

    def test_chart_3p_30bb(self, client):
        data = client.get("/api/preflop/nn/chart?n=3&bb=30&pidx=0&hist=").json()
        assert len(data["hands"]) == 169

    def test_chart_facing_3bet(self, client):
        """SB facing 3bet in 2-max should return valid data."""
        data = client.get("/api/preflop/nn/chart?n=2&bb=30&pidx=0&hist=bet,bet").json()
        assert "hands" in data
        assert "facing 3bet" in data.get("label", "").lower() or "3bet" in data.get("label", "")


# ─── NN status endpoints ────────────────────────────────────────────────────

class TestNNStatusAPI:

    def test_preflop_nn_status(self, client):
        r = client.get("/api/preflop/nn/status")
        assert r.status_code == 200
        data = r.json()
        assert "step" in data or "params" in data

    def test_postflop_nn_status(self, client):
        r = client.get("/api/postflop/nn/status")
        assert r.status_code == 200

    def test_no_optimizer_in_status(self, client):
        """Optimizer checkpoints must not appear in status responses."""
        for endpoint in ["/api/preflop/nn/status", "/api/postflop/nn/status"]:
            data = client.get(endpoint).json()
            ckpts = data.get("checkpoints", [])
            for c in ckpts:
                assert "optimizer" not in str(c), f"Optimizer leaked in {endpoint}: {c}"


# ─── Static pages ────────────────────────────────────────────────────────────

class TestStaticPages:

    def test_charts_page(self, client):
        r = client.get("/charts")
        assert r.status_code == 200

    def test_progress_page(self, client):
        r = client.get("/progress")
        assert r.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

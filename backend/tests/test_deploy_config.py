"""What the deploy job relies on: the featured runs ship in the image, the data is pinned,
and the live site says which commit it runs."""

import re
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

from wellpilot.config import Settings
from wellpilot.main import create_app

ROOT = Path(__file__).parents[2]


def test_the_featured_runs_ship_in_the_image() -> None:
    """fly.toml's FEATURED_RUNS are what the public site lists; the image holds only the runs
    frozen in tests/fixtures/runs, so a featured run missing there would vanish from it."""
    featured = tomllib.loads((ROOT / "fly.toml").read_text())["env"]["FEATURED_RUNS"]
    runs = featured.split(",")
    assert runs and len(set(runs)) == len(runs)
    missing = [r for r in runs if not (ROOT / "backend/tests/fixtures/runs" / f"{r}.json").exists()]
    assert missing == []


def test_the_data_is_pinned() -> None:
    pin = dict(
        line.split("=", 1)
        for line in (ROOT / "deploy/data.env").read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert re.fullmatch(r"data-\d{8}-\d{4}", pin["DATA_RELEASE"])
    assert re.fullmatch(r"[0-9a-f]{64}", pin["DATA_SHA256"])


def test_health_names_the_commit(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, frontend_dir=tmp_path / "none", git_commit="abc123")
    health = TestClient(create_app(settings)).get("/api/health").json()
    assert health["commit"] == "abc123"

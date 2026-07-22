import json

import pytest

from app.services.hybrid import HybridArtifact, ServingError, get_artifact, score_candidates


def make_artifact_file(tmp_path, version="v2"):
    payload = {
        "model": "hybrid",
        "version": version,
        "params": {"weights": {"similarity": 0.6, "popularity": 0.4}},
        "popularity": {"1": 1.0, "2": 0.5, "3": 0.1},
        "neighbors": {"1": [[2, 0.9], [3, 0.2]]},
    }
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_missing_artifact_raises_serving_error(tmp_path):
    with pytest.raises(ServingError):
        get_artifact(str(tmp_path / "nope.json"))


def test_corrupt_artifact_raises_serving_error(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ServingError):
        get_artifact(str(path))


def test_version_mismatch_raises_serving_error(tmp_path):
    path = make_artifact_file(tmp_path, version="v2")
    with pytest.raises(ServingError):
        get_artifact(str(path), expected_version="v3")


def test_artifact_reloaded_after_rewrite(tmp_path):
    path = make_artifact_file(tmp_path, version="v2")
    assert get_artifact(str(path)).version == "v2"
    make_artifact_file(tmp_path, version="v3")
    import os
    import time
    os.utime(path, (time.time() + 10, time.time() + 10))
    assert get_artifact(str(path)).version == "v3"


def test_cold_start_scores_by_popularity(tmp_path):
    artifact = HybridArtifact.load(make_artifact_file(tmp_path))
    scored = score_candidates(artifact, liked_ids=[], candidate_ids=[1, 2, 3])
    ranked = sorted(scored, key=lambda g: -scored[g][0])
    assert ranked == [1, 2, 3]
    assert all(source is None for _, source in scored.values())


def test_similarity_dominates_and_reports_source(tmp_path):
    artifact = HybridArtifact.load(make_artifact_file(tmp_path))
    scored = score_candidates(artifact, liked_ids=[1], candidate_ids=[2, 3])
    assert scored[2][0] > scored[3][0]
    assert scored[2][1] == 1

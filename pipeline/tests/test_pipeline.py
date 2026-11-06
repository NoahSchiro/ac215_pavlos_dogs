"""Tests for the pipeline definition.

These compile the pipeline, which needs no GCP credentials, so CI can catch
a malformed spec without touching a cloud account.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from kfp import compiler

from gaze_pipeline import IMAGE, gaze_pipeline, gcs_mount

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spec(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("spec") / "pipeline.json"
    compiler.Compiler().compile(gaze_pipeline, str(out))
    return json.loads(out.read_text())


def test_pipeline_compiles_with_the_expected_steps(spec: dict) -> None:
    assert sorted(spec["root"]["dag"]["tasks"]) == [
        "evaluate-op",
        "preprocess-op",
        "train-op",
    ]


def test_steps_run_in_order(spec: dict) -> None:
    """Training must not start before the cache exists, nor evaluation before training."""
    tasks = spec["root"]["dag"]["tasks"]
    assert tasks["preprocess-op"].get("dependentTasks", []) == []
    assert tasks["train-op"]["dependentTasks"] == ["preprocess-op"]
    assert tasks["evaluate-op"]["dependentTasks"] == ["train-op"]


def test_every_step_uses_the_repository_image(spec: dict) -> None:
    """No step may introduce its own image: the pipeline orders work, it does
    not reimplement it."""
    containers = spec["deploymentSpec"]["executors"]
    images = {e["container"]["image"] for e in containers.values()}
    assert images == {IMAGE}


def test_training_flags_mirror_the_cli(spec: dict) -> None:
    executor = spec["deploymentSpec"]["executors"]["exec-train-op"]["container"]
    assert executor["command"] == ["uv", "run", "train.py"]
    flags = [a for a in executor["args"] if isinstance(a, str) and a.startswith("--")]
    assert {"--data-root", "--checkpoint-dir", "--epochs", "--val", "--test"} <= set(
        flags
    )


def test_pipeline_parameters_have_defaults(spec: dict) -> None:
    params = spec["root"]["inputDefinitions"]["parameters"]
    assert {"data_root", "checkpoint_dir", "epochs"} <= set(params)
    assert all("defaultValue" in p for p in params.values())


def test_gcs_mount_rewrites_bucket_uris() -> None:
    assert gcs_mount("gs://bucket/path") == "/gcs/bucket/path"
    assert gcs_mount("gs://bucket") == "/gcs/bucket"


def test_gcs_mount_rejects_a_non_bucket_uri() -> None:
    with pytest.raises(ValueError, match="gs:// uri"):
        gcs_mount("/local/path")


def test_image_is_overridable_from_the_environment() -> None:
    """submit.py sets GAZE_IMAGE before import, because KFP bakes the image
    into the compiled spec rather than taking it as a runtime parameter."""
    result = subprocess.run(
        [sys.executable, "-c", "import gaze_pipeline; print(gaze_pipeline.IMAGE)"],
        cwd=ROOT,
        env={**os.environ, "GAZE_IMAGE": "example.dev/custom:v2"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "example.dev/custom:v2"


def test_submit_compile_only_needs_no_credentials(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "submit.py", "--compile-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (ROOT / "gaze_pipeline.json").is_file()


def test_submit_refuses_the_placeholder_image() -> None:
    """The default image still says PROJECT; submitting it would fail opaquely."""
    result = subprocess.run(
        [sys.executable, "submit.py", "--project", "some-project"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "PROJECT placeholder" in result.stderr

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

from gaze_pipeline import _PUBLISH_SH, _TRAIN_SH, IMAGE, gaze_pipeline, gcs_mount
from submit import models_bucket_name

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
        "publish-op",
        "train-op",
    ]


def test_steps_run_in_order(spec: dict) -> None:
    """Training must not start before the cache exists, nor evaluation before training."""
    tasks = spec["root"]["dag"]["tasks"]
    assert tasks["preprocess-op"].get("dependentTasks", []) == []
    assert tasks["train-op"]["dependentTasks"] == ["preprocess-op"]
    assert tasks["evaluate-op"]["dependentTasks"] == ["train-op"]
    assert tasks["publish-op"]["dependentTasks"] == ["evaluate-op"]


def test_every_step_uses_the_repository_image(spec: dict) -> None:
    """No step may introduce its own image: the pipeline orders work, it does
    not reimplement it."""
    containers = spec["deploymentSpec"]["executors"]
    images = {e["container"]["image"] for e in containers.values()}
    assert images == {IMAGE}


def test_training_flags_mirror_the_cli(spec: dict) -> None:
    executor = spec["deploymentSpec"]["executors"]["exec-train-op"]["container"]
    assert executor["command"] == ["bash", "-c"]
    assert executor["args"][0] == _TRAIN_SH
    flags = [a for a in executor["args"] if isinstance(a, str) and a.startswith("--")]
    assert {"--data-root", "--checkpoint-dir", "--epochs", "--val", "--test"} <= set(
        flags
    )


def test_wandb_key_never_enters_the_pipeline_spec(spec: dict) -> None:
    """Only the key's path is a parameter; the spec is visible in the console."""
    params = spec["root"]["inputDefinitions"]["parameters"]
    assert params["wandb_key_path"]["defaultValue"].startswith("/gcs/")
    assert "WANDB_API_KEY" not in json.dumps(spec["root"])


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


def publish(tmp_path: Path, checkpoints: Path) -> subprocess.CompletedProcess[str]:
    """Run the publish step's script for real, with tmp_path standing in for /gcs."""
    out = tmp_path / "outputs" / "model_url"
    return subprocess.run(
        ["bash", "-c", _PUBLISH_SH, str(checkpoints), "models-bucket", str(out)],
        env={**os.environ, "GCS_MOUNT": str(tmp_path / "gcs")},
        capture_output=True,
        text=True,
        check=False,  # the tests assert on failures too
    )


def make_run(checkpoints: Path, name: str, model: bytes) -> None:
    run = checkpoints / name
    run.mkdir(parents=True)
    (run / "best.onnx").write_bytes(model)
    (run / "best_metrics.json").write_text('{"mean": 0.139}')


def test_publish_writes_the_newest_run_where_the_ui_loads_it(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    make_run(checkpoints, "20261101-090000", b"old")
    make_run(checkpoints, "20261107-120000", b"new")

    result = publish(tmp_path, checkpoints)

    assert result.returncode == 0, result.stderr
    published = tmp_path / "gcs" / "models-bucket" / "models" / "20261107-120000"
    assert (published / "gazenet.onnx").read_bytes() == b"new"
    assert (published / "metrics.json").read_text() == '{"mean": 0.139}'
    url = (tmp_path / "outputs" / "model_url").read_text()
    assert url == (
        "https://storage.googleapis.com/models-bucket/models/20261107-120000/gazenet.onnx"
    )


def test_publish_is_idempotent_but_never_overwrites_a_version(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    make_run(checkpoints, "20261107-120000", b"weights")
    assert publish(tmp_path, checkpoints).returncode == 0
    # A retried step republishes the same bytes: fine.
    assert publish(tmp_path, checkpoints).returncode == 0
    # Different bytes under a published version: refused, original intact.
    (checkpoints / "20261107-120000" / "best.onnx").write_bytes(b"different")
    result = publish(tmp_path, checkpoints)
    assert result.returncode != 0
    assert "immutable" in result.stderr
    published = tmp_path / "gcs/models-bucket/models/20261107-120000/gazenet.onnx"
    assert published.read_bytes() == b"weights"


def test_publish_fails_when_training_left_no_onnx(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "20261107-120000").mkdir(parents=True)
    result = publish(tmp_path, checkpoints)
    assert result.returncode != 0
    assert "no best.onnx" in result.stderr


def test_models_bucket_defaults_as_setup_gcp_creates_it() -> None:
    assert models_bucket_name(None, "my-proj") == "my-proj-gaze-models"
    assert models_bucket_name("gs://other-bucket/", "my-proj") == "other-bucket"


def train(
    tmp_path: Path, project: str, key: str | None, shm: Path | None = None
) -> tuple[int, str, str]:
    """Run the train step's wrapper with a fake `uv` that records how it ran."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        '#!/bin/sh\necho "argv=$*"\necho "key=${WANDB_API_KEY:-<unset>}"\n'
    )
    fake_uv.chmod(0o755)
    key_file = tmp_path / "wandb_api_key"
    if key is not None:
        key_file.write_text(key)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    env.pop("WANDB_API_KEY", None)
    # Default to a roomy /dev/shm (any real directory on this disk); tests that
    # care pass their own.
    env["SHM_DIR"] = str(shm if shm is not None else tmp_path)
    result = subprocess.run(
        ["bash", "-c", _TRAIN_SH, str(key_file), project, "--epochs", "2"],
        env=env,
        capture_output=True,
        text=True,
        check=False,  # the tests assert on failures too
    )
    return result.returncode, result.stdout, result.stderr


def test_train_logs_to_wandb_with_the_key_from_the_bucket(tmp_path: Path) -> None:
    code, out, _ = train(tmp_path, "gaze", key="secret-key")
    assert code == 0
    assert "argv=run train.py --wandb gaze --epochs 2" in out
    assert "key=secret-key" in out


def test_train_fails_fast_when_wandb_has_no_key(tmp_path: Path) -> None:
    code, out, err = train(tmp_path, "gaze", key=None)
    assert code != 0
    assert "no API key" in err
    assert "argv=" not in out  # training never started


def test_train_without_a_wandb_project_skips_wandb(tmp_path: Path) -> None:
    code, out, _ = train(tmp_path, "", key=None)
    assert code == 0
    assert "argv=run train.py --epochs 2" in out
    assert "--wandb" not in out


def test_train_keeps_parallel_loading_with_a_large_dev_shm(tmp_path: Path) -> None:
    code, out, _ = train(tmp_path, "", key=None)
    assert code == 0
    assert "keeping parallel data loading" in out
    assert "--workers" not in out


def test_train_loads_in_process_without_a_usable_dev_shm(tmp_path: Path) -> None:
    code, out, _ = train(tmp_path, "gaze", key="k", shm=tmp_path / "missing")
    assert code == 0
    assert "argv=run train.py --wandb gaze --epochs 2 --workers 0" in out

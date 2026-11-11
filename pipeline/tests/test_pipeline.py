"""Tests for the pipeline definition.

These compile the pipeline, which needs no GCP credentials, so CI can catch
a malformed spec without touching a cloud account.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from kfp import compiler

from gaze_pipeline import (
    _EVALUATE_SH,
    _PUBLISH_SH,
    _TRAIN_SH,
    IMAGE,
    gaze_pipeline,
    gcs_mount,
)
from submit import models_bucket_name, resolve_run_id

Spec = dict[str, Any]

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spec(tmp_path_factory: pytest.TempPathFactory) -> Spec:
    out = tmp_path_factory.mktemp("spec") / "pipeline.json"
    compiler.Compiler().compile(gaze_pipeline, str(out))
    loaded: Spec = json.loads(out.read_text())
    return loaded


def test_pipeline_compiles_with_the_expected_steps(spec: Spec) -> None:
    assert sorted(spec["root"]["dag"]["tasks"]) == [
        "evaluate-op",
        "preprocess-op",
        "publish-op",
        "train-op",
    ]


def test_steps_run_in_order(spec: Spec) -> None:
    """Training must not start before the cache exists, nor evaluation before training."""
    tasks = spec["root"]["dag"]["tasks"]
    assert tasks["preprocess-op"].get("dependentTasks", []) == []
    assert tasks["train-op"]["dependentTasks"] == ["preprocess-op"]
    assert tasks["evaluate-op"]["dependentTasks"] == ["train-op"]
    assert tasks["publish-op"]["dependentTasks"] == ["evaluate-op"]


def test_every_step_uses_the_repository_image(spec: Spec) -> None:
    """No step may introduce its own image: the pipeline orders work, it does
    not reimplement it."""
    containers = spec["deploymentSpec"]["executors"]
    images = {e["container"]["image"] for e in containers.values()}
    assert images == {IMAGE}


def test_training_flags_mirror_the_cli(spec: Spec) -> None:
    executor = spec["deploymentSpec"]["executors"]["exec-train-op"]["container"]
    assert executor["command"] == ["bash", "-c"]
    assert executor["args"][0] == _TRAIN_SH
    flags = [a for a in executor["args"] if isinstance(a, str) and a.startswith("--")]
    assert {"--data-root", "--epochs", "--val", "--test"} <= set(flags)
    # --checkpoint-dir is added by the script, as <checkpoint_dir>/<run_id>.
    assert "--checkpoint-dir" not in flags


def test_every_checkpoint_step_is_scoped_to_the_run(spec: Spec) -> None:
    """train, evaluate and publish all receive run_id, so none of them can
    reach into another run's folder."""
    tasks = spec["root"]["dag"]["tasks"]
    for name in ("train-op", "evaluate-op", "publish-op"):
        inputs = tasks[name]["inputs"]["parameters"]
        assert inputs["run_id"]["componentInputParameter"] == "run_id", name


def test_wandb_key_never_enters_the_pipeline_spec(spec: Spec) -> None:
    """Only the key's path is a parameter; the spec is visible in the console."""
    params = spec["root"]["inputDefinitions"]["parameters"]
    assert params["wandb_key_path"]["defaultValue"].startswith("/gcs/")
    assert "WANDB_API_KEY" not in json.dumps(spec["root"])


def test_pipeline_parameters_have_defaults(spec: Spec) -> None:
    params = spec["root"]["inputDefinitions"]["parameters"]
    assert {"data_root", "checkpoint_dir", "run_id", "epochs"} <= set(params)
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


RUN = "20261111-090000"  # a pipeline run id, as submit.py makes them


def publish(
    tmp_path: Path, checkpoints: Path, run_id: str = RUN
) -> subprocess.CompletedProcess[str]:
    """Run the publish step's script for real, with tmp_path standing in for /gcs."""
    out = tmp_path / "outputs" / "model_url"
    return subprocess.run(
        [
            "bash",
            "-c",
            _PUBLISH_SH,
            str(checkpoints),
            run_id,
            "models-bucket",
            str(out),
        ],
        env={**os.environ, "GCS_MOUNT": str(tmp_path / "gcs")},
        capture_output=True,
        text=True,
        check=False,  # the tests assert on failures too
    )


def make_run(
    checkpoints: Path,
    name: str,
    model: bytes,
    run_id: str = RUN,
    metrics: str = '{"mean": 0.139}',
) -> None:
    """A training attempt as train.py leaves it: <run_id>/<timestamp>/best.*"""
    run = checkpoints / run_id / name
    run.mkdir(parents=True)
    (run / "best.onnx").write_bytes(model)
    (run / "best_metrics.json").write_text(metrics)


def test_publish_writes_the_newest_attempt_where_the_ui_loads_it(
    tmp_path: Path,
) -> None:
    """Within one pipeline run, a retried train step can leave two attempts;
    the newest is the one that finished."""
    checkpoints = tmp_path / "checkpoints"
    make_run(checkpoints, "20261111-090000", b"old")
    make_run(checkpoints, "20261111-120000", b"new")

    result = publish(tmp_path, checkpoints)

    assert result.returncode == 0, result.stderr
    published = tmp_path / "gcs" / "models-bucket" / "models" / "20261111-120000"
    assert (published / "gazenet.onnx").read_bytes() == b"new"
    assert (published / "metrics.json").read_text() == '{"mean": 0.139}'
    url = (tmp_path / "outputs" / "model_url").read_text()
    assert url == (
        "https://storage.googleapis.com/models-bucket/models/20261111-120000/gazenet.onnx"
    )


def test_publish_is_idempotent_but_never_overwrites_a_version(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    make_run(checkpoints, "20261107-120000", b"weights")
    assert publish(tmp_path, checkpoints).returncode == 0
    # A retried step republishes the same bytes: fine.
    assert publish(tmp_path, checkpoints).returncode == 0
    # Different bytes under a published version: refused, original intact.
    (checkpoints / RUN / "20261107-120000" / "best.onnx").write_bytes(b"different")
    result = publish(tmp_path, checkpoints)
    assert result.returncode != 0
    assert "immutable" in result.stderr
    published = tmp_path / "gcs/models-bucket/models/20261107-120000/gazenet.onnx"
    assert published.read_bytes() == b"weights"


def test_publish_fails_when_training_left_no_onnx(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / RUN / "20261107-120000").mkdir(parents=True)
    result = publish(tmp_path, checkpoints)
    assert result.returncode != 0
    assert "no best.onnx" in result.stderr


def test_publish_never_takes_a_concurrent_runs_model(tmp_path: Path) -> None:
    """Another pipeline run finished later and its model is newer, but this
    run must publish its own."""
    checkpoints = tmp_path / "checkpoints"
    make_run(checkpoints, "20261111-090500", b"mine", run_id="run-a")
    make_run(checkpoints, "20261111-091500", b"theirs", run_id="run-b")

    result = publish(tmp_path, checkpoints, run_id="run-a")

    assert result.returncode == 0, result.stderr
    models = tmp_path / "gcs" / "models-bucket" / "models"
    assert (models / "20261111-090500" / "gazenet.onnx").read_bytes() == b"mine"
    assert not (models / "20261111-091500").exists()


def test_publish_fails_when_this_run_trained_nothing(tmp_path: Path) -> None:
    """Even if other runs have models, a run with none must not borrow one."""
    checkpoints = tmp_path / "checkpoints"
    make_run(checkpoints, "20261111-091500", b"theirs", run_id="run-b")
    result = publish(tmp_path, checkpoints, run_id="run-a")
    assert result.returncode != 0
    assert "no best.onnx" in result.stderr


@pytest.mark.parametrize("bad", ["", "..", "a/b", "../other"])
def test_steps_reject_a_run_id_that_is_not_one_folder(tmp_path: Path, bad: str) -> None:
    checkpoints = tmp_path / "checkpoints"
    make_run(checkpoints, "20261111-090500", b"m")
    result = publish(tmp_path, checkpoints, run_id=bad)
    assert result.returncode != 0
    assert "invalid run id" in result.stderr


def evaluate(tmp_path: Path, checkpoints: Path, run_id: str) -> tuple[int, str]:
    out = tmp_path / "outputs" / "metrics"
    result = subprocess.run(
        ["bash", "-c", _EVALUATE_SH, str(checkpoints), run_id, str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, out.read_text() if out.exists() else result.stderr


def test_evaluate_reports_its_own_runs_metrics(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    make_run(
        checkpoints, "20261111-090500", b"m", run_id="run-a", metrics='{"mean": 0.1}'
    )
    make_run(
        checkpoints, "20261111-091500", b"m", run_id="run-b", metrics='{"mean": 0.9}'
    )
    code, out = evaluate(tmp_path, checkpoints, "run-a")
    assert code == 0
    assert out == '{"mean": 0.1}'


def test_evaluate_fails_when_the_run_has_no_folder(tmp_path: Path) -> None:
    code, err = evaluate(tmp_path, tmp_path / "checkpoints", "run-a")
    assert code != 0
    assert "no run directory" in err


def test_run_id_defaults_to_a_fresh_utc_timestamp() -> None:
    now = datetime(2026, 11, 11, 15, 30, 5, tzinfo=UTC)
    assert resolve_run_id(None, now) == "20261111-153005"


def test_run_id_must_be_a_single_safe_folder_name() -> None:
    assert resolve_run_id("replay-1") == "replay-1"
    for bad in ["", "..", "a/b", "has space"]:
        with pytest.raises(SystemExit):
            resolve_run_id(bad)


def test_models_bucket_defaults_as_setup_gcp_creates_it() -> None:
    assert models_bucket_name(None, "my-proj") == "my-proj-gaze-models"
    assert models_bucket_name("gs://other-bucket/", "my-proj") == "other-bucket"


CKPT = f"--checkpoint-dir /gcs/b/checkpoints/{RUN}"


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
        [
            "bash",
            "-c",
            _TRAIN_SH,
            str(key_file),
            project,
            "/gcs/b/checkpoints",
            RUN,
            "--epochs",
            "2",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,  # the tests assert on failures too
    )
    return result.returncode, result.stdout, result.stderr


def test_train_logs_to_wandb_with_the_key_from_the_bucket(tmp_path: Path) -> None:
    code, out, _ = train(tmp_path, "gaze", key="secret-key")
    assert code == 0
    assert f"argv=run train.py --wandb gaze {CKPT} --epochs 2" in out
    assert "key=secret-key" in out


def test_train_fails_fast_when_wandb_has_no_key(tmp_path: Path) -> None:
    code, out, err = train(tmp_path, "gaze", key=None)
    assert code != 0
    assert "no API key" in err
    assert "argv=" not in out  # training never started


def test_train_without_a_wandb_project_skips_wandb(tmp_path: Path) -> None:
    code, out, _ = train(tmp_path, "", key=None)
    assert code == 0
    assert f"argv=run train.py {CKPT} --epochs 2" in out
    assert "--wandb" not in out


def test_train_keeps_parallel_loading_with_a_large_dev_shm(tmp_path: Path) -> None:
    code, out, _ = train(tmp_path, "", key=None)
    assert code == 0
    assert "keeping parallel data loading" in out
    assert "--workers" not in out


def test_train_loads_in_process_without_a_usable_dev_shm(tmp_path: Path) -> None:
    code, out, _ = train(tmp_path, "gaze", key="k", shm=tmp_path / "missing")
    assert code == 0
    assert f"argv=run train.py --wandb gaze {CKPT} --epochs 2 --workers 0" in out


def test_train_writes_under_this_runs_folder(tmp_path: Path) -> None:
    """train.py adds its <timestamp> under --checkpoint-dir, so this run's
    weights land in <checkpoints>/<run_id>/<timestamp>/."""
    code, out, _ = train(tmp_path, "", key=None)
    assert code == 0
    assert f"--checkpoint-dir /gcs/b/checkpoints/{RUN} " in out

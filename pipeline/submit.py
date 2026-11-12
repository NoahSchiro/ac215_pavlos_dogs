"""Compile the pipeline and optionally submit it to Vertex AI.

    uv run submit.py --compile-only                 # writes gaze_pipeline.json
    uv run submit.py --project P --image IMG        # compiles and submits

Compiling needs no credentials, which is what CI checks. Submitting needs
an authenticated gcloud and the Vertex AI API enabled on the project.
"""

import argparse
import string
from datetime import UTC, datetime
from pathlib import Path

from google.cloud import aiplatform
from kfp import compiler

from gaze_pipeline import PLACEHOLDER_IMAGE, gcs_mount, make_pipeline

DEFAULT_REGION = "us-east1"
DEFAULT_BUCKET = "gs://ac215-gaze-data-rh"
SPEC_PATH = Path(__file__).parent / "gaze_pipeline.json"
# A run id names a folder, so it must be one plain path segment.
RUN_ID_CHARS = frozenset(string.ascii_letters + string.digits + "._-")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", help="GCP project id; required to submit")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--image",
        help="training image, e.g."
        " us-east1-docker.pkg.dev/<project>/gaze/gaze-model:latest."
        " Written into every step's container spec when the pipeline is built.",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="bucket holding the dataset; also where Vertex stages run artifacts",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val", type=int, default=13)
    parser.add_argument("--test", type=int, default=14)
    parser.add_argument(
        "--wandb-project",
        default="gaze",
        help="W&B project to log to (the team's is `gaze`); '' to skip W&B",
    )
    parser.add_argument(
        "--wandb-key",
        default="secrets/wandb_api_key",
        help="object in --bucket holding the submitter's own W&B API key, so a"
        " run logs as whoever launched it (e.g. secrets/wandb_api_key_<name>)",
    )
    parser.add_argument(
        "--models-bucket",
        help="public bucket the trained model is published to for the UI;"
        " defaults to <project>-gaze-models, as deploy/setup-gcp.sh creates it",
    )
    parser.add_argument(
        "--run-id",
        help="folder this run's checkpoints go in, under <bucket>/checkpoints/;"
        " defaults to a new UTC timestamp, so every submission trains fresh."
        " Reuse an earlier id to let Vertex replay that run from its cache",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="rerun every step; by default Vertex reuses any step whose image"
        " and inputs are unchanged, which would skip training on a rerun",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="write the spec and stop; needs no GCP credentials",
    )
    return parser


def models_bucket_name(arg: str | None, project: str) -> str:
    """The models bucket as a bare name, defaulting as deploy/setup-gcp.sh does."""
    return (arg or f"{project}-gaze-models").removeprefix("gs://").rstrip("/")


def resolve_run_id(arg: str | None, now: datetime | None = None) -> str:
    """The given run id, checked; otherwise a fresh UTC timestamp like 20261111-153000."""
    if arg is None:
        return (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    if not arg or not set(arg) <= RUN_ID_CHARS or arg in {".", ".."}:
        raise SystemExit(
            f"--run-id must be letters, digits, '.', '_' or '-', got {arg!r}"
        )
    return arg


def main() -> None:
    args = build_parser().parse_args()
    run_id = resolve_run_id(args.run_id)

    image = args.image or PLACEHOLDER_IMAGE
    compiler.Compiler().compile(make_pipeline(image), str(SPEC_PATH))
    print(f"compiled {SPEC_PATH} using image {image}")

    if args.compile_only:
        return
    if not args.project:
        raise SystemExit(
            "--project is required to submit; use --compile-only otherwise"
        )
    if image == PLACEHOLDER_IMAGE or "PROJECT" in image:
        raise SystemExit("--image is required to submit (got the placeholder)")

    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=f"{args.bucket}/vertex-staging",
    )
    job = aiplatform.PipelineJob(
        display_name="gaze-training",
        template_path=str(SPEC_PATH),
        pipeline_root=f"{args.bucket}/pipeline-root",
        parameter_values={
            "data_root": f"/gcs/{args.bucket.removeprefix('gs://')}/MPIIFaceGaze",
            "checkpoint_dir": f"/gcs/{args.bucket.removeprefix('gs://')}/checkpoints",
            "run_id": run_id,
            "epochs": args.epochs,
            "val_participant": args.val,
            "test_participant": args.test,
            "wandb_project": args.wandb_project,
            "wandb_key_path": f"{gcs_mount(args.bucket)}/{args.wandb_key.lstrip('/')}",
            "models_bucket": models_bucket_name(args.models_bucket, args.project),
        },
        enable_caching=not args.no_cache,
    )
    job.submit()
    print(f"submitted run {run_id}: {job._dashboard_uri()}")


if __name__ == "__main__":
    main()

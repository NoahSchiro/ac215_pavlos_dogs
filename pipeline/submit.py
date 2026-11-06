"""Compile the pipeline and optionally submit it to Vertex AI.

    uv run submit.py --compile-only                 # writes gaze_pipeline.json
    uv run submit.py --project P --image IMG        # compiles and submits

Compiling needs no credentials, which is what CI checks. Submitting needs
an authenticated gcloud and the Vertex AI API enabled on the project.
"""

import argparse
import os
from pathlib import Path

DEFAULT_REGION = "us-east1"
DEFAULT_BUCKET = "gs://ac215-gaze-data-rh"
SPEC_PATH = Path(__file__).parent / "gaze_pipeline.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", help="GCP project id; required to submit")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--image",
        help="training image, e.g."
        " us-east1-docker.pkg.dev/<project>/gaze/gaze-model:latest."
        " Resolved at compile time, so it is set before the pipeline is imported.",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="bucket holding the dataset; also where Vertex stages run artifacts",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val", type=int, default=13)
    parser.add_argument("--test", type=int, default=14)
    parser.add_argument("--wandb-project", default="gaze-maze")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="write the spec and stop; needs no GCP credentials",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # The image is baked into the compiled spec, so it must be set before the
    # pipeline module is imported.
    if args.image:
        os.environ["GAZE_IMAGE"] = args.image

    from kfp import compiler

    from gaze_pipeline import IMAGE, gaze_pipeline

    compiler.Compiler().compile(gaze_pipeline, str(SPEC_PATH))
    print(f"compiled {SPEC_PATH} using image {IMAGE}")

    if args.compile_only:
        return
    if not args.project:
        raise SystemExit(
            "--project is required to submit; use --compile-only otherwise"
        )
    if "PROJECT" in IMAGE:
        raise SystemExit("--image still contains the PROJECT placeholder")

    from google.cloud import aiplatform

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
            "epochs": args.epochs,
            "val_participant": args.val,
            "test_participant": args.test,
            "wandb_project": args.wandb_project,
        },
        enable_caching=True,
    )
    job.submit()
    print(f"submitted: {job._dashboard_uri()}")


if __name__ == "__main__":
    main()

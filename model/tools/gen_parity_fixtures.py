"""Regenerate the TypeScript parity fixtures from the Python pipeline.

`ui/src/vision/` reimplements this package's preprocessing so the model can
run in the browser, and `ui/src/vision/parity.test.ts` checks that port
against reference outputs recorded here. Those references were previously
committed with no way to recreate them, which made the check one-sided: a
change to `src/data/utils.py` left the fixtures stale, the TypeScript suite
kept passing against them, and the browser quietly fed the model different
crops than training produced.

Run this whenever the preprocessing changes:

    uv run python tools/gen_parity_fixtures.py

CI runs it with --check, which regenerates into a temporary directory and
fails if anything differs from what is committed.

MediaPipe aborts on macOS (it fails inside Metal), so run this in the
training container rather than on a Mac:

    docker run --rm -v "$PWD/..:/repo" -w /repo/model --entrypoint uv \\
        gaze-model:dev run python tools/gen_parity_fixtures.py
"""

import argparse
import filecmp
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import CameraIntrinsics, PointPx
from src.data.utils import (
    _GENERIC_FACE_MODEL,
    _MODEL_LANDMARK_IDS,
    NORM_EYE_SIZE_PX,
    NORM_FACE_SIZE_PX,
    get_landmarker,
    head_pose_est,
    landmark_detection,
    normalization_transforms,
    normalize_face,
)

MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = MODEL_ROOT / "tests" / "data" / "face.jpg"
DEFAULT_OUT = MODEL_ROOT.parent / "ui" / "src" / "vision" / "fixtures"

# The calibration face.jpg was captured with. parity.test.ts hard-codes the
# same numbers; they must agree or every recorded matrix below is meaningless.
MPII_INTRINSICS = CameraIntrinsics(
    focal_length_px=996.4509,
    principal_point=PointPx(x=624.6634, y=364.0874),
)

# Probe geometry. Small enough to keep the JSON readable, large enough that
# every interpolation and border path in cv2 gets exercised.
WARP_SRC_SHAPE = (40, 50)
WARP_DSIZE = (32, 24)  # (width, height)
SUBPIX_SRC_SHAPE = (30, 40)
SUBPIX_WIN = (20, 10)  # (width, height)
PROBE_K = (1000.0, 640.0, 360.0)  # (focal, cx, cy)


def _json(obj: Any) -> str:
    """Match the formatting of the fixtures already in the tree: one line."""
    return json.dumps(obj)


def _reference(source: Path) -> tuple[dict[str, Any], dict[str, NDArray[np.uint8]]]:
    """Run the real pipeline over `source` and record every intermediate.

    This is the fixture that matters: it is the only one that changes when
    preprocessing changes, so it is what makes drift visible.
    """
    bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read {source}")
    image = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)

    landmarks = landmark_detection(get_landmarker(), image)
    if landmarks is None:
        raise SystemExit(f"no face detected in {source}")

    rotation, translation = head_pose_est(landmarks, MPII_INTRINSICS)
    transforms = normalization_transforms(
        landmarks, rotation, translation, MPII_INTRINSICS
    )
    face, left, right, r_virtual = normalize_face(
        image, landmarks, rotation, translation, MPII_INTRINSICS
    )

    # normalize_face must agree with the transforms it is built from; if this
    # ever trips, the refactor that split them apart has drifted.
    assert np.array_equal(r_virtual, transforms.r_virtual)
    assert face.shape == (NORM_FACE_SIZE_PX, NORM_FACE_SIZE_PX, 3), face.shape
    assert left.shape == (*NORM_EYE_SIZE_PX, 3), left.shape

    reference = {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "landmarks": [[p.x, p.y] for p in landmarks],
        "R": rotation.tolist(),
        "t": translation.tolist(),
        "R_virtual": transforms.r_virtual.tolist(),
        "H_norm": transforms.h_norm.tolist(),
        "H_src_to_dst": transforms.h_src_to_dst.tolist(),
        "left_xy": list(transforms.left_xy),
        "right_xy": list(transforms.right_xy),
    }
    return reference, {"face": face, "left": left, "right": right}


def _warp_probes(rng: np.random.Generator) -> dict[str, Any]:
    """cv2.warpPerspective on random homographies, borders included."""
    rows, cols = WARP_SRC_SHAPE
    src = rng.integers(0, 256, (rows, cols, 3), dtype=np.uint8)
    cases = []
    for _ in range(6):
        h = np.eye(3)
        h[:2, :2] += rng.uniform(-0.3, 0.3, (2, 2))
        h[:2, 2] = rng.uniform(-8.0, 8.0, 2)
        # A small perspective term; too large and the whole output falls
        # outside the source, which tests nothing.
        h[2, :2] = rng.uniform(-0.004, 0.004, 2)
        dst = cv2.warpPerspective(
            src, h, WARP_DSIZE, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        )
        cases.append({"H": h.tolist(), "dst": np.asarray(dst).tolist()})
    return {
        "src_shape": list(WARP_SRC_SHAPE),
        "src": src.tolist(),
        "dsize": list(WARP_DSIZE),
        "cases": cases,
    }


def _subpix_probes(rng: np.random.Generator) -> dict[str, Any]:
    """cv2.getRectSubPix, including centres that hang off every edge."""
    rows, cols = SUBPIX_SRC_SHAPE
    src = rng.integers(0, 256, (rows, cols, 3), dtype=np.uint8)
    centres = [
        (10.5, 10.25),  # interior, half-pixel
        (0.0, 0.0),  # top-left corner, fully clamped
        (float(cols - 1), float(rows - 1)),  # bottom-right corner
        (float(cols) / 2, 0.75),  # off the top edge
        (0.75, float(rows) / 2),  # off the left edge
        (float(cols) - 0.75, float(rows) / 2),  # off the right edge
        (float(cols) / 2, float(rows) - 0.75),  # off the bottom edge
        (-3.5, -2.25),  # entirely outside, top-left
        (float(cols) + 3.5, float(rows) + 2.25),  # entirely outside, bottom-right
        (17.3333, 9.6667),  # interior, awkward fraction
    ]
    cases = [
        {
            "center": list(c),
            "patch": np.asarray(cv2.getRectSubPix(src, SUBPIX_WIN, c)).tolist(),
        }
        for c in centres
    ]
    return {
        "src_shape": list(SUBPIX_SRC_SHAPE),
        "src": src.tolist(),
        "win": list(SUBPIX_WIN),
        "cases": cases,
    }


def _rodrigues_probes(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Rodrigues both ways, with the degenerate rotations spelled out.

    The round trip is only well defined away from 0 and pi, so `R3` is null
    for the cases where cv2 is free to return either of two equivalent
    vectors and the TypeScript port is not required to agree.
    """
    vectors = [
        np.zeros(3),  # identity
        np.array([np.pi, 0.0, 0.0]),  # pi about x: the ambiguous case
        np.array([0.0, np.pi, 0.0]),
        np.array([0.0, 0.0, np.pi]),
        np.array([np.pi / 2, 0.0, 0.0]),
        np.array([0.0, np.pi / 2, 0.0]),
        np.array([0.0, 0.0, np.pi / 2]),
        np.array([1e-9, 0.0, 0.0]),  # near-identity, exercises the small-angle path
    ]
    vectors += [rng.uniform(-2.0, 2.0, 3) for _ in range(5)]

    cases = []
    for v in vectors:
        r = np.asarray(cv2.Rodrigues(v)[0], dtype=np.float64)
        v2 = np.asarray(cv2.Rodrigues(r)[0], dtype=np.float64).reshape(3)
        angle = float(np.linalg.norm(v2))
        ambiguous = angle < 1e-8 or abs(angle - np.pi) < 1e-8
        r3 = None if ambiguous else np.asarray(cv2.Rodrigues(v2)[0]).tolist()
        cases.append(
            {"v": np.asarray(v, dtype=np.float64).tolist(), "R": r.tolist(),
             "v2": v2.tolist(), "R3": r3}
        )
    return cases


def _synthetic_landmarks(
    rotation: NDArray[np.float64], translation: NDArray[np.float64]
) -> list[list[float]]:
    """Project the generic face model under a known pose into a full mesh.

    solvePnP only reads the eight indices in `_MODEL_LANDMARK_IDS`, but the
    TypeScript test indexes into a 478-point array, so the rest is padded.
    """
    rvec = cv2.Rodrigues(rotation)[0]
    k = np.array(
        [[PROBE_K[0], 0.0, PROBE_K[1]], [0.0, PROBE_K[0], PROBE_K[2]], [0.0, 0.0, 1.0]]
    )
    projected, _ = cv2.projectPoints(_GENERIC_FACE_MODEL, rvec, translation, k, None)
    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)

    mesh = [[0.0, 0.0] for _ in range(478)]
    for point, landmark_id in zip(projected, _MODEL_LANDMARK_IDS, strict=True):
        mesh[landmark_id] = point.tolist()
    return mesh


def _pnp_probes(rng: np.random.Generator) -> list[dict[str, Any]]:
    """solvePnP against poses it should recover exactly (no noise, no outliers)."""
    poses = [
        (np.zeros(3), np.array([0.0, 0.0, 60.0])),  # straight on
        (np.array([0.2, -0.3, 0.05]), np.array([2.0, -1.0, 55.0])),
        (np.array([-0.35, 0.25, -0.1]), np.array([-3.0, 2.0, 65.0])),
        (rng.uniform(-0.3, 0.3, 3), np.array([1.0, 1.0, 58.0])),
    ]
    k = np.array(
        [[PROBE_K[0], 0.0, PROBE_K[1]], [0.0, PROBE_K[0], PROBE_K[2]], [0.0, 0.0, 1.0]]
    )
    cases = []
    for rvec_true, tvec_true in poses:
        r_true = np.asarray(cv2.Rodrigues(rvec_true)[0], dtype=np.float64)
        mesh = _synthetic_landmarks(r_true, tvec_true)
        image_points = np.array(
            [mesh[i] for i in _MODEL_LANDMARK_IDS], dtype=np.float64
        )
        found, rvec, tvec = cv2.solvePnP(
            _GENERIC_FACE_MODEL, image_points, k, None, flags=cv2.SOLVEPNP_SQPNP
        )
        if not found:
            raise SystemExit("solvePnP failed on a synthetic probe")
        cases.append({
            "landmarks": mesh,
            "K": list(PROBE_K),
            "rvec_est": np.asarray(rvec, dtype=np.float64).reshape(3).tolist(),
            "tvec_est": np.asarray(tvec, dtype=np.float64).reshape(3).tolist(),
            "R_true": r_true.tolist(),
            "tvec_true": np.asarray(tvec_true, dtype=np.float64).tolist(),
        })
    return cases


def _project_probe() -> dict[str, Any]:
    """cv2.projectPoints on the generic model under one fixed pose."""
    rvec = np.array([0.15, -0.25, 0.05])
    t = np.array([1.5, -2.0, 57.0])
    k = np.array(
        [[PROBE_K[0], 0.0, PROBE_K[1]], [0.0, PROBE_K[0], PROBE_K[2]], [0.0, 0.0, 1.0]]
    )
    pts, _ = cv2.projectPoints(_GENERIC_FACE_MODEL, rvec, t, k, None)
    return {
        "rvec": rvec.tolist(),
        "t": t.tolist(),
        "K": list(PROBE_K),
        "pts": np.asarray(pts, dtype=np.float64).reshape(-1, 2).tolist(),
    }


def generate(out_dir: Path, source: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    reference, crops = _reference(source)
    (out_dir / "ref.json").write_text(_json(reference), encoding="utf-8")
    for name, array in crops.items():
        (out_dir / f"{name}_crop.bin").write_bytes(array.tobytes())

    probes = {
        "warp": _warp_probes(rng),
        "rodrigues": _rodrigues_probes(rng),
        "pnp": _pnp_probes(rng),
        "misc": {"projectPoints_probe": _project_probe()},
        "subpix": _subpix_probes(rng),
    }
    (out_dir / "cv2_probes.json").write_text(_json(probes), encoding="utf-8")

    # face.jpg is the input, not an output: keep the copy the UI imports in
    # step with the one the model tests use.
    (out_dir / "face.jpg").write_bytes(source.read_bytes())


FIXTURES = (
    "ref.json",
    "cv2_probes.json",
    "face_crop.bin",
    "left_crop.bin",
    "right_crop.bin",
    "face.jpg",
)

# MediaPipe and OpenCV are not bit-reproducible across CPU architectures:
# XNNPACK and the OpenCV kernels pick different code paths, so landmarks move
# by ~0.01px on arm64 vs x86-64 and the crops shift by at most one LSB. The
# committed fixtures were recorded on x86-64 (parity.test.ts says so), so a
# byte comparison would fail for anyone on an Apple Silicon Mac while passing
# in CI, which is worse than useless.
#
# These bounds are two orders of magnitude tighter than any real change to
# preprocessing -- altering the normalization distance, an interpolation flag
# or a landmark index moves these numbers by percent, not by 1e-4 -- while
# staying well clear of that cross-platform noise.
TOLERANCES = {
    "landmarks": 0.1,  # px; cross-arch drift measured at 0.014
    "R": 1e-3,
    "t": 1e-2,
    "R_virtual": 1e-3,
    "H_norm": 10.0,  # entries are ~1e5, so this is 1e-4 relative
    "H_src_to_dst": 1e-2,
    "left_xy": 0.1,  # px
    "right_xy": 0.1,
}
MAX_CROP_LSB = 1
MAX_CROP_FRACTION = 0.15


def _compare_reference(fresh: Path, committed: Path) -> list[str]:
    a = json.loads((fresh / "ref.json").read_text(encoding="utf-8"))
    b = json.loads((committed / "ref.json").read_text(encoding="utf-8"))
    problems = []
    for key, value in a.items():
        if key not in b:
            problems.append(f"ref.json: {key} is missing from the committed file")
            continue
        if key in ("width", "height"):
            if value != b[key]:
                problems.append(f"ref.json: {key} {b[key]} -> {value}")
            continue
        x = np.asarray(value, dtype=np.float64)
        y = np.asarray(b[key], dtype=np.float64)
        if x.shape != y.shape:
            problems.append(f"ref.json: {key} shape {y.shape} -> {x.shape}")
            continue
        worst = float(np.abs(x - y).max())
        limit = TOLERANCES[key]
        if worst > limit:
            problems.append(
                f"ref.json: {key} differs by {worst:.6g} (tolerance {limit:g})"
            )
    return problems


def _compare_crops(fresh: Path, committed: Path) -> list[str]:
    problems = []
    for name in ("face", "left", "right"):
        filename = f"{name}_crop.bin"
        x = np.frombuffer((fresh / filename).read_bytes(), dtype=np.uint8)
        y = np.frombuffer((committed / filename).read_bytes(), dtype=np.uint8)
        if x.shape != y.shape:
            problems.append(f"{filename}: {y.size} bytes -> {x.size} bytes")
            continue
        diff = np.abs(x.astype(np.int16) - y.astype(np.int16))
        worst, fraction = int(diff.max()), float((diff > 0).mean())
        if worst > MAX_CROP_LSB:
            problems.append(f"{filename}: differs by up to {worst} (max {MAX_CROP_LSB})")
        elif fraction > MAX_CROP_FRACTION:
            problems.append(
                f"{filename}: {fraction:.1%} of bytes differ"
                f" (max {MAX_CROP_FRACTION:.0%})"
            )
    return problems


def _verify_probes(committed: Path) -> list[str]:
    """Re-run cv2 on the committed probe *inputs* and confirm its *outputs*.

    The probes are random, so a regenerated set cannot be diffed against the
    committed one without knowing the original seed. Replaying them is the
    better check anyway: it is what catches an OpenCV upgrade that changes a
    result the TypeScript port is pinned to.
    """
    probes = json.loads((committed / "cv2_probes.json").read_text(encoding="utf-8"))
    problems = []

    def worst(name: str, got: Any, want: Any, limit: float) -> None:
        a = np.asarray(got, dtype=np.float64)
        b = np.asarray(want, dtype=np.float64)
        if a.shape != b.shape:
            problems.append(f"cv2_probes.json: {name} shape {b.shape} -> {a.shape}")
            return
        diff = float(np.abs(a - b).max())
        if diff > limit:
            problems.append(
                f"cv2_probes.json: {name} differs by {diff:.6g} (tolerance {limit:g})"
            )

    warp = probes["warp"]
    src = np.asarray(warp["src"], dtype=np.uint8)
    dsize = (warp["dsize"][0], warp["dsize"][1])
    for i, case in enumerate(warp["cases"]):
        out = cv2.warpPerspective(
            src,
            np.asarray(case["H"], dtype=np.float64),
            dsize,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        worst(f"warp[{i}]", out, case["dst"], MAX_CROP_LSB)

    subpix = probes["subpix"]
    src = np.asarray(subpix["src"], dtype=np.uint8)
    win = (subpix["win"][0], subpix["win"][1])
    for i, case in enumerate(subpix["cases"]):
        centre = (float(case["center"][0]), float(case["center"][1]))
        worst(f"subpix[{i}]", cv2.getRectSubPix(src, win, centre), case["patch"], 0)

    for i, case in enumerate(probes["rodrigues"]):
        rotation = np.asarray(cv2.Rodrigues(np.asarray(case["v"], dtype=np.float64))[0])
        worst(f"rodrigues[{i}].R", rotation, case["R"], 1e-9)
        vector = np.asarray(cv2.Rodrigues(rotation)[0]).reshape(3)
        worst(f"rodrigues[{i}].v2", vector, case["v2"], 1e-9)
        if case["R3"] is not None:
            worst(f"rodrigues[{i}].R3", cv2.Rodrigues(vector)[0], case["R3"], 1e-9)

    for i, case in enumerate(probes["pnp"]):
        focal, cx, cy = case["K"]
        k = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
        image_points = np.array(
            [case["landmarks"][j] for j in _MODEL_LANDMARK_IDS], dtype=np.float64
        )
        found, rvec, tvec = cv2.solvePnP(
            _GENERIC_FACE_MODEL, image_points, k, None, flags=cv2.SOLVEPNP_SQPNP
        )
        if not found:
            problems.append(f"cv2_probes.json: pnp[{i}] no longer converges")
            continue
        worst(f"pnp[{i}].rvec", np.asarray(rvec).reshape(3), case["rvec_est"], 1e-6)
        worst(f"pnp[{i}].tvec", np.asarray(tvec).reshape(3), case["tvec_est"], 1e-6)

    probe = probes["misc"]["projectPoints_probe"]
    focal, cx, cy = probe["K"]
    k = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    pts, _ = cv2.projectPoints(
        _GENERIC_FACE_MODEL,
        np.asarray(probe["rvec"], dtype=np.float64),
        np.asarray(probe["t"], dtype=np.float64),
        k,
        None,
    )
    worst("projectPoints", np.asarray(pts).reshape(-1, 2), probe["pts"], 1e-9)
    return problems


def check(out_dir: Path, source: Path, seed: int, strict: bool = False) -> int:
    """Regenerate into a temporary directory and compare. Returns an exit code."""
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp)
        generate(fresh, source, seed)

        missing = [n for n in FIXTURES if not (out_dir / n).is_file()]
        if missing:
            print("missing fixtures: " + ", ".join(missing), file=sys.stderr)
            return 1

        if strict:
            problems = [
                f"{n}: bytes differ"
                for n in FIXTURES
                if not filecmp.cmp(fresh / n, out_dir / n, shallow=False)
            ]
        else:
            problems = _compare_reference(fresh, out_dir)
            problems += _compare_crops(fresh, out_dir)
            problems += _verify_probes(out_dir)
            if not filecmp.cmp(fresh / "face.jpg", out_dir / "face.jpg", shallow=False):
                problems.append("face.jpg: the UI copy is not the model's test asset")

    if problems:
        print(
            "the committed TypeScript fixtures no longer match the Python"
            " pipeline:\n",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "\npreprocessing in src/data/ changed without the references being"
            "\nrefreshed. Regenerate them (on linux/amd64, see the module"
            "\ndocstring) with:"
            "\n    uv run python tools/gen_parity_fixtures.py",
            file=sys.stderr,
        )
        return 1
    print(
        "committed fixtures agree with the Python pipeline"
        + (" byte for byte" if strict else " within tolerance")
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--seed", type=int, default=0, help="seeds the synthetic cv2 probes"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed fixtures instead of rewriting them",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="with --check, require a byte-for-byte match (same CPU architecture)",
    )
    args = parser.parse_args()

    if args.check:
        raise SystemExit(check(args.out, args.source, args.seed, strict=args.strict))
    if args.strict:
        raise SystemExit("--strict only applies with --check")
    generate(args.out, args.source, args.seed)
    print(f"wrote {len(FIXTURES)} fixtures to {args.out}")


if __name__ == "__main__":
    main()

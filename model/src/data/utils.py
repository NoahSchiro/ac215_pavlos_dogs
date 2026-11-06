"""
Primary preprocessing tools:

1. Landmarker pulls out key features of face
2. head_pose_est estimates head rotation/translation from those landmarks
3. normalize_face warps the image into a canonical face/eye frame
"""

import io
import json
import math
import os
import tarfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import torch
import webdataset as wds
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from numpy.typing import NDArray
from tqdm import tqdm

from src.data.schema import CameraIntrinsics, ImageArray, PointPx, Sample, SizePx


def get_landmarker() -> Any:
    """Create the FaceLandmarker."""
    # This lands at project_root/model/models/landmarker.task
    # TODO: there might be an argument that we want these weights
    # stored somewhere more generally, since we need it in the
    # browser too... Keep it here for now
    model_dir = Path(__file__).resolve().parents[2] / "models"
    model_path = model_dir / "landmarker.task"

    # If model isn't locally present, fetch and cache
    os.makedirs(model_dir, exist_ok=True)
    if not model_path.is_file():
        print(f"downloading face landmarker model to {model_path}")
        tmp_path = model_path.with_name(model_path.name + ".part")
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task",
            tmp_path,
        )
        tmp_path.replace(model_path)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def landmark_detection(landmarker, image: ImageArray) -> list[PointPx] | None:
    """Detect facial landmarks in an RGB uint8 (H, W, 3) image.

    Returns keypoints in pixel coordinates, or None when no face is
    detected.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    (face,) = result.face_landmarks  # num_faces=1 -> exactly zero or one
    height, width = image.shape[0], image.shape[1]
    return [PointPx(x=landmark.x * width, y=landmark.y * height) for landmark in face]


"""
Generic 3D face model for head pose: 

Taken from MediaPipe's canonical face model (google-ai-edge/mediapipe,
canonical_face_model.obj) reference points. Coordinates are [x,y,z]
where y is up, z faces the camera. Units are centimeters

These values are what we would expect from a generic face 
looking straight into the camera 
"""
_GENERIC_FACE_MODEL = np.array(
    [
        [-4.446, -2.664, -3.173],  # right eye, outer corner (MediaPipe landmark 33)
        [-1.856, -2.585, -3.758],  # right eye, inner corner (MediaPipe landmark 133)
        [1.856, -2.585, -3.758],  # left eye, inner corner  (MediaPipe landmark 362)
        [4.446, -2.664, -3.173],  # left eye, outer corner  (MediaPipe landmark 263)
        [-2.456, 4.343, -4.284],  # mouth, right corner     (MediaPipe landmark 61)
        [2.456, 4.343, -4.284],  # mouth, left corner      (MediaPipe landmark 291)
        [0.0, 1.127, -7.476],  # nose tip                (MediaPipe landmark 1)
        [0.0, 9.403, -4.264],  # chin                    (MediaPipe landmark 152)
    ]
)

# MediaPipe Face Mesh indices of key points we are interested in
_MODEL_LANDMARK_IDS = (33, 133, 362, 263, 61, 291, 1, 152)


def head_pose_est(
    landmarks: list[PointPx], intrinsics: CameraIntrinsics
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate head pose by computing a rotation + translation matrix from a
    generic 3D face to the observed subject face.

    Args:
        landmarks: pixel-space keypoints from `landmark_detection` (MediaPipe
            Face Mesh, 468+ points).
        intrinsics: camera calibration for the image; estimated ones stay
            flagged via `CameraIntrinsics.is_estimate` in the schema.

    Returns:
        (R, t) mapping the 3D model into the OpenCV camera frame
        (+X image right, +Y image down, +Z forward): R is the 3x3 rotation
        matrix, t the model origin's translation in the model's units
        (~centimetres; expect t[2] ~ 50-60 for MPIIFaceGaze webcam distance).
        A face looking straight at the camera gives R ~ identity.
    """
    # We need to ensure out landmark feature vector has all of the features we want
    if len(landmarks) < max(_MODEL_LANDMARK_IDS) + 1:
        raise ValueError(
            f"head pose needs the full Face Mesh (>= {max(_MODEL_LANDMARK_IDS) + 1} points),"
            f" got {len(landmarks)}"
        )

    # Extract only the points we care about
    image_points = np.array(
        [[landmarks[i].x, landmarks[i].y] for i in _MODEL_LANDMARK_IDS],
        dtype=np.float64,
    )
    camera_matrix = np.array(
        [
            [intrinsics.focal_length_px, 0.0, intrinsics.principal_point.x],
            [0.0, intrinsics.focal_length_px, intrinsics.principal_point.y],
            [0.0, 0.0, 1.0],
        ]
    )
    # This func finds the 6 DOF rotation + translation vector (rvec, tvec)
    # between the subjects current pose and the "looking straight-on" pose
    found, rvec, tvec = cv2.solvePnP(
        _GENERIC_FACE_MODEL,
        image_points,
        camera_matrix,
        None,  # lens distortion is not modeled
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not found:
        raise ValueError("solvePnP did not converge")

    # Vector -> matrix
    rotation = np.asarray(cv2.Rodrigues(rvec)[0], dtype=np.float64)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    return rotation, translation


# Fixed normalization geometry. Every sample from every dataset must use
# these, in both Python and TypeScript -- they define the canonical frame the
# model is trained in, so a mismatch silently shifts every crop at inference.
NORM_DIST_CM = 60.0
NORM_FOCAL_PX = 960.0
NORM_FACE_SIZE_PX = 224
NORM_EYE_SIZE_PX = (36, 60)  # (height, width)

# Iris centres from the Face Mesh.
LEFT_IRIS_ID = 473
RIGHT_IRIS_ID = 468


@dataclass(frozen=True)
class NormalizationTransforms:
    """The geometry `normalize_face` warps with, without the warping.

    Split out so the TypeScript port can be checked against the exact
    matrices Python computes rather than against its own reimplementation
    of them; `model/tools/gen_parity_fixtures.py` records these.
    """

    r_virtual: NDArray[np.float64]
    """(3, 3) virtual camera rotation."""
    h_norm: NDArray[np.float64]
    """(3, 3) real pixel -> virtual pixel homography."""
    h_src_to_dst: NDArray[np.float64]
    """(3, 3) its inverse, used to place the eye crops."""
    left_xy: tuple[float, float]
    """Left iris centre in the normalized face crop."""
    right_xy: tuple[float, float]
    """Right iris centre in the normalized face crop."""


def normalization_transforms(
    landmarks: list[PointPx],
    R: NDArray[np.float64],
    t: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
) -> NormalizationTransforms:
    """Virtual-camera geometry for one face, given its pose.

    Args:
        landmarks: pixel-space Face Mesh points (needs the iris points
            468/473).
        R, t: head pose from `head_pose_est` (model -> camera frame).
        intrinsics: camera calibration for this image.
    """
    if len(landmarks) < max(LEFT_IRIS_ID, RIGHT_IRIS_ID) + 1:
        raise ValueError(
            f"normalization needs the full refined Face Mesh"
            f" (>= {max(LEFT_IRIS_ID, RIGHT_IRIS_ID) + 1} points),"
            f" got {len(landmarks)}"
        )

    def unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
        return v / np.linalg.norm(v)

    # Virtual camera rotation:
    # z-axis = camera to face
    # x-axis = straight up
    z_axis = unit(R @ np.array([0.0, 0.0, 1.0]))
    head_x = R @ np.array([1.0, 0.0, 0.0])
    # this trick ensures that the x_axis is orthogonal with the z-axis,
    # even if it is not *exactly* the x rotation of the head
    x_axis = unit(head_x - (head_x @ z_axis) * z_axis)
    y_axis = np.cross(z_axis, x_axis)
    r_virtual = np.column_stack((x_axis, y_axis, z_axis))

    # Virtual camera center
    c_virtual = t - NORM_DIST_CM * z_axis
    k = np.array(
        [  # camera specs
            [intrinsics.focal_length_px, 0.0, intrinsics.principal_point.x],
            [0.0, intrinsics.focal_length_px, intrinsics.principal_point.y],
            [0.0, 0.0, 1.0],
        ]
    )
    k_norm = np.array(
        [  # virtual camera specs
            [NORM_FOCAL_PX, 0.0, NORM_FACE_SIZE_PX / 2],
            [0.0, NORM_FOCAL_PX, NORM_FACE_SIZE_PX / 2],
            [0.0, 0.0, 1.0],
        ]
    )

    # h_norm answers: for each pixel captured by a real camera, where would
    # that be on an image taken by our virtual camera
    h_norm = NORM_DIST_CM * k @ r_virtual @ np.linalg.inv(k_norm) + (k @ c_virtual)[
        :, None
    ] * np.array([[0.0, 0.0, 1.0]])
    h_src_to_dst = np.linalg.inv(h_norm)

    def warped(point: PointPx) -> tuple[float, float]:
        out = h_src_to_dst @ np.array([point.x, point.y, 1.0])
        return out[0] / out[2], out[1] / out[2]

    return NormalizationTransforms(
        r_virtual=r_virtual,
        h_norm=h_norm,
        h_src_to_dst=h_src_to_dst,
        left_xy=warped(landmarks[LEFT_IRIS_ID]),
        right_xy=warped(landmarks[RIGHT_IRIS_ID]),
    )


def normalize_face(
    image: ImageArray,
    landmarks: list[PointPx],
    R: NDArray[np.float64],
    t: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
) -> tuple[ImageArray, ImageArray, ImageArray, NDArray[np.float64]]:
    """Warp the image into a canonical, head-centered frame.

    We place a virtual camera at a fixed position and normalize the face to
    be at a fixed position away from the camera with a fixed rotation

    Args:
        image: input RGB image.
        landmarks: pixel-space Face Mesh points (needs the iris points 468/473).
        R, t: head pose from `head_pose_est` (model -> camera frame).
        intrinsics: camera calibration for this image.

    Returns:
        normalize face crop, normalized eye crops, and the virtual
        camera rotation R_norm.
    """
    transforms = normalization_transforms(landmarks, R, t, intrinsics)
    face = np.asarray(
        cv2.warpPerspective(
            image,
            transforms.h_norm,
            (NORM_FACE_SIZE_PX, NORM_FACE_SIZE_PX),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        ),
        dtype=np.uint8,
    )
    height, width = NORM_EYE_SIZE_PX
    left = np.asarray(
        cv2.getRectSubPix(face, (width, height), transforms.left_xy),
        dtype=np.uint8,
    )
    right = np.asarray(
        cv2.getRectSubPix(face, (width, height), transforms.right_xy),
        dtype=np.uint8,
    )
    return face, left, right, transforms.r_virtual


def transform_label(
    gaze_target_px: PointPx,
    screen_size_px: SizePx,
    R_virtual: NDArray[np.float64],
    gaze_direction_cam: NDArray[np.float64] | None = None,
) -> tuple[tuple[float, float], NDArray[np.float64] | None]:
    """Convert a raw gaze label to match the normalized image.

    We output two formats of labels
        - a fraction of the screen (x/W, y/H)
        - Normalized gaze direction (the angle off of the z-axis)

    Args:
        gaze_target_px: raw on-screen gaze position.
        screen_size_px: resolution of the screen the position refers to.
        R_virtual: virtual camera rotation from `normalize_face`.
        gaze_direction_cam: optional 3D gaze direction in the camera frame
            (e.g. gt - fc when a dataset provides it) to also get the
            angle-based label.

    Returns:
        (screen_fraction, direction_norm)
    """
    fraction = (
        gaze_target_px.x / screen_size_px.width,
        gaze_target_px.y / screen_size_px.height,
    )

    direction: NDArray[np.float64] | None = None
    if gaze_direction_cam is not None:
        direction = R_virtual.T @ np.asarray(gaze_direction_cam, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if not math.isfinite(length) or length == 0.0:
            raise ValueError("gaze direction must be a finite, non-zero vector")
        direction = direction / length

    return fraction, direction


def check_sample(
    landmarker: Any, sample: Sample, max_reproj: float = 0.1
) -> dict[str, Any] | str:
    """Run every preprocessing step on one sample.

    Quality checks first (cheapest first):
    - gaze target inside the screen
    - image readable
    - face detection succeeds
    - head pose solvable
    - reprojection error
    - iris points near their eye corners

    Samples passing all checks continue through face normalization and
    the label transform.

    Args:
        landmarker: FaceLandmarker instance (e.g. from `get_landmarker`).
        sample: parsed sample whose image lives at `image_path`.
        max_reproj: mean reprojection error, as a fraction of the
            projected eye-corner distance, above which the sample is
            dropped.

    Returns:
        The per-item dict
        ``{"sample_id", "face", "left_eye", "right_eye", "label", "R_norm"}``
        or a string naming the reason the sample was dropped.
    """
    fx = sample.gaze_target_px.x / sample.screen_size_px.width
    fy = sample.gaze_target_px.y / sample.screen_size_px.height
    if not (math.isfinite(fx) and math.isfinite(fy)) or not (
        0 <= fx <= 1 and 0 <= fy <= 1
    ):
        return "gaze_out_of_bounds"

    image_bgr = cv2.imread(str(sample.image_path))
    if image_bgr is None:
        return "image_unreadable"
    image = np.asarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).astype(np.uint8)

    landmarks = landmark_detection(landmarker, image)
    if landmarks is None:
        return "face_not_detected"

    try:
        R, t = head_pose_est(landmarks, sample.camera_intrinsics)
    except ValueError:
        return "pose_failed"

    # reprojection error, normalized by the projected eye-corner distance
    camera_focal = sample.camera_intrinsics.focal_length_px
    camera_x = sample.camera_intrinsics.principal_point.x
    camera_y = sample.camera_intrinsics.principal_point.y
    K = np.array(
        [
            [camera_focal, 0.0, camera_x],
            [0.0, camera_focal, camera_y],
            [0.0, 0.0, 1.0],
        ]
    )
    observed = np.array([[landmarks[i].x, landmarks[i].y] for i in _MODEL_LANDMARK_IDS])
    projected, _ = cv2.projectPoints(
        _GENERIC_FACE_MODEL, cv2.Rodrigues(R)[0], t, K, None
    )
    projected = projected.reshape(-1, 2)
    eye_span = float(np.linalg.norm(projected[0] - projected[3]))
    reproj = float(np.linalg.norm(projected - observed, axis=1).mean()) / max(
        eye_span, 1e-9
    )
    if reproj > max_reproj:
        return "reprojection_error"

    # iris centres must sit near their eye corners' bounding box
    for iris_id, corner_a, corner_b in ((468, 33, 133), (473, 263, 362)):
        a, b, iris = landmarks[corner_a], landmarks[corner_b], landmarks[iris_id]
        margin = 0.25 * max(abs(b.x - a.x), abs(b.y - a.y))
        in_box = (
            min(a.x, b.x) - margin <= iris.x <= max(a.x, b.x) + margin
            and min(a.y, b.y) - margin <= iris.y <= max(a.y, b.y) + margin
        )
        if not in_box:
            return "eyes_off_face"

    face, left_eye, right_eye, R_virtual = normalize_face(
        image, landmarks, R, t, sample.camera_intrinsics
    )
    fraction, direction = transform_label(
        sample.gaze_target_px,
        sample.screen_size_px,
        R_virtual,
        gaze_direction_cam=sample.gaze_direction_cam,
    )
    return {
        "sample_id": sample.sample_id,
        "face": face,
        "left_eye": left_eye,
        "right_eye": right_eye,
        "R_norm": R_virtual,
        "label": np.array(fraction, dtype=np.float32),
        **(
            {"direction": direction.astype(np.float32)}
            if direction is not None
            else {}
        ),
    }


def build_webdataset(
    samples: list[Sample], path: str | os.PathLike[str], num_threads: int | None = None
) -> None:
    """Preprocess samples with `check_sample` and cache them as a webdataset.

    Drops invalid samples. Kept samples are written to shard files under
    `path` (a directory), keyed by their sample_id:

    - `{key}.json`      manifest metadata, plus `label` and `R_norm`
    - `{key}.png`       normalized face crop (224x224x3 RGB)
    - `{key}.eyes.npy`  uint8 (2, 36, 60, 3); [0] left, [1] right eye

    Work is split between worker threads. Every thread runs the whole
    pipeline (load -> process -> write its own shard files).

    Args:
        samples: parsed samples (e.g. from `parse_mpii`).
        path: directory for the shard files; created if missing.
        num_threads: pool size; defaults to the number of CPU cores.
    """
    path = Path(path)
    workers = num_threads or os.cpu_count() or 1
    workers = max(1, min(workers, len(samples)))

    path.mkdir(parents=True, exist_ok=True)
    for stale in path.glob("*.tar"):
        # Prevents a rebuild from mixing shards of a previous run
        stale.unlink()  

    landmarker_lock = threading.Lock()
    progress = tqdm(total=len(samples))

    def process_chunk(
        chunk: list[Sample], worker_id: int
    ) -> tuple[int, Counter[tuple[str, str]]]:
        """Preprocess and write one thread's share of samples, one at a time."""
        with landmarker_lock:
            landmarker = get_landmarker()
        written = 0
        drop_counts: Counter[tuple[str, str]] = Counter()
        # the worker id keeps each thread's shards disjoint, so threads
        # never write to the same file
        with wds.ShardWriter(
            str(path / f"{path.name}-w{worker_id:03d}-%06d.tar"),
            maxcount=1000,  # ~150 MB per shard
        ) as sink:
            for sample in chunk:
                result = check_sample(landmarker, sample)
                if isinstance(result, str):
                    drop_counts[(sample.source_dataset.value, result)] += 1
                else:
                    sink.write({
                        "__key__": result["sample_id"],
                        "json": {
                            **sample.to_manifest_entry(),
                            "label": result["label"].tolist(),
                            "R_norm": result["R_norm"].tolist(),
                            **(
                                {"direction": result["direction"].tolist()}
                                if "direction" in result
                                else {}
                            ),
                        },
                        "png": result["face"],
                        "eyes.npy": np.stack(
                            [result["left_eye"], result["right_eye"]]
                        ),
                    })
                    written += 1
                progress.update(1)
        return written, drop_counts

    # one interleaved chunk per thread, so the workload stays balanced
    chunks = [samples[i::workers] for i in range(workers)]
    written = 0
    drop_counts: Counter[tuple[str, str]] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, drops in pool.map(process_chunk, chunks, range(workers)):
            written += count
            drop_counts.update(drops)
    progress.close()

    print(
        f"wrote {written}/{len(samples)} samples to {path}"
        f" ({len(samples) - written} dropped)"
    )
    for (dataset, reason), count in sorted(drop_counts.items()):
        print(f"  {dataset} / {reason}: {count}")


@cache
def shard_index(shard: str) -> dict[str, tuple[int, int]]:
    """Scan a shard once: member name -> (data offset, size) for every member.

    The index is tiny (a few ints per member), so caching it for every
    shard costs nothing and reads never have to rescan the file or keep
    tarfile handles open.
    """
    with tarfile.open(shard, "r") as tf:
        return {
            info.name: (info.offset_data, info.size) for info in tf if info.isfile()
        }


def read_shard_member(shard: str, name: str) -> bytes:
    """Read one member's bytes from a shard."""
    index = shard_index(shard)
    if name not in index:
        raise FileNotFoundError(f"member {name} missing in {shard}")
    offset, size = index[name]
    with open(shard, "rb") as f:
        f.seek(offset)
        return f.read(size)


def shard_lookup(cache_dir: str | os.PathLike[str]) -> dict[str, str]:
    """Map every sample_id in a webdataset cache to its shard file.

    Built by a header-only scan of all shards and persisted next to them
    (``index.json``), so repeated runs -- and the train/val/test Dataset
    constructions within one run -- skip the scan entirely. Rebuilt
    automatically whenever any shard file changes.
    """
    cache_dir = Path(cache_dir)
    shards = sorted(cache_dir.glob("*.tar"))
    fingerprint = [
        [shard.name, shard.stat().st_size, shard.stat().st_mtime_ns]
        for shard in shards
    ]
    index_path = cache_dir / "index.json"
    if index_path.is_file():
        saved = json.loads(index_path.read_text(encoding="utf-8"))
        if saved["fingerprint"] == fingerprint:
            return saved["lookup"]

    lookup: dict[str, str] = {}
    for shard in shards:
        for name in shard_index(str(shard)):
            if name.endswith(".json"):
                lookup[name[: -len(".json")]] = str(shard)
    index_path.write_text(
        json.dumps({"fingerprint": fingerprint, "lookup": lookup}),
        encoding="utf-8",
    )
    return lookup


def read_webdataset_sample(shard: str, key: str) -> dict[str, Any]:
    """Read one preprocessed sample from a shard written by `build_webdataset`.

    Args:
        shard: shard file the sample lives in.
        key: the sample's sample_id.

    Returns:
        The per-item dict
        ``{"sample_id", "face", "left_eye", "right_eye", "label", "R_norm"}``
        - exactly what `check_sample` computed when the sample was cached.
    """
    meta = json.loads(read_shard_member(shard, f"{key}.json"))
    if not isinstance(meta, dict):
        raise TypeError(f"metadata for {key} in {shard} is not a JSON object")

    bgr = cv2.imdecode(
        np.frombuffer(read_shard_member(shard, f"{key}.png"), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if bgr is None:
        raise ValueError(f"corrupt face crop for {key} in {shard}")
    face = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)

    eyes = np.load(io.BytesIO(read_shard_member(shard, f"{key}.eyes.npy")))

    return {
        "sample_id": str(meta["sample_id"]),
        "face": face,
        "left_eye": eyes[0],
        "right_eye": eyes[1],
        "label": np.array(meta["label"], dtype=np.float32),
        "R_norm": np.array(meta["R_norm"], dtype=np.float64),
        **(
            {"direction": np.array(meta["direction"], dtype=np.float32)}
            if "direction" in meta
            else {}
        ),
    }


def prepare_images(
    batch: torch.Tensor, device: torch.device, *, imagenet: bool = True
) -> torch.Tensor:
    """uint8 (B, H, W, 3) crops from a Dataset -> normalized float (B, 3, H, W).

    Datasets hand back HWC uint8 because that is what OpenCV produces. The
    conversion happens on device so the DataLoader stays cheap.

    Args:
        batch: stacked crops as uint8.
        device: where to move and convert them.
        imagenet: normalize with ImageNet statistics, which a model
            initialized from ImageNet weights expects. False scales to
            [0, 1] only, for a randomly initialized model.
    """
    if batch.ndim != 4 or batch.shape[-1] != 3:
        raise ValueError(f"expected (B, H, W, 3), got {tuple(batch.shape)}")
    images = batch.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    if not imagenet:
        return images
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (images - mean) / std

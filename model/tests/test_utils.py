"""Tests for landmark_detection."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from src.data.schema import (
    CameraIntrinsics,
    DeviceType,
    ImageArray,
    PointPx,
    Sample,
    SizeMm,
    SizePx,
    SourceDataset,
    make_sample_id,
)
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
    transform_label,
)

# I pulled this image from MPII
FACE_IMAGE_PATH = Path(__file__).parent / "data" / "face.jpg"


def load_face_image() -> ImageArray:
    return np.asarray(Image.open(FACE_IMAGE_PATH).convert("RGB"))


def test_detects_landmarks_on_face() -> None:
    image = load_face_image()
    model = get_landmarker()
    landmarks = landmark_detection(model, image)

    assert landmarks is not None
    # 468 face mesh points + 10 iris points
    assert len(landmarks) == 478
    assert all(isinstance(point, PointPx) for point in landmarks)


def test_landmarks_are_in_pixel_coordinates() -> None:
    image = load_face_image()
    model = get_landmarker()
    landmarks = landmark_detection(model, image)

    assert landmarks is not None
    height, width = image.shape[0], image.shape[1]
    assert all(0 <= point.x <= width for point in landmarks)
    assert all(0 <= point.y <= height for point in landmarks)


def test_eye_landmark_anatomy() -> None:
    """Subject's right eye (33) must sit image-left of their left eye (263)."""
    image = load_face_image()
    model = get_landmarker()
    landmarks = landmark_detection(model, image)

    assert landmarks is not None
    height = int(image.shape[0])
    assert landmarks[33].x < landmarks[263].x
    assert landmarks[33].y == pytest.approx(landmarks[263].y, abs=height * 0.05)


def test_detection_is_deterministic() -> None:
    image = load_face_image()
    model = get_landmarker()

    first = landmark_detection(model, image)
    second = landmark_detection(model, image)

    assert first is not None and second is not None
    assert first == second


def test_returns_none_without_face() -> None:
    noise = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    model = get_landmarker()

    assert landmark_detection(model, noise) is None


def make_landmarks_from_projection(
    rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray
) -> list[PointPx]:
    """Apply rotation / translation vectors to generic 3D face."""
    projected, _ = cv2.projectPoints(
        _GENERIC_FACE_MODEL, rvec, tvec, camera_matrix, None
    )
    landmarks = [PointPx(x=0.0, y=0.0)] * 478
    for i, (x, y) in zip(_MODEL_LANDMARK_IDS, projected.reshape(-1, 2)):
        landmarks[i] = PointPx(x=float(x), y=float(y))
    return landmarks


def test_recovers_known_pose_from_synthetic_projection() -> None:
    """solvePnP must invert the projection of a known pose exactly."""
    intrinsics = CameraIntrinsics(
        focal_length_px=1000.0, principal_point=PointPx(x=640.0, y=360.0)
    )
    camera_matrix = np.array(
        [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]]
    )
    rvec_true = np.array([[0.15], [-0.25], [0.05]])  # yaw/pitch/roll
    tvec_true = np.array([[3.0], [-4.0], [60.0]])  # x/y/z

    landmarks = make_landmarks_from_projection(rvec_true, tvec_true, camera_matrix)
    r_mat, t_mat = head_pose_est(landmarks, intrinsics)

    R_true, _ = cv2.Rodrigues(rvec_true)
    assert np.allclose(r_mat, R_true, atol=1e-6)
    assert np.allclose(t_mat, tvec_true.reshape(3), atol=1e-6)


def test_fronto_parallel_face_gives_identity_rotation() -> None:
    intrinsics = CameraIntrinsics(
        focal_length_px=1000.0, principal_point=PointPx(x=0.0, y=0.0)
    )
    # Face with no rotation, 60 cm in front of camera
    landmarks = make_landmarks_from_projection(
        np.zeros((3, 1)), np.array([[0.0], [0.0], [60.0]]), np.eye(3) * 1000.0
    )

    r_mat, t_mat = head_pose_est(landmarks, intrinsics)

    assert np.allclose(r_mat, np.eye(3), atol=1e-6)
    assert np.allclose(t_mat, [0.0, 0.0, 60.0], atol=1e-6)


def test_rejects_landmark_lists_without_the_model_points() -> None:
    intrinsics = CameraIntrinsics(
        focal_length_px=1000.0, principal_point=PointPx(x=0.0, y=0.0)
    )

    with pytest.raises(ValueError, match="Face Mesh"):
        head_pose_est([PointPx(x=0.0, y=0.0)] * 10, intrinsics)


NORM_INTRINSICS = CameraIntrinsics(
    focal_length_px=1000.0, principal_point=PointPx(x=640.0, y=360.0)
)
NORM_K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])


def black_canvas() -> ImageArray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def draw_red_square(image: ImageArray, x: float, y: float, half: int = 4) -> None:
    image[int(y) - half : int(y) + half, int(x) - half : int(x) + half] = (255, 0, 0)


def test_eye_crops_follow_anatomical_convention() -> None:
    """Subject's LEFT eye (image-right in non-mirrored frames) -> `left_eye`."""
    t = np.array([0.0, 0.0, 60.0])
    landmarks = make_landmarks_from_projection(np.zeros(3), t, NORM_K)
    # iris positions ~ +-2.88 units off-centre, projected by K at depth 60
    landmarks[473] = PointPx(x=688.0, y=316.4)  # subject's left -> image right
    landmarks[468] = PointPx(x=592.0, y=316.4)  # subject's right -> image left

    image = black_canvas()
    draw_red_square(image, 688.0, 316.4)  # marker on the subject's left eye only

    _face, left_eye, right_eye, _R = normalize_face(
        image, landmarks, np.eye(3), t, NORM_INTRINSICS
    )

    # crops centre on the warped iris positions (158, 70) and (66, 70)
    assert (left_eye[18, 30] == (255, 0, 0)).all()
    assert (right_eye[18, 30] == (0, 0, 0)).all()


def test_real_face_normalization_produces_valid_crops() -> None:
    image = load_face_image()
    model = get_landmarker()
    landmarks = landmark_detection(model, image)

    assert landmarks is not None
    intrinsics = CameraIntrinsics(
        focal_length_px=996.4509,
        principal_point=PointPx(x=624.6634, y=364.0874),
    )
    R, t = head_pose_est(landmarks, intrinsics)

    face, left, right, _ = normalize_face(image, landmarks, R, t, intrinsics)

    assert face.shape == (224, 224, 3)
    assert left.shape == (36, 60, 3)
    assert right.shape == (36, 60, 3)
    assert face.max() > 0
    assert left.max() > 0
    assert right.max() > 0
    assert not np.array_equal(left, right)

    face2, left2, right2, _ = normalize_face(image, landmarks, R, t, intrinsics)
    assert np.array_equal(face, face2)
    assert np.array_equal(left, left2)
    assert np.array_equal(right, right2)


def test_screen_fraction_is_resolution_independent() -> None:
    fraction, direction = transform_label(
        PointPx(x=562.5, y=1218.0), SizePx(width=1125, height=2436), np.eye(3)
    )

    assert fraction == pytest.approx((0.5, 0.5))
    assert direction is None


def test_out_of_bounds_targets_pass_through_unclamped() -> None:
    fraction, _ = transform_label(
        PointPx(x=-10.0, y=5000.0), SizePx(width=1125, height=2436), np.eye(3)
    )

    assert fraction == pytest.approx((-10.0 / 1125.0, 5000.0 / 2436.0))


def test_direction_is_rotated_into_normalized_frame() -> None:
    # R_virtual is a +90-degree rotation about z. Its transpose (what the
    # function applies) rotates the camera x-axis into the normalized -y-axis
    R_virtual = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    fraction, direction = transform_label(
        PointPx(x=0.0, y=0.0),
        SizePx(width=100, height=100),
        R_virtual,
        np.array([2.0, 0.0, 0.0]),  # unnormalized on purpose
    )

    assert fraction == (0.0, 0.0)
    assert direction is not None
    assert np.allclose(direction, [0.0, -1.0, 0.0])
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_rejects_degenerate_direction() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        transform_label(
            PointPx(x=0.0, y=0.0),
            SizePx(width=100, height=100),
            np.eye(3),
            np.zeros(3),
        )


def make_filter_sample(**overrides: Any) -> Sample:
    """A sample pointing at the real fixture image, so detection succeeds."""
    values: dict[str, Any] = {
        "sample_id": make_sample_id(SourceDataset.MPIIFaceGaze, "p00", "day01", "0005"),
        "source_dataset": SourceDataset.MPIIFaceGaze,
        "subject_id": "p00",
        "image_path": FACE_IMAGE_PATH,
        "camera_intrinsics": CameraIntrinsics(
            focal_length_px=996.4509,
            principal_point=PointPx(x=624.6634, y=364.0874),
        ),
        "screen_size_mm": SizeMm(width=286.5, height=179.0),
        "screen_size_px": SizePx(width=1125, height=2436),
        "gaze_target_px": PointPx(x=562.5, y=1218.0),
        "device_type": DeviceType.Laptop,
    }
    values.update(overrides)
    return Sample(**values)


def test_normalization_transforms_reproduce_normalize_face() -> None:
    """The split-out geometry is the geometry normalize_face actually warps with.

    `tools/gen_parity_fixtures.py` records these matrices as the reference the
    TypeScript port is checked against, so they have to be the same ones the
    crops come from -- not a second implementation that happens to agree today.
    """
    image = load_face_image()
    landmarks = landmark_detection(get_landmarker(), image)
    assert landmarks is not None
    intrinsics = CameraIntrinsics(
        focal_length_px=996.4509,
        principal_point=PointPx(x=624.6634, y=364.0874),
    )
    R, t = head_pose_est(landmarks, intrinsics)

    face, left, right, r_virtual = normalize_face(image, landmarks, R, t, intrinsics)
    transforms = normalization_transforms(landmarks, R, t, intrinsics)

    assert np.array_equal(r_virtual, transforms.r_virtual)

    replayed_face = cv2.warpPerspective(
        image,
        transforms.h_norm,
        (NORM_FACE_SIZE_PX, NORM_FACE_SIZE_PX),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )
    assert np.array_equal(face, replayed_face)

    height, width = NORM_EYE_SIZE_PX
    assert np.array_equal(
        left, cv2.getRectSubPix(face, (width, height), transforms.left_xy)
    )
    assert np.array_equal(
        right, cv2.getRectSubPix(face, (width, height), transforms.right_xy)
    )


def test_normalization_transforms_homographies_are_inverses() -> None:
    image = load_face_image()
    landmarks = landmark_detection(get_landmarker(), image)
    assert landmarks is not None
    intrinsics = CameraIntrinsics(
        focal_length_px=996.4509,
        principal_point=PointPx(x=624.6634, y=364.0874),
    )
    R, t = head_pose_est(landmarks, intrinsics)

    transforms = normalization_transforms(landmarks, R, t, intrinsics)
    assert np.allclose(transforms.h_src_to_dst @ transforms.h_norm, np.eye(3))
    # The virtual camera rotation is a rotation: orthonormal, right-handed.
    assert np.allclose(transforms.r_virtual.T @ transforms.r_virtual, np.eye(3))
    assert np.isclose(np.linalg.det(transforms.r_virtual), 1.0)


def test_normalization_transforms_rejects_landmark_lists_without_the_irises() -> None:
    intrinsics = CameraIntrinsics(
        focal_length_px=996.4509,
        principal_point=PointPx(x=624.6634, y=364.0874),
    )
    with pytest.raises(ValueError, match="refined Face Mesh"):
        normalization_transforms(
            [PointPx(x=0.0, y=0.0)] * 10, np.eye(3), np.zeros(3), intrinsics
        )

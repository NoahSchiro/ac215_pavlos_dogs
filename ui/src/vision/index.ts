/**
 * Browser gaze preprocessing pipeline.
 *
 * Port of the image preprocessing steps of `model/src/data`, so the ONNX
 * GazeNet can run in the browser on artifacts identical to the ones the
 * Python pipeline produces. Inference consumes the crops, `R_norm` and the
 * tensors.
 *
 * The cv2 geometry is @techstark/opencv-js (same version the Python side pins)
 * loaded once per page and wrapped in ./cv.ts.
 */

export type { Size2i, Point2 } from './cv'
export {
  loadCv,
  warpPerspective,
  getRectSubPix,
  solvePnP,
  rodrigues,
  rodriguesToVector,
  projectPoints,
  matMul3x3,
  invert3x3,
} from './cv'
export type {
  ImageArray,
  PointPx,
  CameraIntrinsics,
  NormalizedFace,
  PreprocessedFrame,
  DropReason,
} from './utils'
export {
  getLandmarker,
  landmarkDetection,
  headPoseEst,
  normalizeFace,
  transformLabel,
  preprocess,
  prepareImages,
  GENERIC_FACE_MODEL,
  MODEL_LANDMARK_IDS,
} from './utils'

/**
 * GazeNet inference in the browser via onnxruntime-web.
 *
 * The ONNX export is produced by the training pipeline.
 *
 * Inputs are the tensors `prepareImages` produces from the preprocessing
 * crops (ImageNet normalization, NCHW float32); the output is (B, 2)
 * sigmoid-activated screen fractions.
 */

import * as ort from 'onnxruntime-web/wasm'
import type { FaceLandmarker } from '@mediapipe/tasks-vision'
import {
  prepareImages,
  preprocess,
  type CameraIntrinsics,
  type DropReason,
  type ImageArray,
  type PreprocessedFrame,
} from './utils'

/** Gaze prediction as a fraction of the screen, each axis in [0, 1]. */
export interface ScreenFraction {
  x: number
  y: number
}

/** A loaded GazeNet session, validated against the expected interface. */
export interface GazeModel {
  session: ort.InferenceSession
  inputNames: readonly string[]
  outputName: string
}

let cachedModel: Promise<GazeModel> | null = null

/** Load the ONNX GazeNet model, once per page. */
export function loadGazeModel(): Promise<GazeModel> {
  const expected_inputs = ['face', 'left_eye', 'right_eye'] as const
  const expected_output = 'prediction'
  cachedModel ??= (async () => {
    // The wasm runtime must come from the same onnxruntime-web version this
    // module is compiled against (onnxruntime pins loader and binary).
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/`
    const session = await ort.InferenceSession.create(`${import.meta.env.BASE_URL}models/gazenet.onnx`, {
      executionProviders: ['wasm'],
    })

    const inputNames = [...session.inputNames]
    const missing = expected_inputs.filter((name) => !inputNames.includes(name))
    if (missing.length > 0 || inputNames.length !== expected_inputs.length) {
      throw new Error(
        `expected GazeNet inputs [${expected_inputs.join(', ')}], got [${inputNames.join(', ')}]`,
      )
    }
    const outputName = session.outputNames[0]
    if (session.outputNames.length !== 1 || outputName !== expected_output) {
      throw new Error(
        `expected GazeNet output "${expected_output}", got [${session.outputNames.join(', ')}]`,
      )
    }
    return { session, inputNames, outputName }
  })()
  return cachedModel
}

/** Run GazeNet on one preprocessed frame. */
export async function predictScreenFraction(
  model: GazeModel,
  frame: PreprocessedFrame,
): Promise<ScreenFraction> {
  const face = prepareImages([frame.face])
  const leftEye = prepareImages([frame.leftEye])
  const rightEye = prepareImages([frame.rightEye])

  const results = await model.session.run({
    face: new ort.Tensor('float32', face.data, face.shape),
    left_eye: new ort.Tensor('float32', leftEye.data, leftEye.shape),
    right_eye: new ort.Tensor('float32', rightEye.data, rightEye.shape),
  })
  const prediction = results[model.outputName] as ort.Tensor

  const dims = prediction.dims as number[]
  if (dims.length !== 2 || dims[0] !== 1 || dims[1] !== 2) {
    throw new Error(`expected prediction shape [1, 2], got [${dims.join(', ')}]`)
  }
  const data = prediction.data as Float32Array
  return { x: data[0], y: data[1] }
}

/** Webcam frame -> gaze screen fraction. */
export async function predictGaze(
  landmarker: FaceLandmarker,
  model: GazeModel,
  image: ImageArray,
  intrinsics: CameraIntrinsics,
): Promise<ScreenFraction | DropReason> {
  const frame = await preprocess(landmarker, image, intrinsics)
  // It is possible for the preprocessing to fail. If so, pass
  // the error up
  if (typeof frame === 'string') return frame
  return predictScreenFraction(model, frame)
}

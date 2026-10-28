/**
 * Thin wrappers over @techstark/opencv-js — OpenCV 5.0.0 compiled to
 * WebAssembly. Covers exactly the cv2 functions the preprocessing
 * pipeline uses: warpPerspective, getRectSubPix, Rodrigues, projectPoints,
 * solvePnP (SOLVEPNP_SQPNP) and the small matrix ops of normalize_face.
 *
 * - The module is a ~13 MB WASM bundle (3.6 MB gzipped) that initializes
 *   asynchronously, so every wrapper is async.
 * - The runtime is loaded once per page (`loadCv`) because loading a 3.6
 *   MB module is expensive and will be noticable my the user
 * - opencv.js memory is not garbage collected: Mats are deleted on every
 *   exit path.
 * - The WASM backend has no FMA instruction, so last-ulp rounding can
 *   differ from the native x86 wheel by ±1 in the final uint8 pixel.
 */

import type { CV } from '@techstark/opencv-js'
import type { ImageArray } from './utils'

export interface Size2i {
  width: number
  height: number
}

export interface Point2 {
  x: number
  y: number
}

let cached: Promise<CV> | null = null

/** Load and initialize the OpenCV WASM runtime once per page. */
export function loadCv(): Promise<CV> {
  cached ??= (async () => {
    const mod = (await import('@techstark/opencv-js')) as unknown as {
      default?: unknown
    }
    return (await (mod.default ?? mod)) as CV
  })()
  return cached
}

/** Convert an image to a CV matrix. */
function matFromImage(cv: CV, image: ImageArray) {
  const mat = new cv.Mat(image.rows, image.cols, cv.CV_8UC3)
  mat.data.set(image.data)
  return mat
}

/** Convert a CV matrix to an image. */
function imageFromMat(mat: { data: Uint8Array; rows: number; cols: number }): ImageArray {
  return { data: mat.data.slice(), rows: mat.rows, cols: mat.cols }
}

/** Manual GC. */
function deleteAll(mats: unknown[]): void {
  for (const mat of mats) (mat as { delete(): void }).delete()
}

/** Wrapper around cv2.warpPerspective() */
export async function warpPerspective(
  src: ImageArray,
  M: ArrayLike<number>,
  dsize: Size2i,
): Promise<ImageArray> {
  const cv = await loadCv()
  const srcMat = matFromImage(cv, src)
  const mMat = cv.matFromArray(3, 3, cv.CV_64FC1, Array.from(M))
  const dst = new cv.Mat()
  try {
    cv.warpPerspective(
      srcMat,
      dst,
      mMat,
      new cv.Size(dsize.width, dsize.height),
      cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
    )
    return imageFromMat(dst)
  } finally {
    deleteAll([srcMat, mMat, dst])
  }
}

/** Wrapper around cv2.getRectSubPix() */
export async function getRectSubPix(
  src: ImageArray,
  patchSize: Size2i,
  center: Point2,
): Promise<ImageArray> {
  const cv = await loadCv()
  const srcMat = matFromImage(cv, src)
  const patch = new cv.Mat()
  try {
    cv.getRectSubPix(
      srcMat,
      new cv.Size(patchSize.width, patchSize.height),
      new cv.Point(center.x, center.y),
      patch,
      cv.CV_8U,
    )
    return imageFromMat(patch)
  } finally {
    deleteAll([srcMat, patch])
  }
}

/** Wrapper around cv2.solvePnP() */
export async function solvePnP(
  objectPoints: ArrayLike<number>,
  imagePoints: ArrayLike<number>,
  cameraMatrix: ArrayLike<number>,
): Promise<{ rvec: Float64Array; tvec: Float64Array }> {
  const cv = await loadCv()
  const n = objectPoints.length / 3
  const obj = cv.matFromArray(n, 1, cv.CV_64FC3, Array.from(objectPoints))
  const img = cv.matFromArray(n, 1, cv.CV_64FC2, Array.from(imagePoints))
  const kMat = cv.matFromArray(3, 3, cv.CV_64FC1, Array.from(cameraMatrix))
  const dist = new cv.Mat()
  const rvec = new cv.Mat()
  const tvec = new cv.Mat()
  try {
    // SOLVEPNP_SQPNP exists at runtime but is missing from the shipped
    // typings; its value in OpenCV 5.0.0 is 6.
    const SOLVEPNP_SQPNP = (cv as { SOLVEPNP_SQPNP?: number }).SOLVEPNP_SQPNP ?? 6
    cv.solvePnP(obj, img, kMat, dist, rvec, tvec, false, SOLVEPNP_SQPNP)
    return {
      rvec: Float64Array.from(rvec.data64F),
      tvec: Float64Array.from(tvec.data64F),
    }
  } finally {
    deleteAll([obj, img, kMat, dist, rvec, tvec])
  }
}

/** Wrapper around cv2.Rodrigues() that returns a rotation matrix. */
export async function rodrigues(rvec: ArrayLike<number>): Promise<Float64Array> {
  const cv = await loadCv()
  const src = cv.matFromArray(3, 1, cv.CV_64FC1, Array.from(rvec))
  const dst = new cv.Mat()
  const jacobian = new cv.Mat()
  try {
    cv.Rodrigues(src, dst, jacobian)
    return Float64Array.from(dst.data64F)
  } finally {
    deleteAll([src, dst, jacobian])
  }
}

/** Wrapper around cv2.Rodrigues() that returns a rotation vector. */
export async function rodriguesToVector(
  R: ArrayLike<number>,
): Promise<Float64Array> {
  const cv = await loadCv()
  const src = cv.matFromArray(3, 3, cv.CV_64FC1, Array.from(R))
  const dst = new cv.Mat()
  const jacobian = new cv.Mat()
  try {
    cv.Rodrigues(src, dst, jacobian)
    return Float64Array.from(dst.data64F)
  } finally {
    deleteAll([src, dst, jacobian])
  }
}

/** Wrapper around cv2.projectPoints() */
export async function projectPoints(
  objectPoints: ArrayLike<number>,
  rvec: ArrayLike<number>,
  tvec: ArrayLike<number>,
  cameraMatrix: ArrayLike<number>,
): Promise<Float64Array> {
  const cv = await loadCv()
  const n = objectPoints.length / 3
  const obj = cv.matFromArray(n, 1, cv.CV_64FC3, Array.from(objectPoints))
  const isMatrix = rvec.length === 9
  const rot = isMatrix ? await rodriguesToVector(rvec) : new Float64Array(rvec)
  const rMat = cv.matFromArray(3, 1, cv.CV_64FC1, Array.from(rot))
  const tMat = cv.matFromArray(3, 1, cv.CV_64FC1, Array.from(tvec))
  const kMat = cv.matFromArray(3, 3, cv.CV_64FC1, Array.from(cameraMatrix))
  const dist = new cv.Mat()
  const out = new cv.Mat()
  try {
    cv.projectPoints(obj, rMat, tMat, kMat, dist, out)
    return Float64Array.from(out.data64F)
  } finally {
    deleteAll([obj, rMat, tMat, kMat, dist, out])
  }
}

/** Wrapper around cv2.gemm() */
export async function matMul3x3(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
): Promise<Float64Array> {
  const cv = await loadCv()
  const aMat = cv.matFromArray(3, 3, cv.CV_64FC1, Array.from(a))
  const bMat = cv.matFromArray(3, 3, cv.CV_64FC1, Array.from(b))
  const dst = new cv.Mat()
  try {
    cv.gemm(aMat, bMat, 1, new cv.Mat(), 0, dst)
    return Float64Array.from(dst.data64F)
  } finally {
    deleteAll([aMat, bMat, dst])
  }
}

/** Wrapper around cv2.invert() */
export async function invert3x3(m: ArrayLike<number>): Promise<Float64Array> {
  const cv = await loadCv()
  const mMat = cv.matFromArray(3, 3, cv.CV_64FC1, Array.from(m))
  const dst = new cv.Mat()
  try {
    cv.invert(mMat, dst, cv.DECOMP_LU)
    return Float64Array.from(dst.data64F)
  } finally {
    deleteAll([mMat, dst])
  }
}

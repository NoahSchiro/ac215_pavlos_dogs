/// <reference types="node" />
import { Blob as NodeBlob } from 'node:buffer'
import { IDBFactory, IDBObjectStore } from 'fake-indexeddb'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { createCalibrationRecord } from './record'
import { testSession } from './fixtures/testSession'
import {
  CALIBRATION_DB_NAME, CALIBRATION_KEY, CALIBRATION_STORE,
  clearCalibration, loadCalibration, saveCalibration,
} from './storage'

// Write raw data to simulate an older version or a corrupt existing record.
async function putRaw(value: unknown) {
  const request = indexedDB.open(CALIBRATION_DB_NAME, 1)
  const db = await new Promise<IDBDatabase>((resolve, reject) => {
    request.onupgradeneeded = () => request.result.createObjectStore(CALIBRATION_STORE)
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(CALIBRATION_STORE, 'readwrite')
      tx.oncomplete = () => resolve()
      tx.onabort = () => reject(tx.error)
      tx.objectStore(CALIBRATION_STORE).put(value, CALIBRATION_KEY)
    })
  } finally { db.close() }
}

beforeEach(() => {
  vi.stubGlobal('indexedDB', new IDBFactory())
  // Node's structuredClone supports Node Blob; jsdom's Blob is not cloneable.
  vi.stubGlobal('Blob', NodeBlob)
})
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

test('restores a complete record, including Blobs, fit, lambda, and capture metadata', async () => {
  const record = testSession().record
  await saveCalibration(record)
  const loaded = await loadCalibration()
  expect(loaded.status).toBe('restored')
  if (loaded.status !== 'restored') throw new Error('Expected restored record')
  expect(loaded.record.calibration.fit).toEqual(record.calibration.fit)
  expect(loaded.record.calibration.validFrameCounts).toEqual(Array(9).fill(1))
  expect(loaded.record.compatibility).toEqual(record.compatibility)
  expect(loaded.record.calibration.capture.camera).toEqual(record.calibration.capture.camera)
  expect(loaded.record.calibration.capture.samples[0].image).toBeInstanceOf(Blob)
  expect(await loaded.record.calibration.capture.samples[0].image.text()).toBe('jpeg')
  expect(loaded.record.calibration.frames[0].sample).toBe(loaded.record.calibration.capture.samples[0])
})

test('replaces one complete record and clear removes it', async () => {
  const first = testSession().record
  await saveCalibration(first)
  const next = { ...first, savedAt: first.savedAt + 1 }
  await saveCalibration(next)
  expect(await loadCalibration()).toEqual({ status: 'restored', record: next })
  await clearCalibration()
  expect(await loadCalibration()).toEqual({ status: 'empty' })
})

test.each([
  ['format version', (r: ReturnType<typeof createCalibrationRecord>) => ({ ...r, version: 99 })],
  ['model version', (r: ReturnType<typeof createCalibrationRecord>) => ({ ...r, compatibility: { ...r.compatibility, modelId: 'old-model' } })],
  ['preprocessing version', (r: ReturnType<typeof createCalibrationRecord>) => ({ ...r, compatibility: { ...r.compatibility, preprocessingVersion: 99 } })],
  ['intrinsics version', (r: ReturnType<typeof createCalibrationRecord>) => ({ ...r, compatibility: { ...r.compatibility, intrinsicsVersion: 99 } })],
  ['orientation', (r: ReturnType<typeof createCalibrationRecord>) => ({ ...r, compatibility: { ...r.compatibility, imageOrientation: 'mirrored' } })],
  ['non-finite fit', (r: ReturnType<typeof createCalibrationRecord>) => ({ ...r, calibration: { ...r.calibration, fit: { ...r.calibration.fit, intercept: [NaN, 0] } } })],
  ['wrong counts', (r: ReturnType<typeof createCalibrationRecord>) => ({ ...r, calibration: { ...r.calibration, validFrameCounts: Array(9).fill(3) } })],
  ['missing samples', (r: ReturnType<typeof createCalibrationRecord>) => ({ ...r, calibration: { ...r.calibration, capture: { ...r.calibration.capture, samples: [] } } })],
  ['unknown camera', (r: ReturnType<typeof createCalibrationRecord>) => ({ ...r, calibration: { ...r.calibration, capture: { ...r.calibration.capture, camera: null } } })],
])('rejects a record with %s', async (_, corrupt) => {
  await putRaw(corrupt(testSession().record))
  expect(await loadCalibration()).toMatchObject({ status: 'invalid', reason: expect.any(String) })
})

test('rejects finite coefficients that no longer match the saved prediction-target pairs', async () => {
  const record = testSession().record
  await putRaw({ ...record, calibration: { ...record.calibration, fit: { ...record.calibration.fit, intercept: [0.4, 0.6] } } })
  expect(await loadCalibration()).toMatchObject({ status: 'invalid' })
})

test('transaction abort rolls back a replacement even after put succeeds', async () => {
  const first = testSession().record
  await saveCalibration(first)
  const put = IDBObjectStore.prototype.put
  const spy = vi.spyOn(IDBObjectStore.prototype, 'put').mockImplementation(function (this: IDBObjectStore, value, key) {
    const request = put.call(this, value, key)
    request.addEventListener('success', () => this.transaction.abort())
    return request
  })
  await expect(saveCalibration({ ...first, savedAt: first.savedAt + 1 })).rejects.toThrow()
  spy.mockRestore()
  expect(await loadCalibration()).toEqual({ status: 'restored', record: first })
})

test('cancelled writes do not replace the previous record', async () => {
  const first = testSession().record
  await saveCalibration(first)
  const controller = new AbortController()
  const put = IDBObjectStore.prototype.put
  const spy = vi.spyOn(IDBObjectStore.prototype, 'put').mockImplementation(function (this: IDBObjectStore, value, key) {
    const request = put.call(this, value, key)
    request.addEventListener('success', () => controller.abort())
    return request
  })
  await expect(saveCalibration({ ...first, savedAt: first.savedAt + 1 }, controller.signal)).rejects.toMatchObject({ name: 'AbortError' })
  spy.mockRestore()
  expect(await loadCalibration()).toEqual({ status: 'restored', record: first })
})

test('a failed clear preserves the existing record', async () => {
  const first = testSession().record
  await saveCalibration(first)
  const remove = IDBObjectStore.prototype.delete
  const spy = vi.spyOn(IDBObjectStore.prototype, 'delete').mockImplementation(function (this: IDBObjectStore, key) {
    const request = remove.call(this, key)
    request.addEventListener('success', () => this.transaction.abort())
    return request
  })
  await expect(clearCalibration()).rejects.toThrow()
  spy.mockRestore()
  expect(await loadCalibration()).toEqual({ status: 'restored', record: first })
})

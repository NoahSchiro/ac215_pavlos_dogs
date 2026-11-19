import { validateCalibrationRecord, type CalibrationRecord } from './record'

export const CALIBRATION_DB_NAME = 'gaze-maze-calibration'
export const CALIBRATION_STORE = 'calibrations'
export const CALIBRATION_KEY = 'current'

export type CalibrationLoadResult =
  | { status: 'empty' }
  | { status: 'restored'; record: CalibrationRecord }
  | { status: 'invalid'; reason: string }

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (!globalThis.indexedDB) { reject(new Error('IndexedDB storage is unavailable.')); return }
    const request = indexedDB.open(CALIBRATION_DB_NAME, 2)
    let blocked = false
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(CALIBRATION_STORE)) request.result.createObjectStore(CALIBRATION_STORE)
      // Version 1 records contain photos; discard them when upgrading to compact records.
      else request.transaction!.objectStore(CALIBRATION_STORE).clear()
    }
    request.onblocked = () => { blocked = true; reject(new Error('Calibration storage is blocked by another open page. Close it and retry.')) }
    request.onerror = () => reject(request.error ?? new Error('Could not open calibration storage.'))
    request.onsuccess = () => {
      if (blocked) { request.result.close(); return }
      request.result.onversionchange = () => request.result.close()
      resolve(request.result)
    }
  })
}

async function transaction<T>(
  mode: IDBTransactionMode,
  action: (store: IDBObjectStore) => IDBRequest<T>,
  signal?: AbortSignal,
): Promise<T> {
  signal?.throwIfAborted()
  const db = await openDatabase()
  try {
    signal?.throwIfAborted()
    return await new Promise<T>((resolve, reject) => {
      const tx = db.transaction(CALIBRATION_STORE, mode)
      let result: T
      let error: unknown
      const abort = () => { try { tx.abort() } catch { /* Already completed. */ } }
      const cleanup = () => signal?.removeEventListener('abort', abort)
      tx.oncomplete = () => { cleanup(); resolve(result) }
      tx.onabort = () => { cleanup(); reject(signal?.aborted ? signal.reason : error ?? tx.error ?? new Error('Calibration storage transaction aborted.')) }
      signal?.addEventListener('abort', abort, { once: true })
      try {
        const request = action(tx.objectStore(CALIBRATION_STORE))
        request.onsuccess = () => { result = request.result }
        request.onerror = () => { error = request.error }
      } catch (caught) {
        error = caught
        abort()
      }
    })
  } finally { db.close() }
}

export async function loadCalibration(signal?: AbortSignal): Promise<CalibrationLoadResult> {
  const value = await transaction('readonly', (store) => store.get(CALIBRATION_KEY), signal)
  if (value === undefined) return { status: 'empty' }
  try {
    validateCalibrationRecord(value)
    return { status: 'restored', record: value }
  } catch (error) {
    return { status: 'invalid', reason: error instanceof Error ? error.message : 'Invalid saved calibration.' }
  }
}

export async function saveCalibration(record: CalibrationRecord, signal?: AbortSignal): Promise<void> {
  signal?.throwIfAborted()
  validateCalibrationRecord(record)
  await transaction('readwrite', (store) => store.put(record, CALIBRATION_KEY), signal)
}

export async function clearCalibration(signal?: AbortSignal): Promise<void> {
  await transaction('readwrite', (store) => store.delete(CALIBRATION_KEY), signal)
}

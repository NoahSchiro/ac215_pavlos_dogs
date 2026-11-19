/**
 * The OpenCV runtime, behind a function.
 *
 * `@techstark/opencv-js` is CommonJS, and its default export is the emscripten
 * runtime's Promise. Imported dynamically, the production bundler (rolldown)
 * wraps that export in its CJS interop object, which copies the Promise's
 * prototype without being a Promise; resolving the import with it throws
 * "Promise.prototype.then called on incompatible receiver". The Vite dev
 * server does not wrap it, so only the production build broke.
 *
 * Importing it statically here and handing it out through a function means
 * the dynamic import in cv.ts resolves with this ES module's namespace, which
 * is not thenable, and the runtime Promise is awaited directly.
 */
import cv from '@techstark/opencv-js'

export const getCv = (): unknown => cv

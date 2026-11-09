/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { preview } from '@vitest/browser-preview'
import { defineConfig } from 'vite'

// Two test projects:
// - "app": the jsdom component tests, as before.
// - "vision": the preprocessing pipeline tests, which need a real browser
export default defineConfig({
  plugins: [react()],
  test: {
    projects: [
      {
        extends: true,
        test: {
          name: 'app',
          environment: 'jsdom',
          setupFiles: ['./src/test-setup.ts'],
          include: ['src/**/*.{test,spec}.{ts,tsx}'],
          exclude: ['src/vision/**'],
        },
      },
      {
        extends: true,
        test: {
          name: 'vision',
          environment: 'node',
          include: ['src/vision/**/*.{test,spec}.{ts,tsx}'],
          browser: {
            enabled: true,
            provider: preview(),
            instances: [{ browser: 'chrome' }],
          },
        },
      },
    ],
  },
})

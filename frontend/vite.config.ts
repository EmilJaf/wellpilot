import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // FastAPI runs on :8000 in development; in production it serves the built frontend itself.
    proxy: { '/api': 'http://localhost:8000' },
  },
})

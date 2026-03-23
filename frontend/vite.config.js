import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/ask/stream': {
        target: 'http://localhost:8000',
        // SSE requires no response buffering
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['X-Accel-Buffering'] = 'no'
            proxyRes.headers['Cache-Control'] = 'no-cache'
            proxyRes.headers['Connection'] = 'keep-alive'
          })
        },
      },
      '/ask/speak': {
        target: 'http://localhost:8000',
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['X-Accel-Buffering'] = 'no'
            proxyRes.headers['Cache-Control'] = 'no-cache'
            proxyRes.headers['Connection'] = 'keep-alive'
          })
        },
      },
      '/ask': 'http://localhost:8000',
      '/narrate': 'http://localhost:8000',
      '/schemes': 'http://localhost:8000',
      '/categories': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})

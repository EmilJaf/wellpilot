import { useEffect, useState } from 'react'

type Health = { status: string; version: string; llm_provider: string }

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/health')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setHealth)
      .catch((e: Error) => setError(e.message))
  }, [])

  const backendStatus = error
    ? `unreachable (${error})`
    : health
      ? `${health.status}, v${health.version}, LLM provider: ${health.llm_provider}`
      : 'checking...'

  return (
    <main className="min-h-screen bg-slate-950 p-8 text-slate-100">
      <h1 className="text-2xl font-semibold">WellPilot</h1>
      <p className="mt-2 text-slate-400">
        AI agent for well performance analysis on the Volve field.
      </p>
      <p className="mt-6 font-mono text-sm">backend: {backendStatus}</p>
    </main>
  )
}

export default App

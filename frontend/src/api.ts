// Types and calls for the WellPilot API (FastAPI, proxied at /api in development).

export type Table = { columns: string[]; rows: (string | number | boolean | null)[][] }

export type ToolCallView = {
  name: string
  arguments: Record<string, unknown>
  evidence_id?: string | null
  warnings?: string[]
  preview?: string
  error?: string
  accepted?: boolean
  problems?: string[]
}

export type StepEvent = {
  type: 'step'
  index: number
  content: string | null
  reasoning?: string | null // Muse Spark Open's visible reasoning (Baseten); hidden on Meta's API
  tool_calls: ToolCallView[]
  prompt_tokens: number
  completion_tokens: number
  seconds: number
}

export type DeclaredNumber = { text: string; value: number; evidence_id: string }

export type Finding = {
  claim: string
  evidence_ids: string[]
  confidence: 'fact' | 'inference'
  numbers?: DeclaredNumber[] | null
}

export type EvidenceMeta = {
  id: string
  tool: string
  arguments: Record<string, unknown>
  code: string
  created_at: string
  warnings: string[]
}

export type Evidence = EvidenceMeta & { result: unknown }

export type FinalEvent = {
  type: 'final'
  question: string
  summary: string | null
  summary_numbers?: DeclaredNumber[]
  findings: Finding[]
  verified: boolean
  finished: boolean
  /** Why the run ended early: step or time limit, a model provider error. */
  stopped?: string | null
  check_rounds: string[][]
  steps: number
  tokens: { input: number; output: number }
  evidence: EvidenceMeta[]
}

export type RunEvent = { seq: number } & (
  | { type: 'run_started'; id: string; question: string; as_of: string | null; model: string; provider: string }
  | { type: 'llm_call'; step: number }
  | StepEvent
  | FinalEvent
  | { type: 'run_error'; message: string }
)

export type RunSummary = {
  id: string
  question: string
  as_of: string | null
  created_at: string
  status: 'running' | 'done' | 'error'
  verified: boolean | null
}

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (typeof body?.detail === 'string') detail = body.detail
      // FastAPI validation errors are a list; say what to fix instead of dumping it.
      else if (response.status === 422) detail = 'Ask a question of at least 5 characters.'
    } catch {
      // keep the status line
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export const api = {
  startRun: (question: string, asOf?: string) =>
    fetch('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, as_of: asOf || null }),
    }).then((r) => json<RunSummary>(r)),
  runs: () => fetch('/api/runs').then((r) => json<RunSummary[]>(r)),
  evidence: (runId: string, evidenceId: string) =>
    fetch(`/api/runs/${runId}/evidence/${evidenceId}`).then((r) => json<Evidence>(r)),
  daily: (well: string) => fetch(`/api/wells/${encodeURIComponent(well)}/daily`).then((r) => json<Table>(r)),
  pageImageUrl: (docId: string, page: number) => `/api/reports/${docId}/pages/${page}/image`,
}

/**
 * Follow a run's event stream. Returns a function that closes it.
 *
 * A dropped connection reconnects on its own and the server resumes after the last event
 * id it sent, so nothing is repeated. onFail fires only when the stream is closed for good
 * before the run ended (unknown run, server gone).
 */
export function followRun(runId: string, onEvent: (event: RunEvent) => void, onFail: (message: string) => void): () => void {
  const source = new EventSource(`/api/runs/${runId}/events`)
  let ended = false
  const types: RunEvent['type'][] = ['run_started', 'llm_call', 'step', 'final', 'run_error']
  for (const type of types) {
    source.addEventListener(type, (message) => {
      if (!(message instanceof MessageEvent)) return
      const event = JSON.parse(message.data) as RunEvent
      onEvent(event)
      if (type === 'final' || type === 'run_error') {
        ended = true
        source.close()
      }
    })
  }
  source.onerror = () => {
    if (!ended && source.readyState === EventSource.CLOSED) {
      onFail('The run could not be loaded: it does not exist, or the server is unreachable.')
    }
  }
  return () => {
    ended = true
    source.close()
  }
}

/** Column index helper for column-wise tables. */
export function column(table: Table, name: string): number {
  return table.columns.indexOf(name)
}

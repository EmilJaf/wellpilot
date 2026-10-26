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
  tool_calls: ToolCallView[]
  prompt_tokens: number
  completion_tokens: number
  seconds: number
}

export type Finding = {
  claim: string
  evidence_ids: string[]
  confidence: 'fact' | 'inference'
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
  findings: Finding[]
  verified: boolean
  finished: boolean
  check_rounds: string[][]
  steps: number
  tokens: { input: number; output: number }
  evidence: EvidenceMeta[]
}

export type RunEvent =
  | { type: 'run_started'; id: string; question: string; as_of: string | null; model: string; provider: string }
  | { type: 'llm_call'; step: number }
  | StepEvent
  | FinalEvent
  | { type: 'error'; message: string }

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
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
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

/** Follow a run's event stream. Returns a function that closes it. */
export function followRun(runId: string, onEvent: (event: RunEvent) => void, onClose: () => void): () => void {
  const source = new EventSource(`/api/runs/${runId}/events`)
  const types: RunEvent['type'][] = ['run_started', 'llm_call', 'step', 'final', 'error']
  for (const type of types) {
    source.addEventListener(type, (message) => {
      const event = JSON.parse((message as MessageEvent).data) as RunEvent
      onEvent(event)
      if (type === 'final' || type === 'error') {
        source.close()
        onClose()
      }
    })
  }
  source.onerror = () => {
    // The server ends the stream after the final event; anything else is a dropped link.
    if (source.readyState === EventSource.CLOSED) onClose()
  }
  return () => source.close()
}

/** Column index helper for column-wise tables. */
export function column(table: Table, name: string): number {
  return table.columns.indexOf(name)
}

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

/** Where to mark a text, computed by the backend checker's own rules (positions index the text as written). */
export type Span =
  | { kind: 'quote'; start: number; end: number }
  | { kind: 'number'; start: number; end: number; value: number; evidence_id: string }

export type Finding = {
  claim: string
  evidence_ids: string[]
  confidence: 'fact' | 'inference'
  numbers?: DeclaredNumber[] | null
  spans?: Span[]
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
  summary_spans?: Span[]
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

/** The operator's key, set once in the browser console (localStorage.operatorKey = '...'): lifts the public limits. */
function operatorHeader(): Record<string, string> {
  try {
    const key = localStorage.getItem('operatorKey')
    return key ? { 'X-Operator-Key': key } : {}
  } catch {
    return {}
  }
}

export const api = {
  startRun: (question: string, asOf?: string) =>
    fetch('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...operatorHeader() },
      body: JSON.stringify({ question, as_of: asOf || null }),
    }).then((r) => json<RunSummary>(r)),
  runs: () => fetch('/api/runs').then((r) => json<RunSummary[]>(r)),
  evidence: (runId: string, evidenceId: string) =>
    fetch(`/api/runs/${runId}/evidence/${evidenceId}`).then((r) => json<Evidence>(r)),
  fieldPlan: () => fetch('/api/field/plan').then((r) => json<unknown>(r)),
  fieldDesign: () => fetch('/api/field/design').then((r) => json<unknown>(r)),
  daily: (well: string) => fetch(`/api/wells/${encodeURIComponent(well)}/daily`).then((r) => json<Table>(r)),
  pageQuotes: (docId: string, page: number) =>
    fetch(`/api/reports/${docId}/pages/${page}/quotes`).then((r) => json<{ quote: string; boxes: number[][] }[]>(r)),
  pageImageUrl: (docId: string, page: number) => `/api/reports/${docId}/pages/${page}/image`,
  backtest: () => fetch('/api/backtest').then((r) => json<Backtest>(r)),
  backtestCase: (well: string, cutoff: string) =>
    fetch(`/api/backtest/cases/${encodeURIComponent(well)}/${cutoff}`).then((r) => json<BacktestCase>(r)),
}

export type MethodSummary = { cases: number; median_abs_error: number; mean_abs_error: number; median_bias: number }

/** The backtest: 12-month oil forecasts made at past dates, against what the wells then produced. */
export type Backtest = {
  summary: {
    all_cases: Record<string, MethodSummary>
    agent_closer?: Record<string, { cases: number; closer: number }>
    horizon_days: number
  }
  results: Table
}

type Curve = { date: string[]; rate: number[] }

export type BacktestCase = {
  well: string
  cutoff: string
  horizon_end: string
  actual_sm3: number
  rate: Curve
  methods: {
    method: string
    model: string
    window_start: string
    window_end: string
    fitted: Curve
    forecast: Curve
    p50_sm3: number
    error: number
  }[]
  agent: { question: string; verified: boolean; summary: string; inferences: string[] } | null
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

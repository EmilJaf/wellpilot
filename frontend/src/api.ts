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
  | { kind: 'quote'; start: number; end: number; evidence_id?: string }
  | { kind: 'number'; start: number; end: number; value: number; evidence_id: string }

export type Finding = {
  claim: string
  detail?: string | null // the fit and exact values, for an engineer; checked like the claim
  detail_spans?: Span[]
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
      else if (response.status === 422) {
        const field = body?.detail?.[0]?.loc?.at(-1)
        detail = field === 'question' ? 'Ask a question of at least 5 characters.' : `The request was not valid (${field ?? 'body'}).`
      }
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
  // With `end`, only the days up to it: what an agent cut at that date could see.
  daily: (well: string, end?: string | null) =>
    fetch(`/api/wells/${encodeURIComponent(well)}/daily${end ? `?end=${end}` : ''}`).then((r) => json<Table>(r)),
  pageQuotes: (docId: string, page: number) =>
    fetch(`/api/reports/${docId}/pages/${page}/quotes`).then((r) => json<{ quote: string; boxes: number[][] }[]>(r)),
  pageImageUrl: (docId: string, page: number) => `/api/reports/${docId}/pages/${page}/image`,
  code: (ref: string) => fetch(`/api/code/${encodeURIComponent(ref)}`).then((r) => json<ToolCode>(r)),
  backtest: () => fetch('/api/backtest').then((r) => json<Backtest>(r)),
  hindsight: (runId: string) => fetch(`/api/runs/${runId}/hindsight`).then((r) => json<Hindsight>(r)),
  backtestCase: (well: string, cutoff: string) =>
    fetch(`/api/backtest/cases/${encodeURIComponent(well)}/${cutoff}`).then((r) => json<BacktestCase>(r)),
  film: (well: string) => fetch(`/api/backtest/film/${encodeURIComponent(well)}`).then((r) => json<Film>(r)),
  // A reader's edit of a finished answer, checked against the run's saved evidence (no model call).
  check: (runId: string, edit: CheckRequest) =>
    fetch(`/api/runs/${runId}/check`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(edit),
    }).then((r) => json<CheckResult>(r)),
}

export type CheckRequest = {
  findings: { index: number; claim: string; confidence: 'fact' | 'inference' }[]
  summary?: string
}

export type CheckResult = {
  accepted: boolean
  problems: string[]
  findings: { index: number; claim: string; confidence: 'fact' | 'inference'; numbers: DeclaredNumber[]; spans: Span[] }[]
}

/** The source of the tool function that produced an evidence item. */
export type ToolCode = { ref: string; same_as_run: boolean; file: string; line: number; source: string }

export type MethodSummary = {
  cases: number
  median_abs_error: number
  mean_abs_error: number
  median_bias: number
  misses_over_50pct?: number
  bootstrap_band_coverage?: number
  hindcast_band_coverage?: number | null
  hindcast_band_cases?: number
}

/** The backtest: 12-month oil forecasts made at past dates, against what the wells then produced. */
export type Backtest = {
  summary: {
    all_cases: Record<string, MethodSummary>
    agent_closer?: Record<string, { cases: number; closer: number }>
    best_fixed_window?: string
    agent_models?: Record<string, number>
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
  agent: {
    question: string
    verified: boolean
    summary: string | null
    inferences: string[]
    window_asked?: (string | null)[]
  } | null
}

/** Every forecast made for one well, in cutoff order, on one rate series: the accuracy tab's film. */
export type Film = {
  well: string
  rate: Curve
  data_ends: string
  frames: {
    cutoff: string
    horizon_end: string
    actual_sm3: number
    methods: BacktestCase['methods']
    band: { low: number; high: number; holds: boolean } | null
    agent: BacktestCase['agent']
  }[]
}

/** A run made with data up to a past date, against what happened next. */
export type Hindsight = {
  as_of: string
  forecasts: (BacktestCase & {
    evidence_id: string
    cited?: boolean
    forecast_from: string
    data_ends: string | null
    classic_warned: boolean
    agent_range: { p90: number; p10: number }
    p50_by_model?: Record<string, number> | null
    calibrated_band: { p90: number; p10: number; holds_actual: boolean } | null
  })[]
  wells: {
    well: string
    oil_year_before_sm3: number
    oil_year_after_sm3: number
    water_cut_at_cut: number | null
    water_cut_year_later: number | null
    gauge_dead_at_cut: boolean
    gauge_back_on: string | null
  }[]
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

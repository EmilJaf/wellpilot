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
  review?: ReviewEvent | null
  steps: number
  tokens: { input: number; output: number }
  evidence: EvidenceMeta[]
  /** What each Meta model did for this answer. */
  meta_models?: MetaModels
}

export type MetaModels = {
  muse: {
    model: string | null
    steps: number
    tool_calls: number
    reasoning_shown: boolean
    pages: string[]
    second_opinion: string | null
  }
  sam: { figures: { evidence_id: string; agreed: number; compared: number; not_compared?: number }[]; logs: string[] }
}

export type RunEvent = { seq: number } & (
  | { type: 'run_started'; id: string; question: string; as_of: string | null; model: string; provider: string }
  | { type: 'llm_call'; step: number }
  | StepEvent
  | FinalEvent
  | ({ type: 'review' } & ReviewEvent)
  | { type: 'run_error'; message: string }
)

/** The second opinion on the first accepted answer: the reviewer's points, and what came
    of them (revised, kept because the revision did not pass, or sound). */
export type ReviewEvent = {
  verdict: 'sound' | 'concerns'
  points: { point: string; check: string }[]
  tokens: number
  after_step: number
  outcome?: 'asked' | 'revised' | 'kept' | 'sound'
}

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
  // With `end`, the field only up to that day (the time machine before its reveal).
  fieldPlan: (end?: string | null) => fetch(`/api/field/plan${end ? `?end=${end}` : ''}`).then((r) => json<unknown>(r)),
  fieldDesign: (end?: string | null) => fetch(`/api/field/design${end ? `?end=${end}` : ''}`).then((r) => json<unknown>(r)),
  // With `end`, only the days up to it: what an agent cut at that date could see.
  daily: (well: string, end?: string | null) =>
    fetch(`/api/wells/${encodeURIComponent(well)}/daily${end ? `?end=${end}` : ''}`).then((r) => json<Table>(r)),
  timeline: () => fetch('/api/field/timeline').then((r) => json<Timeline>(r)),
  planFigure: (end?: string | null) =>
    fetch(`/api/field/plan-figure${end ? `?end=${end}` : ''}`).then((r) => json<PlanFigure>(r)),
  planFigureImageUrl: () => '/api/field/plan-figure/image',
  // The well's daily-report jobs; with `end`, only those done by then.
  operations: (well: string, end?: string | null) =>
    fetch(`/api/wells/${encodeURIComponent(well)}/operations${end ? `?end=${end}` : ''}`).then((r) => json<Operations>(r)),
  // The well along its depth: production logs, perforations, plugs; with `end`, only by then.
  inside: (well: string, end?: string | null) =>
    fetch(`/api/wells/${encodeURIComponent(well)}/inside${end ? `?end=${end}` : ''}`).then((r) => json<Inside>(r)),
  logPlotImageUrl: (plotId: string) => `/api/log-plots/${plotId}/image`,
  pageQuotes: (docId: string, page: number) =>
    fetch(`/api/reports/${docId}/pages/${page}/quotes`).then((r) => json<{ quote: string; boxes: number[][] }[]>(r)),
  pageImageUrl: (docId: string, page: number) => `/api/reports/${docId}/pages/${page}/image`,
  code: (ref: string) => fetch(`/api/code/${encodeURIComponent(ref)}`).then((r) => json<ToolCode>(r)),
  backtest: () => fetch('/api/backtest').then((r) => json<Backtest>(r)),
  hindsight: (runId: string) => fetch(`/api/runs/${runId}/hindsight`).then((r) => json<Hindsight>(r)),
  backtestCase: (well: string, cutoff: string) =>
    fetch(`/api/backtest/cases/${encodeURIComponent(well)}/${cutoff}`).then((r) => json<BacktestCase>(r)),
  health: () => fetch('/api/health').then((r) => json<Health>(r)),
  // A recorded question (16 kHz mono WAV, base64) transcribed by the model.
  transcribe: (audioWavB64: string) =>
    fetch('/api/transcribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...operatorHeader() },
      body: JSON.stringify({ audio_wav_b64: audioWavB64 }),
    }).then((r) => json<{ question: string; language: string; tokens: number }>(r)),
  film: (well: string) => fetch(`/api/backtest/film/${encodeURIComponent(well)}`).then((r) => json<Film>(r)),
  // The model's own attempts to get a false version of a finding past the checker.
  attack: (runId: string, index: number, count = 3) =>
    fetch(`/api/runs/${runId}/attack`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...operatorHeader() },
      body: JSON.stringify({ index, count }),
    }).then((r) => json<AttackResult>(r)),
  // A reader's edit of a finished answer, checked against the run's saved evidence (no model call).
  check: (runId: string, edit: CheckRequest) =>
    fetch(`/api/runs/${runId}/check`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(edit),
    }).then((r) => json<CheckResult>(r)),
}

/** Work done on a well, from its daily drilling and well-work reports. */
export type OperationJob = {
  start: string
  end: string
  kind: 'drilling' | 'completion' | 'workover' | 'other'
  report_days: number
  contractor: string
  perforations: string[]
  summaries: { date: string; summary: string }[]
}

export type Operations = { well: string; jobs: OperationJob[] }

type Production = { water_cut: number; oil_sm3_per_day: number; days: number; from: string; to: string }

/** A well along its depth: where its oil and water came in, and the work done on it. */
export type Inside = {
  well: string
  zones_m: [number, number][]
  surveys: {
    survey: string
    date: string
    read_from: string
    rate_unit?: string
    check?: {
      its_plot_read_by?: string
      plot_against_files_median_share_gap?: Record<string, Record<string, number>>
      agreement_with_colour_trace_median_share_gap?: Record<string, number>
      upper_two_zones_share?: Record<string, number>
      report_says?: { quote: string; share: number; page: number }
      perforated_zones_m?: [number, number][]
    }
    production_30_days_before: { water_cut: number; oil_sm3_per_producing_day: number }
    plots: { id: string; pass: string; page: number; read_by: Record<string, string>; prompts: Record<string, string> }[]
    passes: {
      pass: string
      water_cut_at_top: number
      depth_m: number[]
      oil_share: number[]
      water_share: number[] | null
    }[]
  }[]
  jobs: {
    start: string
    end: string
    kind: string
    production_log: boolean
    plugs_m: number[]
    perforations_m: [number, number][]
    summaries: { date: string; summary: string }[]
    before: Production | null
    after: Production | null
  }[]
}

/** The plan's forecast curve read off its Figure 4.2, with the field's monthly oil rate. */
export type PlanFigure = {
  read_by: string
  prompt?: string
  objects_found?: number
  confidence?: number | null
  calibration: { gridlines: number; rate_axis_sm3_per_day: [number, number]; year_axis: [number, number] }
  curve: { month: string[]; rate: number[] }
  agreement_with_colour?: { columns: number; median_sm3_per_day: number; p90_sm3_per_day: number; within_200: number }
  actual: { month: string[]; rate: number[] }
  plateau: { sm3_per_day: number; first: string; last: string; months: number }
  plan_text: { page: number; quote: string; statement_en: string }[]
}

/** The field screen replayed month by month, with each producer's rate and its wells' work. */
export type Timeline = {
  months: [string, string]
  replayed: number
  wells: {
    well: string
    first_oil: string
    oil_sm3: number
    oil_rate: { month: string[]; rate: number[] }
    issues: {
      kind: string
      level: 'alarm' | 'caution' | 'note'
      first_flagged: string
      since: string | null
      cleared: string | null
      text: string
    }[]
    work: { start: string; end: string; perforations: string[]; summary: string }[]
  }[]
}

export type Health = {
  status: string
  version: string
  llm_provider: string
  llm_configured: boolean
  voice: boolean
  data_ready: boolean
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

/** One of the model's attempts on a verified finding, and the checker's verdict on it. */
export type Attempt = {
  finding: number
  original: string
  kind: 'number' | 'well' | 'date' | 'direction' | 'quote' | 'cause' | 'scale' | 'unsupported'
  claim: string
  confidence: 'fact' | 'inference'
  what_is_false: string
  caught: boolean
  problems: string[]
  unchanged: boolean
}

export type AttackResult = { attempts: Attempt[]; model: string; tokens: number }

/** The source of the tool function that produced an evidence item. */
export type ToolCode = {
  ref: string
  same_as_run: boolean
  file: string
  line: number
  source: string
  depends_on?: string[]
  /** For a short wrapper: the analysis functions it calls, which computed the numbers. */
  called?: { name: string; file: string; line: number; source: string }[]
}

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
export type ReasoningCheck =
  | { available: false; note: string }
  | {
      available: true
      steps: number
      characters: number
      later_dates: { step: number; dates: string[]; sentence: string }[]
      hidden_values: { step: number; number: number; matches: string; sentence: string }[]
      forecast_mentions: number
    }

export type Hindsight = {
  as_of: string
  /** The model's reasoning, read for dates and values from the hidden period. */
  reasoning?: ReasoningCheck
  forecasts: (BacktestCase & {
    evidence_id: string
    cited?: boolean
    forecast_from: string
    data_ends: string | null
    classic_warned: boolean
    // Fits on the last 3, 6, 12 and 24 months, made the same day: the fair baselines.
    fixed_windows?: { months: number; p50_sm3: number; error: number | null }[]
    decomposition?: {
      rate_forecast: number
      rate_actual: number
      rate_error: number
      uptime_forecast: number
      uptime_actual: number
      uptime_error: number
    } | null
    window_from_change?: { date: string; evidence_id: string; kind: 'change' | 'work'; text: string | null } | null
    agent_range: { p90: number; p10: number }
    p50_by_model?: Record<string, number> | null
    calibrated_band: { p90: number; p10: number; holds_actual: boolean; past_forecasts?: number | null; coverage?: number | null } | null
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
  /** Work done on the wells the answer names soon after the date, and what it was worth. */
  work_next?: {
    well: string
    start: string
    end: string
    kind: string
    summary: string
    plug_m: number | null
    perforations: string[]
    before: { water_cut: number; oil_sm3_per_day: number; days: number }
    back_on: string
    after: { water_cut: number; oil_sm3_per_day: number; days: number } | null
    oil_year_after_sm3: number
    trend_year_after_sm3: number | null
    trend_fit: { model: string; from: string; r2_log?: number } | null
    trend_windows?: { months: number; model: string; r2_log: number; trend_sm3: number; extra_sm3: number }[]
    extra_oil_range_sm3?: [number, number] | null
    oil_rest_of_life_sm3?: number
    extra_oil_sm3: number | null
    extra_oil_bbl: number | null
    price_usd_per_bbl: number
    extra_value_usd: number | null
    extra_value_range_usd?: [number, number] | null
    volumes_note?: string
    depth?: {
      perforated_m: [number, number][]
      blank_pipe_m: [number, number][]
      earlier_plugs_m: number[]
      answer_plug_m: number[]
      new_perforations_m: [number, number][]
    }
    /** The field's water and oil per day on the days the well flowed, before and after. */
    field_before?: { water_sm3_per_day: number; oil_sm3_per_day: number; days: number } | null
    field_after?: { water_sm3_per_day: number; oil_sm3_per_day: number; days: number } | null
    /** What a plug at the operator's depth and at the answer's sits below on the last log
     * before the job: shares of the water and the oil, low to high over its passes. */
    on_latest_log?: {
      survey: string
      date: string
      depths: { depth_m: number; water_share?: [number, number]; oil_share?: [number, number] }[]
    } | null
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

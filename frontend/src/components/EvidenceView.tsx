import { createContext, useContext, useEffect, useRef, useState } from 'react'
import { api, type Evidence, type Table, type ToolCode } from '../api'
import { CautionIcon } from '../icons'
import { describeArgs, sameQuote, toolName } from '../labels'
import type { Focus } from './Answer'
import { DesignChart, type DesignResult } from './DesignChart'
import { ScreenList, type ScreenResult } from './FieldScreen'
import { WaterfloodChart, type WaterfloodResult } from './WaterfloodChart'
import { PageImage } from './PageImage'
import { PlanChart, type PlanResult } from './PlanChart'

type Props = { runId: string | null; evidenceId: string | null; attempt: number; focus?: Focus | null }

/**
 * Does a value in a tool result match the number the reader clicked? The declared value
 * may be a fraction written as a percent (0.72 for "72%") or the other way round, and the
 * text rounds it: a value within 0.6% is that number. Text matches the number as written.
 */
function matcher(focus: Focus | null | undefined): (v: unknown) => boolean {
  if (!focus) return () => false
  const targets = [focus.value, focus.value / 100, focus.value * 100].map(Math.abs).filter((t) => t > 0)
  return (v) => {
    if (typeof v === 'number') return targets.some((t) => Math.abs(Math.abs(v) - t) <= t * 0.006)
    if (typeof v !== 'string') return false
    return focus.quote ? sameQuote(v, focus.text) : v.includes(focus.text)
  }
}

const HitContext = createContext<(v: unknown) => boolean>(() => false)

function countHits(value: unknown, hit: (v: unknown) => boolean): number {
  if (Array.isArray(value)) return value.reduce((n: number, v) => n + countHits(v, hit), 0)
  if (value && typeof value === 'object') {
    return Object.values(value).reduce((n: number, v) => n + countHits(v, hit), 0)
  }
  return hit(value) ? 1 : 0
}

type Loaded = { key: string; evidence?: Evidence; error?: string }

/** One piece of evidence: what ran, on which arguments, and exactly what it returned. */
export function EvidenceView({ runId, evidenceId, attempt, focus }: Props) {
  // Results are keyed by what was asked for, so a stale result or error never shows under
  // another id, and selecting the same id again retries.
  const key = `${runId}/${evidenceId}/${attempt}`
  const [loaded, setLoaded] = useState<Loaded | null>(null)

  useEffect(() => {
    if (!runId || !evidenceId) return
    let current = true
    api
      .evidence(runId, evidenceId)
      .then((evidence) => current && setLoaded({ key, evidence }))
      .catch((e: Error) => current && setLoaded({ key, error: e.message }))
    return () => {
      current = false
    }
  }, [runId, evidenceId, key])

  const evidence = loaded?.key === key ? loaded.evidence : undefined
  const error = loaded?.key === key ? loaded.error : undefined
  const hit = matcher(focus)
  const host = useRef<HTMLDivElement>(null)

  // Opened from a number: bring its first match into view.
  useEffect(() => {
    if (evidence && focus) host.current?.querySelector('.hit')?.scrollIntoView({ block: 'center' })
  }, [evidence, focus])

  if (!evidenceId) {
    return (
      <div className="panel-body evidence">
        <p className="empty">
          Click a number in the answer, a source tag or a step in the agent log to see exactly
          what the tool returned and which code produced it.
        </p>
      </div>
    )
  }
  if (error) return <div className="panel-body evidence"><p className="empty">Evidence {evidenceId}: {error}</p></div>
  if (!evidence) return <div className="panel-body evidence"><p className="empty">Loading {evidenceId}</p></div>

  const hits = focus ? countHits(evidence.result, hit) : 0
  return (
    <div className="panel-body evidence" ref={host}>
      {focus && (
        <p className="focus-banner">
          {focus.quote ? (
            <>The quoted passage was read from this report{hits ? ': its page is shown below, with the passage marked.' : '.'}</>
          ) : (
            <>
              <strong>{focus.text}</strong> in the answer comes from this result{hits ? ', marked below.' : '.'}
            </>
          )}
        </p>
      )}
      <h3 className="evidence-title">
        {evidence.id}: {toolName(evidence.tool)}
      </h3>
      <dl className="evidence-meta">
        <dt>Asked for</dt>
        <dd>{describeArgs(evidence.tool, evidence.arguments) || 'no arguments'}</dd>
        <dt>Code</dt>
        <dd>
          <CodeRef key={evidence.code} codeRef={evidence.code} />
        </dd>
        <dt>Ran at</dt>
        <dd>{evidence.created_at.replace('T', ' ').replace('+00:00', ' UTC')}</dd>
      </dl>
      {evidence.warnings.map((w, i) => (
        <div key={i} className="flag flag-caution">
          <CautionIcon />
          <span>{w}</span>
        </div>
      ))}
      <HitContext.Provider value={hit}>
        <Result tool={evidence.tool} result={evidence.result} focus={focus} />
      </HitContext.Provider>
    </div>
  )
}

const REPO = 'https://github.com/EmilJaf/wellpilot/blob/main'

/** The code ref, with the tool's Python source one click away: the code that ran. */
function CodeRef({ codeRef }: { codeRef: string }) {
  const [code, setCode] = useState<{ value: ToolCode | null; error?: string } | null>(null)
  const [open, setOpen] = useState(false)
  const toggle = () => {
    setOpen(!open)
    if (!code)
      api
        .code(codeRef)
        .then((value) => setCode({ value }))
        .catch((e: Error) => setCode({ value: null, error: e.message }))
  }
  return (
    <>
      <code>{codeRef}</code>{' '}
      <button type="button" className="link-button" aria-expanded={open} onClick={toggle}>
        {open ? 'Hide the code' : 'Show the code'}
      </button>
      {open &&
        (code?.value ? (
          <div className="tool-code">
            <p className="evidence-sub">
              {code.value.same_as_run
                ? 'The Python function that produced this result, exactly as it ran (its hash matches).'
                : 'This tool changed after the run; today’s version is shown.'}{' '}
              <a href={`${REPO}/${code.value.file}#L${code.value.line}`} target="_blank" rel="noreferrer">
                {code.value.file.split('/').pop()}, line {code.value.line}
              </a>
            </p>
            <pre className="code">{code.value.source}</pre>
          </div>
        ) : (
          <p className="evidence-sub">{code?.error ? `The code could not load: ${code.error}` : 'Loading the code'}</p>
        ))}
    </>
  )
}

const isTable = (v: unknown): v is Table =>
  typeof v === 'object' && v !== null && Array.isArray((v as Table).columns) && Array.isArray((v as Table).rows)

function Result({ tool, result, focus }: { tool: string; result: unknown; focus?: Focus | null }) {
  if (tool === 'calculate') {
    const r = result as { expression: string; value: number }
    return <CalcResult expression={r.expression} value={r.value} />
  }
  if (tool === 'detect_events') return <Events events={result as EventRow[]} />
  if (tool === 'plan_vs_actual') return <PlanEvidence result={result as PlanResult} />
  if (tool === 'look_at_page') return <PageReading result={result as ReadingResult} />
  if (tool === 'field_screen') return <ScreenList result={result as ScreenResult} />
  if (tool === 'waterflood') {
    const wf = result as WaterfloodResult
    return (
      <>
        <WaterfloodChart result={wf} />
        <p className="evidence-sub">By year: liquid rate held while the choke opened means water took the well&rsquo;s capacity.</p>
        <DataTable table={wf.by_year} />
        <p className="evidence-sub">The field&rsquo;s water injected against liquid produced (surface volumes):</p>
        <DataTable table={wf.field_voidage} />
        <DataTable table={wf.monthly} />
      </>
    )
  }
  if (tool === 'field_vs_design') {
    const design = result as DesignResult
    return (
      <>
        <DesignChart result={design} />
        <DataTable table={design.months} />
      </>
    )
  }
  if (tool === 'search_reports') return <Facts result={result as SearchResult} focus={focus} />
  if (tool === 'read_report_page') return <ReportPage page={result as PageResult} focus={focus} />
  if (tool === 'fit_decline' || tool === 'forecast_production') return <Fit result={result as FitResult} />
  if (isTable(result)) return <DataTable table={result} />
  return <pre className="code">{JSON.stringify(result, null, 2)}</pre>
}

type ReadingResult = {
  doc_id: string
  page: number
  question: string
  answer: string
  readings: { label: string; value: number; unit: string; where: string }[]
  read_by: string
}

/** What the model read off a report page image during the run, next to the page itself. */
function PageReading({ result }: { result: ReadingResult }) {
  const hit = useContext(HitContext)
  return (
    <>
      <p className="evidence-sub">
        The agent showed this page to {result.read_by.replace('meta-models/', '')} and asked: “{result.question}”
      </p>
      <p className="reading-answer">{result.answer}</p>
      {result.readings.length > 0 && (
        <table className="data-table reading-table">
          <thead>
            <tr>
              <th scope="col">Read</th>
              <th scope="col">Value</th>
              <th scope="col">From the</th>
            </tr>
          </thead>
          <tbody>
            {result.readings.map((r, i) => (
              <tr key={i}>
                <td>{r.label}</td>
                <td className={hit(r.value) ? 'hit' : undefined}>
                  {r.value.toLocaleString('en')} {r.unit}
                </td>
                <td>{r.where}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <PageImage doc={result.doc_id} page={result.page} alt={`${result.doc_id} page ${result.page}`} marks={[]} />
    </>
  )
}

function PlanEvidence({ result }: { result: PlanResult }) {
  const hit = useContext(HitContext)
  return (
    <>
      <PlanChart result={result} />
      <p className="evidence-sub">The plan's rows as quoted from the page (year, mean, P90, P50, P10):</p>
      <ul className="quoted-rows">
        {result.plan_rows_quoted.map((q) => (
          <li key={q} lang="no" className={hit(q) ? 'hit' : undefined}>
            {q}
          </li>
        ))}
      </ul>
      <DataTable table={result.years} />
    </>
  )
}

function CalcResult({ expression, value }: { expression: string; value: number }) {
  const hit = useContext(HitContext)
  return (
    <p className="calc">
      <code>{expression}</code> = <span className={hit(value) ? 'hit' : undefined}>{value}</span>
    </p>
  )
}

function DataTable({ table, maxRows = 200 }: { table: Table; maxRows?: number }) {
  const hit = useContext(HitContext)
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            {table.columns.map((c) => (
              <th key={c} className={typeof table.rows[0]?.[table.columns.indexOf(c)] === 'number' ? 'num' : ''}>
                {c.replace(/_/g, ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.slice(0, maxRows).map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td key={j} className={`${typeof v === 'number' ? 'num' : ''}${hit(v) ? ' hit' : ''}`}>
                  {v === null ? '' : String(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

type EventRow = {
  kind: string
  signal: string
  start: string
  end: string | null
  summary: string
  details?: Record<string, unknown>
}

const EVENT_KINDS: Record<string, string> = {
  gauge_failure: 'Gauge failure',
  regime_change: 'Regime change',
  threshold_crossing: 'Threshold crossed',
  shut_in: 'Shut-in',
}

function Events({ events }: { events: EventRow[] }) {
  const hit = useContext(HitContext)
  return (
    <ul className="event-list">
      {events.map((e, i) => (
        <li key={i} className={countHits(e, hit) ? 'hit' : undefined}>
          <span className="event-kind">
            {EVENT_KINDS[e.kind] ?? e.kind}, {e.signal.replace(/_/g, ' ')}
          </span>
          <div>{e.summary}</div>
        </li>
      ))}
    </ul>
  )
}

type SearchResult = { documents: Record<string, { title: string; issued: string }>; facts?: Table; pages?: Table }

function Facts({ result, focus }: { result: SearchResult; focus?: Focus | null }) {
  const hit = useContext(HitContext)
  const table = result.facts
  // Opened from a quote in the answer: show that fact's page straight away.
  const quoteAt = table ? table.columns.indexOf('quote') : -1
  const quoted = focus?.quote && table ? table.rows.findIndex((r) => hit(r[quoteAt])) : -1
  const [open, setOpen] = useState<number | null>(quoted >= 0 ? quoted : null) // the fact showing its page
  if (!table) return result.pages ? <DataTable table={result.pages} /> : null
  const at = (row: Table['rows'][number], name: string) => row[table.columns.indexOf(name)]
  if (!table.rows.length) return <p className="empty">No report facts matched this search.</p>
  return (
    <>
      {Object.entries(result.documents).map(([id, doc]) => (
        <p key={id} className="narration">
          {doc.title}, issued {doc.issued.slice(0, 7)}
        </p>
      ))}
      <ul className="fact-list">
        {table.rows.map((row, i) => {
          const doc = String(at(row, 'doc_id'))
          const page = Number(at(row, 'page'))
          const check = String(at(row, 'quote_check'))
          return (
            <li key={i} className={row.some(hit) ? 'hit' : undefined}>
              <span className="fact-source">
                Page {page}, {String(at(row, 'source'))}{' '}
                <span className={`check-badge${check === 'unverified' ? ' check-badge-unverified' : ''}`}>
                  {check === 'exact'
                    ? 'quote found on the page'
                    : check === 'tokens'
                      ? 'words found on the page out of order; quote not shown'
                      : 'quote not found on the page; not shown'}
                </span>
              </span>
              <div>{String(at(row, 'statement_en'))}</div>
              {at(row, 'quote') !== null && <p className="fact-quote" lang="no">{String(at(row, 'quote'))}</p>}
              <button type="button" className="link-button" onClick={() => setOpen(open === i ? null : i)}>
                {open === i ? 'Hide page' : `Show page ${page}`}
              </button>
              {open === i && (
                <PageImage doc={doc} page={page} alt={`${doc} page ${page}`} marks={[String(at(row, 'quote') ?? '')]} />
              )}
            </li>
          )
        })}
      </ul>
    </>
  )
}

type PageResult = { document: { title: string }; page: number; text: string; image: string; summary_en?: string; facts?: Table }

function ReportPage({ page, focus }: { page: PageResult; focus?: Focus | null }) {
  const doc = page.image.split('/')[1]
  return (
    <>
      {page.summary_en && <p>{page.summary_en}</p>}
      <PageImage
        doc={doc}
        page={page.page}
        alt={`${page.document.title}, page ${page.page}`}
        marks={focus?.quote ? [focus.text] : []}
      />
    </>
  )
}

type FitResult = {
  best?: Record<string, number | string>
  fit?: Record<string, number | string>
  cumulative_oil_sm3?: Record<string, number>
  monthly_rate_sm3_per_producing_day?: Table
}

const FIT_LABELS: Record<string, string> = {
  model: 'Model',
  qi_sm3_per_producing_day: 'Initial rate qi, Sm³ per producing day',
  di_per_year: 'Initial decline Di, per year',
  b: 'Arps b',
  window_start: 'Window start',
  window_end: 'Window end',
  n_points: 'Producing days fitted',
  r2_log: 'R² on log rate',
  residual_trend: 'Residual trend at window end',
  uptime: 'Uptime in window',
}

function Fit({ result }: { result: FitResult }) {
  const hit = useContext(HitContext)
  const fit = result.best ?? result.fit ?? {}
  return (
    <>
      <table className="data-table">
        <tbody>
          {Object.entries(FIT_LABELS).map(([k, label]) =>
            k in fit ? (
              <tr key={k}>
                <td>{label}</td>
                <td className={`num${hit(fit[k]) ? ' hit' : ''}`}>{String(fit[k])}</td>
              </tr>
            ) : null,
          )}
        </tbody>
      </table>
      {result.cumulative_oil_sm3 && (
        <p>
          Forecast cumulative oil, Sm³: low (P90) {result.cumulative_oil_sm3.p90}, mid (P50){' '}
          {result.cumulative_oil_sm3.p50}, high (P10) {result.cumulative_oil_sm3.p10}
        </p>
      )}
      {result.monthly_rate_sm3_per_producing_day && <DataTable table={result.monthly_rate_sm3_per_producing_day} />}
    </>
  )
}

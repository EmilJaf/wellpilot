import { createContext, Fragment, useContext, useEffect, useRef, useState } from 'react'
import { api, type Evidence, type Table, type ToolCode } from '../api'
import { CautionIcon, CheckIcon } from '../icons'
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
 * text rounds it: a value within 0.6% is that number. When the result holds the declared
 * value itself, only that is marked: a table has neighbours within 0.6% (the uptime 0.9626
 * of 2008-05, beside the water cut 0.9665 of 2014-10 the reader clicked), and the view
 * scrolls to the first mark. Text matches the number as written.
 */
function matcher(focus: Focus | null | undefined, result?: unknown): (v: unknown) => boolean {
  if (!focus) return () => false
  const value = Math.abs(focus.value)
  const close = (v: number, within: (scale: number) => number) =>
    value > 0 && [1, 0.01, 100].some((scale) => Math.abs(Math.abs(v) - value * scale) <= within(scale))
  const near = (v: number) => close(v, (scale) => value * scale * 0.006)
  const halfUnit = 0.5 * 10 ** -(String(value).split('.')[1]?.length ?? 0)
  const same = (v: number) => close(v, (scale) => Math.min(value * scale * 0.006, halfUnit * scale))
  const sharp = countHits(result, (v) => typeof v === 'number' && same(v)) > 0
  return (v) => {
    if (typeof v === 'number') return sharp ? same(v) : near(v)
    if (typeof v !== 'string') return false
    return focus.quote ? sameQuote(v, focus.text) : v.includes(focus.text)
  }
}

const HitContext = createContext<(v: unknown) => boolean>(() => false)

/** The field screen as the app shows it, with the clicked number marked. */
function ScreenEvidence({ result }: { result: ScreenResult }) {
  return <ScreenList result={result} hit={useContext(HitContext)} />
}

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
  const hit = matcher(focus, evidence?.result)
  const host = useRef<HTMLDivElement>(null)
  // What the view actually marked, counted once it rendered: a match in the raw result
  // that the view does not show must not be promised ("marked below" over no mark).
  const [marked, setMarked] = useState<{ key: string; count: number } | null>(null)

  // Opened from a number: show the banner that says where it came from, then its first
  // match, if the match is not already in view below it.
  useEffect(() => {
    if (!evidence || !focus) return
    setMarked({ key: `${key}/${focus.text}`, count: host.current?.querySelectorAll('.hit').length ?? 0 })
    // Stacked on a phone, the evidence sits below the answer: a step that opens a source
    // (step 3's first finding) must not pull the page down to it. Scroll only when it shows.
    const frame = host.current?.getBoundingClientRect()
    if (window.innerWidth < 1100 && frame && (frame.top > window.innerHeight || frame.bottom < 0)) return
    const banner = host.current?.querySelector('.focus-banner')
    const hitEl = host.current?.querySelector('.hit')
    banner?.scrollIntoView({ block: 'start' })
    const box = hitEl?.getBoundingClientRect()
    if (hitEl && box && (box.bottom > window.innerHeight - 24 || box.top < 0)) {
      hitEl.scrollIntoView({ block: 'center', inline: 'nearest' })
    } else hitEl?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }, [evidence, focus, key])

  if (!evidenceId) {
    if (!runId) return null
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

  const hits = focus && marked?.key === `${key}/${focus.text}` ? marked.count : 0
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
        {describeArgs(evidence.tool, evidence.arguments) && (
          <>
            <dt>Asked for</dt>
            <dd>{describeArgs(evidence.tool, evidence.arguments)}</dd>
          </>
        )}
        <dt>Code</dt>
        <dd title={`Ran at ${evidence.created_at.replace('T', ' ').replace('+00:00', ' UTC')}`}>
          <CodeRef key={evidence.code} codeRef={evidence.code} />
        </dd>
      </dl>
      {evidence.warnings.map((w, i) => (
        <div key={i} className="flag flag-caution">
          <CautionIcon />
          <span>{w}</span>
        </div>
      ))}
      <HitContext.Provider value={hit}>
        <Result tool={evidence.tool} result={evidence.result} args={evidence.arguments} focus={focus} />
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
      {/* The code ref (function and hash) is for engineers: it shows with the source. */}
      <button type="button" className="link-button" aria-expanded={open} onClick={toggle} title={codeRef}>
        {open ? 'Hide the code' : 'Show the code'}
      </button>
      {open &&
        (code?.value ? (
          <div className="tool-code">
            <p className="evidence-sub">
              <code>{codeRef}</code>
            </p>
            <p className="evidence-sub">
              {code.value.same_as_run
                ? 'The Python function that produced this result, exactly as it ran (its hash, taken with the analysis code it calls, matches).'
                : 'This tool or the analysis code it calls changed after the run; today’s version is shown.'}{' '}
              <a href={`${REPO}/${code.value.file}#L${code.value.line}`} target="_blank" rel="noreferrer">
                {code.value.file.split('/').pop()}, line {code.value.line}
              </a>
              {code.value.depends_on?.length ? (
                <>
                  {', with '}
                  {code.value.depends_on.map((f, i) => (
                    <span key={f}>
                      {i > 0 && ', '}
                      <a href={`${REPO}/${f}`} target="_blank" rel="noreferrer">
                        {f.split('/').pop()}
                      </a>
                    </span>
                  ))}
                </>
              ) : null}
            </p>
            <pre className="code">{code.value.source}</pre>
            {code.value.called?.map((c) => (
              <Fragment key={c.name}>
                <p className="evidence-sub">
                  It calls <code>{c.name}</code>, which computes the result:{' '}
                  <a href={`${REPO}/${c.file}#L${c.line}`} target="_blank" rel="noreferrer">
                    {c.file.split('/').pop()}, line {c.line}
                  </a>
                </p>
                <pre className="code">{c.source}</pre>
              </Fragment>
            ))}
          </div>
        ) : (
          <p className="evidence-sub">{code?.error ? `The code could not load: ${code.error}` : 'Loading the code'}</p>
        ))}
    </>
  )
}

const isTable = (v: unknown): v is Table =>
  typeof v === 'object' && v !== null && Array.isArray((v as Table).columns) && Array.isArray((v as Table).rows)

function Result({
  tool,
  result,
  args,
  focus,
}: {
  tool: string
  result: unknown
  args?: Record<string, unknown>
  focus?: Focus | null
}) {
  if (tool === 'run_python') return <PythonRun result={result as PythonResult} code={String(args?.code ?? '')} />
  if (tool === 'calculate') {
    const r = result as { expression?: string; value?: number; results?: { expression: string; value: number }[] }
    const rows = r.results ?? (r.expression !== undefined ? [{ expression: r.expression, value: r.value as number }] : [])
    return (
      <>
        {rows.map((row, i) => (
          <CalcResult key={i} expression={row.expression} value={row.value} />
        ))}
      </>
    )
  }
  if (tool === 'detect_events') return <Events events={result as EventRow[]} />
  if (tool === 'plan_vs_actual') return <PlanEvidence result={result as PlanResult} />
  if (tool === 'look_at_page') return <PageReading result={result as ReadingResult} />
  if (tool === 'field_screen') return <ScreenEvidence result={result as ScreenResult} />
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
  return <Tree value={result} />
}

type Scalar = string | number | boolean | null
const isScalar = (v: unknown): v is Scalar => v === null || typeof v !== 'object'

/** A list of records that share their keys and hold plain values: a table. */
function recordKeys(value: unknown[]): string[] | null {
  if (!value.length || !value.every((v) => v && typeof v === 'object' && !Array.isArray(v))) return null
  const keys = Object.keys(value[0] as object)
  const same = value.every((v) => {
    const k = Object.keys(v as object)
    return k.length === keys.length && k.every((x, i) => x === keys[i]) && Object.values(v as object).every(isScalar)
  })
  return same ? keys : null
}

/** Any result the views above do not know (well work, production logs, SQL in records),
    as nested fields and tables with the clicked number marked. A raw JSON dump had no
    marks and ran thousands of pixels wide on a phone. */
function Tree({ value, depth = 0 }: { value: unknown; depth?: number }) {
  const hit = useContext(HitContext)
  if (isScalar(value)) {
    if (value === null) return <span className="tree-none">none</span>
    return <span className={hit(value) ? 'hit' : undefined}>{String(value)}</span>
  }
  if (Array.isArray(value)) {
    if (!value.length) return <span className="tree-none">none</span>
    if (value.every(isScalar)) {
      return (
        <span>
          {value.map((v, i) => (
            <Fragment key={i}>
              {i ? ', ' : ''}
              <Tree value={v} depth={depth + 1} />
            </Fragment>
          ))}
        </span>
      )
    }
    const keys = recordKeys(value)
    const body = keys ? (
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              {keys.map((k) => (
                <th key={k} scope="col">
                  {k.replace(/_/g, ' ')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {value.map((row, i) => (
              <tr key={i}>
                {keys.map((k) => {
                  const v = (row as Record<string, Scalar>)[k]
                  return (
                    <td key={k} className={`${typeof v === 'number' ? 'num' : ''}${hit(v) ? ' hit' : ''}`}>
                      {v === null ? '' : String(v)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    ) : (
      <ol className="tree-list">
        {value.map((v, i) => (
          <li key={i}>
            <Tree value={v} depth={depth + 1} />
          </li>
        ))}
      </ol>
    )
    if (value.length <= 12 || depth === 0) return body
    return (
      <details open={countHits(value, hit) > 0}>
        <summary>{value.length} items</summary>
        {body}
      </details>
    )
  }
  if (isTable(value)) return <DataTable table={value} />
  return (
    <dl className="tree">
      {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
        <Fragment key={k}>
          <dt>{k.replace(/_/g, ' ')}</dt>
          <dd>
            <Tree value={v} depth={depth + 1} />
          </dd>
        </Fragment>
      ))}
    </dl>
  )
}

type PythonResult = {
  purpose: string
  result: unknown
  printed: string
  rows_read: number
}

/** The agent's own Python and what it returned: the code that computed the number, as it
    ran in the sandbox on the data up to the run's date. */
function PythonRun({ result, code }: { result: PythonResult; code: string }) {
  const hit = useContext(HitContext)
  const value = result.result
  const scalars =
    typeof value === 'object' && value !== null && !Array.isArray(value) && !isTable(value)
      ? Object.entries(value as Record<string, unknown>)
      : null
  return (
    <>
      <p className="evidence-sub">
        The code the agent wrote for this, run in an isolated sandbox (no network) on the field&rsquo;s data up to the
        run&rsquo;s date. It read {result.rows_read.toLocaleString('en')} rows of data.
      </p>
      <pre className="code python-code">{code}</pre>
      <h4 className="evidence-subhead">What it returned</h4>
      {isTable(value) ? (
        <DataTable table={value} />
      ) : scalars ? (
        <table className="data-table">
          <tbody>
            {scalars.map(([k, v]) => (
              <tr key={k}>
                <th scope="row">{k.replace(/_/g, ' ')}</th>
                <td className={`${typeof v === 'number' ? 'num' : ''}${hit(v) ? ' hit' : ''}`}>
                  {typeof v === 'object' && v !== null ? <Nested value={v} /> : String(v)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <pre className={`code${hit(value) ? ' hit' : ''}`}>{JSON.stringify(value, null, 2)}</pre>
      )}
      {result.printed && (
        <>
          <h4 className="evidence-subhead">What it printed</h4>
          <pre className="code">{result.printed}</pre>
        </>
      )}
    </>
  )
}

/** A list or record inside a returned value, laid out one field a line: written on one line
    it has no place to break and pushes the evidence pane thousands of pixels wide. */
function Nested({ value }: { value: object }) {
  const text = JSON.stringify(value, null, 1)
  if (!Array.isArray(value)) return <pre className="code nested-json">{text}</pre>
  return (
    <details>
      <summary>
        {value.length} {value.length === 1 ? 'row' : 'rows'}
      </summary>
      <pre className="code nested-json">{text}</pre>
    </details>
  )
}

type ReadingResult = {
  doc_id: string
  page: number
  question: string
  answer: string
  readings: { label: string; value: number; unit: string; where: string }[]
  read_by: string
  // Set against SAM 3.1's measurement of the same figure, when it measured one.
  measured_by_sam?: {
    what: string
    plateau: { sm3_per_day: number; first: string; last: string; months: number }
    colour_check?: { median_sm3_per_day: number } | null
    tolerances: { rate_sm3_per_day: number; months: number }
  }
  cross_check?: {
    label: string
    muse?: number | string | null
    sam: number | null
    sam_unit?: string
    at?: string
    difference?: number
    tolerance?: string
    agree: boolean | null
    note?: string
  }[]
}

/** What the model read off a report page image during the run, next to the page itself. */
function PageReading({ result }: { result: ReadingResult }) {
  const hit = useContext(HitContext)
  const sam = result.measured_by_sam
  const checks = result.cross_check
  return (
    <>
      <p className="evidence-sub">
        The agent showed this page to {result.read_by.replace('meta-models/', '')} and asked: “{result.question}”
      </p>
      <p className="reading-answer">{result.answer}</p>
      {sam && checks && (
        <p className="evidence-sub cross-lead">
          <strong>Two Meta models on one picture.</strong> SAM 3.1 measured the same figure at ingest: a plateau of{' '}
          {sam.plateau.sm3_per_day.toLocaleString('en')} Sm³/d for {sam.plateau.months} months
          {sam.colour_check ? `, matching a colour trace of the same pixels within ${sam.colour_check.median_sm3_per_day} Sm³/d` : ''}.
          Each reading is set against it within a quarter of a gridline, fixed before any comparison (
          {sam.tolerances.rate_sm3_per_day} Sm³/d on a rate, {sam.tolerances.months} months on a length). They agree on{' '}
          {checks.filter((c) => c.agree).length} of {checks.filter((c) => c.agree !== null).length} compared
          {(() => {
            const open = checks.filter((c) => c.agree === null && !String(c.note ?? '').includes('not the figure')).length
            return open ? `; ${open} figure reading${open > 1 ? 's' : ''} could not be compared` : ''
          })()}
          .
        </p>
      )}
      {checks ? (
        <ul className="cross-list">
          {result.readings.map((r, i) => {
            const c = checks[i]
            return (
              <li key={i} className={c?.agree === false ? 'is-off' : c?.agree ? 'is-agreed' : undefined}>
                <span className="cross-label">{r.label}</span>
                <span>
                  Muse Spark Open read{' '}
                  <strong className={hit(r.value) ? 'hit' : undefined}>
                    {typeof c?.muse === 'string'
                      ? c.muse
                      : r.value >= 1900 && r.value <= 2100 && Number.isInteger(r.value)
                        ? `${r.value} (a year)`
                        : `${r.value.toLocaleString('en')} ${r.unit.split(/[,~(=]/)[0].trim()}`}
                  </strong>
                </span>
                {c?.sam != null ? (
                  <span>
                    SAM 3.1 measured{' '}
                    <strong className={hit(c.sam) ? 'hit' : undefined}>
                      {c.sam.toLocaleString('en')} {c.sam_unit === 'Sm3/d' ? 'Sm³/d' : c.sam_unit}
                    </strong>
                    {c.at && c.at !== 'the plateau' ? ` in ${c.at}` : ''}
                  </span>
                ) : null}
                <span className="cross-verdict">
                  {c?.agree === true ? (
                    <><CheckIcon /> they agree, {c.difference?.toLocaleString('en')} apart</>
                  ) : c?.agree === false ? (
                    <><CautionIcon /> they differ by {c.difference?.toLocaleString('en')}{c.sam_unit === 'months' ? ' months' : ' Sm³/d'}</>
                  ) : (
                    c?.note ?? 'nothing SAM measured to set it against'
                  )}
                </span>
              </li>
            )
          })}
        </ul>
      ) : result.readings.length > 0 && (
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
  calibrated_band_sm3?: { p90: number; p10: number; past_forecasts: number; median_abs_error: number } | null
  calibrated_band_note?: string
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
        <dl className="forecast-bands">
          <div>
            <dt>Forecast, P50</dt>
            <dd className={hit(result.cumulative_oil_sm3.p50) ? 'hit' : undefined}>
              {result.cumulative_oil_sm3.p50.toLocaleString('en')} Sm³
            </dd>
          </div>
          {result.calibrated_band_sm3 && (
            <div className="is-honest">
              <dt>Range from past errors (P90 to P10)</dt>
              <dd>
                <span className={hit(result.calibrated_band_sm3.p90) ? 'hit' : undefined}>
                  {result.calibrated_band_sm3.p90.toLocaleString('en')}
                </span>{' '}
                to{' '}
                <span className={hit(result.calibrated_band_sm3.p10) ? 'hit' : undefined}>
                  {result.calibrated_band_sm3.p10.toLocaleString('en')}
                </span>
              </dd>
            </div>
          )}
          <div>
            <dt>Range of the fit&rsquo;s scatter (bootstrap)</dt>
            <dd>
              <span className={hit(result.cumulative_oil_sm3.p90) ? 'hit' : undefined}>
                {result.cumulative_oil_sm3.p90.toLocaleString('en')}
              </span>{' '}
              to{' '}
              <span className={hit(result.cumulative_oil_sm3.p10) ? 'hit' : undefined}>
                {result.cumulative_oil_sm3.p10.toLocaleString('en')}
              </span>
            </dd>
          </div>
        </dl>
      )}
      {result.calibrated_band_note && <p className="evidence-sub">{result.calibrated_band_note}</p>}
      {result.monthly_rate_sm3_per_producing_day && <DataTable table={result.monthly_rate_sm3_per_producing_day} />}
    </>
  )
}

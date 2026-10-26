import { useEffect, useState } from 'react'
import { api, type Evidence, type Table } from '../api'
import { CautionIcon } from '../icons'
import { describeArgs, toolName } from '../labels'

type Props = { runId: string | null; evidenceId: string | null }

/** One piece of evidence: what ran, on which arguments, and exactly what it returned. */
export function EvidenceView({ runId, evidenceId }: Props) {
  const [evidence, setEvidence] = useState<Evidence | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!runId || !evidenceId) return
    let current = true
    api
      .evidence(runId, evidenceId)
      .then((e) => current && (setEvidence(e), setError(null)))
      .catch((e: Error) => current && setError(e.message))
    return () => {
      current = false
    }
  }, [runId, evidenceId])

  if (!evidenceId) {
    return (
      <div className="panel-body evidence">
        <p className="empty">
          Select a step in the agent log or an evidence tag in the answer to see exactly what the
          tool returned and which code produced it.
        </p>
      </div>
    )
  }
  if (error) return <div className="panel-body evidence"><p className="empty">Evidence {evidenceId}: {error}</p></div>
  if (!evidence || evidence.id !== evidenceId) return <div className="panel-body evidence"><p className="empty">Loading {evidenceId}</p></div>

  return (
    <div className="panel-body evidence">
      <h3 className="evidence-title">
        {evidence.id}: {toolName(evidence.tool)}
      </h3>
      <dl className="evidence-meta">
        <dt>Asked for</dt>
        <dd>{describeArgs(evidence.tool, evidence.arguments) || 'no arguments'}</dd>
        <dt>Code</dt>
        <dd>
          <code>{evidence.code}</code>
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
      <Result tool={evidence.tool} result={evidence.result} />
    </div>
  )
}

const isTable = (v: unknown): v is Table =>
  typeof v === 'object' && v !== null && Array.isArray((v as Table).columns) && Array.isArray((v as Table).rows)

function Result({ tool, result }: { tool: string; result: unknown }) {
  if (tool === 'calculate') {
    const r = result as { expression: string; value: number }
    return (
      <p className="calc">
        <code>{r.expression}</code> = {r.value}
      </p>
    )
  }
  if (tool === 'detect_events') return <Events events={result as EventRow[]} />
  if (tool === 'search_reports') return <Facts result={result as SearchResult} />
  if (tool === 'read_report_page') return <ReportPage page={result as PageResult} />
  if (tool === 'fit_decline' || tool === 'forecast_production') return <Fit result={result as FitResult} />
  if (isTable(result)) return <DataTable table={result} />
  return <pre className="code">{JSON.stringify(result, null, 2)}</pre>
}

function DataTable({ table, maxRows = 200 }: { table: Table; maxRows?: number }) {
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
                <td key={j} className={typeof v === 'number' ? 'num' : ''}>
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

type EventRow = { kind: string; signal: string; start: string; end: string | null; summary: string }

const EVENT_KINDS: Record<string, string> = {
  gauge_failure: 'Gauge failure',
  regime_change: 'Regime change',
  threshold_crossing: 'Threshold crossed',
  shut_in: 'Shut-in',
}

function Events({ events }: { events: EventRow[] }) {
  return (
    <ul className="event-list">
      {events.map((e, i) => (
        <li key={i}>
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

function Facts({ result }: { result: SearchResult }) {
  const [open, setOpen] = useState<number | null>(null) // row index of the fact showing its page
  const table = result.facts
  if (!table) return result.pages ? <DataTable table={result.pages} /> : null
  const at = (row: Table['rows'][number], name: string) => row[table.columns.indexOf(name)]
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
            <li key={i}>
              <span className="fact-source">
                Page {page}, {String(at(row, 'source'))}{' '}
                <span className={`check-badge${check === 'unverified' ? ' check-badge-unverified' : ''}`}>
                  {check === 'exact' ? 'quote found on the page' : check === 'tokens' ? 'all words found on the page' : 'quote not found on the page'}
                </span>
              </span>
              <div>{String(at(row, 'statement_en'))}</div>
              <p className="fact-quote" lang="no">{String(at(row, 'quote'))}</p>
              <button type="button" className="link-button" onClick={() => setOpen(open === i ? null : i)}>
                {open === i ? 'Hide page' : `Show page ${page}`}
              </button>
              {open === i && <img className="page-image" src={api.pageImageUrl(doc, page)} alt={`${doc} page ${page}`} />}
            </li>
          )
        })}
      </ul>
    </>
  )
}

type PageResult = { document: { title: string }; page: number; text: string; image: string; summary_en?: string; facts?: Table }

function ReportPage({ page }: { page: PageResult }) {
  const doc = page.image.split('/')[1]
  return (
    <>
      {page.summary_en && <p>{page.summary_en}</p>}
      <img className="page-image" src={api.pageImageUrl(doc, page.page)} alt={`${page.document.title}, page ${page.page}`} />
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
  const fit = result.best ?? result.fit ?? {}
  return (
    <>
      <table className="data-table">
        <tbody>
          {Object.entries(FIT_LABELS).map(([k, label]) =>
            k in fit ? (
              <tr key={k}>
                <td>{label}</td>
                <td className="num">{String(fit[k])}</td>
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

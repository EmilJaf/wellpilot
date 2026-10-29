import { useEffect, useState } from 'react'
import { api, column, type Backtest, type BacktestCase } from '../api'
import { CheckIcon } from '../icons'
import { SHORT, signed, volume } from '../labels'
import { useWidth } from '../useWidth'
import { ForecastFilm } from './ForecastFilm'

const METHODS: { key: string; label: string }[] = [
  { key: 'agent', label: 'WellPilot agent, its own window' },
  { key: 'after_change', label: 'Rule, no model: fit after the last change or well job' },
  { key: 'recent_3m', label: 'Decline fit, last 3 months' },
  { key: 'recent_6m', label: 'Decline fit, last 6 months' },
  { key: 'recent_12m', label: 'Decline fit, last 12 months' },
  { key: 'recent_24m', label: 'Decline fit, last 24 months' },
  { key: 'classic', label: 'Decline fit, whole history (textbook default)' },
]

type Case = { key: string; well: string; cutoff: string; actual: number; error: Record<string, number> }

/** One row per forecast (well, cutoff) with each method's error, in the backtest's order. */
function toCases(backtest: Backtest): Case[] {
  const t = backtest.results
  const [well, cutoff, method, actual, error] = ['well', 'cutoff', 'method', 'actual', 'error'].map((c) => column(t, c))
  const cases = new Map<string, Case>()
  for (const row of t.rows) {
    const key = `${row[well]}|${row[cutoff]}`
    const found = cases.get(key) ?? {
      key,
      well: String(row[well]),
      cutoff: String(row[cutoff]),
      actual: Number(row[actual]),
      error: {},
    }
    found.error[String(row[method])] = Number(row[error])
    cases.set(key, found)
  }
  return [...cases.values()]
}

/**
 * The backtest, told for someone who has never fitted a decline curve: forecasts made at
 * past dates against what the wells then produced, all of them, then one case up close.
 */
export function AccuracyView() {
  const [data, setData] = useState<{ backtest: Backtest | null; error?: string } | null>(null)
  const [picked, setPicked] = useState<string | null>(null)

  useEffect(() => {
    api
      .backtest()
      .then((backtest) => setData({ backtest }))
      .catch((e: Error) => setData({ backtest: null, error: e.message }))
  }, [])

  if (!data) return <p className="empty">Loading the backtest</p>
  if (!data.backtest) return <p className="empty">The backtest could not load: {data.error}</p>

  const { summary } = data.backtest
  const cases = toCases(data.backtest)
  const compared = cases.filter((c) => 'agent' in c.error && 'classic' in c.error)
  // Opens on the case where the two differ most; every case is on the strip to compare.
  const widest = [...compared].sort(
    (a, b) => Math.abs(b.error.classic) - Math.abs(b.error.agent) - (Math.abs(a.error.classic) - Math.abs(a.error.agent)),
  )[0]
  const selected = cases.find((c) => c.key === picked) ?? widest ?? cases[0]
  // The film runs over wells with enough forecasts to tell a story; the biggest producer first.
  const counts = new Map<string, number>()
  for (const c of compared) counts.set(c.well, (counts.get(c.well) ?? 0) + 1)
  const filmWells = [...counts].filter(([, n]) => n >= 3).sort((a, b) => b[1] - a[1]).map(([w]) => w)
  const methods = METHODS.filter((m) => summary.all_cases[m.key])
  const worst = Math.max(...methods.map((m) => summary.all_cases[m.key].median_abs_error))
  const closer = summary.agent_closer
  const best = summary.best_fixed_window
  // How the agent compares with the hindsight bar, in words, from the numbers themselves.
  const agentStats = summary.all_cases.agent
  const bestStats = best ? summary.all_cases[best] : undefined
  const versusBest =
    agentStats && bestStats
      ? [
          agentStats.mean_abs_error < bestStats.mean_abs_error ? 'a lower mean error' : 'a higher mean error',
          agentStats.misses_over_50pct !== undefined && bestStats.misses_over_50pct !== undefined
            ? agentStats.misses_over_50pct < bestStats.misses_over_50pct
              ? 'fewer large misses'
              : agentStats.misses_over_50pct > bestStats.misses_over_50pct
                ? 'more large misses'
                : 'as many large misses'
            : null,
        ]
          .filter(Boolean)
          .join(' and ')
      : ''

  return (
    <div className="accuracy">
      <h3 className="accuracy-title">How close were the forecasts?</h3>
      <p className="accuracy-lead">
        For each producing well, every 6 months: the data is cut at that day, the next 12 months of oil are
        forecast, and the forecast is compared with what the well actually produced. The tools never return data
        after the cut. A decline fit needs a stretch of history to fit on. The quick standard is the whole
        history; a fixed recent window does better, but which length is best shows only afterwards. The agent
        looks at each well and chooses its own window.
      </p>

      <p className="accuracy-label">
        Median error over {summary.all_cases.classic?.cases ?? cases.length} forecasts, then the mean and the forecasts
        off by more than half
      </p>
      <ul className="error-bars">
        {methods.map((m) => {
          const s = summary.all_cases[m.key]
          return (
            <li key={m.key} className={m.key === 'agent' ? 'is-agent' : undefined}>
              <span className="error-bar-label">
                {m.label}
                {m.key === best && <em className="error-bar-note">best fixed window, known only afterwards</em>}
              </span>
              <span className="error-bar-track">
                <span className="error-bar-area">
                  <span className="error-bar" style={{ width: `${(s.median_abs_error / worst) * 100}%` }} />
                </span>
                <span className="error-bar-value">{Math.round(s.median_abs_error * 100)}%</span>
                <span className="error-bar-more">
                  mean {Math.round(s.mean_abs_error * 100)}%
                  {s.misses_over_50pct !== undefined && `, ${s.misses_over_50pct} over 50%`}
                </span>
              </span>
            </li>
          )
        })}
      </ul>
      {closer?.classic && (
        <p className="accuracy-closer">
          {best && closer[best] && (
            <>
              Against the best fixed window, which nobody could have chosen in advance, the agent was closer in{' '}
              <strong>
                {closer[best].closer} of {closer[best].cases}
              </strong>{' '}
              forecasts{versusBest && `, with ${versusBest}`}.{' '}
            </>
          )}
          Against the textbook default, a fit on the whole history, it was closer in {closer.classic.closer} of{' '}
          {closer.classic.cases} and cut the median error from{' '}
          {Math.round((summary.all_cases.classic?.median_abs_error ?? 0) * 100)}% to{' '}
          {Math.round((agentStats?.median_abs_error ?? 0) * 100)}%.
          {closer.after_change && summary.all_cases.after_change && (
            <>
              {' '}
              A rule that does what the agent is told, with no model (fit after the last change point or the last well
              job), had a median error of {Math.round(summary.all_cases.after_change.median_abs_error * 100)}%; the
              agent was closer in {closer.after_change.closer} of {closer.after_change.cases}, with a mean error of{' '}
              {Math.round((agentStats?.mean_abs_error ?? 0) * 100)}% against the rule&rsquo;s{' '}
              {Math.round(summary.all_cases.after_change.mean_abs_error * 100)}%: about even case by case, with smaller
              misses where the rule picked a bad window.
            </>
          )}{' '}
          All forecasts are below, misses included; they come from three wells of one field, so read them as a check,
          not a benchmark.
        </p>
      )}
      {agentStats?.bootstrap_band_coverage !== undefined && (
        <p className="chart-note-line">
          Uncertainty: a range from the fit&rsquo;s scatter held the actual volume in{' '}
          {Math.round(agentStats.bootstrap_band_coverage * 100)}% of the agent&rsquo;s forecasts, far from the 80% it
          claims.
          {agentStats.hindcast_band_coverage !== null && agentStats.hindcast_band_coverage !== undefined && (
            <>
              {' '}
              The range each forecast now reports is built from the errors of forecasts made the same way from earlier
              dates; it held the actual in {Math.round(agentStats.hindcast_band_coverage * 100)}% (
              {agentStats.hindcast_band_cases} forecasts with enough history to calibrate on).
            </>
          )}
        </p>
      )}
      {summary.agent_models && (
        <p className="chart-note-line">
          The agent’s runs:{' '}
          {Object.entries(summary.agent_models)
            .map(([model, n]) =>
              model === 'not recorded' ? `${n} from before the model was recorded` : `${n} on ${model.replace('meta-models/', '')}`,
            )
            .join(', ')}
          .
        </p>
      )}

      {filmWells.length > 0 && <ForecastFilm wells={filmWells} initial={filmWells[0]} />}

      <ErrorStrip cases={cases} selected={selected?.key ?? null} onSelect={setPicked} />
      {selected && <CaseView key={selected.key} entry={selected} />}
    </div>
  )
}

function caseLine(c: Case) {
  const parts = ['agent', 'classic', 'recent_12m'].filter((m) => m in c.error).map((m) => `${SHORT[m]} ${signed(c.error[m])}`)
  return `${c.well}, forecast made ${c.cutoff}: actual ${volume(c.actual)} in the next 12 months; ${parts.join(', ')}`
}

/** Every forecast as a column: the agent's error (filled) and the classic fit's (hollow). */
function ErrorStrip({ cases, selected, onSelect }: { cases: Case[]; selected: string | null; onSelect: (key: string) => void }) {
  const [host, width] = useWidth(520)
  const [hover, setHover] = useState<number | null>(null)
  const height = 190
  const pad = { left: 44, right: 8, top: 10, bottom: 38 }
  const plotW = width - pad.left - pad.right
  const plotH = height - pad.top - pad.bottom
  const errors = cases.flatMap((c) => [c.error.agent, c.error.classic]).filter((e) => e !== undefined)
  const lo = Math.min(-0.5, ...errors) - 0.05
  const hi = Math.max(0.5, ...errors) + 0.05
  const yAt = (e: number) => pad.top + plotH * ((hi - e) / (hi - lo))

  // Columns grouped by well, with a gap between wells.
  const gap = 0.8
  const groupOf = cases.map((_, i) => cases.slice(1, i + 1).filter((c, j) => c.well !== cases[j].well).length)
  const step = plotW / (cases.length + (groupOf[groupOf.length - 1] ?? 0) * gap)
  const xs = cases.map((_, i) => pad.left + step * (i + groupOf[i] * gap + 0.5))
  const wells = [...new Set(cases.map((c) => c.well))].map((w) => {
    const at = cases.map((c, i) => (c.well === w ? xs[i] : null)).filter((x): x is number => x !== null)
    return { well: w, from: at[0], to: at[at.length - 1] }
  })
  const ticks = [-0.5, 0, 0.5, 1].filter((v) => v > lo && v < hi)
  // Each well's years, labelled under the middle of their forecasts.
  const yearLabels = [...new Set(cases.map((c) => `${c.well}|${c.cutoff.slice(0, 4)}`))].map((key) => {
    const at = cases.map((c, i) => (`${c.well}|${c.cutoff.slice(0, 4)}` === key ? xs[i] : null)).filter((x): x is number => x !== null)
    return { key, year: key.slice(-2), x: at.reduce((a, b) => a + b, 0) / at.length }
  }).reduce<{ key: string; year: string; x: number }[]>(
    (kept, label) => (kept.length && label.x - kept[kept.length - 1].x < 24 ? kept : [...kept, label]),
    [],
  )
  const shown = hover !== null ? cases[hover] : cases.find((c) => c.key === selected)

  return (
    <div className="error-strip" ref={host}>
      <p className="readout">{shown ? caseLine(shown) : 'Hover a forecast to read it; click to see it up close.'}</p>
      <svg width={width} height={height} role="img" aria-label="Forecast error of every backtest case, agent against the whole-history fit">
        {ticks.map((v) => (
          <g key={v}>
            <line className={v === 0 ? 'strip-zero' : 'plan-grid'} x1={pad.left} x2={width - pad.right} y1={yAt(v)} y2={yAt(v)} />
            <text className="plan-axis" x={pad.left - 6} y={yAt(v) + 4} textAnchor="end">
              {v === 0 ? 'exact' : signed(v)}
            </text>
          </g>
        ))}
        {cases.map((c, i) =>
          c.key === selected ? (
            <rect key="sel" className="strip-selected" x={xs[i] - step / 2} y={pad.top} width={step} height={plotH} />
          ) : null,
        )}
        {cases.map((c, i) => {
          const { agent, classic } = c.error
          return (
            <g key={c.key}>
              {agent !== undefined && classic !== undefined && (
                <line className="strip-link" x1={xs[i]} x2={xs[i]} y1={yAt(classic)} y2={yAt(agent)} />
              )}
              {classic !== undefined && <circle className="strip-classic" cx={xs[i]} cy={yAt(classic)} r={4} />}
              {agent !== undefined && <circle className="strip-agent" cx={xs[i]} cy={yAt(agent)} r={4} />}
            </g>
          )
        })}
        {yearLabels.map((y) => (
          <text key={y.key} className="plan-axis" x={y.x} y={height - 22} textAnchor="middle">
            {`'${y.year}`}
          </text>
        ))}
        {wells.map((w) => (
          <g key={w.well}>
            <line className="strip-group" x1={w.from - step / 3} x2={w.to + step / 3} y1={height - 14} y2={height - 14} />
            <text className="strip-well" x={(w.from + w.to) / 2} y={height - 2} textAnchor="middle">
              {w.well}
            </text>
          </g>
        ))}
        {cases.map((c, i) => (
          <rect
            key={c.key}
            className="strip-hit"
            x={xs[i] - step / 2}
            y={pad.top}
            width={step}
            height={plotH}
            role="button"
            tabIndex={0}
            aria-label={caseLine(c)}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
            onClick={() => onSelect(c.key)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onSelect(c.key)
              }
            }}
          >
            <title>{caseLine(c)}</title>
          </rect>
        ))}
      </svg>
      <p className="chart-key">
        <span><i className="strip-key strip-key-agent" /> agent</span>
        <span><i className="strip-key strip-key-classic" /> whole-history decline fit</span>
        <span>above the line: forecast too high; below: too low</span>
      </p>
    </div>
  )
}

/** One forecast up close: the rate history, the cut, each method's window and forecast. */
function CaseView({ entry }: { entry: Case }) {
  const [detail, setDetail] = useState<{ value: BacktestCase | null; error?: string } | null>(null)

  useEffect(() => {
    api
      .backtestCase(entry.well, entry.cutoff)
      .then((value) => setDetail({ value }))
      .catch((e: Error) => setDetail({ value: null, error: e.message }))
  }, [entry.well, entry.cutoff])

  if (!detail) return <p className="empty">Loading {entry.well} from {entry.cutoff}</p>
  if (!detail.value) return <p className="empty">This forecast could not load: {detail.error}</p>
  const d = detail.value
  const agent = d.methods.find((m) => m.method === 'agent')

  return (
    <div className="case">
      <h4 className="case-title">
        {d.well}, forecast made on {d.cutoff}
      </h4>
      <CaseChart detail={d} />
      {agent && d.agent && (
        <div className="case-why">
          <p className="case-why-head">
            {d.agent.window_asked?.[0]
              ? `The agent chose the window from ${d.agent.window_asked[0]}${d.agent.window_asked[1] ? ` to ${d.agent.window_asked[1]}` : ''}, `
              : 'The agent fitted the whole history, '}
            producing days {agent.window_start} to {agent.window_end}, with a {agent.model} decline. In its own words:
          </p>
          <blockquote>{d.agent.summary}</blockquote>
          {d.agent.inferences.map((claim) => (
            <p key={claim} className="case-inference">
              <span className="pill">Inference</span> {claim}
            </p>
          ))}
          {d.agent.verified && (
            <p className="case-verified">
              <CheckIcon /> Its answer passed the checker.
            </p>
          )}
        </div>
      )}
      <table className="case-table">
        <thead>
          <tr>
            <th scope="col">Method</th>
            <th scope="col">Fit window</th>
            <th scope="col">Model</th>
            <th scope="col" className="num-col">Forecast</th>
            <th scope="col" className="num-col">Error</th>
          </tr>
        </thead>
        <tbody>
          {d.methods.map((m) => (
            <tr key={m.method}>
              <th scope="row">{SHORT[m.method]}</th>
              <td>
                {m.window_start} to {m.window_end}
              </td>
              <td>{m.model}</td>
              <td className="num-col">{volume(m.p50_sm3)}</td>
              <td className="num-col">{signed(m.error)}</td>
            </tr>
          ))}
          <tr className="case-actual">
            <th scope="row">actual</th>
            <td>
              {d.cutoff} to {d.horizon_end}
            </td>
            <td />
            <td className="num-col">{volume(d.actual_sm3)}</td>
            <td />
          </tr>
        </tbody>
      </table>
      {d.agent && (
        <details className="case-question">
          <summary>What the agent was asked</summary>
          <p>{d.agent.question}</p>
        </details>
      )}
    </div>
  )
}

const DAY = 86_400_000
const time = (iso: string) => Date.parse(`${iso}T00:00:00Z`)

/** Fixed axes for a film: x in ms since the epoch, y in log10 of the rate. */
export type Domain = { x: [number, number]; y: [number, number] }

/** Oil rate per producing day on a log scale, with the cut and the two forecasts. With a
    domain the axes are fixed (the film); without it they fit this case. */
export function CaseChart({ detail, domain, height = 240 }: { detail: BacktestCase; domain?: Domain; height?: number }) {
  const [host, width] = useWidth(520)
  const [hover, setHover] = useState<number | null>(null)
  const pad = { left: 44, right: 92, top: 18, bottom: 50 }
  const plotW = width - pad.left - pad.right
  const plotH = height - pad.top - pad.bottom
  const shown = detail.methods.filter((m) => m.method === 'agent' || m.method === 'classic')
  const t0 = domain?.x[0] ?? Math.min(time(detail.rate.date[0]), ...shown.map((m) => time(m.window_start)))
  const t1 = domain?.x[1] ?? time(detail.horizon_end) + DAY
  const cut = time(detail.cutoff) + DAY
  const values = [...detail.rate.rate, ...shown.flatMap((m) => [...m.forecast.rate, ...m.fitted.rate])].filter((v) => v > 0)
  const lo = domain?.y[0] ?? Math.log10(Math.min(...values) * 0.85)
  const hi = domain?.y[1] ?? Math.log10(Math.max(...values) * 1.15)
  const horizon = time(detail.horizon_end) + DAY
  const xAt = (t: number) => pad.left + (plotW * (t - t0)) / (t1 - t0)
  const yAt = (v: number) => pad.top + plotH * ((hi - Math.log10(Math.max(v, 1e-9))) / (hi - lo))
  const path = (dates: string[], rates: number[]) => dates.map((d, i) => `${xAt(time(d))},${yAt(rates[i])}`).join(' ')

  const ticks = [1, 2, 5]
    .flatMap((m) => [10, 100, 1000, 10000].map((p) => m * p))
    .filter((v) => Math.log10(v) > lo && Math.log10(v) < hi)
  const years: number[] = []
  for (let y = new Date(t0).getUTCFullYear() + 1; Date.UTC(y, 0, 1) < t1; y++) years.push(y)

  const rateDates = detail.rate.date
  const split = rateDates.findIndex((d) => time(d) >= cut)
  const before = split < 0 ? rateDates.length : split
  // A film's rate series runs on past the forecast year; what comes after is drawn faint.
  const later = rateDates.findIndex((d) => time(d) > horizon)
  const afterEnd = later < 0 ? rateDates.length : later
  // Direct labels at the end of each forecast, nudged apart when they would touch.
  const ends = shown
    .map((m) => ({ m, y: yAt(m.forecast.rate[m.forecast.rate.length - 1]) }))
    .sort((a, b) => a.y - b.y)
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 15)

  const at = (dates: string[], rates: number[], t: number) => {
    let best = -1
    for (let i = 0; i < dates.length; i++) if (Math.abs(time(dates[i]) - t) <= 4 * DAY) best = i
    return best >= 0 ? rates[best] : null
  }
  const hoverT = hover !== null ? time(rateDates[hover]) : null
  const readout = hoverT !== null && hover !== null && (
    <>
      <strong>week of {rateDates[hover]}</strong>
      <span>actual {Math.round(detail.rate.rate[hover]).toLocaleString('en')}</span>
      {shown.map((m) => {
        const v = hoverT > cut ? at(m.forecast.date, m.forecast.rate, hoverT) : at(m.fitted.date, m.fitted.rate, hoverT)
        return v !== null ? (
          <span key={m.method}>
            {SHORT[m.method]} {Math.round(v).toLocaleString('en')}
          </span>
        ) : null
      })}
      <span>Sm³ per producing day</span>
    </>
  )

  return (
    <div className="case-chart" ref={host}>
      <p className="readout">{readout || 'Oil rate per producing day, log scale. Hover to read the weeks.'}</p>
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`${detail.well} oil rate with the forecasts made on ${detail.cutoff}`}
        onMouseMove={(e) => {
          const box = e.currentTarget.getBoundingClientRect()
          const t = t0 + ((e.clientX - box.left - pad.left) / plotW) * (t1 - t0)
          let best = 0
          for (let i = 1; i < rateDates.length; i++) {
            if (Math.abs(time(rateDates[i]) - t) < Math.abs(time(rateDates[best]) - t)) best = i
          }
          setHover(t >= t0 && t <= t1 ? best : null)
        }}
        onMouseLeave={() => setHover(null)}
      >
        <rect className="case-hidden" x={xAt(cut)} y={pad.top} width={xAt(horizon) - xAt(cut)} height={plotH} />
        {ticks.map((v) => (
          <g key={v}>
            <line className="plan-grid" x1={pad.left} x2={pad.left + plotW} y1={yAt(v)} y2={yAt(v)} />
            <text className="plan-axis" x={pad.left - 6} y={yAt(v) + 4} textAnchor="end">
              {v >= 1000 ? `${v / 1000}k` : v}
            </text>
          </g>
        ))}
        {years.map((y) => (
          <text key={y} className="plan-axis" x={xAt(Date.UTC(y, 0, 1))} y={pad.top + plotH + 14} textAnchor="middle">
            {y}
          </text>
        ))}
        <polyline className="case-actual-line is-past" points={path(rateDates.slice(0, before), detail.rate.rate.slice(0, before))} />
        <polyline
          className="case-actual-line"
          points={path(rateDates.slice(Math.max(0, before - 1), afterEnd), detail.rate.rate.slice(Math.max(0, before - 1), afterEnd))}
        />
        {afterEnd < rateDates.length && (
          <polyline className="case-actual-line is-later" points={path(rateDates.slice(afterEnd - 1), detail.rate.rate.slice(afterEnd - 1))} />
        )}
        {shown.map((m) => (
          <g key={m.method} className={`case-method case-${m.method}`}>
            <polyline className="case-fitted" points={path(m.fitted.date, m.fitted.rate)} />
            <polyline className="case-forecast" points={path(m.forecast.date, m.forecast.rate)} />
          </g>
        ))}
        <line className="case-cut" x1={xAt(cut)} x2={xAt(cut)} y1={pad.top - 6} y2={pad.top + plotH} />
        <text className="plan-note" x={xAt(cut) + 5} y={pad.top + 6}>
          hidden from the forecasts
        </text>
        {ends.map(({ m, y }) => (
          <text key={m.method} className={`case-end case-end-${m.method}`} x={pad.left + plotW + 6} y={y + 4}>
            {SHORT[m.method]} {signed(m.error)}
          </text>
        ))}
        {/* each method's fit window, under the axis */}
        {shown.map((m, i) => {
          const y = pad.top + plotH + 26 + i * 13
          return (
            <g key={m.method} className={`case-window case-${m.method}`}>
              <line x1={xAt(time(m.window_start))} x2={xAt(time(m.window_end) + DAY)} y1={y} y2={y} />
              <text x={xAt(time(m.window_end) + DAY) + 6} y={y + 4}>
                {SHORT[m.method]} window
              </text>
            </g>
          )
        })}
        {hoverT !== null && <line className="case-hover" x1={xAt(hoverT)} x2={xAt(hoverT)} y1={pad.top} y2={pad.top + plotH} />}
      </svg>
      <p className="chart-key">
        <span><i className="case-key case-key-actual" /> actual oil, from the data</span>
        <span><i className="case-key case-key-agent" /> agent's fit and forecast</span>
        <span><i className="case-key case-key-classic" /> whole-history fit and forecast</span>
        <span><i className="key-band case-key-hidden" /> the 12 months forecast</span>
      </p>
    </div>
  )
}

import { useEffect, useState } from 'react'
import { api, type Hindsight, type ReasoningCheck } from '../api'
import { CautionIcon, CheckIcon } from '../icons'
import { BARRELS_NOTE, barrels, signed, volume } from '../labels'
import { CaseChart } from './AccuracyView'

/** What the model's reasoning holds from the hidden period, in words that promise no more
    than was checked: what it wrote, not what it may remember. */
function ReasoningLine({ r, asOf }: { r: ReasoningCheck; asOf: string }) {
  if (!r.available) return <p className="reasoning-line">{r.note}.</p>
  const found = r.later_dates.length + r.hidden_values.length
  if (!found)
    return (
      <p className="reasoning-line">
        <CheckIcon /> The model&rsquo;s reasoning ({r.steps} steps, {r.characters.toLocaleString('en')} characters) holds no
        date or value from after {asOf}
        {r.forecast_mentions > 0 &&
          `; ${r.forecast_mentions} ${r.forecast_mentions === 1 ? 'mention' : 'mentions'} of the year it forecasts ${r.forecast_mentions === 1 ? 'is' : 'are'} not counted`}
        . That is what it wrote, not what it may remember; the README shows what it remembers of this field.
      </p>
    )
  return (
    <div className="flag flag-caution reasoning-found">
      <CautionIcon />
      <div>
        <strong>
          The model&rsquo;s reasoning mentions {found === 1 ? 'something' : `${found} things`} from after {asOf}:
        </strong>
        <ul>
          {r.later_dates.map((d, i) => (
            <li key={`d${i}`}>
              step {d.step}, {d.dates.join(', ')}: &ldquo;{d.sentence}&rdquo;
            </li>
          ))}
          {r.hidden_values.map((v, i) => (
            <li key={`v${i}`}>
              step {v.step}, {v.number.toLocaleString('en')} ({v.matches}): &ldquo;{v.sentence}&rdquo;
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

/**
 * The time machine's reveal: the agent answered with data up to a past date; this shows the
 * data it could not see. Each forecast against the oil the well then produced (and against
 * the classic fit made the same day), and every well a year later.
 */
export function HindsightView({ runId }: { runId: string }) {
  const [loaded, setLoaded] = useState<{ id: string; value?: Hindsight; error?: string } | null>(null)

  useEffect(() => {
    let current = true
    api
      .hindsight(runId)
      .then((value) => current && setLoaded({ id: runId, value }))
      .catch((e: Error) => current && setLoaded({ id: runId, error: e.message }))
    return () => {
      current = false
    }
  }, [runId])

  if (loaded?.id !== runId) return <p className="empty field-view">Loading what happened next</p>
  if (!loaded.value) return <p className="empty field-view">What happened next could not load: {loaded.error}</p>
  const h = loaded.value
  // The forecasts the answer gives; the others the agent tried on the way fold away. An
  // answer that cites none shows them all.
  const cited = h.forecasts.filter((f) => f.cited)
  const answered = cited.length > 0 ? cited : h.forecasts
  const tried = cited.length > 0 ? h.forecasts.filter((f) => !f.cited) : []

  return (
    <div className="hindsight field-view">
      <h3 className="accuracy-title">What happened after {h.as_of}</h3>
      <p className="accuracy-lead">
        The agent answered with the data up to <strong>{h.as_of}</strong>. Below is the data it could not see,
        set against what it said.
      </p>
      {h.reasoning && <ReasoningLine r={h.reasoning} asOf={h.as_of} />}
      {h.forecasts.length === 0 && !h.work_next?.length && (
        <p className="empty">The agent made no forecast in this run.</p>
      )}
      {answered.map((f) => (
        <ForecastCase key={f.evidence_id} f={f} />
      ))}
      {tried.length > 0 && (
        <details className="hindsight-tried">
          <summary>
            {tried.length === 1 ? 'One more forecast' : `${tried.length} more forecasts`} the agent made on the way
          </summary>
          <p className="chart-note-line">
            The agent tried {tried.length === 1 ? 'this window' : 'these windows'} too; its answer does not use{' '}
            {tried.length === 1 ? 'it' : 'them'}.
          </p>
          {tried.map((f) => (
            <ForecastCase key={f.evidence_id} f={f} />
          ))}
        </details>
      )}

      {h.work_next?.map((w) => (
        <WorkNext key={`${w.well}-${w.start}`} w={w} asOf={h.as_of} />
      ))}

      <h4 className="case-title hindsight-wells-title">Every producing well, a year later</h4>
      <table className="case-table hindsight-wells">
        <thead>
          <tr>
            <th scope="col">Well</th>
            <th scope="col" className="num-col">Oil, year before</th>
            <th scope="col" className="num-col">Oil, year after</th>
            <th scope="col" className="num-col">Water cut then, a year on</th>
            <th scope="col">Downhole gauge</th>
          </tr>
        </thead>
        <tbody>
          {h.wells.map((w) => (
            <tr key={w.well}>
              <th scope="row">{w.well}</th>
              <td className="num-col">{volume(w.oil_year_before_sm3)}</td>
              <td className="num-col">
                {volume(w.oil_year_after_sm3)}
                {w.oil_year_before_sm3 > 0 && (
                  <span className="hindsight-change"> {signed(w.oil_year_after_sm3 / w.oil_year_before_sm3 - 1)}</span>
                )}
              </td>
              <td className="num-col">
                {cut(w.water_cut_at_cut)} → {cut(w.water_cut_year_later)}
              </td>
              <td>{w.gauge_dead_at_cut ? (w.gauge_back_on ? `dead, back on ${w.gauge_back_on}` : 'dead, never came back') : 'working'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="chart-note-line">
        Year before and after: the 365 days on each side of {h.as_of}. Water cut: the median over the flowing days of
        the 30 days before each date.
      </p>
    </div>
  )
}

/** The work done on a well the answer named, soon after the date: what it was, the well
 * before and after it, and the year after it against the decline it was on. */
function WorkNext({ w, asOf }: { w: NonNullable<Hindsight['work_next']>[number]; asOf: string }) {
  const millions = (usd: number) => `$${(usd / 1e6).toFixed(usd >= 1e7 ? 0 : 1)} million`
  return (
    <section className="work-next">
      <h4 className="case-title">
        What was done to {w.well} next: {w.start === w.end ? w.start : `${w.start} to ${w.end}`}
      </h4>
      <p className="work-next-what">
        {w.plug_m !== null ? `A plug set at ${w.plug_m.toLocaleString('en-US')} m` : `A ${w.kind} job`}
        {w.perforations.length > 0 && `, perforations at ${w.perforations.join(', ')}`}. The daily report says: &ldquo;
        {w.summary}&rdquo;
      </p>
      <table className="case-table work-next-table">
        <thead>
          <tr>
            <th scope="col" />
            <th scope="col" className="num-col">Oil per producing day</th>
            <th scope="col" className="num-col">Water cut</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">60 days before</th>
            <td className="num-col">{w.before.oil_sm3_per_day.toLocaleString('en-US')} Sm³/d</td>
            <td className="num-col">{cut(w.before.water_cut)}</td>
          </tr>
          {w.after && (
            <tr>
              <th scope="row">Back on stream from {w.back_on}</th>
              <td className="num-col">{w.after.oil_sm3_per_day.toLocaleString('en-US')} Sm³/d</td>
              <td className="num-col">{cut(w.after.water_cut)}</td>
            </tr>
          )}
        </tbody>
      </table>
      {w.extra_oil_sm3 !== null && w.trend_year_after_sm3 !== null && (
        <p className="work-next-worth">
          In the year after it the well made <strong>{volume(w.oil_year_after_sm3)}</strong> of oil; on the decline it
          was on before the job ({w.trend_fit?.model} fit from {w.trend_fit?.from}) it would have made about{' '}
          {volume(w.trend_year_after_sm3)}. The difference, about <strong>{volume(w.extra_oil_sm3)}</strong> (
          {((w.extra_oil_bbl ?? 0) / 1e6).toFixed(2)} million barrels), is worth about{' '}
          <strong>{millions(w.extra_value_usd ?? 0)}</strong> at an assumed ${w.price_usd_per_bbl} a barrel.
        </p>
      )}
      <p className="chart-note-line">
        The agent answered as of {asOf} and could not see this job. Set it beside the answer: did the agent point at
        the depth where the water came in?
      </p>
    </section>
  )
}

function ForecastCase({ f }: { f: Hindsight['forecasts'][number] }) {
  const [agent, classic] = f.methods
  const band = f.calibrated_band
  return (
    <section className="case hindsight-forecast" aria-label={`${f.well} forecast against the outcome`}>
      <h4 className="case-title">{f.well}: forecast against the outcome</h4>
      <p className="hindsight-verdict">
        The agent forecast <strong>{volume(agent.p50_sm3)}</strong>
        <span className="bbl" title={BARRELS_NOTE}> (about {barrels(agent.p50_sm3)} barrels)</span> of oil for{' '}
        {f.forecast_from} to {f.horizon_end}
        {f.data_ends && ` (the data ends on ${f.data_ends})`}. The well produced <strong>{volume(f.actual_sm3)}</strong>
        <span className="bbl" title={BARRELS_NOTE}> ({barrels(f.actual_sm3)} barrels)</span>:
        the forecast was <strong>{signed(agent.error)}</strong>.
      </p>
      <CaseChart detail={f} />
      {f.fixed_windows && f.fixed_windows.length > 0 && <ErrorBars f={f} />}
      {f.decomposition && <Decomposition d={f.decomposition} />}
      {f.fixed_windows && f.fixed_windows.length > 0 && (
        <p className="hindsight-fixed">
          {f.window_from_change ? (
            <>
              <strong>It chose its own window:</strong> it fitted from {f.window_from_change.date},{' '}
              {f.window_from_change.kind === 'work'
                ? `after the work on the well it had found (${f.window_from_change.text}, ${f.window_from_change.evidence_id})`
                : `where it had found ${f.well}’s rate shifting (${f.window_from_change.evidence_id})`}
              , not a fixed length of history.{' '}
            </>
          ) : (
            <>It fitted from {agent.window_start}, a window it chose itself. </>
          )}
          Fits made the same day on fixed windows said{' '}
          {f.fixed_windows
            .filter((w) => w.error !== null)
            .map((w) => `${signed(w.error!)} (last ${w.months} months)`)
            .join(', ')}
          : one of them came closer, but which one was known only afterwards. The textbook fit on the whole history, which
          averages in the years before the change, said {signed(classic.error)}.
        </p>
      )}
      {!f.fixed_windows?.length && (
        <p className="hindsight-fixed">
          The decline fit over the whole history, made the same day, said {volume(classic.p50_sm3)} (
          {signed(classic.error)}).
        </p>
      )}
      {f.p50_by_model && Object.keys(f.p50_by_model).length > 1 && (
        <p className="hindsight-models">
          On the same window the three Arps models said{' '}
          {Object.entries(f.p50_by_model)
            .map(([model, p50]) => `${volume(p50)} (${model}, ${signed(p50 / f.actual_sm3 - 1)})`)
            .join(', ')}
          : the choice of model is part of the uncertainty.
        </p>
      )}
      {band && (
        <p className={`hindsight-band${band.holds_actual ? '' : ' is-miss'}`}>
          {band.holds_actual && <CheckIcon />}
          Its range from {band.past_forecasts ? `${band.past_forecasts} ` : ''}past forecast errors
          {band.coverage ? ` (a range that holds about ${Math.round(band.coverage * 100)}% of outcomes)` : ''},{' '}
          {volume(band.p90)} to {volume(band.p10)},{' '}
          {band.holds_actual ? 'held the outcome' : 'missed the outcome'}.
        </p>
      )}
    </section>
  )
}

/** Every forecast made the same day, as its error against the outcome: the agent's own
 * window, each fixed window and the whole history, so the textbook fit's miss is one bar of
 * several, not the only comparison. */
function ErrorBars({ f }: { f: Hindsight['forecasts'][number] }) {
  const [agent, classic] = f.methods
  const rows = [
    { label: 'Agent, its own window', error: agent.error, agent: true },
    ...(f.fixed_windows ?? [])
      .filter((w) => w.error !== null)
      .map((w) => ({ label: `Last ${w.months} months`, error: w.error!, agent: false })),
    { label: 'Whole history (textbook)', error: classic.error, agent: false },
  ]
  const reach = Math.max(0.25, ...rows.map((r) => Math.min(Math.abs(r.error), 1.2)))
  const at = (e: number) => 50 + (Math.max(-reach, Math.min(reach, e)) / reach) * 50
  return (
    <figure className="error-bars-case" aria-label="Forecast error of each method made the same day">
      <figcaption>Each forecast made on {f.cutoff}, against what the well then produced</figcaption>
      <ul>
        {rows.map((r) => (
          <li key={r.label} className={r.agent ? 'is-agent' : undefined}>
            <span className="error-bars-label">{r.label}</span>
            <span className="error-bars-track">
              <span className="error-bars-zero" />
              <span
                className="error-bars-bar"
                style={{ left: `${Math.min(50, at(r.error))}%`, width: `${Math.abs(at(r.error) - 50)}%` }}
              />
            </span>
            <span className="error-bars-value">{signed(r.error)}</span>
          </li>
        ))}
      </ul>
    </figure>
  )
}

/** A volume is a rate per producing day times the time on stream; say which was off. */
function Decomposition({ d }: { d: NonNullable<Hindsight['forecasts'][number]['decomposition']> }) {
  const offset = Math.sign(d.rate_error) !== Math.sign(d.uptime_error) && Math.abs(d.rate_error) > 0.03
  return (
    <p className="hindsight-fixed">
      {offset ? 'Two errors that partly cancel: ' : 'In parts: '}
      the rate per producing day was {signed(d.rate_error)} ({d.rate_forecast.toLocaleString('en-US')} forecast,{' '}
      {d.rate_actual.toLocaleString('en-US')} Sm³/d actual) and the time on stream {signed(d.uptime_error)} (the fit
      window&rsquo;s {Math.round(d.uptime_forecast * 100)}% against {Math.round(d.uptime_actual * 100)}% that year).
    </p>
  )
}

const cut = (v: number | null) => (v === null ? '–' : v.toFixed(2))

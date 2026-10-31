import { useEffect, useState } from 'react'
import { api, type Hindsight, type ReasoningCheck } from '../api'
import { CautionIcon, CheckIcon } from '../icons'
import { BARRELS_NOTE, barrels, signed, volume } from '../labels'
import { CaseChart } from './AccuracyView'

/** What the model's reasoning holds from the hidden period, in words that promise no more
    than was checked: what it wrote, not what it may remember. */
/** A date kept on one line: a phone breaks "2011-12-31" at a hyphen otherwise. */
const Day = ({ d }: { d: string }) => <span className="nobr">{d}</span>

function ReasoningLine({ r, asOf }: { r: ReasoningCheck; asOf: string }) {
  if (!r.available) return <p className="reasoning-line">{r.note}.</p>
  const found = r.later_dates.length + r.hidden_values.length
  if (!found)
    return (
      <p className="reasoning-line">
        <CheckIcon /> The model&rsquo;s reasoning ({r.steps} steps, {r.characters.toLocaleString('en')} characters) holds no
        date or value from after <Day d={asOf} />
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
          The model&rsquo;s reasoning mentions {found === 1 ? 'something' : `${found} things`} from after <Day d={asOf} />:
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
export function HindsightView({ runId, onReady }: { runId: string; onReady?: () => void }) {
  const [loaded, setLoaded] = useState<{ id: string; value?: Hindsight; error?: string } | null>(null)
  const ready = loaded?.id === runId
  useEffect(() => {
    if (ready) onReady?.()
  }, [ready, onReady])

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
      {h.forecasts.length === 0 && !h.work_next?.length && (
        <p className="empty">The agent made no forecast in this run.</p>
      )}
      {h.work_next?.map((w) => (
        <WorkNext key={`${w.well}-${w.start}`} w={w} asOf={h.as_of} />
      ))}
      {answered.map((f) =>
        h.work_next?.some((w) => w.well === f.well) ? (
          <details key={f.evidence_id} className="hindsight-tried">
            <summary>
              The forecast the answer gave for {f.well}, made for the well as it was: off by {signed(f.methods[0].error)}, as
              no forecast could see the job
            </summary>
            <ForecastCase f={f} overtaken />
          </details>
        ) : (
          <ForecastCase key={f.evidence_id} f={f} />
        ),
      )}
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

      {h.reasoning && <ReasoningLine r={h.reasoning} asOf={h.as_of} />}
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

/** The work done on a well the answer named, soon after the date: what it was, set against
 * what the answer recommended, the well before and after it, and the year after it against
 * the decline it was on (a range over three fit windows: no one baseline is right). */
function WorkNext({ w, asOf }: { w: NonNullable<Hindsight['work_next']>[number]; asOf: string }) {
  const millions = (usd: number) => `$${Math.round(usd / 1e6)} million`
  const pick = w.depth?.answer_plug_m[0]
  const range = w.extra_oil_range_sm3
  const value = w.extra_value_range_usd
  // The answer's depth at or below a plug already in the hole (on the last log's terms).
  const isolatedBy = w.on_latest_log?.depths.some((d) => d.depth_m === pick && d.already_isolated)
    ? w.on_latest_log.plugs_set_since?.[0]?.depth_m
    : undefined
  return (
    <section className="work-next">
      <h4 className="case-title">
        What was done to {w.well} next: <Day d={w.start} />{w.start === w.end ? null : <> to <Day d={w.end} /></>}
      </h4>
      {w.plug_m !== null && (
        <p className="work-next-verdict">
          The operator set a plug at <strong>{w.plug_m.toLocaleString('en-US')} m</strong>
          {w.depth && w.depth.blank_pipe_m.some(([a, b]) => a <= w.plug_m! && w.plug_m! <= b) && ', in blank pipe'}
          {newPerforations(w.depth)}.{' '}
          {pick !== undefined && (
            <>
              The answer&rsquo;s plug depth was about <strong>{pick.toLocaleString('en-US')} m</strong>
              {isolatedBy !== undefined
                ? `, below the plug already set at ${isolatedBy.toLocaleString('en-US')} m: a plug there would shut off nothing new.`
                : '.'}
            </>
          )}
        </p>
      )}
      {w.on_latest_log && <OnLatestLog log={w.on_latest_log} plug={w.plug_m} after={w.after} backOn={w.back_on} />}
      {w.depth && <DepthStrip d={w.depth} plug={w.plug_m} />}
      <p className="work-next-what">
        The daily report says: &ldquo;{w.summary}&rdquo;
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
            <th scope="row">The 60 days before</th>
            <td className="num-col">{w.before.oil_sm3_per_day.toLocaleString('en-US')} Sm³/d</td>
            <td className="num-col">{cut(w.before.water_cut)}</td>
          </tr>
          {w.after && (
            <tr>
              <th scope="row">
                The 60 days from {w.back_on}, back on stream ({w.after.days} producing days)
              </th>
              <td className="num-col">{w.after.oil_sm3_per_day.toLocaleString('en-US')} Sm³/d</td>
              <td className="num-col">{cut(w.after.water_cut)}</td>
            </tr>
          )}
        </tbody>
      </table>
      {w.field_before && w.field_after && (
        <p className="work-next-field">
          The whole field on the same days: water {w.field_before.water_sm3_per_day.toLocaleString('en-US')} →{' '}
          {w.field_after.water_sm3_per_day.toLocaleString('en-US')} Sm³/d, oil{' '}
          {w.field_before.oil_sm3_per_day.toLocaleString('en-US')} →{' '}
          {w.field_after.oil_sm3_per_day.toLocaleString('en-US')} Sm³/d.
        </p>
      )}
      {range && w.trend_year_after_sm3 !== null && (
        <p className="work-next-worth">
          In the year after it the well made <strong>{volume(w.oil_year_after_sm3)}</strong> of oil. On the decline it
          was on before the job, fitted on the 12, 6 and 3 months before it, it would have made about{' '}
          {volume(Math.min(...(w.trend_windows ?? []).map((t) => t.trend_sm3)))} to{' '}
          {volume(Math.max(...(w.trend_windows ?? []).map((t) => t.trend_sm3)))}: the job added about{' '}
          <strong>
            {volume(range[0])} to {volume(range[1])}
          </strong>{' '}
          of oil that year.
          {w.oil_rest_of_life_sm3 !== undefined &&
            ` From the job to the end of its life the well made ${volume(w.oil_rest_of_life_sm3)} in all (a total, not the gain).`}
          {value && (
            <>
              {' '}
              At an assumed ${w.price_usd_per_bbl} a barrel that is {millions(value[0])} to {millions(value[1])} of gross
              revenue, before the job's cost and taxes.
            </>
          )}
        </p>
      )}
      <p className="chart-note-line">
        {w.trend_fit?.r2_log !== undefined && `The decline fits are poor (r² ${w.trend_fit.r2_log} at best): the range is wide for that reason. `}
        {w.volumes_note} The agent answered as of <Day d={asOf} /> and could not see this job.
      </p>
    </section>
  )
}

/** ", and opened new perforations at A-B m, just above the top zone (C-D m)". */
function newPerforations(d: NonNullable<Hindsight['work_next']>[number]['depth']) {
  if (!d?.new_perforations_m.length) return ''
  const spans = d.new_perforations_m.map(([a, b]) => `${a}-${b} m`).join(', ')
  const top = d.perforated_m.find(([a]) => d.new_perforations_m.some(([, b]) => a - b >= 0 && a - b < 2))
  return `, and opened new perforations at ${spans}${top ? `, just above the top zone (${top[0]}-${top[1]} m)` : ' above it'}`
}

/** Each depth against the last log before the job: the share of the water and of the oil
 * entering below it, which is what a plug there shuts off. */
function OnLatestLog({
  log,
  plug,
  after,
  backOn,
}: {
  log: NonNullable<NonNullable<Hindsight['work_next']>[number]['on_latest_log']>
  plug: number | null
  after: NonNullable<Hindsight['work_next']>[number]['after']
  backOn: string
}) {
  const pct = (r?: [number, number]) => {
    if (!r) return null
    const [a, b] = r.map((v) => Math.round(v * 100))
    return a === b ? `${a}%` : `${a}-${b}%`
  }
  // A plug set after the log already shut off what entered below it: what a new plug cuts is
  // then a share of the flow left above that one, and nothing at or below it.
  const old = log.plugs_set_since?.[0]
  // What the log said a plug at the operator's depth would leave flowing, against what the
  // well then made: the log's prediction, checked.
  const operator = log.depths.find((d) => d.depth_m === plug && !d.already_isolated)
  return (
    <>
    <table className="case-table work-next-log">
      <caption>
        On the last log before the job ({log.survey}, <Day d={log.date} />), a plug shuts off what enters below it
        {old && (
          <>
            ; the plug set at {old.depth_m.toLocaleString('en-US')} m on <Day d={old.date} /> already shut off what entered
            below that, so the shares are of the flow left above it
          </>
        )}
      </caption>
      <thead>
        <tr>
          <th scope="col">A plug at</th>
          <th scope="col" className="num-col">Water it cuts</th>
          <th scope="col" className="num-col">Oil it cuts</th>
          <th scope="col" className="num-col">Water cut it leaves</th>
        </tr>
      </thead>
      <tbody>
        {log.depths.map((d) => {
          const shares = d.of_flow_left ?? d
          return (
            <tr key={d.depth_m} className={d.already_isolated ? 'is-isolated' : ''}>
              <th scope="row">
                {d.depth_m.toLocaleString('en-US')} m, {d.depth_m === plug ? 'the operator’s' : 'the answer’s'}
              </th>
              {d.already_isolated && old ? (
                <td className="num-col" colSpan={3}>
                  nothing new: already shut off by the {old.depth_m.toLocaleString('en-US')} m plug
                </td>
              ) : (
                <>
                  <td className="num-col">{pct(shares.water_share) ?? 'not logged'}</td>
                  <td className="num-col">{pct(shares.oil_share) ?? 'not logged'}</td>
                  <td className="num-col">{pct(d.water_cut_left_above) ?? 'not logged'}</td>
                </>
              )}
            </tr>
          )
        })}
      </tbody>
    </table>
    {operator?.water_cut_left_above && after && (
      <p className="work-next-field">
        So the {log.date.slice(0, 4)} log said a plug at {operator.depth_m.toLocaleString('en-US')} m would leave a flow of
        about {pct(operator.water_cut_left_above)} water; in the 60 days from {backOn} the well made{' '}
        {Math.round(after.water_cut * 100)}% water. The passes ran below the well&rsquo;s own rate, so the log is a
        guide, not a measurement of the well as it flowed.
      </p>
    )}
    </>
  )
}

/** The well along its depth: perforated zones, blank pipe, the earlier plug, where the answer
 * pointed and where the operator set the plug, and the new perforations. */
function DepthStrip({
  d,
  plug,
}: {
  d: NonNullable<NonNullable<Hindsight['work_next']>[number]['depth']>
  plug: number | null
}) {
  const marks = [...d.perforated_m.flat(), ...d.new_perforations_m.flat(), ...d.earlier_plugs_m, ...d.answer_plug_m, plug ?? NaN].filter(
    (v) => Number.isFinite(v),
  )
  if (!marks.length) return null
  const top = Math.floor((Math.min(...marks) - 10) / 10) * 10
  const bottom = Math.ceil((Math.max(...marks) + 10) / 10) * 10
  const width = 640
  const x = (m: number) => 70 + ((m - top) / (bottom - top)) * (width - 90)
  const line = (m: number, label: string, cls: string, row: number) => (
    <g key={`${cls}-${m}`} className={cls}>
      <line x1={x(m)} x2={x(m)} y1={18} y2={52} />
      <text x={x(m)} y={row} textAnchor="middle">
        {label}
      </text>
    </g>
  )
  return (
    <figure className="depth-strip">
      <svg viewBox={`0 0 ${width} 92`} role="img" aria-label="The well along its depth around the plug">
        <text x={0} y={40} className="depth-strip-axis">
          depth, m
        </text>
        <rect x={x(top)} y={30} width={x(bottom) - x(top)} height={10} className="depth-pipe" />
        {d.perforated_m.map(([a, b]) => (
          <rect key={`z${a}`} x={x(a)} y={26} width={Math.max(2, x(b) - x(a))} height={18} className="depth-zone" />
        ))}
        {d.new_perforations_m.map(([a, b]) => (
          <rect key={`n${a}`} x={x(a)} y={26} width={Math.max(3, x(b) - x(a))} height={18} className="depth-new" />
        ))}
        {d.earlier_plugs_m.map((m) => line(m, `2010 plug ${m}`, 'depth-old', 64))}
        {d.answer_plug_m.slice(0, 1).map((m) => line(m, `agent ${m}`, 'depth-agent', 12))}
        {plug !== null && line(plug, `operator ${plug}`, 'depth-plug', 64)}
        {[top, bottom].map((m) => (
          <text key={m} x={x(m)} y={86} textAnchor="middle" className="depth-strip-axis">
            {m}
          </text>
        ))}
      </svg>
      <figcaption>
        Grey: perforated zones; dark: perforations opened in the job; the gaps are blank pipe, where a plug can be set.
      </figcaption>
    </figure>
  )
}

function ForecastCase({ f, overtaken = false }: { f: Hindsight['forecasts'][number]; overtaken?: boolean }) {
  const [agent, classic] = f.methods
  const band = f.calibrated_band
  return (
    <section className="case hindsight-forecast" aria-label={`${f.well} forecast against the outcome`}>
      <h4 className="case-title">{f.well}: forecast against the outcome</h4>
      {/* The turn of the story, in three numbers before any prose; not for a forecast a job
          on the well overtook. */}
      <dl className={`verdict-figures${overtaken ? ' is-quiet' : ''}`}>
        <div>
          <dt>The agent forecast</dt>
          <dd>{volume(agent.p50_sm3)}</dd>
        </div>
        <div>
          <dt>The well produced</dt>
          <dd>{volume(f.actual_sm3)}</dd>
        </div>
        <div className={Math.abs(agent.error) <= 0.15 ? 'is-close' : ''}>
          <dt>Off by</dt>
          <dd>{signed(agent.error)}</dd>
        </div>
      </dl>
      <p className="hindsight-verdict">
        Oil from <Day d={f.forecast_from} /> to <Day d={f.horizon_end} />
        {f.data_ends && <> (the data ends on <Day d={f.data_ends} />)</>}
        <span className="bbl" title={BARRELS_NOTE}>
          : about {barrels(agent.p50_sm3)} barrels forecast, {barrels(f.actual_sm3)} produced
        </span>
        .
      </p>
      <CaseChart detail={f} height={320} />
      {f.fixed_windows && f.fixed_windows.length > 0 && <ErrorBars f={f} />}
      {/* What made the difference, in one sentence; the rest folds for engineers. */}
      <p className="hindsight-fixed">
        {f.window_from_change ? (
          <>
            <strong>It chose its own window:</strong> it fitted from <Day d={f.window_from_change.date} />,{' '}
            {f.window_from_change.kind === 'work'
              ? `after the work on the well it had found (${f.window_from_change.text})`
              : `where it had found ${f.well}’s rate shifting`}
            .{' '}
          </>
        ) : agent.window_start === classic.window_start ? (
          <>It fitted the whole history, as the textbook does. </>
        ) : (
          <>It fitted from <Day d={agent.window_start} />, a window it chose itself. </>
        )}
        The textbook fit on the whole history said {signed(classic.error)}
        {f.fixed_windows && f.fixed_windows.length > 0 && (
          <>
            ; fixed recent windows said {fixedRange(f.fixed_windows)}, and which of them was best showed only afterwards
          </>
        )}
        .
      </p>
      {(f.decomposition || (f.p50_by_model && Object.keys(f.p50_by_model).length > 1) || band) && (
        <details className="hindsight-more">
          <summary>More on this forecast</summary>
          {f.decomposition && <Decomposition d={f.decomposition} />}
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
              {volume(band.p90)} to {volume(band.p10)}, {band.holds_actual ? 'held the outcome' : 'missed the outcome'}.
            </p>
          )}
          <p className="chart-note-line">Its source: {f.evidence_id}{f.window_from_change ? `; the change it fitted from: ${f.window_from_change.evidence_id}` : ''}.</p>
        </details>
      )}
    </section>
  )
}

/** The fixed windows' errors as a range, "+2% to +24%". */
function fixedRange(windows: NonNullable<Hindsight['forecasts'][number]['fixed_windows']>): string {
  const errors = windows.map((w) => w.error).filter((e): e is number => e !== null)
  if (!errors.length) return 'nothing'
  const low = Math.min(...errors)
  const high = Math.max(...errors)
  return low === high ? signed(low) : `${signed(low)} to ${signed(high)}`
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
      <figcaption>Each forecast made on <Day d={f.cutoff} />, against what the well then produced</figcaption>
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

// The API rounds to three places: toFixed alone would turn 0.965 into 0.96.
const cut = (v: number | null) => (v === null ? '–' : (Math.round(v * 100 + 1e-9) / 100).toFixed(2))

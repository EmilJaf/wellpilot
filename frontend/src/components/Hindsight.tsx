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
      {h.forecasts.length === 0 && <p className="empty">The agent made no forecast in this run.</p>}
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
      {f.fixed_windows && f.fixed_windows.length > 0 && (
        <p className="hindsight-fixed">
          Decline fits made the same day on fixed windows said{' '}
          {f.fixed_windows
            .filter((w) => w.error !== null)
            .map((w) => `${signed(w.error!)} (last ${w.months} months)`)
            .join(', ')}
          , and on the whole history {signed(classic.error)}. Which window was right is known only afterwards; the agent
          chose its own, from {agent.window_start}.
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
          Its range from past forecast errors, {volume(band.p90)} to {volume(band.p10)},{' '}
          {band.holds_actual ? 'held the outcome' : 'missed the outcome'}.
        </p>
      )}
      <CaseChart detail={f} />
    </section>
  )
}

const cut = (v: number | null) => (v === null ? '–' : v.toFixed(2))

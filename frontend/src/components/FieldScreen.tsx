import { useEffect, useState } from 'react'
import { AlarmIcon, CautionIcon, CheckIcon } from '../icons'

type Issue = { level: 'alarm' | 'caution' | 'note'; kind: string; since: string | null; text: string }

type ScreenWell = {
  well: string
  role: 'producer' | 'injector'
  first_oil: string | null
  cum_oil_sm3: number
  share_of_field_oil: number
  oil_rate_sm3_per_day?: number | null
  water_cut?: number | null
  uptime?: number | null
  oil_rate_change_90d?: number | null
  water_inj_sm3_per_day?: number | null
  attention: number
  issues: Issue[]
}

/** The field_screen tool's result (also served at /api/field/screen). */
export type ScreenResult = { as_of: string; window_days: number; field_oil_sm3: number; wells: ScreenWell[] }

const pct = (v: number | null | undefined) => (v === null || v === undefined ? '–' : `${Math.round(v * 100)}%`)
const signed = (v: number | null | undefined) =>
  v === null || v === undefined ? '–' : `${v >= 0 ? '+' : '−'}${Math.abs(Math.round(v * 100))}%`
const rate = (v: number | null | undefined) => (v === null || v === undefined ? '–' : Math.round(v).toLocaleString('en'))

function Level({ issues }: { issues: Issue[] }) {
  if (issues.some((i) => i.level === 'alarm'))
    return (
      <span className="level level-alarm">
        <AlarmIcon /> Act now
      </span>
    )
  if (issues.some((i) => i.level === 'caution'))
    return (
      <span className="level level-caution">
        <CautionIcon /> Watch
      </span>
    )
  if (issues.length)
    return <span className="level level-note">Stopped</span>
  return (
    <span className="level">
      <CheckIcon /> No issue
    </span>
  )
}

/**
 * Every well at once, ranked by what needs attention: the view an engineer looking after a
 * field starts from. Deterministic detectors, no model; the date box above sets the day.
 */
export function FieldScreen({ asOf, onOpenWell }: { asOf: string; onOpenWell: (well: string) => void }) {
  const [loaded, setLoaded] = useState<{ key: string; result?: ScreenResult; error?: string } | null>(null)
  const key = asOf || 'end'

  useEffect(() => {
    let current = true
    fetch(`/api/field/screen${asOf ? `?as_of=${asOf}` : ''}`)
      .then((r) => (r.ok ? (r.json() as Promise<ScreenResult>) : Promise.reject(new Error(r.statusText))))
      .then((result) => current && setLoaded({ key, result }))
      .catch((e: Error) => current && setLoaded({ key, error: e.message }))
    return () => {
      current = false
    }
  }, [asOf, key])

  const result = loaded?.key === key ? loaded.result : undefined
  if (loaded?.key === key && loaded.error) return <p className="empty field-view">The field screen could not load: {loaded.error}</p>
  if (!result) return <p className="empty field-view">Screening every well</p>
  const flagged = result.wells.filter((w) => w.issues.some((i) => i.level !== 'note')).length

  return (
    <div className="screen">
      <h3 className="accuracy-title">Which wells need a look</h3>
      <p className="accuracy-lead">
        Every well as of <strong>{result.as_of}</strong>
        {asOf ? ' (the date in the box above)' : ', the last day the field produced; set a date above to screen another day'}.
        The detectors that the agent uses run over each one: dead gauges, water cut against the platform&rsquo;s design
        limit, steep declines, shut-ins, steps in rate. {flagged} of {result.wells.length} wells have something to look at.
      </p>
      <ScreenList result={result} onOpenWell={onOpenWell} />
    </div>
  )
}

/** The ranked wells; also how a field_screen result shows as evidence. */
export function ScreenList({ result, onOpenWell }: { result: ScreenResult; onOpenWell?: (well: string) => void }) {
  return (
    <>
      <ol className="screen-list">
        {result.wells.map((w) => (
          <li key={w.well} className={`screen-row attention-${w.attention}`}>
            <div className="screen-head">
              <Level issues={w.issues} />
              {onOpenWell ? (
                <button type="button" className="screen-well" onClick={() => onOpenWell(w.well)} title={`Open ${w.well}'s charts`}>
                  {w.well}
                </button>
              ) : (
                <strong className="screen-well">{w.well}</strong>
              )}
              <span className="screen-role">
                {w.role}
                {w.role === 'producer' && `, ${pct(w.share_of_field_oil)} of the field's oil`}
              </span>
            </div>
            {w.role === 'producer' ? (
              <dl className="screen-metrics">
                <div>
                  <dt>Oil, Sm³/d</dt>
                  <dd>{rate(w.oil_rate_sm3_per_day)}</dd>
                </div>
                <div>
                  <dt>Water cut</dt>
                  <dd>{w.water_cut === null || w.water_cut === undefined ? '–' : w.water_cut.toFixed(2)}</dd>
                </div>
                <div>
                  <dt>Uptime</dt>
                  <dd>{pct(w.uptime)}</dd>
                </div>
                <div>
                  <dt>Oil, 90 days</dt>
                  <dd>{signed(w.oil_rate_change_90d)}</dd>
                </div>
              </dl>
            ) : (
              <dl className="screen-metrics">
                <div>
                  <dt>Injection, Sm³/d</dt>
                  <dd>{rate(w.water_inj_sm3_per_day)}</dd>
                </div>
              </dl>
            )}
            {w.issues.length > 0 && (
              <ul className="screen-issues">
                {w.issues.map((i, n) => (
                  <li key={n} className={`issue issue-${i.level}`}>
                    {i.level === 'alarm' ? <AlarmIcon /> : i.level === 'caution' ? <CautionIcon /> : <span className="issue-dot" />}
                    <span>
                      {i.text}
                      {i.since && <span className="issue-since"> since {i.since}</span>}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ol>
      <p className="chart-note-line">
        Rates and water cut are medians over the last {result.window_days} days on stream; the 90-day change compares
        them with the {result.window_days} days before.{onOpenWell && ' Click a well to open its charts.'}
      </p>
    </>
  )
}

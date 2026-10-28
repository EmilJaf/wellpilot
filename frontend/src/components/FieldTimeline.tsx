import { useEffect, useState } from 'react'
import { api, type Timeline } from '../api'
import { useWidth } from '../useWidth'

const ISSUE_NAMES: Record<string, string> = {
  dead_gauge: 'dead downhole gauge',
  water_cut: 'water cut past the design limit',
  decline: 'steep decline',
  shut_in: 'long shut-ins',
  step_down: 'step down in rate',
  shift_down: 'shift to a lower rate',
  stopped: 'stopped',
}

const time = (iso: string) => Date.parse(iso.length === 7 ? `${iso}-15T00:00:00Z` : `${iso}T00:00:00Z`)

/**
 * The field's history as the screen would have told it: one row per producer, its monthly
 * oil rate, the issues the screen raised at each month end with only the data up to that day,
 * and the work done on the well from its daily reports. Hover to read; click a row to open
 * the well.
 */
/**
 * The history as far as `end` (the time machine before its reveal): wells that had started,
 * their months, the issues flagged and the work begun by then. An issue cleared later reads
 * as open; its text, written at a later month end, is left out.
 */
function upTo(t: Timeline, end: string): Timeline {
  const month = end.slice(0, 7)
  return {
    ...t,
    wells: t.wells
      .filter((w) => w.first_oil <= end)
      .map((w) => {
        const keep = w.oil_rate.month.map((m) => m <= month)
        const rate = w.oil_rate.rate.filter((_, i) => keep[i])
        const days = w.oil_rate.month.filter((_, i) => keep[i])
        return {
          ...w,
          // The life total would give the later years away; the rates are per producing day,
          // so no total is rebuilt from them.
          oil_sm3: -1,
          oil_rate: { month: days, rate },
          issues: w.issues
            .filter((i) => i.first_flagged <= end)
            .map((i) => (i.cleared && i.cleared <= end ? i : { ...i, cleared: null, text: '' })),
          work: w.work.filter((j) => j.start <= end).map((j) => (j.end <= end ? j : { ...j, end })),
        }
      }),
  }
}

export function FieldTimeline({
  asOf,
  end,
  onOpenWell,
}: {
  asOf: string | null
  end?: string | null
  onOpenWell: (well: string) => void
}) {
  const [loaded, setLoaded] = useState<{ value?: Timeline; error?: string } | null>(null)

  useEffect(() => {
    api
      .timeline()
      .then((value) => setLoaded({ value }))
      .catch((e: Error) => setLoaded({ error: e.message }))
  }, [])

  if (!loaded) return <p className="empty field-view">Loading the field&rsquo;s history</p>
  if (!loaded.value) return <p className="empty field-view">The history could not load: {loaded.error}</p>
  // The plot mounts with the data, so its width is measured on an element that exists.
  return <TimelinePlot t={end ? upTo(loaded.value, end) : loaded.value} asOf={asOf} onOpenWell={onOpenWell} />
}

function TimelinePlot({ t, asOf, onOpenWell }: { t: Timeline; asOf: string | null; onOpenWell: (well: string) => void }) {
  const [host, width] = useWidth(560)
  const [hover, setHover] = useState<string | null>(null)

  const pad = { left: 62, right: 10, top: 22, bottom: 24 }
  const row = 70
  const height = pad.top + t.wells.length * row + pad.bottom
  const t0 = Date.UTC(2008, 0, 1)
  const t1 = Date.UTC(2016, 11, 31)
  const plotW = width - pad.left - pad.right
  const xAt = (ms: number) => pad.left + (plotW * (ms - t0)) / (t1 - t0)
  // One log scale for every well, so their rates compare at a glance.
  const lo = Math.log10(50)
  const hi = Math.log10(10000)
  const sparkH = 34
  const yAt = (rate: number, top: number) =>
    top + sparkH * (1 - (Math.log10(Math.min(Math.max(rate, 50), 10000)) - lo) / (hi - lo))

  const years: number[] = []
  for (let y = 2008; y <= 2016; y++) years.push(y)

  const readouts: Record<string, string> = {}
  t.wells.forEach((w) => {
    w.issues.forEach((i, n) => {
      readouts[`${w.well}|i${n}`] =
        `${w.well}, ${ISSUE_NAMES[i.kind] ?? i.kind} (${i.level}): first flagged at the ${i.first_flagged} month end` +
        `${i.since ? `, dated from ${i.since}` : ''}; ${i.cleared ? `cleared by ${i.cleared}` : 'still open at the end'}. ${i.text}`
    })
    w.work.forEach((j, n) => {
      const before = monthRate(w, j.start, -1)
      const after = firstRateAfter(w, j.end)
      readouts[`${w.well}|w${n}`] =
        `${w.well}, work on the well ${j.start} to ${j.end}` +
        `${j.perforations.length ? `, perforated ${j.perforations.join(', ')}` : ''}` +
        `${before && after ? `; oil ${Math.round(before)} Sm³/d the month before, ${Math.round(after)} the first month back` : ''}. ` +
        `Report: ${keySentences(j.summary)}`
    })
  })

  return (
    <div className="timeline field-view">
      <h3 className="accuracy-title">The field&rsquo;s history, as the screen would have told it</h3>
      <p className="accuracy-lead">
        The well screen run at every month end from {t.months[0]} to {t.months[1]}, each time with only the data up
        to that day ({t.replayed} replays): when each issue first showed, and the work done on the wells from their
        daily reports. Hover to read; click a well to open it.
      </p>
      <p className="readout timeline-readout">{hover ? readouts[hover] : 'Hover an issue or a job.'}</p>
      {/* Measured without the panel's padding, so the plot never runs past its edge. */}
      <div ref={host}>
      <svg width={width} height={height} role="img" aria-label="Each well's oil rate, issues and work on one time axis">
        {years.map((y) => (
          <g key={y}>
            <line className="plan-grid" x1={xAt(Date.UTC(y, 0, 1))} x2={xAt(Date.UTC(y, 0, 1))} y1={pad.top - 6} y2={height - pad.bottom} />
            <text className="plan-axis" x={xAt(Date.UTC(y, 6, 1))} y={height - 8} textAnchor="middle">
              {y}
            </text>
          </g>
        ))}
        {t.wells.map((w, r) => {
          const top = pad.top + r * row
          const points = w.oil_rate.month
            .map((m, i) => (w.oil_rate.rate[i] > 0 ? `${xAt(time(m))},${yAt(w.oil_rate.rate[i], top)}` : null))
            .filter(Boolean)
            .join(' ')
          const alarms = w.issues.map((i, n) => ({ ...i, n })).filter((i) => i.level !== 'note')
          return (
            <g key={w.well} className="timeline-row" onClick={() => onOpenWell(w.well)}>
              <rect className="timeline-hit" x={0} y={top - 4} width={width} height={row - 4} />
              <text className="timeline-well" x={4} y={top + 14}>
                {w.well}
              </text>
              {w.oil_sm3 >= 0 && (
                <text className="timeline-sub" x={4} y={top + 29}>
                  {(w.oil_sm3 / 1e6).toFixed(2)} MSm³
                </text>
              )}
              <polyline className="timeline-oil" points={points} />
              {alarms.map((i) => {
                const x0 = xAt(time(i.since && i.since < i.first_flagged ? i.first_flagged : i.first_flagged))
                const x1 = xAt(i.cleared ? time(i.cleared) : t1)
                const lane = i.level === 'alarm' ? 0 : 1
                return (
                  <rect
                    key={`i${i.n}`}
                    className={`timeline-issue is-${i.level}`}
                    x={x0}
                    y={top + sparkH + 6 + lane * 8}
                    width={Math.max(x1 - x0, 3)}
                    height={6}
                    onMouseEnter={() => setHover(`${w.well}|i${i.n}`)}
                    onMouseLeave={() => setHover(null)}
                  >
                    <title>{readouts[`${w.well}|i${i.n}`]}</title>
                  </rect>
                )
              })}
              {w.work.map((j, n) => {
                const x = xAt(time(j.start))
                return (
                  <g
                    key={`w${n}`}
                    className="timeline-work"
                    onMouseEnter={() => setHover(`${w.well}|w${n}`)}
                    onMouseLeave={() => setHover(null)}
                  >
                    <line x1={x} x2={x} y1={top - 2} y2={top + row - 10} />
                    <path d={`M${x - 5},${top - 4} L${x + 5},${top - 4} L${x},${top + 4} Z`} />
                    <rect className="timeline-hit-work" x={x - 7} y={top - 6} width={14} height={row - 4} />
                    <title>{readouts[`${w.well}|w${n}`]}</title>
                  </g>
                )
              })}
            </g>
          )
        })}
        {asOf && time(asOf) > t0 && time(asOf) < t1 && (
          <g className="timeline-asof">
            <line x1={xAt(time(asOf))} x2={xAt(time(asOf))} y1={pad.top - 14} y2={height - pad.bottom} />
            <text x={xAt(time(asOf)) + 4} y={pad.top - 8}>
              data up to {asOf}
            </text>
          </g>
        )}
      </svg>
      </div>
      <p className="chart-key">
        <span><i className="case-key case-key-actual" /> oil rate per producing day, monthly, log scale</span>
        <span><i className="timeline-key is-alarm" /> alarm</span>
        <span><i className="timeline-key is-caution" /> caution</span>
        <span><i className="key-work" /> work on the well, from its daily reports</span>
      </p>
    </div>
  )
}

/** The well's monthly rate `offset` months from the month of `iso`. */
function monthRate(w: Timeline['wells'][number], iso: string, offset: number): number | null {
  const d = new Date(`${iso}T00:00:00Z`)
  d.setUTCMonth(d.getUTCMonth() + offset)
  const key = d.toISOString().slice(0, 7)
  const i = w.oil_rate.month.indexOf(key)
  return i >= 0 && w.oil_rate.rate[i] > 0 ? w.oil_rate.rate[i] : null
}

/** The first month with oil after `iso`: a well often stays shut in for weeks after a job. */
function firstRateAfter(w: Timeline['wells'][number], iso: string): number | null {
  const after = iso.slice(0, 7)
  const i = w.oil_rate.month.findIndex((m, k) => m > after && w.oil_rate.rate[k] > 0)
  return i >= 0 ? w.oil_rate.rate[i] : null
}

/** The report sentences that say what was done to the well (a plug, perforations, a log),
    or its opening when none does. */
function keySentences(summary: string): string {
  const sentences = summary.split(/(?<=\.)\s+/)
  const done = sentences.filter((s) => /plug|perforat|\bPLT\b|logg|acid|squeeze|stimul|gauge/i.test(s))
  const text = (done.length ? done : sentences.slice(0, 2)).join(' ')
  return text.length > 320 ? `${text.slice(0, 317)}…` : text
}

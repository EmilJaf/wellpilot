import { useEffect, useState } from 'react'
import { api, type Inside } from '../api'
import { useWidth } from '../useWidth'

type Survey = Inside['surveys'][number]
type Pass = Survey['passes'][number]
type Job = Inside['jobs'][number]

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
const monthYear = (iso: string) => `${MONTHS[Number(iso.slice(5, 7)) - 1]} ${iso.slice(0, 4)}`
const pct = (share: number) => `${Math.round(share * 100)}%`

/** The pass two surveys share best: medium rate when both have it. */
function pick(survey: Survey): Pass {
  return survey.passes.find((p) => p.pass === 'medium rate') ?? survey.passes[0]
}

/** A pass's shares at the depth nearest `depth` (null outside the log). */
function at(pass: Pass, depth: number): { oil: number; water: number | null } | null {
  const d = pass.depth_m
  if (depth < d[0] - 1 || depth > d[d.length - 1] + 1) return null
  let i = 0
  for (let k = 1; k < d.length; k++) if (Math.abs(d[k] - depth) < Math.abs(d[i] - depth)) i = k
  return { oil: pass.oil_share[i], water: pass.water_share ? pass.water_share[i] : null }
}

/**
 * A well along its depth: where its oil and water came in at each production log, its
 * perforated zones and the plugs set in it, with what it produced before and after each job.
 * Logs kept only as pictures were read off their plots; the view says so, with the checks.
 */
export function InsideWell({ well, asOf, onOpenWell }: { well: string; asOf: string | null; onOpenWell: (well: string) => void }) {
  const [loaded, setLoaded] = useState<{ well: string; value?: Inside; error?: string } | null>(null)

  useEffect(() => {
    let live = true
    api
      .inside(well, asOf)
      .then((value) => live && setLoaded({ well, value }))
      .catch((e: Error) => live && setLoaded({ well, error: e.message }))
    return () => {
      live = false
    }
  }, [well, asOf])

  if (!loaded || loaded.well !== well) return <p className="empty field-view">Loading {well} along its depth</p>
  if (!loaded.value)
    return (
      <div className="empty field-view">
        <p>{loaded.error}</p>
        {well !== 'F-12' && (
          <button type="button" className="link-button" onClick={() => onOpenWell('F-12')}>
            Open F-12, the well with production logs
          </button>
        )}
      </div>
    )
  return <WellSection data={loaded.value} />
}

function WellSection({ data }: { data: Inside }) {
  const [host, width] = useWidth(620)
  const [hover, setHover] = useState<number | null>(null)
  const surveys = data.surveys
  const passes = surveys.map(pick)
  const plugs = data.jobs.flatMap((j) => j.plugs_m.map((depth) => ({ depth, date: j.start })))
  const opened = data.jobs.filter((j) => j.kind !== 'completion').flatMap((j) => j.perforations_m.map((z) => ({ zone: z, date: j.start })))

  const d0 = 3060
  const d1 = 3290
  const height = 520
  const pad = { left: 46, top: 46, bottom: 14, right: 8 }
  const wellW = 74
  const gap = 22
  const trackW = Math.max(110, (width - pad.left - pad.right - wellW - gap * (surveys.length + 1)) / Math.max(surveys.length, 1))
  const y = (depth: number) => pad.top + ((height - pad.top - pad.bottom) * (depth - d0)) / (d1 - d0)
  const wellX = pad.left + gap / 2
  const casing = [wellX + 26, wellX + wellW - 26]
  const trackX = (i: number) => pad.left + wellW + gap * (i + 1) + trackW * i

  /** The pass drawn as parts of the whole flow at the top: water from zero, the oil beside. */
  const areas = (pass: Pass, x0: number) => {
    const wc = pass.water_cut_at_top
    const water = pass.depth_m.map((d, k) => ({ d, w: wc * (pass.water_share?.[k] ?? 0) }))
    const total = pass.depth_m.map((d, k) => ({ d, t: water[k].w + (1 - wc) * pass.oil_share[k] }))
    const edge = (pts: { d: number; v: number }[]) => pts.map((p) => `${x0 + p.v * trackW},${y(p.d)}`).join(' ')
    const top = pass.depth_m[0]
    const bottom = pass.depth_m[pass.depth_m.length - 1]
    const waterPath = `${x0},${y(top)} ${edge(water.map((p) => ({ d: p.d, v: p.w })))} ${x0},${y(bottom)}`
    const oilPath = `${edge(water.map((p) => ({ d: p.d, v: p.w })))} ${edge(
      [...total].reverse().map((p) => ({ d: p.d, v: p.t })),
    )}`
    return { waterPath, oilPath }
  }

  const readout =
    hover === null
      ? 'Hover the logs to read the shares at a depth.'
      : `${hover} m: ` +
        surveys
          .map((s, i) => {
            const v = at(passes[i], hover)
            if (!v) return `${monthYear(s.date)}, not logged`
            return `${monthYear(s.date)}, ${pct(v.oil)} of the oil${v.water !== null ? ` and ${pct(v.water)} of the water` : ''} came from below`
          })
          .join('; ')

  return (
    <div className="inside field-view">
      <h3 className="accuracy-title">Inside {data.well}: where its oil and water came in</h3>
      <p className="accuracy-lead">
        {surveys.length > 1
          ? `${surveys.length} production logs, ${monthYear(surveys[0].date)} and ${monthYear(surveys[surveys.length - 1].date)}, `
          : 'Its production log, '}
        with the zones it produced from and the plugs set in it, on one depth axis. A log measures the flow passing each
        depth; each area is a share of the flow at the top, water in blue and oil in green.
      </p>
      <p className="readout">{readout}</p>
      <div ref={host}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`${data.well}'s inflow by depth at each production log, with perforations and plugs`}
          onMouseMove={(e) => {
            const box = e.currentTarget.getBoundingClientRect()
            const depth = d0 + ((e.clientY - box.top - pad.top) / (height - pad.top - pad.bottom)) * (d1 - d0)
            setHover(depth >= d0 && depth <= d1 && e.clientX - box.left > pad.left + wellW ? Math.round(depth) : null)
          }}
          onMouseLeave={() => setHover(null)}
        >
          {Array.from({ length: Math.floor((d1 - d0) / 25) + 1 }, (_, k) => Math.ceil(d0 / 25) * 25 + 25 * k)
            .filter((d) => d <= d1)
            .map((d) => (
              <g key={d}>
                <line className="plan-grid" x1={pad.left} x2={width - pad.right} y1={y(d)} y2={y(d)} />
                <text className="plan-axis" x={pad.left - 6} y={y(d) + 4} textAnchor="end">
                  {d}
                </text>
              </g>
            ))}
          <text className="inside-head" x={wellX + wellW / 2} y={18} textAnchor="middle">
            The well
          </text>
          <text className="inside-sub" x={wellX + wellW / 2} y={33} textAnchor="middle">
            m along hole
          </text>
          {/* The casing, its perforated zones (ticks through it) and the zones opened later. */}
          <line className="inside-casing" x1={casing[0]} x2={casing[0]} y1={pad.top} y2={height - pad.bottom} />
          <line className="inside-casing" x1={casing[1]} x2={casing[1]} y1={pad.top} y2={height - pad.bottom} />
          {data.zones_m.map(([a, b]) => (
            <g key={a} className="inside-perf">
              {Array.from({ length: Math.max(2, Math.round((y(b) - y(a)) / 4)) }, (_, k) => y(a) + ((y(b) - y(a)) * (k + 0.5)) / Math.max(2, Math.round((y(b) - y(a)) / 4))).map((ty) => (
                <g key={ty}>
                  <line x1={casing[0] - 9} x2={casing[0] + 2} y1={ty} y2={ty} />
                  <line x1={casing[1] - 2} x2={casing[1] + 9} y1={ty} y2={ty} />
                </g>
              ))}
            </g>
          ))}
          {opened.map(({ zone: [a, b], date }) => (
            <g key={`${a}-${date}`} className="inside-perf is-new">
              <rect x={casing[0] - 10} y={y(a)} width={casing[1] - casing[0] + 20} height={Math.max(y(b) - y(a), 3)} />
              <text x={casing[0] - 12} y={y(a) - 3} textAnchor="start">
                opened {date.slice(0, 7)}
              </text>
            </g>
          ))}
          {plugs.map(({ depth, date }) => (
            <g key={`${depth}-${date}`} className="inside-plug">
              <rect x={casing[0] + 1} y={y(depth) - 3} width={casing[1] - casing[0] - 2} height={6} />
              <line x1={casing[1]} x2={width - pad.right} y1={y(depth)} y2={y(depth)} />
              <text x={width - pad.right - 2} y={y(depth) - 4} textAnchor="end">
                plug at {depth} m, {date.slice(0, 7)}
              </text>
            </g>
          ))}
          {surveys.map((s, i) => {
            const x0 = trackX(i)
            const { waterPath, oilPath } = areas(passes[i], x0)
            return (
              <g key={s.survey}>
                <text className="inside-head" x={x0} y={18}>
                  {monthYear(s.date)}
                </text>
                <text className="inside-sub" x={x0} y={33}>
                  {s.read_from.startsWith('LAS') ? 'from its LAS files' : "read off its report's plots"}, {passes[i].pass}
                </text>
                <line className="inside-zero" x1={x0} x2={x0} y1={pad.top} y2={height - pad.bottom} />
                <polygon className="inside-water" points={waterPath} />
                <polygon className="inside-oil" points={oilPath} />
              </g>
            )
          })}
          {hover !== null && <line className="case-hover" x1={pad.left + wellW} x2={width - pad.right} y1={y(hover)} y2={y(hover)} />}
        </svg>
      </div>
      <p className="chart-key">
        <span><i className="inside-key is-water" /> water</span>
        <span><i className="inside-key is-oil" /> oil, each from zero at the bottom to the whole flow at the top</span>
        <span><i className="inside-key is-perf" /> perforated zones</span>
        <span><i className="inside-key is-plug" /> plug</span>
      </p>
      <Jobs data={data} passes={passes} />
      <HowRead surveys={surveys} />
    </div>
  )
}

/** The work along the well's depth, what the last log before it said about each plug's
    depth, and what the well produced before and after. */
function Jobs({ data, passes }: { data: Inside; passes: Pass[] }) {
  const lastLogBefore = (job: Job) => {
    let found: number | null = null
    data.surveys.forEach((s, i) => {
      if (s.date <= job.end) found = i
    })
    return found
  }
  const work = (job: Job) =>
    [
      job.production_log && 'production log',
      ...job.plugs_m.map((d) => `plug at ${d} m`),
      ...job.perforations_m.map(([a, b]) => `perforated ${a}-${b} m`),
    ]
      .filter(Boolean)
      .join('; ')
  // One line per plug: what the last log said lay below it, and what the well did after.
  const findings = data.jobs
    .filter((job) => job.plugs_m.length && job.before && job.after)
    .map((job) => {
      const i = lastLogBefore(job)
      const depth = job.plugs_m[0]
      const v = i !== null ? at(passes[i], depth) : null
      const { before, after } = job as Required<Job> & { before: NonNullable<Job['before']>; after: NonNullable<Job['after']> }
      const opened = job.perforations_m.length ? `, with ${job.perforations_m.map(([a, b]) => `${a}-${b} m`).join(' and ')} opened above it` : ''
      return {
        key: job.start,
        text:
          `The plug set at ${depth} m in ${monthYear(job.start)}${opened}` +
          (v && i !== null && v.water !== null
            ? ` sat below ${pct(v.water)} of the water and ${pct(v.oil)} of the oil the ${monthYear(data.surveys[i].date)} log measured`
            : '') +
          `. Water cut ${before.water_cut.toFixed(2)} before, ${after.water_cut.toFixed(2)} after; oil ${before.oil_sm3_per_day.toLocaleString('en')} to ${after.oil_sm3_per_day.toLocaleString('en')} Sm³/d.`,
      }
    })
  return (
    <>
      <h3 className="accuracy-title field-view-second">What was done along the well, and what it did after</h3>
      {findings.map((f) => (
        <p key={f.key} className="plan-figure-finding">
          {f.text}
        </p>
      ))}
      <table className="inside-jobs">
        <thead>
          <tr>
            <th scope="col">Dates</th>
            <th scope="col">Work, from the daily reports</th>
            <th scope="col">The last log before it</th>
            <th scope="col">60 days before</th>
            <th scope="col">First 60 days back</th>
          </tr>
        </thead>
        <tbody>
          {data.jobs.map((job) => {
            const i = lastLogBefore(job)
            const said = job.plugs_m
              .map((depth) => {
                if (i === null) return null
                const v = at(passes[i], depth)
                return v && `below ${depth} m: ${v.water !== null ? `${pct(v.water)} of the water, ` : ''}${pct(v.oil)} of the oil`
              })
              .filter(Boolean)
            return (
              <tr key={job.start}>
                <td>
                  {job.start}
                  {job.end !== job.start && <> to {job.end}</>}
                </td>
                <td>{work(job)}</td>
                <td>{said.length ? said.join('; ') : '–'}</td>
                <td>{job.before ? production(job.before) : '–'}</td>
                <td>{job.after ? production(job.after) : '–'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </>
  )
}

function production(p: NonNullable<Job['before']>) {
  return (
    <>
      water cut {p.water_cut.toFixed(2)}, oil {p.oil_sm3_per_day.toLocaleString('en')} Sm³/d
      <span className="inside-days"> ({p.days} days on stream)</span>
    </>
  )
}

/** How each log was read: the files, or pictures read by SAM 3.1 with the checks. */
function HowRead({ surveys }: { surveys: Survey[] }) {
  return (
    <details className="plan-figure-source inside-how">
      <summary>How the logs were read</summary>
      {surveys.map((s) => {
        const sam = s.plots.some((p) => Object.values(p.read_by).includes('sam-3.1'))
        const upper = s.check?.upper_two_zones_share
        const files = s.check?.plot_against_files_median_share_gap
        return (
          <div key={s.survey} className="inside-how-survey">
            <h4>{monthYear(s.date)}</h4>
            <p>
              {s.read_from.startsWith('LAS') ? (
                <>
                  From the survey&rsquo;s LAS files.{' '}
                  {files && (
                    <>
                      Its plot was also read{sam ? ' by SAM 3.1' : ' by its colours'}, the way the pictures-only survey was,
                      and agrees with the files: at each depth, the shares differ by a median of{' '}
                      {Math.round(Math.max(...Object.values(files).flatMap((g) => Object.values(g))) * 1000) / 10}% of the
                      flow or less.
                    </>
                  )}
                </>
              ) : (
                <>
                  This survey exists only as plots in its interpretation report, pictures with no numbers behind them.{' '}
                  {sam
                    ? `SAM 3.1, Meta's segmentation model, found the rate area on each plot from the words "${Object.values(s.plots[0].prompts)[0]}"; a trace of the same colour checks each reading.`
                    : 'They were read by their colours; the SAM 3.1 reading has not been run yet.'}{' '}
                  The depths and rates come from the plot's own labels, found in the image.
                  {upper && s.check?.report_says && (
                    <>
                      {' '}
                      The upper two zones gave{' '}
                      {Object.entries(upper)
                        .map(([pass, v]) => `${pct(v)} of the oil at the ${pass}`)
                        .join(', ')}
                      ; the report says <q>{s.check.report_says.quote}</q> (page {s.check.report_says.page}).
                    </>
                  )}
                </>
              )}
            </p>
            <div className="inside-plots">
              {s.plots.map((p) => (
                <figure key={p.id}>
                  <a href={api.logPlotImageUrl(p.id)} target="_blank" rel="noreferrer">
                    <img src={api.logPlotImageUrl(p.id)} alt={`The ${p.pass} rate track with the edges read drawn in red`} />
                  </a>
                  <figcaption>
                    {p.pass}
                    {s.read_from.startsWith('LAS') ? '' : `, page ${p.page}`}
                  </figcaption>
                </figure>
              ))}
            </div>
          </div>
        )
      })}
    </details>
  )
}

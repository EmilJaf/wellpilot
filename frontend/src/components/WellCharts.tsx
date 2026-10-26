import { useEffect, useMemo, useRef } from 'react'
import uPlot from 'uplot'
import { column, type Table } from '../api'

export type Marker = {
  at: string // ISO date
  until?: string // ISO date, for a band
  kind: 'regime' | 'crossing' | 'gauge'
  track: TrackKey | 'all'
  label?: string
}

type TrackKey = 'oil' | 'water' | 'bhp'

type Track = {
  key: TrackKey
  column: string
  title: string
  unit: string
  color: string // CSS variable name
  log?: boolean
  range?: [number, number]
  height: number
}

// Small multiples on one time axis, never a dual axis: oil on a log scale (decline reads
// as a straight line), then water cut, then downhole pressure.
const TRACKS: Track[] = [
  { key: 'oil', column: 'oil_rate_per_producing_day', title: 'Oil rate', unit: 'Sm³ per producing day, log scale', color: '--oil', log: true, height: 150 },
  { key: 'water', column: 'water_cut', title: 'Water cut', unit: 'share of produced liquid', color: '--water', range: [0, 1], height: 105 },
  { key: 'bhp', column: 'bhp_bar', title: 'Downhole pressure', unit: 'bar', color: '--pressure', height: 105 },
]

const toSeconds = (iso: string) => Date.parse(iso.length === 7 ? `${iso}-01` : iso) / 1000

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

type Props = { table: Table | null; markers: Marker[]; highlights: string[] }

export function WellCharts({ table, markers, highlights }: Props) {
  const data = useMemo(() => {
    if (!table) return null
    const kind = column(table, 'flow_kind')
    const rows = table.rows.filter((r) => r[kind] === 'production')
    const date = column(table, 'date')
    const x = rows.map((r) => toSeconds(String(r[date])))
    const series = Object.fromEntries(
      TRACKS.map((t) => {
        const i = column(table, t.column)
        return [t.key, rows.map((r) => {
          const v = r[i]
          return typeof v === 'number' && (!t.log || v > 0) ? v : null
        })]
      }),
    ) as Record<TrackKey, (number | null)[]>
    return { x, series }
  }, [table])

  const syncing = useRef(false)
  const plots = useRef<uPlot[]>([])

  if (!data) return <p className="empty charts">Loading production data</p>
  if (data.x.length === 0) return <p className="empty charts">This well has no production rows to plot.</p>

  return (
    <div className="charts">
      {TRACKS.map((track, i) => (
        <ChartTrack
          key={track.key}
          track={track}
          x={data.x}
          y={data.series[track.key]}
          markers={markers}
          highlights={highlights}
          showTime={i === TRACKS.length - 1}
          register={(plot) => {
            plots.current[i] = plot
          }}
          onZoom={(min, max) => {
            if (syncing.current) return
            syncing.current = true
            plots.current.forEach((p, j) => j !== i && p?.setScale('x', { min, max }))
            syncing.current = false
          }}
        />
      ))}
    </div>
  )
}

function ChartTrack({
  track,
  x,
  y,
  markers,
  highlights,
  showTime,
  register,
  onZoom,
}: {
  track: Track
  x: number[]
  y: (number | null)[]
  markers: Marker[]
  highlights: string[]
  showTime: boolean
  register: (plot: uPlot) => void
  onZoom: (min: number, max: number) => void
}) {
  const host = useRef<HTMLDivElement>(null)
  const plot = useRef<uPlot | null>(null)
  const overlay = useRef({ markers, highlights })
  overlay.current = { markers, highlights }

  useEffect(() => {
    const el = host.current
    if (!el) return
    const ink3 = css('--ink-3')
    const rule = css('--rule')
    const font = `11px ${css('--font')}`
    const axis = {
      stroke: ink3,
      font,
      grid: { stroke: rule, width: 1 },
      ticks: { stroke: rule, width: 1, size: 4 },
    }
    const opts: uPlot.Options = {
      width: el.clientWidth,
      height: track.height,
      cursor: { sync: { key: 'well' }, drag: { x: true, y: false } },
      legend: { show: true, live: true },
      scales: {
        x: { time: true },
        y: track.log ? { distr: 3 } : track.range ? { range: track.range } : {},
      },
      axes: [
        showTime ? { ...axis, size: 26 } : { ...axis, size: 4, values: () => [] },
        {
          ...axis,
          size: 46,
          // On the log scale keep the 1-2-5 ticks only; every decade's nine ticks are noise.
          filter: track.log
            ? (_u, splits) => splits.map((v) => ([1, 2, 5].includes(Number(v.toPrecision(1)[0])) ? v : null))
            : undefined,
          values: (_u, ticks) => ticks.map((v) => (v == null ? '' : v >= 1000 ? `${v / 1000}k` : String(v))),
        },
      ],
      series: [
        { label: 'Date', value: (_u, v) => (v == null ? '–' : new Date(v * 1000).toISOString().slice(0, 10)) },
        {
          label: track.title,
          stroke: css(track.color),
          width: 1.5,
          spanGaps: false,
          points: { show: false },
          value: (_u, v) => (v == null ? '–' : track.key === 'water' ? v.toFixed(3) : v.toFixed(0)),
        },
      ],
      hooks: {
        setScale: [
          (u, key) => {
            if (key === 'x' && u.scales.x.min != null && u.scales.x.max != null) {
              onZoom(u.scales.x.min, u.scales.x.max)
            }
          },
        ],
        draw: [(u) => drawOverlay(u, track.key, overlay.current.markers, overlay.current.highlights)],
      },
    }
    const instance = new uPlot(opts, [x, y], el)
    plot.current = instance
    register(instance)
    const resize = new ResizeObserver(() => instance.setSize({ width: el.clientWidth, height: track.height }))
    resize.observe(el)
    return () => {
      resize.disconnect()
      instance.destroy()
      plot.current = null
    }
    // onZoom and register are stable for the life of the track
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [track, x, y, showTime])

  useEffect(() => {
    plot.current?.redraw(false)
  }, [markers, highlights])

  return (
    <div className="chart-track">
      <p className="chart-title">
        <span className="swatch" style={{ background: `var(${track.color})` }} aria-hidden />
        {track.title}
        <span className="chart-note">{track.unit}</span>
      </p>
      <div ref={host} />
    </div>
  )
}

function drawOverlay(u: uPlot, key: TrackKey, markers: Marker[], highlights: string[]) {
  const ctx = u.ctx
  const { top, height, left, width } = u.bbox
  const ratio = uPlot.pxRatio
  const xAt = (iso: string) => u.valToPos(toSeconds(iso), 'x', true)
  ctx.save()
  ctx.beginPath()
  ctx.rect(left, top, width, height)
  ctx.clip()
  ctx.font = `${11 * ratio}px ${css('--font')}`
  ctx.textBaseline = 'top'
  ctx.textAlign = 'left'

  for (const m of markers) {
    if (m.track !== 'all' && m.track !== key) continue
    const x0 = xAt(m.at)
    if (m.until) {
      const x1 = xAt(m.until)
      ctx.fillStyle = css('--caution-wash')
      ctx.fillRect(x0, top, Math.max(x1 - x0, ratio), height)
      ctx.fillStyle = css('--caution')
      ctx.fillRect(x0, top, 2 * ratio, height)
      if (m.label) {
        ctx.fillStyle = css('--caution-ink')
        ctx.fillText(m.label, x0 + 6 * ratio, top + 4 * ratio)
      }
    } else {
      ctx.strokeStyle = css('--rule-strong')
      ctx.lineWidth = ratio
      ctx.beginPath()
      ctx.moveTo(Math.round(x0) + 0.5, top)
      ctx.lineTo(Math.round(x0) + 0.5, top + height)
      ctx.stroke()
    }
  }

  // Dates named in the selected finding: a solid ink rule across every track.
  ctx.strokeStyle = css('--ink')
  ctx.fillStyle = css('--ink')
  ctx.lineWidth = 2 * ratio
  for (const iso of highlights) {
    const x0 = xAt(iso)
    if (x0 < left || x0 > left + width) continue
    ctx.beginPath()
    ctx.moveTo(x0, top)
    ctx.lineTo(x0, top + height)
    ctx.stroke()
    if (key === 'oil') ctx.fillText(iso, x0 + 4 * ratio, top + 4 * ratio)
  }
  ctx.restore()
}

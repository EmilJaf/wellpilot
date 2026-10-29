import { useEffect, useMemo, useRef, useState } from 'react'
import uPlot from 'uplot'
import { column, type Table } from '../api'

export type Marker = {
  at: string // ISO date
  until?: string // ISO date, for a band
  kind: 'regime' | 'crossing' | 'gauge' | 'shut_in' | 'hidden' | 'work'
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

// Data dates are UTC days; show them as such whatever the viewer's time zone.
const utc = (ts: number) => uPlot.tzDate(new Date(ts * 1e3), 'Etc/UTC')

// Shorter charts on shorter screens, so the evidence below them keeps room.
function heightFor(track: Track): number {
  const scale = Math.min(1, Math.max(0.6, (window.innerHeight - 300) / 700))
  return Math.round(track.height * scale)
}

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

/** A span the reader zoomed into, as ISO dates: the charts offer to ask about it. */
export type Span = { start: string; end: string }

type Props = {
  table: Table | null
  error?: string
  markers: Marker[]
  highlights: string[]
  onSelect?: (span: Span | null) => void
}

export function WellCharts({ table, error, markers, highlights, onSelect }: Props) {
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
  const [cursor, setCursor] = useState<number | null>(null)

  if (error) return <p className="empty charts">Production data could not be loaded: {error}</p>
  if (!data) return <p className="empty charts">Loading production data</p>
  if (data.x.length === 0) return <p className="empty charts">This well has no production rows to plot.</p>

  const at = cursor !== null && cursor < data.x.length ? cursor : null
  const reading = (key: TrackKey, digits: number, unit = '') => {
    const v = at === null ? null : data.series[key][at]
    return v === null || v === undefined ? 'no reading' : `${v.toLocaleString('en-US', { maximumFractionDigits: digits })}${unit}`
  }
  return (
    <div className="charts">
      <p className="readout" aria-live="off">
        {at === null ? (
          'Hover to read the values; drag across a chart to zoom in, double-click to zoom out.'
        ) : (
          <>
            <strong>{new Date(data.x[at] * 1000).toISOString().slice(0, 10)}</strong>
            <span>oil {reading('oil', 0, ' Sm³/d')}</span>
            <span>water cut {reading('water', 3)}</span>
            <span>pressure {reading('bhp', 0, ' bar')}</span>
          </>
        )}
      </p>
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
          onCursor={setCursor}
          onZoom={(min, max) => {
            if (syncing.current) return
            syncing.current = true
            plots.current.forEach((p, j) => j !== i && p?.setScale('x', { min, max }))
            syncing.current = false
            // A zoom narrower than the data is a span the reader may ask about; the full
            // range (a double-click) clears it.
            const first = data.x[0]
            const last = data.x[data.x.length - 1]
            if (max - min < last - first - 1) {
              const iso = (s: number) => new Date(s * 1000).toISOString().slice(0, 10)
              onSelect?.({ start: iso(Math.max(min, first)), end: iso(Math.min(max, last)) })
            } else {
              onSelect?.(null)
            }
          }}
        />
      ))}
      {/* The key names only what is on the charts: before a run, the work on the well. */}
      <p className="chart-key">
        {markers.some((m) => m.kind === 'regime' || m.kind === 'crossing') && (
          <span><i className="key-line" /> step change or threshold the agent found</span>
        )}
        {markers.some((m) => m.kind === 'shut_in') && <span><i className="key-band key-shut" /> well shut in</span>}
        {markers.some((m) => m.kind === 'work') && (
          <span><i className="key-work" /> work on the well, from its daily reports</span>
        )}
        {markers.some((m) => m.kind === 'gauge') && <span><i className="key-band key-gauge" /> dead pressure gauge</span>}
        {highlights.length > 0 && <span><i className="key-rule" /> date in the selected finding</span>}
        {markers.some((m) => m.kind === 'hidden') && (
          <span><i className="key-band key-future" /> after the agent&rsquo;s date: data it never saw</span>
        )}
      </p>
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
  onCursor,
}: {
  track: Track
  x: number[]
  y: (number | null)[]
  markers: Marker[]
  highlights: string[]
  showTime: boolean
  register: (plot: uPlot) => void
  onZoom: (min: number, max: number) => void
  onCursor: (index: number | null) => void
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
    const height = heightFor(track)
    const opts: uPlot.Options = {
      width: el.clientWidth,
      height,
      tzDate: utc,
      cursor: { sync: { key: 'well' }, drag: { x: true, y: false } },
      legend: { show: false }, // one readout above all three charts instead
      scales: {
        x: { time: true },
        y: track.log ? { distr: 3 } : track.range ? { range: track.range } : {},
      },
      axes: [
        // 40px leaves room for uPlot's second label line (the year) when zoomed in.
        showTime ? { ...axis, size: 40 } : { ...axis, size: 4, values: () => [] },
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
        setCursor: [(u) => onCursor(u.cursor.idx ?? null)],
      },
    }
    const instance = new uPlot(opts, [x, y], el)
    plot.current = instance
    register(instance)
    const resize = new ResizeObserver(() => instance.setSize({ width: el.clientWidth, height }))
    resize.observe(el)
    return () => {
      resize.disconnect()
      instance.destroy()
      plot.current = null
    }
    // onZoom, onCursor and register are stable for the life of the track
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

  // Labels along the top edge: each takes the first of three rows where it overlaps no
  // other, and goes to the left of its mark where the right edge would cut it. A label
  // with no room is left out (the key and the hover readout still say what the mark is).
  // Labels are drawn after every mark, so a later mark never covers an earlier label, and
  // the top row keeps clear of the other work marks' triangles.
  const rows: [number, number][][] = [[], [], []]
  const triangles =
    key === 'oil'
      ? markers.filter((m) => m.kind === 'work' && (m.track === 'all' || m.track === key)).map((m) => xAt(m.at))
      : []
  const later: (() => void)[] = []
  const label = (text: string, x: number, beside = 7 * ratio) => {
    const ink = ctx.fillStyle
    later.push(() => place(text, x, beside, ink))
  }
  const place = (text: string, x: number, beside: number, ink: string | CanvasGradient | CanvasPattern) => {
    const w = ctx.measureText(text).width
    const right = left + width
    const start = x + beside + w <= right ? x + beside : Math.max(left + 2 * ratio, x - beside - w)
    const clear = (a: number, b: number) => start + w + 6 * ratio < a || start > b + 6 * ratio
    const row = rows.findIndex(
      (taken, i) =>
        taken.every(([a, b]) => clear(a, b)) &&
        (i > 0 || triangles.every((t) => Math.abs(t - x) < ratio || clear(t - 6 * ratio, t + 6 * ratio))),
    )
    if (row < 0) return
    rows[row].push([start, start + w])
    // A wash behind the text, so a label over the data stays legible.
    ctx.fillStyle = css('--panel')
    ctx.globalAlpha = 0.85
    ctx.fillRect(start - 2 * ratio, top + (1 + row * 13) * ratio, w + 4 * ratio, 13 * ratio)
    ctx.globalAlpha = 1
    ctx.fillStyle = ink
    ctx.fillText(text, start, top + (2 + row * 13) * ratio)
  }

  for (const m of markers) {
    if (m.track !== 'all' && m.track !== key) continue
    const x0 = xAt(m.at)
    if (m.kind === 'hidden') {
      // The time machine's reveal: the days after the agent's date, which it never saw.
      const x1 = m.until ? xAt(m.until) : left + width
      ctx.fillStyle = css('--future-wash')
      ctx.fillRect(x0, top, Math.max(x1 - x0, 0), height)
      ctx.fillStyle = css('--ink')
      ctx.fillRect(Math.round(x0), top, 2 * ratio, height)
      if (m.label && key === 'oil') ctx.fillText(m.label, x0 + 6 * ratio, top + 4 * ratio)
    } else if (m.until) {
      const x1 = xAt(m.until)
      const gauge = m.kind === 'gauge'
      ctx.fillStyle = css(gauge ? '--caution-wash' : '--shut-wash')
      ctx.fillRect(x0, top, Math.max(x1 - x0, ratio), height)
      if (gauge) {
        ctx.fillStyle = css('--caution')
        ctx.fillRect(x0, top, 2 * ratio, height)
      }
      if (m.label && x1 > left) {
        // Keep the label in view when zoomed inside the band.
        ctx.fillStyle = css('--caution-ink')
        label(m.label, Math.max(x0, left), 6 * ratio)
      }
    } else if (m.kind === 'work') {
      // Work done on the well (from its daily reports): a dashed rule and, on the oil
      // track, a triangle with what was done.
      ctx.strokeStyle = css('--ink')
      ctx.lineWidth = ratio
      ctx.setLineDash([3 * ratio, 3 * ratio])
      ctx.beginPath()
      ctx.moveTo(Math.round(x0) + 0.5, top)
      ctx.lineTo(Math.round(x0) + 0.5, top + height)
      ctx.stroke()
      ctx.setLineDash([])
      if (key === 'oil') {
        const s = 5 * ratio
        ctx.fillStyle = css('--ink')
        ctx.beginPath()
        ctx.moveTo(x0 - s, top)
        ctx.lineTo(x0 + s, top)
        ctx.lineTo(x0, top + s * 1.4)
        ctx.closePath()
        ctx.fill()
        if (m.label) label(m.label, x0)
      }
    } else if (m.kind === 'crossing') {
      // A threshold crossing: a short tick down from the top edge.
      ctx.fillStyle = css('--ink-2')
      ctx.fillRect(Math.round(x0), top, ratio, 10 * ratio)
    } else {
      ctx.strokeStyle = css('--ink-2')
      ctx.lineWidth = ratio
      ctx.beginPath()
      ctx.moveTo(Math.round(x0) + 0.5, top)
      ctx.lineTo(Math.round(x0) + 0.5, top + height)
      ctx.stroke()
    }
  }

  for (const draw of later) draw()

  // Dates named in the selected finding: a solid ink rule across every track, labelled in
  // the same rows as the work labels, so a date never prints over one.
  ctx.strokeStyle = css('--ink')
  ctx.lineWidth = 2 * ratio
  for (const iso of [...highlights].sort()) {
    const x0 = xAt(iso)
    if (x0 < left || x0 > left + width) continue
    ctx.beginPath()
    ctx.moveTo(x0, top)
    ctx.lineTo(x0, top + height)
    ctx.stroke()
    if (key === 'oil') place(iso, x0, 4 * ratio, css('--ink'))
  }
  ctx.restore()
}

import { useState } from 'react'
import { column, type Table } from '../api'
import { useWidth } from '../useWidth'

type Limit = { limit: string; value: number; quote: string }
type Against = { first_month_at_or_above: string | null; months_at_or_above: number; max: number; max_month: string }

/** The field_vs_design tool's result (also served at /api/field/design). */
export type DesignResult = {
  design_limits: Record<string, Limit>
  field_against_limits: Record<string, Against>
  months: Table
  plan_source: string
}

const MONTH = (m: string) => Date.parse(`${m}-15T00:00:00Z`)

/** A round tick step (1, 2 or 5 times a power of ten) giving two or three steps up to max. */
function niceStep(max: number): number {
  const raw = max / 2.5
  const power = 10 ** Math.floor(Math.log10(raw))
  return ([1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? 10 * power)
}

type Track = {
  key: string
  title: string
  unit: string
  series: { column: string; label: string; className: string }[]
  limit?: Limit
  format: (v: number) => string
}

/**
 * The field month by month against the limits the 2005 plan designed the platform for:
 * water cut against its maximum, liquid and oil against capacity. One axis per track.
 */
export function DesignChart({ result }: { result: DesignResult }) {
  const [host, width] = useWidth(560)
  const [hover, setHover] = useState<number | null>(null)
  const t = result.months
  const months = t.rows.map((r) => String(r[column(t, 'month')]))
  const values = (name: string) => t.rows.map((r) => Number(r[column(t, name)]))
  const limits = result.design_limits
  const tracks: Track[] = [
    {
      key: 'water_cut',
      title: 'Field water cut',
      unit: 'share of produced liquid',
      series: [{ column: 'water_cut', label: 'water cut', className: 'design-water' }],
      limit: limits.water_cut,
      format: (v) => v.toFixed(2),
    },
    {
      key: 'liquid',
      title: 'Field liquid and oil',
      unit: 'Sm³ per day',
      series: [
        { column: 'liquid_sm3_per_day', label: 'liquid', className: 'design-liquid' },
        { column: 'oil_sm3_per_day', label: 'oil', className: 'design-oil' },
      ],
      limit: limits.liquid_sm3_per_day,
      format: (v) => Math.round(v).toLocaleString('en'),
    },
  ]
  const height = 120
  const pad = { left: 52, right: 12, top: 8, bottom: 18 }
  const plotW = width - pad.left - pad.right
  const plotH = height - pad.top - pad.bottom
  const t0 = MONTH(months[0])
  const t1 = MONTH(months[months.length - 1])
  const xAt = (i: number) => pad.left + (plotW * (MONTH(months[i]) - t0)) / Math.max(1, t1 - t0)
  const years = [...new Set(months.map((m) => m.slice(0, 4)))].filter((y) => months.includes(`${y}-01`))
  const water = result.field_against_limits.water_cut

  return (
    <div className="design-chart" ref={host}>
      <p className="readout">
        {hover !== null ? (
          <>
            <strong>{months[hover]}</strong>
            {tracks.flatMap((track) =>
              track.series.map((s) => (
                <span key={s.column}>
                  {s.label} {track.format(values(s.column)[hover])}
                </span>
              )),
            )}
          </>
        ) : (
          'Hover a month to read the field’s rates.'
        )}
      </p>
      {tracks.map((track) => {
        const all = track.series.flatMap((s) => values(s.column)).filter(Number.isFinite)
        const peak = Math.max(...all, track.limit?.value ?? 0)
        const step = niceStep(peak)
        // Ticks up to the first one at or above the peak, which is the top of the axis.
        const ticks = [0, step, 2 * step, 3 * step].filter((_, i, list) => i === 0 || list[i - 1] < peak)
        const top = ticks[ticks.length - 1]
        const yAt = (v: number) => pad.top + plotH * (1 - v / top)
        const line = (name: string) =>
          values(name)
            .map((v, i) => (Number.isFinite(v) ? `${xAt(i)},${yAt(v)}` : null))
            .filter(Boolean)
            .join(' ')
        return (
          <div key={track.key} className="design-track">
            <p className="chart-title">
              <strong>{track.title}</strong> <span>{track.unit}</span>
            </p>
            <svg
              width={width}
              height={height}
              role="img"
              aria-label={`${track.title} by month against the plan's ${track.limit?.limit ?? 'limit'}`}
              onMouseMove={(e) => {
                const x = e.clientX - e.currentTarget.getBoundingClientRect().left
                let best = 0
                for (let i = 1; i < months.length; i++) if (Math.abs(xAt(i) - x) < Math.abs(xAt(best) - x)) best = i
                setHover(x >= pad.left && x <= pad.left + plotW ? best : null)
              }}
              onMouseLeave={() => setHover(null)}
            >
              {ticks.map((v) => (
                <g key={v}>
                  <line className="plan-grid" x1={pad.left} x2={pad.left + plotW} y1={yAt(v)} y2={yAt(v)} />
                  <text className="plan-axis" x={pad.left - 6} y={yAt(v) + 4} textAnchor="end">
                    {track.format(v)}
                  </text>
                </g>
              ))}
              {track.limit && (
                <g>
                  <line className="design-limit" x1={pad.left} x2={pad.left + plotW} y1={yAt(track.limit.value)} y2={yAt(track.limit.value)} />
                  <text className="design-limit-label" x={pad.left + 6} y={yAt(track.limit.value) - 5}>
                    plan: {track.limit.quote}
                  </text>
                </g>
              )}
              {track.series.map((s) => (
                <polyline key={s.column} className={`design-line ${s.className}`} points={line(s.column)} />
              ))}
              {years.map((y) => (
                <text key={y} className="plan-axis" x={xAt(months.indexOf(`${y}-01`))} y={height - 4} textAnchor="middle">
                  {width < 420 ? `'${y.slice(2)}` : y}
                </text>
              ))}
              {hover !== null && <line className="case-hover" x1={xAt(hover)} x2={xAt(hover)} y1={pad.top} y2={pad.top + plotH} />}
            </svg>
          </div>
        )
      })}
      <p className="chart-key">
        <span><i className="case-key design-key-water" /> water cut</span>
        <span><i className="case-key design-key-liquid" /> liquid (oil and water)</span>
        <span><i className="case-key case-key-actual" /> oil</span>
        <span><i className="case-key design-key-limit" /> the plan’s design limit</span>
      </p>
      <p className="plan-summary">
        {water?.first_month_at_or_above ? (
          <>
            The field’s water cut reached the plan’s maximum of {limits.water_cut.value.toFixed(2)} in{' '}
            <strong>{water.first_month_at_or_above}</strong> and stayed at or above it for {water.months_at_or_above}{' '}
            months.
          </>
        ) : (
          <>The field’s water cut stayed below the plan’s maximum.</>
        )}{' '}
        Liquid peaked at {Math.round(result.field_against_limits.liquid_sm3_per_day.max).toLocaleString('en')} Sm³/d (
        {result.field_against_limits.liquid_sm3_per_day.max_month}), against a capacity of{' '}
        {limits.liquid_sm3_per_day.value.toLocaleString('en')}.
      </p>
      <p className="chart-note-line">Limits: {result.plan_source}, as read from the page. Monthly averages per calendar day.</p>
    </div>
  )
}

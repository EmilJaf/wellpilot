import { useEffect, useRef, useState } from 'react'
import { column, type Table } from '../api'

/** The plan_vs_actual tool's result (also served at /api/field/plan). */
export type PlanResult = {
  years: Table
  totals_msm3: { actual: number; plan_p90: number; plan_p50: number; plan_p10: number }
  plan_p50_last_year: number
  actual_last_year: number | null
  actual_after_plan_p50_ends_msm3: number
  wells_first_oil_after_plan_p50_ends: Table
  plan_source: string
  plan_rows_quoted: string[]
}

type Year = { year: number; actual: number | null; p90: number | null; p50: number | null; p10: number | null }

const fmt = (v: number | null) => (v === null ? 'none' : `${v.toFixed(2)} MSm³`)

/**
 * The field's oil per year (bars, from the data) against the 2005 plan (line: P50, band:
 * P90 to P10, from Table 4.5 of the plan as Muse Spark read it). One axis, one unit.
 */
export function PlanChart({ result }: { result: PlanResult }) {
  const host = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(560)
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    const el = host.current
    if (!el) return
    const observer = new ResizeObserver(() => setWidth(Math.max(300, el.clientWidth)))
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const t = result.years
  const at = (row: Table['rows'][number], name: string) => {
    const v = row[column(t, name)]
    return typeof v === 'number' ? v : null
  }
  const years: Year[] = t.rows.map((r) => ({
    year: Number(r[column(t, 'year')]),
    actual: at(r, 'actual_oil_msm3'),
    p90: at(r, 'plan_p90_msm3'),
    p50: at(r, 'plan_p50_msm3'),
    p10: at(r, 'plan_p10_msm3'),
  }))

  const height = 230
  const pad = { left: 40, right: 12, top: 14, bottom: 26 }
  const plotW = width - pad.left - pad.right
  const plotH = height - pad.top - pad.bottom
  const top = Math.max(3.5, ...years.map((y) => Math.max(y.actual ?? 0, y.p10 ?? 0))) * 1.05
  const band = plotW / years.length
  const xMid = (i: number) => pad.left + band * (i + 0.5)
  const yAt = (v: number) => pad.top + plotH * (1 - v / top)
  const planned = years.map((y, i) => ({ ...y, i })).filter((y) => y.p50 !== null)
  const afterIndex = years.findIndex((y) => y.year > result.plan_p50_last_year)
  const lateWells = result.wells_first_oil_after_plan_p50_ends.rows.map((r) => String(r[0]))
  const shown = hover !== null ? years[hover] : null

  return (
    <div className="plan-chart" ref={host}>
      <p className="readout">
        {shown ? (
          <>
            <strong>{shown.year}</strong>
            <span>actual {fmt(shown.actual)}</span>
            <span>
              plan P50 {fmt(shown.p50)}
              {shown.p90 !== null && shown.p10 !== null && ` (P90 ${shown.p90.toFixed(2)} to P10 ${shown.p10.toFixed(2)})`}
            </span>
          </>
        ) : (
          'Hover a year to read actual and planned oil.'
        )}
      </p>
      <svg width={width} height={height} role="img" aria-label="Field oil production per year against the 2005 plan">
        {/* the years the plan had no production in: what the field made anyway */}
        {afterIndex >= 0 && (
          <rect
            className="plan-after"
            x={pad.left + band * afterIndex}
            y={pad.top}
            width={band * (years.length - afterIndex)}
            height={plotH}
          />
        )}
        {[0, 1, 2, 3].filter((v) => v <= top).map((v) => (
          <g key={v}>
            <line className="plan-grid" x1={pad.left} x2={width - pad.right} y1={yAt(v)} y2={yAt(v)} />
            <text className="plan-axis" x={pad.left - 6} y={yAt(v) + 4} textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        {/* P90 to P10 range of the plan */}
        <path
          className="plan-band"
          d={
            planned
              .map((y, k) => `${k ? 'L' : 'M'}${xMid(y.i)},${yAt(y.p10 ?? 0)}`)
              .join(' ') +
            ' ' +
            [...planned]
              .reverse()
              .map((y) => `L${xMid(y.i)},${yAt(y.p90 ?? 0)}`)
              .join(' ') +
            ' Z'
          }
        />
        {years.map((y, i) =>
          y.actual !== null ? (
            <rect
              key={y.year}
              className="plan-bar"
              x={xMid(i) - band * 0.3}
              y={yAt(y.actual)}
              width={band * 0.6}
              height={Math.max(1, plotH - (yAt(y.actual) - pad.top))}
              rx={2}
            />
          ) : null,
        )}
        <polyline className="plan-line" points={planned.map((y) => `${xMid(y.i)},${yAt(y.p50 ?? 0)}`).join(' ')} />
        {planned.map((y) => (
          <circle key={y.year} className="plan-dot" cx={xMid(y.i)} cy={yAt(y.p50 ?? 0)} r={3} />
        ))}
        {years.map((y, i) => (
          <text key={y.year} className="plan-axis" x={xMid(i)} y={height - 8} textAnchor="middle">
            {width < 420 ? `'${String(y.year).slice(2)}` : y.year}
          </text>
        ))}
        {afterIndex >= 0 && (
          <text className="plan-note" x={pad.left + band * afterIndex + 6} y={pad.top + 14}>
            after the plan
          </text>
        )}
        {/* hover targets: a whole year column, wider than its bar */}
        {years.map((y, i) => (
          <rect
            key={y.year}
            x={pad.left + band * i}
            y={pad.top}
            width={band}
            height={plotH}
            fill="transparent"
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          >
            <title>
              {y.year}: actual {fmt(y.actual)}, plan P50 {fmt(y.p50)}
            </title>
          </rect>
        ))}
      </svg>
      <p className="chart-key">
        <span><i className="key-band plan-key-bar" /> actual oil, from the data</span>
        <span><i className="plan-key-line" /> 2005 plan, P50</span>
        <span><i className="key-band plan-key-band" /> plan's P90 to P10 range</span>
        <span>MSm³ per year</span>
      </p>
      <p className="plan-summary">
        The plan expected <strong>{result.totals_msm3.plan_p50.toFixed(1)} MSm³</strong> by{' '}
        {result.plan_p50_last_year}. The field produced{' '}
        <strong>{result.totals_msm3.actual.toFixed(1)} MSm³</strong> by {result.actual_last_year},{' '}
        <strong>{result.actual_after_plan_p50_ends_msm3.toFixed(1)} MSm³</strong> of it after the plan's last
        year{lateWells.length > 0 && `, with ${lateWells.join(', ')} coming online then`}.
      </p>
      <p className="chart-note-line">Plan figures: {result.plan_source}, as read from the page.</p>
    </div>
  )
}

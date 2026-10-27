import { useState } from 'react'
import { column, type Table } from '../api'
import { useWidth } from '../useWidth'

type Limit = {
  water_cut: number
  wor: number
  cum_oil_at_limit_sm3: number | null
  remaining_oil_sm3: number | null
  already_passed: boolean
}

/** The waterflood tool's result. */
export type WaterfloodResult = {
  well: string
  monthly: Table
  wor_fit: {
    window_start: string
    window_end: string
    log10_wor_per_million_sm3: number
    intercept: number
    r2: number
    cum_oil_now_sm3: number
    wor_now: number
    to_limits: Limit[]
  } | null
  by_year: Table
  field_voidage: Table
  notes: string
}

const MIN_WOR = 0.005 // drier months than this are below the plot's floor
const kSm3 = (v: number) => `${Math.round(v / 1000).toLocaleString('en')}k`

/**
 * WOR against cumulative oil on a log scale, the waterflood engineer's plot: months as dots,
 * the fitted trend extended to the water-cut limits, each limit as a dashed rule.
 */
export function WaterfloodChart({ result }: { result: WaterfloodResult }) {
  const [host, width] = useWidth(520)
  const [hover, setHover] = useState<number | null>(null)
  const t = result.monthly
  const points = t.rows
    .map((r) => ({
      month: String(r[column(t, 'month')]),
      np: Number(r[column(t, 'cum_oil_sm3')]),
      wor: Number(r[column(t, 'wor')]),
    }))
    .filter((p) => p.wor >= MIN_WOR)
  const fit = result.wor_fit
  const limits = fit?.to_limits ?? []
  const npMax = Math.max(...points.map((p) => p.np), ...limits.map((l) => l.cum_oil_at_limit_sm3 ?? 0)) * 1.03
  const worValues = [...points.map((p) => p.wor), ...limits.map((l) => l.wor)]
  const lo = Math.log10(Math.min(...worValues))
  const hi = Math.log10(Math.max(...worValues) * 1.3)
  const height = 240
  const pad = { left: 46, right: 70, top: 12, bottom: 30 }
  const plotW = width - pad.left - pad.right
  const plotH = height - pad.top - pad.bottom
  const xAt = (np: number) => pad.left + (plotW * np) / npMax
  const yAt = (wor: number) => pad.top + plotH * ((hi - Math.log10(wor)) / (hi - lo))
  const ticks = [0.01, 0.1, 1, 10, 100].filter((v) => Math.log10(v) >= lo && Math.log10(v) <= hi)
  const xTicks = [1, 2, 3, 4, 5, 6].map((m) => m * 1e6).filter((v) => v < npMax)
  const trend = (np: number) => 10 ** (fit!.intercept + (fit!.log10_wor_per_million_sm3 / 1e6) * np)
  const shown = hover !== null ? points[hover] : null

  return (
    <div className="waterflood" ref={host}>
      <p className="readout">
        {shown ? (
          <>
            <strong>{shown.month}</strong>
            <span>cumulative oil {kSm3(shown.np)} Sm³</span>
            <span>WOR {shown.wor.toFixed(2)}</span>
          </>
        ) : (
          'WOR (water/oil) against cumulative oil, log scale. Hover a month.'
        )}
      </p>
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`${result.well} water-oil ratio against cumulative oil`}
        onMouseMove={(e) => {
          const x = e.clientX - e.currentTarget.getBoundingClientRect().left
          let best = 0
          for (let i = 1; i < points.length; i++) if (Math.abs(xAt(points[i].np) - x) < Math.abs(xAt(points[best].np) - x)) best = i
          setHover(points.length ? best : null)
        }}
        onMouseLeave={() => setHover(null)}
      >
        {ticks.map((v) => (
          <g key={v}>
            <line className="plan-grid" x1={pad.left} x2={pad.left + plotW} y1={yAt(v)} y2={yAt(v)} />
            <text className="plan-axis" x={pad.left - 6} y={yAt(v) + 4} textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        {xTicks.map((v) => (
          <text key={v} className="plan-axis" x={xAt(v)} y={height - 10} textAnchor="middle">
            {v / 1e6}M
          </text>
        ))}
        <text className="plan-axis" x={pad.left + plotW} y={height - 10} textAnchor="end">
          Sm³ oil
        </text>
        {limits.map((l) => (
          <g key={l.water_cut}>
            <line className="design-limit" x1={pad.left} x2={pad.left + plotW} y1={yAt(l.wor)} y2={yAt(l.wor)} />
            <text className="wf-limit-label" x={pad.left + plotW + 6} y={yAt(l.wor) + 4}>
              wc {l.water_cut.toFixed(2)}
            </text>
          </g>
        ))}
        {points.map((p, i) => (
          <circle key={p.month} className={`wf-point${hover === i ? ' is-hover' : ''}`} cx={xAt(p.np)} cy={yAt(p.wor)} r={2.6} />
        ))}
        {fit && (
          <line
            className="wf-trend"
            x1={xAt(points.find((p) => p.month >= fit.window_start)?.np ?? 0)}
            y1={yAt(trend(points.find((p) => p.month >= fit.window_start)?.np ?? 0))}
            x2={xAt(npMax)}
            y2={yAt(trend(npMax))}
          />
        )}
        {fit && <line className="case-cut" x1={xAt(fit.cum_oil_now_sm3)} x2={xAt(fit.cum_oil_now_sm3)} y1={pad.top} y2={pad.top + plotH} />}
      </svg>
      <p className="chart-key">
        <span><i className="strip-key strip-key-water" /> a month</span>
        <span><i className="case-key" /> the fitted trend, extended</span>
        <span><i className="case-key design-key-limit" /> economic water cut</span>
      </p>
      {fit && (
        <p className="plan-summary">
          Trend fitted on {fit.window_start} to {fit.window_end} (r² {fit.r2.toFixed(2)}). Oil left before the
          limits:{' '}
          {limits
            .map((l) =>
              l.already_passed
                ? `water cut ${l.water_cut.toFixed(2)} already passed`
                : `${kSm3(l.remaining_oil_sm3 ?? 0)} Sm³ to ${l.water_cut.toFixed(2)}`,
            )
            .join('; ')}
          .
        </p>
      )}
    </div>
  )
}

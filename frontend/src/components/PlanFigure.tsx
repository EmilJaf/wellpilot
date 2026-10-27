import { useEffect, useState } from 'react'
import { api, type PlanFigure as Figure } from '../api'
import { useWidth } from '../useWidth'

const time = (month: string) => Date.parse(`${month}-15T00:00:00Z`)

/**
 * The plan's forecast as it was drawn (PUD Figure 4.2), read off the picture by SAM 3.1,
 * against the field's oil rate. The figure is the only place the plan shows the rate it
 * expected day by day; the reading carries its check (a colour trace of the same pixels).
 */
export function PlanFigure() {
  const [loaded, setLoaded] = useState<{ value?: Figure; error?: string } | null>(null)

  useEffect(() => {
    api
      .planFigure()
      .then((value) => setLoaded({ value }))
      .catch((e: Error) => setLoaded({ error: e.message }))
  }, [])

  // Not read yet: say nothing rather than show an empty chart.
  if (!loaded?.value) return null
  return <FigureChart f={loaded.value} />
}

function FigureChart({ f }: { f: Figure }) {
  const [host, width] = useWidth(560)
  const [hover, setHover] = useState<string | null>(null)
  const height = 230
  const pad = { left: 46, right: 12, top: 12, bottom: 26 }
  const t0 = Date.UTC(2007, 0, 1)
  const t1 = Date.UTC(2017, 0, 1)
  const plotW = width - pad.left - pad.right
  const plotH = height - pad.top - pad.bottom
  const xAt = (ms: number) => pad.left + (plotW * (ms - t0)) / (t1 - t0)
  const yAt = (rate: number) => pad.top + plotH * (1 - rate / 9000)
  const line = (months: string[], rates: number[]) => months.map((m, i) => `${xAt(time(m))},${yAt(rates[i])}`).join(' ')
  const plan = Object.fromEntries(f.curve.month.map((m, i) => [m, f.curve.rate[i]]))
  const actual = Object.fromEntries(f.actual.month.map((m, i) => [m, f.actual.rate[i]]))
  const plateau = Math.max(...f.curve.rate)
  const sam = f.read_by === 'sam-3.1'
  const years = Array.from({ length: 6 }, (_, i) => 2007 + 2 * i)

  return (
    <div className="plan-figure">
      <h3 className="accuracy-title field-view-second">The plan&rsquo;s forecast as drawn, against what the field made</h3>
      <p className="accuracy-lead">
        The plan&rsquo;s text gives a plateau of 9 000 Sm³/d; its Figure 4.2 draws the rate it expected month by month,
        with a plateau near {Math.round(plateau / 100) * 100} Sm³/d.{' '}
        {sam ? (
          <>
            SAM 3.1, Meta&rsquo;s segmentation model, found the curve in the picture from the words &ldquo;{f.prompt}
            &rdquo;; the axes were calibrated on the figure&rsquo;s {f.calibration.gridlines} gridlines. A colour trace of
            the same pixels agrees within {f.agreement_with_colour?.median_sm3_per_day} Sm³/d (median).
          </>
        ) : (
          <>The curve was traced by its colour; the SAM 3.1 reading has not been run yet.</>
        )}
      </p>
      <p className="readout">
        {hover ? (
          <>
            <strong>{hover}</strong>
            <span>plan {plan[hover] !== undefined ? `${plan[hover].toLocaleString('en')} Sm³/d` : '–'}</span>
            <span>field {actual[hover] !== undefined ? `${actual[hover].toLocaleString('en')} Sm³/d` : '–'}</span>
          </>
        ) : (
          'Oil per calendar day, by month. Hover to read.'
        )}
      </p>
      <div ref={host}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label="The plan's drawn forecast curve against the field's oil rate"
          onMouseMove={(e) => {
            const box = e.currentTarget.getBoundingClientRect()
            const ms = t0 + ((e.clientX - box.left - pad.left) / plotW) * (t1 - t0)
            const d = new Date(ms)
            setHover(ms >= t0 && ms <= t1 ? d.toISOString().slice(0, 7) : null)
          }}
          onMouseLeave={() => setHover(null)}
        >
          {[0, 3000, 6000, 9000].map((v) => (
            <g key={v}>
              <line className="plan-grid" x1={pad.left} x2={pad.left + plotW} y1={yAt(v)} y2={yAt(v)} />
              <text className="plan-axis" x={pad.left - 6} y={yAt(v) + 4} textAnchor="end">
                {v === 0 ? '0' : `${v / 1000}k`}
              </text>
            </g>
          ))}
          {years.map((y) => (
            <text key={y} className="plan-axis" x={xAt(Date.UTC(y, 0, 1))} y={height - 8} textAnchor="middle">
              {y}
            </text>
          ))}
          <polyline className="plan-figure-actual" points={line(f.actual.month, f.actual.rate)} />
          <polyline className="plan-figure-plan" points={line(f.curve.month, f.curve.rate)} />
          {hover && <line className="case-hover" x1={xAt(time(hover))} x2={xAt(time(hover))} y1={pad.top} y2={pad.top + plotH} />}
        </svg>
      </div>
      <p className="chart-key">
        <span><i className="case-key plan-figure-key-plan" /> the plan&rsquo;s curve, read off Figure 4.2 (PUD p.23)</span>
        <span><i className="case-key case-key-actual" /> the field, from the production data</span>
      </p>
      <details className="plan-figure-source">
        <summary>How the figure was read</summary>
        <p>
          The figure with the curve that was read drawn over it in ink.{' '}
          {sam &&
            `SAM found ${f.objects_found} object${f.objects_found === 1 ? '' : 's'}${f.confidence !== null && f.confidence !== undefined ? ` (confidence ${f.confidence.toFixed(2)})` : ''}. `}
          The plot area and gridlines were found in the image, and the axis ends (0 and 9 000 Sm³/d, 2007 and 2017) read off
          its labels.
        </p>
        <a href={api.planFigureImageUrl()} target="_blank" rel="noreferrer">
          <img src={api.planFigureImageUrl()} alt="Figure 4.2 of the 2005 plan with the read curve drawn over it" />
        </a>
      </details>
    </div>
  )
}

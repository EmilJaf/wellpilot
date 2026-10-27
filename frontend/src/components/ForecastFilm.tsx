import { useEffect, useState } from 'react'
import { api, type BacktestCase, type Film } from '../api'
import { CheckIcon } from '../icons'
import { signed, volume } from '../labels'
import { CaseChart, type Domain } from './AccuracyView'

const FRAME_MS = 1400
const DAY = 86_400_000
const time = (iso: string) => Date.parse(`${iso}T00:00:00Z`)

/**
 * The forecasts of one well in the order they were made: a slider over the cutoffs, each
 * frame the agent's window and forecast against the classic fit's and what the well then
 * produced, on fixed axes so only the forecasts move. Play runs through them.
 */
export function ForecastFilm({ wells, initial }: { wells: string[]; initial: string }) {
  const [well, setWell] = useState(initial)
  const [loaded, setLoaded] = useState<{ well: string; film?: Film; error?: string } | null>(null)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    let current = true
    api
      .film(well)
      .then((film) => current && setLoaded({ well, film }))
      .catch((e: Error) => current && setLoaded({ well, error: e.message }))
    return () => {
      current = false
    }
  }, [well])

  const film = loaded?.well === well ? loaded.film : undefined
  const last = film ? film.frames.length - 1 : 0

  // Play: one frame every FRAME_MS, stopping on the last one.
  useEffect(() => {
    if (!playing || !film || frame >= last) return
    const timer = window.setTimeout(() => {
      setFrame(frame + 1)
      if (frame + 1 >= last) setPlaying(false)
    }, FRAME_MS)
    return () => window.clearTimeout(timer)
  }, [playing, frame, last, film])

  function pickWell(w: string) {
    setWell(w)
    setFrame(0)
    setPlaying(false)
  }

  function play() {
    if (frame >= last) setFrame(0)
    setPlaying(true)
  }

  if (loaded?.well === well && loaded.error) return <p className="empty">The film could not load: {loaded.error}</p>
  if (!film) return <p className="empty">Loading the forecasts of {well}</p>

  const current = film.frames[Math.min(frame, last)]
  const detail: BacktestCase = {
    well: film.well,
    cutoff: current.cutoff,
    horizon_end: current.horizon_end,
    actual_sm3: current.actual_sm3,
    rate: film.rate,
    methods: current.methods,
    agent: current.agent,
  }
  const domain = fixedDomain(film)
  const agent = current.methods.find((m) => m.method === 'agent')
  const classic = current.methods.find((m) => m.method === 'classic')
  const seen = film.frames.slice(0, frame + 1)
  const score = (method: string) =>
    median(seen.map((f) => f.methods.find((m) => m.method === method)?.error).filter((e): e is number => e !== undefined).map(Math.abs))
  const agentScore = score('agent')
  const classicScore = score('classic')

  return (
    <section className="film" aria-label="The forecasts in order">
      <h4 className="case-title">Watch the forecasts in order</h4>
      <p className="accuracy-lead">
        Each frame is a forecast made with the data up to that date, against what the well then produced. The axes
        stay put; only the forecasts move.
      </p>
      <div className="film-controls">
        <button type="button" className="film-play" aria-pressed={playing} onClick={() => (playing ? setPlaying(false) : play())}>
          {playing ? 'Pause' : frame >= last ? 'Play again' : 'Play'}
        </button>
        <label className="film-slider">
          <span className="sr-only">Forecast made on</span>
          <input
            type="range"
            min={0}
            max={last}
            value={Math.min(frame, last)}
            onChange={(e) => {
              setPlaying(false)
              setFrame(Number(e.target.value))
            }}
            aria-valuetext={`forecast made on ${current.cutoff}`}
          />
        </label>
        <span className="film-count">
          {Math.min(frame, last) + 1} of {film.frames.length}
        </span>
        {wells.length > 1 && (
          <select className="well-picker" value={well} onChange={(e) => pickWell(e.target.value)} aria-label="Well">
            {wells.map((w) => (
              <option key={w} value={w}>
                {w}
              </option>
            ))}
          </select>
        )}
      </div>
      <p className="film-caption" aria-live="polite">
        <strong>Data up to {current.cutoff}.</strong>{' '}
        {agent
          ? `The agent fitted from ${agent.window_start} (${agent.model}) and said ${volume(agent.p50_sm3)} (${signed(agent.error)}); `
          : 'The agent made no forecast here; '}
        {classic && `the whole-history fit said ${volume(classic.p50_sm3)} (${signed(classic.error)}); `}
        the well produced <strong>{volume(current.actual_sm3)}</strong>.
        {current.band && (
          <span className={`film-band${current.band.holds ? '' : ' is-miss'}`}>
            {' '}
            {current.band.holds && <CheckIcon />}
            Its range from past errors, {volume(current.band.low)} to {volume(current.band.high)},{' '}
            {current.band.holds ? 'held' : 'missed'}.
          </span>
        )}
      </p>
      <CaseChart detail={detail} domain={domain} />
      {agentScore !== null && classicScore !== null && (
        <p className="film-score">
          After {seen.length} {seen.length === 1 ? 'forecast' : 'forecasts'}: median error{' '}
          <strong>agent {Math.round(agentScore * 100)}%</strong>, classic {Math.round(classicScore * 100)}%.
        </p>
      )}
    </section>
  )
}

/** One x and y range for every frame of a film, so the axes do not jump between them. */
function fixedDomain(film: Film): Domain {
  const curves = film.frames.flatMap((f) => f.methods.flatMap((m) => [...m.fitted.rate, ...m.forecast.rate]))
  const starts = film.frames.flatMap((f) => f.methods.map((m) => time(m.window_start)))
  const ends = film.frames.map((f) => time(f.horizon_end) + DAY)
  const until = Math.max(...ends)
  // The y range covers the frames' years; the faint tail after the last forecast may fall below it.
  const inRange = film.rate.rate.filter((v, i) => v > 0 && time(film.rate.date[i]) <= until)
  const values = [...inRange, ...curves].filter((v) => v > 0)
  return {
    x: [Math.min(time(film.rate.date[0]), ...starts), until],
    y: [Math.log10(Math.min(...values) * 0.85), Math.log10(Math.max(...values) * 1.15)],
  }
}

function median(values: number[]): number | null {
  if (!values.length) return null
  const sorted = [...values].sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

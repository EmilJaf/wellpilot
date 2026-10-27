import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type RunSummary, type Table } from './api'
import { AgentLog } from './components/AgentLog'
import { AccuracyView } from './components/AccuracyView'
import { DesignChart, type DesignResult } from './components/DesignChart'
import { Answer, type Focus } from './components/Answer'
import { ErrorBoundary } from './components/ErrorBoundary'
import { EvidenceView } from './components/EvidenceView'
import { PlanChart, type PlanResult } from './components/PlanChart'
import { type Marker, WellCharts } from './components/WellCharts'
import { datesIn } from './labels'
import { useRun } from './useRun'

// Short labels in the ask bar; a click puts the full question in the box.
const EXAMPLES = [
  { label: 'F-12 decline vs the 2005 plan', question: 'Why did well F-12 decline faster after 2010? Compare with the 2005 development plan.' },
  { label: "F-12's missing pressure: reservoir or sensor?", question: "Is F-12's missing downhole pressure after October 2010 a reservoir problem or a sensor problem?" },
  { label: 'F-12 after the 2014 shutdown', question: "Why did F-12's oil rate jump after the December 2014 shutdown?" },
  { label: 'Forecast F-14 from 2013', question: 'Forecast F-14 oil production for the next 12 months and say how confident the forecast is.' },
]

type EventRow = { kind: string; signal: string; start: string; end: string | null; details: Record<string, unknown> }

const NO_MARKERS: Marker[] = []

function toMarkers(events: EventRow[]): Marker[] {
  return events.flatMap((e): Marker[] => {
    if (e.kind === 'gauge_failure' && e.end) {
      return [{ at: e.start, until: e.end, kind: 'gauge', track: e.signal === 'bhp_bar' ? 'bhp' : 'all', label: 'Gauge reads 0, readings removed' }]
    }
    if (e.kind === 'regime_change') {
      return [{ at: e.start, kind: 'regime', track: e.signal === 'oil_rate' ? 'oil' : 'water' }]
    }
    if (e.kind === 'threshold_crossing') return [{ at: e.start, kind: 'crossing', track: 'water' }]
    if (e.kind === 'shut_in' && e.end) return [{ at: e.start, until: e.end, kind: 'shut_in', track: 'oil' }]
    return []
  })
}

export default function App() {
  const { state, ask, open } = useRun()
  const [question, setQuestion] = useState(EXAMPLES[0].question)
  const [asOf, setAsOf] = useState('')
  const [selectedEvidence, setSelected] = useState<string | null>(null)
  const [focus, setFocus] = useState<Focus | null>(null)
  const [evidenceAttempt, setEvidenceAttempt] = useState(0)
  const evidencePanel = useRef<HTMLDivElement>(null)
  // The data column shows one well's daily data, the field against its 2005 plan, or how
  // close past forecasts came.
  const [view, setView] = useState<'well' | 'plan' | 'accuracy'>('well')
  // The reader can fold the charts away to give the evidence the whole column.
  const [chartsShown, setChartsShown] = useState(true)
  const setSelectedEvidence = (id: string | null, clicked?: Focus) => {
    setSelected(id)
    // The field and accuracy views fill the column: go back to the well, where evidence shows.
    if (id) setView('well')
    setFocus(clicked ?? null)
    setEvidenceAttempt((n) => n + 1)
    // On a narrow screen the evidence sits below the answer: bring it into view.
    if (id && window.innerWidth < 1100) {
      requestAnimationFrame(() => evidencePanel.current?.scrollIntoView({ block: 'start', behavior: 'smooth' }))
    }
  }
  const [selectedFinding, setSelectedFinding] = useState<number | null>(null)
  const [pickedWell, setPickedWell] = useState<string | null>(null)
  const [wells, setWells] = useState<string[]>([])
  const [daily, setDaily] = useState<{ well: string; table: Table | null; error?: string } | null>(null)
  const [markers, setMarkers] = useState<{ key: string; list: Marker[] }>({ key: '', list: [] })
  const [recorded, setRecorded] = useState<RunSummary | null>(null)
  const [plan, setPlan] = useState<{ result: PlanResult | null; design: DesignResult | null; error?: string } | null>(null)

  useEffect(() => {
    if (view !== 'plan' || plan) return
    Promise.all([api.fieldPlan(), api.fieldDesign().catch(() => null)])
      .then(([result, design]) => setPlan({ result: result as PlanResult, design: design as DesignResult | null }))
      .catch((e: Error) => setPlan({ result: null, design: null, error: e.message }))
  }, [view, plan])

  const running = state.status === 'starting' || state.status === 'running'

  // The well the agent is working on, unless the reader picked another one.
  const agentWell = useMemo(() => {
    for (const step of state.steps) {
      for (const call of step.tool_calls) {
        // Only calls that succeeded: a typo like "F12" must not blank the charts.
        if (call.evidence_id && typeof call.arguments.well === 'string') return call.arguments.well
      }
    }
    return null
  }, [state.steps])
  const well = pickedWell ?? agentWell ?? 'F-12'

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const runId = params.get('run')
    if (runId) open(runId, params.get('replay') === '1')
  }, [open])

  useEffect(() => {
    api
      .runs()
      .then((runs) => setRecorded(runs.find((r) => r.status === 'done' && r.verified) ?? null))
      .catch(() => setRecorded(null))
    fetch('/api/wells')
      .then((r) => r.json() as Promise<Table>)
      .then((t) => {
        const oil = t.columns.indexOf('cum_oil_sm3')
        setWells(t.rows.filter((r) => Number(r[oil]) > 0).map((r) => String(r[0])))
      })
      .catch(() => setWells([]))
  }, [])

  useEffect(() => {
    let current = true
    api
      .daily(well)
      .then((table) => current && setDaily({ well, table }))
      .catch((e: Error) => current && setDaily({ well, table: null, error: e.message }))
    return () => {
      current = false
    }
  }, [well])

  // Events the agent detected for this well become marks on the charts.
  const eventsEvidence = useMemo(() => {
    for (const step of state.steps) {
      for (const call of step.tool_calls) {
        if (call.name === 'detect_events' && call.arguments.well === well && call.evidence_id) {
          return call.evidence_id
        }
      }
    }
    return null
  }, [state.steps, well])

  useEffect(() => {
    if (!state.id || !eventsEvidence) return
    const key = `${state.id}/${eventsEvidence}`
    let current = true
    api
      .evidence(state.id, eventsEvidence)
      .then((e) => current && setMarkers({ key, list: toMarkers(e.result as EventRow[]) }))
      .catch(() => {})
    return () => {
      current = false
    }
  }, [state.id, eventsEvidence])

  const finding = selectedFinding !== null ? state.final?.findings[selectedFinding] : undefined
  // Stable arrays, so the charts redraw only when a selection or the markers change.
  const highlights = useMemo(() => (finding ? datesIn(finding.claim) : []), [finding])
  const markersKey = eventsEvidence ? `${state.id}/${eventsEvidence}` : ''
  const chartMarkers = useMemo(
    () => (markersKey && markers.key === markersKey ? markers.list : NO_MARKERS),
    [markersKey, markers],
  )

  const reset = () => {
    setSelectedEvidence(null)
    setSelectedFinding(null)
    setPickedWell(null)
  }

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!question.trim() || running) return
    reset()
    const runId = await ask(question.trim(), asOf || undefined)
    if (runId) window.history.replaceState(null, '', `?run=${runId}`)
  }

  const replay = (runId: string) => {
    reset()
    open(runId, true)
    window.history.replaceState(null, '', `?run=${runId}&replay=1`)
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1 className="wordmark">WellPilot</h1>
          <p className="tagline">
            Well analysis by an AI agent whose every number is checked against the field data. On
            Meta&rsquo;s open-weight Muse Spark Open.
          </p>
        </div>
        <form className="ask" onSubmit={submit}>
          <div className="ask-row">
            <label className="sr-only" htmlFor="question">
              Question
            </label>
            <input
              id="question"
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about a well, e.g. why F-12 declined after 2010"
            />
            <label className="as-of" title="For backtests: the agent sees no data after this day, as if it were that day.">
              Data up to
              <input type="date" value={asOf} min="2008-01-01" max="2016-12-01" onChange={(e) => setAsOf(e.target.value)} />
            </label>
            <button className="run-button" type="submit" disabled={running}>
              {running ? 'Running' : 'Ask the agent'}
            </button>
          </div>
          <div className="suggestions">
            <span className="suggestions-label">Try</span>
            {EXAMPLES.map((e) => (
              <button key={e.label} type="button" className="suggestion" title={e.question} onClick={() => setQuestion(e.question)}>
                {e.label}
              </button>
            ))}
            {recorded && (
              <button type="button" className="link-button" onClick={() => replay(recorded.id)} disabled={running}>
                or watch a recorded run
              </button>
            )}
          </div>
        </form>
      </header>

      <main className="console">
        {/* One boundary per panel: a bad evidence render must not hide the live log. The
            answer comes first in reading order; on a wide screen it sits in the middle. */}
        <ErrorBoundary>
          <Answer
            state={state}
            recorded={recorded}
            selectedFinding={selectedFinding}
            selectedEvidence={selectedEvidence}
            onSelectFinding={(i) => {
              setSelectedFinding(i)
              const first = i !== null ? state.final?.findings[i]?.evidence_ids[0] : undefined
              if (first) setSelectedEvidence(first)
            }}
            onSelectEvidence={setSelectedEvidence}
            onReplay={replay}
          />
        </ErrorBoundary>

        <ErrorBoundary>
          <section aria-label="Well data and evidence" className="center">
            <div className="panel-head">
              <div className="tabs" role="tablist" aria-label="Data to show">
                <button type="button" role="tab" aria-selected={view === 'well'} onClick={() => setView('well')}>
                  Well {well}
                </button>
                <button type="button" role="tab" aria-selected={view === 'plan'} onClick={() => setView('plan')}>
                  Field vs plan
                </button>
                <button type="button" role="tab" aria-selected={view === 'accuracy'} onClick={() => setView('accuracy')}>
                  Forecast accuracy
                </button>
              </div>
              {view === 'well' && (
                <button type="button" className="link-button charts-toggle" aria-pressed={!chartsShown} onClick={() => setChartsShown(!chartsShown)}>
                  {chartsShown ? 'Hide charts' : 'Show charts'}
                </button>
              )}
              {view === 'well' && (
                <select
                  className="well-picker"
                  aria-label="Well to plot"
                  value={well}
                  onChange={(e) => setPickedWell(e.target.value)}
                >
                  {[...new Set([...wells, well])].map((w) => (
                    <option key={w} value={w}>
                      {w}
                    </option>
                  ))}
                </select>
              )}
            </div>
            {view === 'well' && chartsShown && (
              <WellCharts
                table={daily?.well === well ? daily.table : null}
                error={daily?.well === well ? daily.error : undefined}
                markers={chartMarkers}
                highlights={highlights}
              />
            )}
            <div className="evidence-scroll" ref={evidencePanel}>
              {view === 'accuracy' ? (
                <AccuracyView />
              ) : view === 'plan' ? (
                <div className="charts field-view">
                  {plan?.result ? (
                    <>
                      <h3 className="accuracy-title">The field against its 2005 plan</h3>
                      <PlanChart result={plan.result} />
                      {plan.design && (
                        <>
                          <h3 className="accuracy-title field-view-second">Against the limits the platform was designed for</h3>
                          <DesignChart result={plan.design} />
                        </>
                      )}
                    </>
                  ) : (
                    <p className="empty">{plan?.error ? `The plan comparison could not load: ${plan.error}` : 'Loading the plan comparison'}</p>
                  )}
                </div>
              ) : (
                <EvidenceView runId={state.id} evidenceId={selectedEvidence} attempt={evidenceAttempt} focus={focus} />
              )}
            </div>
          </section>
        </ErrorBoundary>

        <ErrorBoundary>
          <AgentLog state={state} selectedEvidence={selectedEvidence} onSelectEvidence={setSelectedEvidence} />
        </ErrorBoundary>
      </main>
      <footer className="credits">
        Volve data by Equinor and the former Volve license partners (ExxonMobil Exploration &amp;
        Production Norway AS, Bayerngas Norge AS) under the{' '}
        <a href="https://www.equinor.com/content/dam/statoil/documents/what-we-do/Equinor-HRS-Terms-and-conditions-for-licence-to-data-Volve.pdf">
          Equinor Open Data Licence
        </a>
        . Not affiliated with or endorsed by Equinor; analyses are generated by an AI agent and
        checked against the data, not reviewed by Equinor.
      </footer>
    </div>
  )
}

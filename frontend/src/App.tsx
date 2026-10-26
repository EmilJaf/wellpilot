import { useEffect, useMemo, useState } from 'react'
import { api, type RunSummary, type Table } from './api'
import { AgentLog } from './components/AgentLog'
import { Answer } from './components/Answer'
import { EvidenceView } from './components/EvidenceView'
import { type Marker, WellCharts } from './components/WellCharts'
import { datesIn } from './labels'
import { useRun } from './useRun'

const EXAMPLES = [
  'Why did well F-12 decline faster after 2010? Compare with the 2005 development plan.',
  "Is F-12's missing downhole pressure after October 2010 a reservoir problem or a sensor problem?",
  "Why did F-12's oil rate jump after the December 2014 shutdown?",
]

type EventRow = { kind: string; signal: string; start: string; end: string | null; details: Record<string, unknown> }

function toMarkers(events: EventRow[]): Marker[] {
  return events.flatMap((e): Marker[] => {
    if (e.kind === 'gauge_failure' && e.end) {
      return [{ at: e.start, until: e.end, kind: 'gauge', track: e.signal === 'bhp_bar' ? 'bhp' : 'all', label: 'Gauge reads 0, readings removed' }]
    }
    if (e.kind === 'regime_change') {
      return [{ at: e.start, kind: 'regime', track: e.signal === 'oil_rate' ? 'oil' : 'water' }]
    }
    if (e.kind === 'threshold_crossing') return [{ at: e.start, kind: 'crossing', track: 'water' }]
    return []
  })
}

export default function App() {
  const { state, ask, open } = useRun()
  const [question, setQuestion] = useState(EXAMPLES[0])
  const [asOf, setAsOf] = useState('')
  const [selectedEvidence, setSelectedEvidence] = useState<string | null>(null)
  const [selectedFinding, setSelectedFinding] = useState<number | null>(null)
  const [pickedWell, setPickedWell] = useState<string | null>(null)
  const [wells, setWells] = useState<string[]>([])
  const [daily, setDaily] = useState<{ well: string; table: Table } | null>(null)
  const [markers, setMarkers] = useState<{ key: string; list: Marker[] }>({ key: '', list: [] })
  const [recorded, setRecorded] = useState<RunSummary | null>(null)

  const running = state.status === 'starting' || state.status === 'running'

  // The well the agent is working on, unless the reader picked another one.
  const agentWell = useMemo(() => {
    for (const step of state.steps) {
      for (const call of step.tool_calls) {
        if (typeof call.arguments.well === 'string') return call.arguments.well
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
      .catch(() => current && setDaily(null))
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
  const highlights = finding ? datesIn(finding.claim) : []
  const chartMarkers = eventsEvidence && markers.key === `${state.id}/${eventsEvidence}` ? markers.list : []

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
        <div>
          <h1 className="wordmark">WellPilot</h1>
          <p className="wordmark-sub">
            An AI agent for well performance on Equinor's open Volve field data, running on Meta
            Muse Spark
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
            <label className="as-of">
              Hide data after
              <input type="date" value={asOf} min="2008-01-01" max="2016-12-01" onChange={(e) => setAsOf(e.target.value)} />
            </label>
            <button className="run-button" type="submit" disabled={running}>
              {running ? 'Running' : 'Run analysis'}
            </button>
          </div>
          <div className="suggestions">
            {EXAMPLES.map((q) => (
              <button key={q} type="button" className="link-button" onClick={() => setQuestion(q)}>
                {q}
              </button>
            ))}
            {recorded && (
              <button type="button" className="link-button" onClick={() => replay(recorded.id)} disabled={running}>
                Replay a recorded run
              </button>
            )}
          </div>
        </form>
      </header>

      <main className="console">
        <AgentLog state={state} selectedEvidence={selectedEvidence} onSelectEvidence={setSelectedEvidence} />

        <section aria-label="Well data and evidence" className="center">
          <div className="panel-head">
            <h2>Well {well}</h2>
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
          </div>
          <WellCharts table={daily?.well === well ? daily.table : null} markers={chartMarkers} highlights={highlights} />
          <div className="evidence-scroll">
            <EvidenceView runId={state.id} evidenceId={selectedEvidence} />
          </div>
        </section>

        <Answer
          state={state}
          selectedFinding={selectedFinding}
          selectedEvidence={selectedEvidence}
          onSelectFinding={(i) => {
            setSelectedFinding(i)
            const first = i !== null ? state.final?.findings[i]?.evidence_ids[0] : undefined
            if (first) setSelectedEvidence(first)
          }}
          onSelectEvidence={setSelectedEvidence}
        />
      </main>
    </div>
  )
}

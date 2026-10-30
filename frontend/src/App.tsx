import { useEffect, useEffectEvent, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { api, type OperationJob, type RunSummary, type Span as TextSpan, type Table } from './api'
import { AgentLog } from './components/AgentLog'
import { StoryBar } from './components/StoryBar'
import { STEPS, type Step } from './story'
import { WorkBar } from './components/WorkBar'
import { PlanFigure } from './components/PlanFigure'
import { FieldTimeline } from './components/FieldTimeline'
import { InsideWell } from './components/InsideWell'
import { AccuracyView } from './components/AccuracyView'
import { DesignChart, type DesignResult } from './components/DesignChart'
import { FieldScreen } from './components/FieldScreen'
import { HindsightView } from './components/Hindsight'
import { Answer, type Focus } from './components/Answer'
import { ErrorBoundary } from './components/ErrorBoundary'
import { EvidenceView } from './components/EvidenceView'
import { PlanByProductionYear, PlanChart, type PlanResult } from './components/PlanChart'
import { type Marker, type Span, WellCharts } from './components/WellCharts'
import { canRecord, startRecording, toBase64, toWav16k, type Recording } from './audio'
import { MicIcon, StopIcon } from './icons'
import { datesIn } from './labels'
import { useRun } from './useRun'

// Short labels in the ask bar; a click puts the full question in the box, and the date the
// agent's data stops at (none: all of it).
const EXAMPLES = [
  {
    label: 'Back to the end of 2010: which well needs attention, and its next year',
    question: 'Which wells need attention right now, and why? Forecast the next 12 months of oil for the well that needs it most.',
    asOf: '2010-12-31',
  },
  {
    label: 'F-12 in 2014, nearly all water: what work to do on it, then what the operator did',
    question:
      "F-12's water cut is near 0.97 and its oil rate has fallen below 200 Sm3/d. What would you recommend doing to the well, and why?",
    asOf: '2014-10-31',
  },
  { label: 'Why F-12 declined faster than the 2005 plan expected', question: 'Why did well F-12 decline faster after 2010? Compare with the 2005 development plan.', asOf: '' },
  { label: "Read a chart in the 2005 plan, cross-checked by a second Meta model (SAM 3.1)", question: 'What oil plateau rate and duration does the forecast figure on page 23 of the 2005 development plan show?', asOf: '' },
  {
    label: 'Why the 2014 plug cut F-12\'s water and the 2010 one did not, from production logs',
    question:
      "F-12's water cut fell from about 0.97 to about 0.3 at the end of 2014. What was done to the well, and why did the plug set in 2010 not do the same?",
    asOf: '',
  },
  {
    label: 'Water cut against choke on F-12: the agent writes and runs its own Python',
    question:
      "Over 2011 to 2013, how closely did F-12's monthly water cut follow its choke opening? Give the correlation and what it suggests.",
    asOf: '',
  },
]

// Questions to run live that the agent answered verified on its first or second check in the
// latest eval (Oct 31), so a visitor's first live run is likely to end checked. A click fills
// the box; Ask the agent runs it (tokens, about a minute).
const LIVE = [
  {
    label: "Dead gauge or dying reservoir? F-12's pressure after October 2010",
    question: "Is F-12's missing downhole pressure after October 2010 a reservoir problem or a sensor problem?",
    asOf: '',
  },
  {
    label: "When the field reached the water cut the 2005 plan designed it for",
    question: 'When did the Volve field reach the maximum water cut the 2005 plan designed the platform for?',
    asOf: '',
  },
  {
    label: 'F-12 in 2011-2013: pressure loss or water taking over? (data up to 2013)',
    question:
      "Was F-12's falling oil rate in 2011-2013 a loss of reservoir pressure or water taking over the well? How much oil did its water-oil ratio trend leave before a 95% water cut?",
    asOf: '2013-12-31',
  },
]

type EventRow = { kind: string; signal: string; start: string; end: string | null; details: Record<string, unknown> }

const NO_MARKERS: Marker[] = []
const PAGE_TOOLS = new Set(['look_at_page', 'read_report_page'])

type View = 'screen' | 'history' | 'well' | 'inside' | 'plan' | 'accuracy' | 'hindsight'
type Group = 'field' | 'well' | 'plan' | 'forecasts'
// Seven views under four tabs: a reader meets four names, the rest is one click inside.
const GROUP: Record<View, Group> = {
  screen: 'field',
  history: 'field',
  well: 'well',
  inside: 'well',
  plan: 'plan',
  accuracy: 'forecasts',
  hindsight: 'forecasts',
}
const TABS: { group: Group; label: string }[] = [
  { group: 'field', label: 'Field' },
  { group: 'well', label: 'Well' },
  { group: 'plan', label: '2005 plan' },
  { group: 'forecasts', label: 'Forecasts' },
]
const FIRST: Record<Group, View> = { field: 'screen', well: 'well', plan: 'plan', forecasts: 'accuracy' }
const SUBS: Record<Group, { view: View; label: string }[]> = {
  field: [
    { view: 'screen', label: 'All wells' },
    { view: 'history', label: 'History' },
  ],
  well: [
    { view: 'well', label: 'Production' },
    { view: 'inside', label: 'Inside the well' },
  ],
  plan: [],
  forecasts: [
    { view: 'hindsight', label: 'What happened next' },
    { view: 'accuracy', label: 'Past forecasts' },
  ],
}

const stepFromUrl = (): Step | null => {
  const n = Number(new URLSearchParams(window.location.search).get('step'))
  return STEPS.some((s) => s.step === n) ? (n as Step) : null
}

/** Bring a part of the story column into view, once it has rendered. */
// Retried for a moment: the target may render a frame or two after the step changes (the
// checker panel under a just-finished answer), and a page opened at an address keeps laying
// out (charts, evidence) after the first scroll, which can put the page back at the top. So
// it scrolls, looks again a moment later, and scrolls again if the page went back to the
// top; a reader who scrolled meanwhile is left where they are. Instant, not smooth: a smooth
// scroll is cut short by that layout.
const scrollToId = (id: string, tries = 20, at: number | null = null): void => {
  requestAnimationFrame(() => {
    const el = document.getElementById(id)
    if (at !== null && window.scrollY > 0) return // in place, or the reader moved on
    if (el) el.scrollIntoView({ block: 'start' })
    if (tries > 0) window.setTimeout(() => scrollToId(id, tries - 1, el ? window.scrollY : null), el ? 250 : 100)
  })
}

/** A date typed as YYYY-MM-DD, the way the data and the answers write it (a browser's date
 * picker shows the reader's locale, 31.12.2010 or 12/31/2010). Empty means all the data. */
function IsoDate({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  // What the reader is typing, until it is a whole date (or they leave the field).
  const [draft, setDraft] = useState<string | null>(null)
  const text = draft ?? value
  const valid = (t: string) => t === '' || (/^\d{4}-\d{2}-\d{2}$/.test(t) && t >= '2008-01-01' && t <= '2016-12-01')
  return (
    <input
      type="text"
      inputMode="numeric"
      placeholder="YYYY-MM-DD"
      aria-label="Data up to, YYYY-MM-DD; empty for all the data"
      aria-invalid={!valid(text)}
      value={text}
      size={10}
      onChange={(e) => {
        const typed = e.target.value.trim()
        if (valid(typed)) {
          onChange(typed)
          setDraft(null)
        } else setDraft(e.target.value)
      }}
      onBlur={() => setDraft(null)}
    />
  )
}

/** A job on the well in a few words, from what its reports say was done. */
function workLabel(job: OperationJob): string {
  const text = job.summaries.map((s) => s.summary).join(' ')
  const done = [
    /\bplug\b/i.test(text) && 'plug',
    (job.perforations.length > 0 || /perforat/i.test(text)) && 'perforation',
    /\bPLT\b|production log/i.test(text) && 'production log',
  ].filter(Boolean)
  return done.length ? `${done.join(', ')} (${job.start})` : `well work (${job.start})`
}

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
  const { state, ask, open, clear } = useRun()
  const [question, setQuestion] = useState(EXAMPLES[0].question)
  const [asOf, setAsOf] = useState(EXAMPLES[0].asOf)
  const [selectedEvidence, setSelected] = useState<string | null>(null)
  const [focus, setFocus] = useState<Focus | null>(null)
  const [evidenceAttempt, setEvidenceAttempt] = useState(0)
  const evidencePanel = useRef<HTMLDivElement>(null)
  // The data column shows one well's daily data, the field against its 2005 plan, or how
  // close past forecasts came.
  // A first visit opens on F-12's charts up to the date: the puzzle (the pressure line
  // stops in October 2010 while oil flows on), not the field screen's verdict on it, which
  // would answer the question before the agent does.
  const [view, setView] = useState<View>('well')
  // A run about the plan alone (its chart, its limits) opens on the plan, once.
  const [viewFor, setViewFor] = useState<string | null>(null)
  if (state.id && state.final && viewFor !== state.id) {
    setViewFor(state.id)
    const used = new Set(state.steps.flatMap((s) => s.tool_calls.map((c) => c.name)))
    const planOnly = ['look_at_page', 'plan_vs_actual', 'field_vs_design'].some((t) => used.has(t))
    const wellTools = ['production_monthly', 'detect_events', 'well_operations', 'production_log', 'fit_decline', 'forecast_production', 'waterflood', 'run_python']
    if (planOnly && !wellTools.some((t) => used.has(t))) setView('plan')
  }
  // The demo's five steps: which one the reader is on, and whether the agent's log is open.
  const [step, setStep] = useState<Step>(() => {
    const params = new URLSearchParams(window.location.search)
    return stepFromUrl() ?? (params.get('replay') === '1' ? 2 : params.get('run') ? 3 : 1)
  })
  const [logOpen, setLogOpen] = useState(false)
  // Set by Back: the address it lands on replaces the history entry instead of adding one.
  const replaceNext = useRef(false)
  // The ask bar opens on request ("Ask your own question"); otherwise the open run's question
  // shows as a line, and a first visit sees one thing to do: watch it work.
  const [askOpen, setAskOpen] = useState(false)
  // While a run plays, the data pane shows what the agent is running (its latest tool result),
  // until the reader opens something there themselves.
  const [paneFollows, setPaneFollows] = useState(false)
  // A step that needs the recorded run to finish loading first (the reveal, the checker).
  const [pending, setPending] = useState<Step | null>(null)
  // The reader can fold the charts away to give the evidence the whole column.
  const [chartsShown, setChartsShown] = useState(true)
  const setSelectedEvidence = (id: string | null, clicked?: Focus) => {
    setPaneFollows(false)
    setSelected(id)
    // A clicked number or quote opens its source with the value marked, and a result that is
    // a page (the model's reading of one, a page of the plan): fold the charts away so the
    // source shows large. "Show charts" brings them back.
    const tool = state.steps.flatMap((step) => step.tool_calls).find((c) => c.evidence_id === id)?.name
    if (clicked || (tool && PAGE_TOOLS.has(tool))) setChartsShown(false)
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
  // A span the reader zoomed into on the charts, offered as a question about that well.
  const [span, setSpan] = useState<(Span & { well: string }) | null>(null)
  // Ask by voice: the mic shows when the browser can record and the server takes audio.
  const [voiceOn, setVoiceOn] = useState<boolean | null>(() => (canRecord() ? null : false))
  const [voice, setVoice] = useState<{ status: 'idle' | 'recording' | 'transcribing'; seconds: number; note?: string }>({
    status: 'idle',
    seconds: 0,
  })
  const recording = useRef<Recording | null>(null)
  const questionBox = useRef<HTMLTextAreaElement>(null)
  // The box is as tall as the question it holds.
  useLayoutEffect(() => {
    const box = questionBox.current
    if (!box) return
    box.style.height = 'auto'
    box.style.height = `${box.scrollHeight + 2}px`
  }, [question, askOpen, state.status])
  const [wells, setWells] = useState<string[]>([])
  const [daily, setDaily] = useState<{ well: string; table: Table | null; error?: string } | null>(null)
  const [work, setWork] = useState<{ key: string; list: Marker[] }>({ key: '', list: [] })
  const [markers, setMarkers] = useState<{ key: string; list: Marker[] }>({ key: '', list: [] })
  const [recorded, setRecorded] = useState<RunSummary | null>(null)
  // Every verified recorded run: an example whose question has one opens it at once (no
  // tokens, no wait); "Ask the agent" still runs any question live.
  const [answered, setAnswered] = useState<RunSummary[]>([])
  const [runsLoaded, setRunsLoaded] = useState(false)
  const [plan, setPlan] = useState<{
    end: string | null
    result: PlanResult | null
    design: DesignResult | null
    error?: string
  } | null>(null)

  const running = state.status === 'starting' || state.status === 'running'
  // A question the server would not start stays in its box, to try again or change.
  const refused = state.status === 'error' && state.steps.length === 0
  const askShown = askOpen || refused
  // The time machine: the date the agent's data stops at (the open run's, or the one in the
  // box before a run). Its data after that stays hidden until the reader asks to see it.
  const pastDate = state.id ? state.asOf : asOf || null
  const [revealed, setRevealed] = useState(false)
  const canReveal = Boolean(state.id && state.asOf && state.final)
  const reveal = () => {
    setRevealed(true)
    setSelected(null)
    setView('hindsight')
    setStep(4)
    // On a narrow screen the data panel sits below the answer: bring it into view once the
    // reveal has loaded (scrolled earlier, the answer above it was still changing height).
    scrollOnReveal.current = window.innerWidth < 1100
  }
  const scrollOnReveal = useRef(false)
  const revealReady = () => {
    if (!scrollOnReveal.current) return
    scrollOnReveal.current = false
    requestAnimationFrame(() => evidencePanel.current?.scrollIntoView({ block: 'start' }))
  }
  // What the field views may show: before the reveal, nothing after the time machine's date.
  const shownEnd = pastDate && !revealed ? pastDate : null

  useEffect(() => {
    if (view !== 'plan' || (plan && plan.end === shownEnd)) return
    let current = true
    Promise.all([api.fieldPlan(shownEnd), api.fieldDesign(shownEnd).catch(() => null)])
      .then(([result, design]) => {
        if (current) setPlan({ end: shownEnd, result: result as PlanResult, design: design as DesignResult | null })
      })
      .catch((e: Error) => {
        if (current) setPlan({ end: shownEnd, result: null, design: null, error: e.message })
      })
    return () => {
      current = false
    }
  }, [view, plan, shownEnd])

  useEffect(() => {
    let current = true
    fetch(`/api/wells${shownEnd ? `?end=${shownEnd}` : ''}`)
      .then((r) => r.json() as Promise<Table>)
      .then((t) => {
        const oil = t.columns.indexOf('cum_oil_sm3')
        if (current) setWells(t.rows.filter((r) => Number(r[oil]) > 0).map((r) => String(r[0])))
      })
      .catch(() => current && setWells([]))
    return () => {
      current = false
    }
  }, [shownEnd])

  // An opened run's question and date go in the ask bar, so the bar never shows another day.
  const [inBar, setInBar] = useState<string | null>(null)
  if (state.id && state.question && state.id !== inBar) {
    setInBar(state.id)
    setQuestion(state.question)
    setAsOf(state.asOf ?? '')
  }

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
    if (!canRecord()) return
    api
      .health()
      .then((h) => setVoiceOn(h.voice))
      .catch(() => setVoiceOn(false))
  }, [])

  useEffect(() => {
    api
      .runs()
      // The time machine first: a verified run of the first example's question and date.
      .then((runs) => {
        const verified = runs.filter((r) => r.status === 'done' && r.verified)
        setAnswered(verified)
        const machine = verified.find((r) => r.question === EXAMPLES[0].question && r.as_of === EXAMPLES[0].asOf)
        setRecorded(machine ?? verified.find((r) => r.as_of) ?? verified[0] ?? null)
      })
      .catch(() => setRecorded(null))
      .finally(() => setRunsLoaded(true))
  }, [])

  useEffect(() => {
    let current = true
    const end = pastDate && !revealed ? pastDate : null
    api
      .daily(well, end)
      .then((table) => current && setDaily({ well, table }))
      .catch((e: Error) => current && setDaily({ well, table: null, error: e.message }))
    // Work done on the well after first oil (plugs, perforations, logs), as chart marks;
    // drilling is before production and stays off the charts.
    api
      .operations(well, end)
      .then(
        (ops) =>
          current &&
          setWork({
            key: `${well}|${end}`,
            list: ops.jobs
              .filter((j) => j.kind === 'workover')
              .map((j) => ({ at: j.start, kind: 'work' as const, track: 'all' as const, label: workLabel(j) })),
          }),
      )
      .catch(() => current && setWork({ key: `${well}|${end}`, list: [] }))
    return () => {
      current = false
    }
  }, [well, pastDate, revealed])

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

  // The latest tool result of a run in progress, for the data pane to show.
  const latestCall = useMemo(() => {
    const calls = state.steps.flatMap((s) => s.tool_calls)
    return [...calls].reverse().find((c) => c.evidence_id && !['submit_findings', 'calculate', 'list_wells'].includes(c.name)) ?? null
  }, [state.steps])
  const [shownCall, setShownCall] = useState<string | null>(null)
  if (paneFollows && running && latestCall?.evidence_id && latestCall.evidence_id !== shownCall) {
    setShownCall(latestCall.evidence_id)
    if (latestCall.name === 'field_screen') setView('screen')
    else if (latestCall.name === 'production_log') setView('inside')
    else {
      setView('well')
      setSelected(latestCall.evidence_id)
      setFocus(null)
      setChartsShown(!PAGE_TOOLS.has(latestCall.name))
      setEvidenceAttempt((n) => n + 1)
    }
  }

  const finding = selectedFinding !== null ? state.final?.findings[selectedFinding] : undefined
  // Stable arrays, so the charts redraw only when a selection or the markers change.
  const highlights = useMemo(() => (finding ? datesIn(finding.claim) : []), [finding])
  const markersKey = eventsEvidence ? `${state.id}/${eventsEvidence}` : ''
  const workKey = `${well}|${pastDate && !revealed ? pastDate : null}`
  const chartMarkers = useMemo(() => {
    const events = markersKey && markers.key === markersKey ? markers.list : NO_MARKERS
    const list = work.key === workKey && work.list.length ? [...events, ...work.list] : events
    if (!(revealed && pastDate)) return list
    return [...list, { at: pastDate, kind: 'hidden' as const, track: 'all' as const, label: `Hidden from the agent after ${pastDate}` }]
  }, [markersKey, markers, revealed, pastDate, work, workKey])

  const reset = () => {
    setSelectedEvidence(null)
    setSelectedFinding(null)
    setPickedWell(null)
    setRevealed(false)
    setPending(null)
    if (view === 'hindsight') setView('screen')
  }

  // A run the reader starts or replays moves the story on: working is step 2, its answer
  // step 3. A run opened already finished does not.
  const [moving, setMoving] = useState(false)
  const follow = () => {
    setMoving(true)
    setStep(2)
    setLogOpen(true)
    setPaneFollows(true)
    setShownCall(null)
  }

  // The answer's first finding, selected: its dates on the charts, its source in the pane.
  // The first number in it is marked there, so the reader sees where an answer's number comes
  // from before trying it.
  const showFirstFinding = () => {
    const finding = state.final?.findings[0]
    const first = finding?.evidence_ids[0]
    if (!finding || !first) return
    const number = finding.spans?.find((sp): sp is Extract<TextSpan, { kind: 'number' }> => sp.kind === 'number')
    setSelectedFinding(0)
    setPaneFollows(false)
    setView('well')
    // A source that is a page (the plan's chart read by the model) shows large, with the
    // well's charts folded away: they are not what the answer is about.
    const source = number?.evidence_id ?? first
    const tool = state.steps.flatMap((step) => step.tool_calls).find((c) => c.evidence_id === source)?.name
    setChartsShown(!(tool && PAGE_TOOLS.has(tool)))
    setSelected(source)
    setFocus(number ? { text: finding.claim.slice(number.start, number.end), value: number.value } : null)
    setEvidenceAttempt((n) => n + 1)
  }

  // Trying to fool the checker: the finding being edited, its source open beside it.
  const showFoolSource = () => {
    const index = selectedFinding ?? 0
    const first = state.final?.findings[index]?.evidence_ids[0]
    if (!first) return
    setSelectedFinding(index)
    setPaneFollows(false)
    setView('well')
    setChartsShown(false)
    setSelected(first)
    setFocus(null)
    setEvidenceAttempt((n) => n + 1)
  }

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!question.trim() || running) return
    reset()
    setAskOpen(false)
    follow()
    await ask(question.trim(), asOf || undefined)
  }

  const replay = (runId: string) => {
    reset()
    open(runId, true)
    follow()
  }

  const [seenStatus, setSeenStatus] = useState(state.status)
  if (state.status !== seenStatus) {
    setSeenStatus(state.status)
    if (moving && (state.status === 'done' || state.status === 'error')) {
      setMoving(false)
      // The answer is in: the log folds to its one line so the answer is in sight. A run that
      // failed keeps its log open; it says where the run stopped.
      if (state.status === 'done') {
        setStep(3)
        setLogOpen(false)
        showFirstFinding()
      } else if (state.steps.length === 0) setStep(1) // refused before it started: back to the question
    }
  }

  // Steps already seen get a tick in the story bar.
  const [visited, setVisited] = useState<ReadonlySet<Step>>(() => new Set())
  const [seenStep, setSeenStep] = useState(step)
  if (step !== seenStep) {
    setSeenStep(step)
    setVisited(new Set([...visited, seenStep]))
  }

  // Any step opens on its own. Steps after the first need a run: the open one, else the one
  // the address names, else the recorded time-machine run.
  const goToStep = (target: Step, named: string | null = null) => {
    setStep(target)
    const load = !state.id && !named && recorded ? recorded.id : null
    if (target === 1) {
      // The screen before the question: a finished answer would make step 1 look like step 3.
      if (state.status !== 'running' && state.status !== 'starting') {
        reset()
        clear()
      }
      setQuestion(EXAMPLES[0].question)
      setAsOf(EXAMPLES[0].asOf)
      setLogOpen(false)
      setAskOpen(false)
      setView('well')
      setChartsShown(true)
      scrollToId('answer')
      return
    }
    if (target === 2) {
      if (load) replay(load)
      else setLogOpen(true)
      scrollToId('work')
      return
    }
    if (load) {
      reset()
      open(load, false)
    }
    setLogOpen(false)
    if (target === 3) scrollToId('answer')
    setPending(target) // once the run's answer is in
  }

  // Steps 4 and 5 act on a finished answer: open what the agent could not see (or, for a run
  // with no past date, how close past forecasts came), or the checker to fool.
  const finishStep = useEffectEvent(() => {
    if (!pending || state.status !== 'done' || !state.final) return
    setPending(null)
    if (pending === 3) showFirstFinding()
    else if (pending === 5) {
      showFoolSource()
      scrollToId('fool')
    } else if (state.asOf) reveal()
    else setView('accuracy')
  })
  // eslint-disable-next-line react/set-state-in-effect -- the answer comes in from the run's stream
  useEffect(() => finishStep(), [pending, state.status, state.final])

  // The address opens what it names: a run (?run=, replayed with &replay=1) and a step
  // (?step=N, for a judge coming back or a video take). Safe to run twice, as React's strict
  // mode does: the run hook stops its stream on a (simulated) unmount.
  const openFromUrl = useEffectEvent(() => {
    const params = new URLSearchParams(window.location.search)
    const runId = params.get('run')
    const target = stepFromUrl()
    if (!runId) return
    if (params.get('replay') === '1' || target === 2) replay(runId)
    else open(runId, false)
    if (target && target !== 2) goToStep(target, runId)
  })
  // eslint-disable-next-line react/set-state-in-effect -- the address is outside React
  useEffect(() => openFromUrl(), [])
  // With no run named, a later step opens on the recorded time-machine run once its id is in.
  const stepFromUrlOnRecorded = useEffectEvent(() => {
    const target = stepFromUrl()
    if (!runsLoaded || !target || target < 2 || new URLSearchParams(window.location.search).get('run')) return
    goToStep(target)
  })
  // eslint-disable-next-line react/set-state-in-effect -- the address is outside React
  useEffect(() => stepFromUrlOnRecorded(), [runsLoaded])

  // ...and keeps saying where the reader is, so a reload or a shared link lands there. Each step
  // the reader moves to is a history entry, so Back returns to the step before; a step the page
  // moves to by itself (a run finishing) and a step reached by Back replace the entry instead.
  useEffect(() => {
    if (!state.id && new URLSearchParams(window.location.search).get('run')) return // still loading it
    const search = state.id ? `?run=${state.id}&step=${step}` : step > 1 ? `?step=${step}` : ''
    if (search !== window.location.search) {
      const url = search || window.location.pathname
      // A run that finishes moves from step 2 to 3 by itself.
      const finished = state.id && window.location.search === `?run=${state.id}&step=2` && step === 3
      if (replaceNext.current || finished) window.history.replaceState(null, '', url)
      else window.history.pushState(null, '', url)
    }
    replaceNext.current = false
  }, [state.id, step])
  const onBack = useEffectEvent(() => {
    replaceNext.current = true
    const params = new URLSearchParams(window.location.search)
    const runId = params.get('run')
    const target = stepFromUrl() ?? (runId ? 3 : 1)
    if (target === 1 || !runId) {
      goToStep(target)
      return
    }
    if (runId !== state.id) {
      reset()
      open(runId, false)
    }
    goToStep(target, runId)
  })
  useEffect(() => {
    const back = () => onBack()
    window.addEventListener('popstate', back)
    return () => window.removeEventListener('popstate', back)
  }, [])

  // Seven views under four tabs; a tab reopens the view last seen under it.
  const lastIn = useRef<Partial<Record<Group, View>>>({})
  useEffect(() => {
    lastIn.current[GROUP[view]] = view
  }, [view])
  const openGroup = (g: Group) => {
    setPaneFollows(false)
    const last = lastIn.current[g]
    setView(last && (last !== 'hindsight' || (revealed && canReveal)) ? last : FIRST[g])
  }
  const subs = SUBS[GROUP[view]].filter((s) => s.view !== 'hindsight' || canReveal)

  const MAX_CLIP_MS = 15_000

  const toggleVoice = async () => {
    if (voice.status === 'recording') {
      recording.current?.stop()
      return
    }
    if (voice.status !== 'idle') return
    try {
      const rec = await startRecording(MAX_CLIP_MS)
      recording.current = rec
      const started = Date.now()
      setVoice({ status: 'recording', seconds: 0 })
      const tick = window.setInterval(() => setVoice({ status: 'recording', seconds: Math.round((Date.now() - started) / 1000) }), 500)
      const clip = await rec.done.finally(() => window.clearInterval(tick))
      recording.current = null
      setVoice({ status: 'transcribing', seconds: 0 })
      const wav = await toWav16k(clip)
      const { question: heard } = await api.transcribe(toBase64(wav))
      setQuestion(heard)
      setVoice({ status: 'idle', seconds: 0 })
      ;questionBox.current?.focus()
    } catch (e) {
      recording.current = null
      const message = e instanceof DOMException && (e.name === 'NotAllowedError' || e.name === 'SecurityError')
        ? 'The browser blocked the microphone; allow it for this site and try again.'
        : (e as Error).message
      setVoice({ status: 'idle', seconds: 0, note: message })
    }
  }

  // The zoomed span as a question: filled in, not sent (a run spends tokens).
  const askAboutSpan = () => {
    if (!span) return
    setQuestion(`What happened to ${span.well} between ${span.start} and ${span.end}, and why?`)
    const input = questionBox.current
    input?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    input?.focus()
  }

  // A recorded example opens its saved answer at once (no tokens, no wait); without one it
  // fills the ask bar.
  const pickExample = (e: (typeof EXAMPLES)[number]) => {
    setQuestion(e.question)
    setAsOf(e.asOf)
    const saved = answered.find((r) => r.question === e.question && (r.as_of ?? '') === e.asOf)
    if (saved) {
      reset()
      setAskOpen(false)
      open(saved.id)
      setStep(3)
      setPending(3)
      setView('well')
    } else setAskOpen(true)
  }
  const exampleButtons = (list: typeof EXAMPLES) =>
    list.map((e) => {
      const saved = answered.find((r) => r.question === e.question && (r.as_of ?? '') === e.asOf)
      return (
        <button
          key={e.label}
          type="button"
          className="suggestion"
          disabled={running}
          title={
            (e.asOf ? `${e.question} (data up to ${e.asOf})` : e.question) +
            (saved ? '. Opens its recorded answer; Ask the agent runs it live.' : '')
          }
          onClick={() => pickExample(e)}
        >
          {e.label}
        </button>
      )
    })
  const liveButtons = LIVE.map((e) => (
    <button
      key={e.label}
      type="button"
      className="suggestion is-live"
      disabled={running}
      title={`${e.asOf ? `${e.question} (data up to ${e.asOf})` : e.question}. Fills the box; Ask the agent runs it live, in about a minute.`}
      onClick={() => {
        setQuestion(e.question)
        setAsOf(e.asOf)
        requestAnimationFrame(() => questionBox.current?.focus())
      }}
    >
      {e.label}
    </button>
  ))
  // From the first screen the box starts empty, on all the data; from an open run it keeps
  // that run's question and date to edit.
  const openAsk = (fresh = false) => {
    if (fresh) {
      setQuestion('')
      setAsOf('')
    }
    setAskOpen(true)
    requestAnimationFrame(() => {
      questionBox.current?.scrollIntoView({ block: 'center' })
      questionBox.current?.focus()
    })
  }

  // The licence's attribution: at the end of the story column beside the data, and at the end
  // of the page when the columns stack (between the answer and the data it would be in the way).
  const credits = (className: string) => (
    <footer className={className}>
            Volve data by Equinor and the former Volve license partners (ExxonMobil Exploration &amp; Production Norway
            AS, Bayerngas Norge AS) under the{' '}
            <a href="https://www.equinor.com/content/dam/statoil/documents/what-we-do/Equinor-HRS-Terms-and-conditions-for-licence-to-data-Volve.pdf">
              Equinor Open Data Licence
            </a>
            . Not affiliated with or endorsed by Equinor; analyses are generated by an AI agent and checked against the
            data, not reviewed by Equinor.
    </footer>
  )

  return (
    <div className="app">
      <header className="topbar">
        <h1 className="wordmark">WellPilot</h1>
        <p className="tagline">
          An AI agent for well analysis. Every number checked against the data. On Meta&rsquo;s open-weight Muse Spark
          Open.
        </p>
      </header>
      <StoryBar current={step} done={(s) => visited.has(s)} onStep={(s) => goToStep(s)} />

      <main className={`console${view === 'hindsight' ? ' is-wide-data' : ''}`}>
        {/* The story column: the question, the agent working on it, its checked answer. One
            boundary per part: a bad evidence render must not hide the live log. */}
        <div className="story-col">
          {!askShown && state.id && state.question && (
            <div id="ask" className="asked">
              <p className="asked-label">{state.asOf ? `Asked with the data up to ${state.asOf}` : 'Asked'}</p>
              <p className="asked-question">{state.question}</p>
              <button type="button" className="link-button" onClick={() => openAsk()} disabled={running}>
                Ask your own question
              </button>
            </div>
          )}
          {askShown && (
          <form id="ask" className="ask" onSubmit={submit} aria-label="Ask the agent">
            <div className="ask-row">
              <label className="sr-only" htmlFor="question">
                Question
              </label>
              {/* Wraps and grows with the question, so a phone shows all of it; Enter still asks. */}
              <textarea
                id="question"
                ref={questionBox}
                rows={1}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault()
                    e.currentTarget.form?.requestSubmit()
                  }
                }}
                placeholder="Ask about a well, e.g. why F-12 declined after 2010"
              />
              {voiceOn !== false && (
                <button
                  type="button"
                  className={`mic-button${voice.status === 'recording' ? ' is-recording' : ''}`}
                  onClick={toggleVoice}
                  disabled={voice.status === 'transcribing' || running}
                  aria-pressed={voice.status === 'recording'}
                  aria-label={voice.status === 'recording' ? 'Stop recording' : 'Ask by voice'}
                  title={voice.status === 'recording' ? 'Stop recording' : 'Ask by voice: up to 15 seconds, transcribed by the same model'}
                >
                  {voice.status === 'recording' ? <StopIcon /> : <MicIcon />}
                  <span className="mic-state">
                    {voice.status === 'recording'
                      ? `0:${String(voice.seconds).padStart(2, '0')}`
                      : voice.status === 'transcribing'
                        ? 'Transcribing'
                        : 'Voice'}
                  </span>
                </button>
              )}
              <label className="as-of" title="For backtests: the agent sees no data after this day, as if it were that day.">
                <span className="as-of-label">Data up to</span>
                <IsoDate value={asOf} onChange={setAsOf} />
              </label>
              <button className={`run-button${state.id ? '' : ' is-quiet'}`} type="submit" disabled={running}>
                {running ? 'Running' : 'Ask the agent'}
              </button>
            </div>
            {voice.note && (
              <p className="voice-note" role="status">
                {voice.note}
              </p>
            )}
            <div className="suggestions">
              <span className="suggestions-label">Run live, about a minute</span>
              {liveButtons}
            </div>
            <div className="suggestions">
              <span className="suggestions-label">Recorded answers</span>
              {exampleButtons(EXAMPLES)}
              <button type="button" className="link-button ask-close" onClick={() => setAskOpen(false)}>
                Close
              </button>
            </div>
          </form>
          )}
          <ErrorBoundary>
            <WorkBar state={state} open={logOpen} onToggle={() => setLogOpen(!logOpen)}>
              <AgentLog state={state} selectedEvidence={selectedEvidence} onSelectEvidence={setSelectedEvidence} />
            </WorkBar>
          </ErrorBoundary>
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
              revealed={revealed}
              onReveal={canReveal ? reveal : undefined}
              wells={wells}
              onAsk={() => openAsk(true)}
              featured={{ question: EXAMPLES[0].question, asOf: EXAMPLES[0].asOf }}
              runsLoaded={runsLoaded}
              examples={
                <details className="examples-menu">
                  <summary>Other recorded answers ({EXAMPLES.length - 1})</summary>
                  <div className="examples-list">{exampleButtons(EXAMPLES.slice(1))}</div>
                </details>
              }
            />
          </ErrorBoundary>
          {credits('credits')}
        </div>

        <ErrorBoundary>
          <section aria-label="Well data and evidence" className="center">
            <div className="panel-head">
              <div className="tabs" role="tablist" aria-label="Data to show">
                {TABS.map((t) => (
                  <button
                    key={t.group}
                    type="button"
                    role="tab"
                    aria-selected={GROUP[view] === t.group}
                    onClick={() => openGroup(t.group)}
                  >
                    {t.group === 'well' ? `Well ${well}` : t.label}
                  </button>
                ))}
              </div>
              {view === 'well' && (
                <button type="button" className="link-button charts-toggle" aria-pressed={!chartsShown} onClick={() => setChartsShown(!chartsShown)}>
                  {chartsShown ? 'Hide charts' : 'Show charts'}
                </button>
              )}
              {(view === 'well' || view === 'inside') && (
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
              {subs.length > 1 && (
                <div className="subtabs" role="tablist" aria-label="Views under this tab">
                  {subs.map((t) => (
                    <button
                      key={t.view}
                      type="button"
                      role="tab"
                      aria-selected={view === t.view}
                      onClick={() => {
                        setPaneFollows(false)
                        if (t.view === 'hindsight') reveal()
                        else setView(t.view)
                      }}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
            {view === 'well' && chartsShown && span?.well === well && (
              <p className="ask-span">
                <button type="button" className="ask-span-button" onClick={askAboutSpan}>
                  Ask what happened to {well} between {span.start} and {span.end}
                </button>
                <button type="button" className="link-button" onClick={() => setSpan(null)} aria-label="Dismiss">
                  &times;
                </button>
              </p>
            )}
            {view === 'well' && chartsShown && !state.id && well === 'F-12' && pastDate === '2010-12-31' && (
              <p className="chart-caption">
                <strong>F-12, the field&rsquo;s biggest well, up to 2010-12-31.</strong> Its oil is falling, its water
                rising, and in October the pressure line stops. What is going on, and what will next year bring?
              </p>
            )}
            {view === 'well' && chartsShown && (
              <WellCharts
                table={daily?.well === well ? daily.table : null}
                error={daily?.well === well ? daily.error : undefined}
                markers={chartMarkers}
                highlights={highlights}
                onSelect={(s) => setSpan(s && { ...s, well })}
              />
            )}
            <div className="evidence-scroll" ref={evidencePanel}>
              {view === 'screen' ? (
                <FieldScreen
                  asOf={pastDate ?? ''}
                  onOpenWell={(w) => {
                    setPickedWell(w)
                    setView('well')
                  }}
                />
              ) : view === 'history' ? (
                <FieldTimeline
                  asOf={pastDate}
                  end={shownEnd}
                  onOpenWell={(w) => {
                    setPickedWell(w)
                    setView('well')
                  }}
                />
              ) : view === 'inside' ? (
                <InsideWell well={well} asOf={pastDate} onOpenWell={setPickedWell} />
              ) : view === 'accuracy' && shownEnd ? (
                <p className="empty">
                  This tab scores forecasts against what the wells produced later, which would give away the
                  period after {shownEnd}. It opens with the rest of the data
                  {canReveal ? ' when you choose "Show what happened next"' : ' once the date is cleared'}.
                </p>
              ) : view === 'accuracy' ? (
                <AccuracyView />
              ) : view === 'hindsight' && state.id ? (
                <HindsightView runId={state.id} onReady={revealReady} />
              ) : view === 'plan' ? (
                <div className="charts field-view">
                  {plan?.result ? (
                    <>
                      <h3 className="accuracy-title">
                        The field against its 2005 plan{shownEnd ? `, with data up to ${shownEnd}` : ''}
                      </h3>
                      <PlanChart result={plan.result} />
                      {plan.result.by_production_year && (
                        <PlanByProductionYear table={plan.result.by_production_year} dataEnds={shownEnd} />
                      )}
                      <PlanFigure end={shownEnd} />
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
      </main>
      {credits('credits credits-page')}
    </div>
  )
}

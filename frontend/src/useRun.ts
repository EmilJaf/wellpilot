import { useCallback, useEffect, useReducer, useRef } from 'react'
import { api, followRun, type FinalEvent, type RunEvent, type StepEvent } from './api'

export type RunState = {
  id: string | null
  question: string
  model: string | null
  asOf: string | null
  status: 'idle' | 'starting' | 'running' | 'done' | 'error'
  steps: StepEvent[]
  thinking: number | null // the step the model is working on right now
  final: FinalEvent | null
  error: string | null
}

const initial: RunState = {
  id: null,
  question: '',
  model: null,
  asOf: null,
  status: 'idle',
  steps: [],
  thinking: null,
  final: null,
  error: null,
}

type Action =
  | { type: 'reset'; id: string | null; question: string }
  | { type: 'event'; event: RunEvent }
  | { type: 'failed'; message: string }

type State = RunState & { lastSeq: number }

function reducer(state: State, action: Action): State {
  if (action.type === 'reset') {
    return {
      ...initial,
      lastSeq: -1,
      id: action.id,
      question: action.question,
      status: action.id ? 'running' : 'starting',
    }
  }
  if (action.type === 'failed') return { ...state, status: 'error', thinking: null, error: action.message }
  // Events carry a sequence number; never apply one twice.
  if (action.event.seq <= state.lastSeq) return state
  const next = applyEvent(state, action.event)
  return { ...next, lastSeq: action.event.seq }
}

function applyEvent(state: State, event: RunEvent): State {
  switch (event.type) {
    case 'run_started':
      return { ...state, id: event.id, question: event.question, model: event.model, asOf: event.as_of, status: 'running' }
    case 'llm_call':
      return { ...state, thinking: event.step }
    case 'step':
      return { ...state, thinking: null, steps: [...state.steps, event] }
    case 'final':
      return { ...state, thinking: null, final: event, status: 'done' }
    case 'run_error':
      return { ...state, thinking: null, status: 'error', error: event.message }
    default: {
      // Runs saved before the rename call a failure "error".
      const legacy = event as { type: string; message?: string }
      if (legacy.type === 'error') return { ...state, thinking: null, status: 'error', error: legacy.message ?? 'failed' }
      return state
    }
  }
}

// Pause after each event when replaying a recorded run, so it reads like a live one.
const REPLAY_PAUSE_MS: Record<RunEvent['type'], number> = {
  run_started: 300,
  llm_call: 1100,
  step: 500,
  final: 0,
  run_error: 0,
}

export function useRun() {
  const [state, dispatch] = useReducer(reducer, { ...initial, lastSeq: -1 })
  const stop = useRef<() => void>(() => {})

  const follow = useCallback((runId: string, replay: boolean) => {
    stop.current()
    const queue: RunEvent[] = []
    let timer: number | undefined
    let closed = false

    // Show one event, then hold the next back for that event's pause.
    const drain = () => {
      const event = queue.shift()
      if (!event || closed) {
        timer = undefined
        return
      }
      dispatch({ type: 'event', event })
      timer = window.setTimeout(drain, REPLAY_PAUSE_MS[event.type])
    }
    const close = followRun(
      runId,
      (event) => {
        if (!replay) return dispatch({ type: 'event', event })
        queue.push(event)
        if (timer === undefined) drain()
      },
      (message) => dispatch({ type: 'failed', message }),
    )
    stop.current = () => {
      closed = true
      close()
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])

  const ask = useCallback(
    async (question: string, asOf?: string) => {
      stop.current()
      dispatch({ type: 'reset', id: null, question })
      try {
        const run = await api.startRun(question, asOf)
        follow(run.id, false)
        return run.id
      } catch (e) {
        dispatch({ type: 'failed', message: (e as Error).message })
        return null
      }
    },
    [follow],
  )

  const open = useCallback(
    (runId: string, replay = false) => {
      dispatch({ type: 'reset', id: runId, question: '' })
      follow(runId, replay)
    },
    [follow],
  )

  useEffect(() => () => stop.current(), [])

  return { state, ask, open }
}

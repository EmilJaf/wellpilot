import type { ReactNode } from 'react'
import { StepsIcon } from '../icons'
import type { RunState } from '../useRun'

/**
 * The agent's work in one line that stays in sight: how long it worked on its own, its
 * steps and tool calls, how often the checker read its answer, and a tick per call. The
 * full log opens beneath it. Folded, never hidden: the agent working alone is the point.
 */
export function WorkBar({ state, open, onToggle, children }: { state: RunState; open: boolean; onToggle: () => void; children: ReactNode }) {
  if (state.status === 'idle') return null
  const calls = state.steps.flatMap((s) => s.tool_calls)
  const tools = calls.filter((c) => c.name !== 'submit_findings')
  const checks = calls.filter((c) => c.name === 'submit_findings')
  const seconds = Math.round(state.steps.reduce((t, s) => t + s.seconds, 0))
  const running = state.status === 'starting' || state.status === 'running'
  return (
    <section id="work" className={`work${open ? ' is-open' : ''}`} aria-label="How the agent worked">
      <button type="button" className="work-bar" aria-expanded={open} onClick={onToggle}>
        <StepsIcon />
        <span className="work-text">
          {running ? (
            <>
              <strong>The agent is working on its own:</strong> step {Math.max(state.thinking ?? state.steps.length, 1)},{' '}
              {tools.length} tool {tools.length === 1 ? 'call' : 'calls'} so far
            </>
          ) : (
            <>
              <strong>The agent worked {seconds} s on its own:</strong> {state.steps.length} steps, {tools.length} tool calls
              {checks.length > 0 && `, its answer checked ${checks.length === 1 ? 'once' : `${checks.length} times`}`}
            </>
          )}
        </span>
        <span className="work-ticks" aria-hidden="true">
          {state.steps.map((s) => (
            <span key={s.index} className="work-step">
              {s.tool_calls.map((c, i) => (
                <i
                  key={i}
                  className={
                    c.name === 'submit_findings' ? (c.accepted ? 'tick-accepted' : 'tick-rejected') : c.error ? 'tick-error' : 'tick-tool'
                  }
                />
              ))}
            </span>
          ))}
        </span>
        <span className="work-toggle">{open ? 'Hide its steps' : 'Show its steps'}</span>
      </button>
      {open && <div className="work-log">{children}</div>}
    </section>
  )
}

import type { ReactNode } from 'react'
import { StepsIcon } from '../icons'
import { modelName, toolName } from '../labels'
import type { RunState } from '../useRun'

/**
 * The agent's work in one line that stays in sight: how long it worked on its own, its
 * steps and tool calls, how often the checker read its answer, and a tick per call. The
 * full log opens beneath it. Folded, never hidden: the agent working alone is the point.
 */
type Call = RunState['steps'][number]['tool_calls'][number]

/** A tick's look and what it stands for, for its hover text. */
function tick(c: Call): [string, string] {
  if (c.name === 'submit_findings')
    return c.accepted ? ['tick-accepted', 'the checker accepted the answer'] : ['tick-rejected', 'the checker sent the answer back']
  return c.error ? ['tick-error', `${toolName(c.name)} failed`] : ['tick-tool', toolName(c.name)]
}

export function WorkBar({ state, open, onToggle, children }: { state: RunState; open: boolean; onToggle: () => void; children: ReactNode }) {
  if (state.status === 'idle') return null
  const calls = state.steps.flatMap((s) => s.tool_calls)
  const tools = calls.filter((c) => c.name !== 'submit_findings')
  const checks = calls.filter((c) => c.name === 'submit_findings')
  const seconds = Math.round(state.steps.reduce((t, s) => t + s.seconds, 0))
  const running = state.status === 'starting' || state.status === 'running'
  // The model by name: a Meta model doing the work is the point of the line.
  const who = state.model ? (
    <>
      <strong>{modelName(state.model)}</strong>, Meta&rsquo;s {state.model.includes('open') ? 'open-weight model' : 'model'},
    </>
  ) : (
    <strong>The agent</strong>
  )
  return (
    <section id="work" className={`work${open ? ' is-open' : ''}`} aria-label="How the agent worked">
      <button type="button" className="work-bar" aria-expanded={open} onClick={onToggle}>
        <StepsIcon />
        <span className="work-text">
          {running ? (
            <>
              {who} is working on its own: step {Math.max(state.thinking ?? state.steps.length, 1)}, {tools.length} tool{' '}
              {tools.length === 1 ? 'call' : 'calls'} so far
            </>
          ) : (
            <>
              {who} worked {seconds} s on its own: {state.steps.length} steps, {tools.length} tool calls
              {checks.length > 0 && `, its answer checked ${checks.length === 1 ? 'once' : `${checks.length} times`}`}
            </>
          )}
        </span>
        <span className="work-ticks">
          {state.steps.map((s) => (
            <span key={s.index} className="work-step" title={`Step ${s.index}`}>
              {s.tool_calls.map((c, i) => {
                const [kind, what] = tick(c)
                return <i key={i} className={kind} title={`Step ${s.index}: ${what}`} />
              })}
            </span>
          ))}
          {calls.length > 0 && (
            <span className="work-key" aria-hidden="true">
              <span><i className="tick-tool" /> tool call</span>
              {checks.some((c) => c.accepted === false) && <span><i className="tick-rejected" /> answer sent back by the checker</span>}
              {checks.some((c) => c.accepted) && <span><i className="tick-accepted" /> answer accepted</span>}
              {tools.some((c) => c.error) && <span><i className="tick-error" /> tool error</span>}
            </span>
          )}
        </span>
        <span className="work-toggle">{open ? 'Hide its steps' : 'Show its steps'}</span>
      </button>
      {open && <div className="work-log">{children}</div>}
    </section>
  )
}

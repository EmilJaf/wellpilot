import { Fragment, useEffect, useRef } from 'react'
import type { StepEvent, ToolCallView } from '../api'
import { AlarmIcon, CautionIcon, CheckIcon } from '../icons'
import { compact, describeArgs, isBookkeeping, modelName, plainProblem, plainToolError, shortWarning, toolName } from '../labels'
import type { RunState } from '../useRun'

type Props = {
  state: RunState
  selectedEvidence: string | null
  onSelectEvidence: (id: string) => void
}

/** The agent's work as a log track: time runs down, every tool call is a tick. */
export function AgentLog({ state, selectedEvidence, onSelectEvidence }: Props) {
  const tokensIn = state.steps.reduce((sum, s) => sum + s.prompt_tokens, 0)
  const tokensOut = state.steps.reduce((sum, s) => sum + s.completion_tokens, 0)
  // Elapsed time at the start of each step, like depth along a log.
  const starts = state.steps.map((_, i) => state.steps.slice(0, i).reduce((sum, s) => sum + s.seconds, 0))
  const elapsed = state.steps.reduce((sum, s) => sum + s.seconds, 0)

  // Follow the newest step while the run goes, unless the reader scrolled up to read.
  const panel = useRef<HTMLElement>(null)
  const following = useRef(true)
  useEffect(() => {
    const el = panel.current
    if (el && following.current && state.status === 'running') el.scrollTop = el.scrollHeight
  }, [state.steps.length, state.thinking, state.status])

  return (
    <section
      aria-label="How the agent worked"
      className="log-panel"
      ref={panel}
      onScroll={(e) => {
        const el = e.currentTarget
        following.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
      }}
    >
      <div className="panel-head">
        <h2>How the agent worked</h2>
        {state.model && (
          <span className="panel-meta">
            {modelName(state.model)}
            {tokensIn > 0 && `, ${compact(tokensIn)} tokens in, ${compact(tokensOut)} out`}
          </span>
        )}
      </div>
      <div className="panel-body">
        {state.status === 'idle' && (
          <p className="empty">
            Each step the agent takes appears here as it happens: the tools it calls, what they
            return, its own reasoning, and the checker's verdicts.
          </p>
        )}
        {state.question && <p className="question">{state.question}</p>}
        {state.asOf && <p className="narration">Data after {state.asOf} is hidden from the agent.</p>}
        <ol className="track" aria-live="polite" aria-label="Agent steps">
          {state.steps.map((step, i) => (
            <Fragment key={step.index}>
              <Step
                step={step}
                startedAt={starts[i]}
                selectedEvidence={selectedEvidence}
                onSelectEvidence={onSelectEvidence}
              />
              {state.review && state.review.after_step === step.index && (
                <li className="track-step">
                  <span className="track-time" />
                  <div className="track-body">
                    <div className={`flag ${state.review.points.length ? 'flag-caution' : 'flag-ok'}`}>
                      {state.review.points.length ? <CautionIcon /> : <CheckIcon />}
                      <span>
                        Second opinion:{' '}
                        {state.review.points.length
                          ? `the reviewer questioned the reasoning (${state.review.points.length} ${state.review.points.length === 1 ? 'point' : 'points'})`
                          : 'the reviewer found the reasoning sound'}
                        {state.review.points.length > 0 && (
                          <ul>
                            {state.review.points.map((p, j) => (
                              <li key={j}>{p.point}</li>
                            ))}
                          </ul>
                        )}
                      </span>
                    </div>
                  </div>
                </li>
              )}
            </Fragment>
          ))}
          {state.thinking !== null && (
            <li className="track-step">
              <span className="track-time">{elapsed.toFixed(1)}s</span>
              <div className="track-body">
                <p className="thinking">
                  <span className="pulse" aria-hidden />
                  Muse Spark is working on step {state.thinking}
                </p>
              </div>
            </li>
          )}
        </ol>
        {state.status === 'starting' && <p className="thinking">Starting the run</p>}
        {/* A run that never started is explained once, in the answer panel. */}
        {state.error && state.id && (
          <div className="flag flag-alarm" role="alert">
            <AlarmIcon />
            <span>
              {!state.id
                ? `The run could not start: ${state.error}`
                : state.model
                  ? `The run stopped: ${state.error}`
                  : state.error}
            </span>
          </div>
        )}
      </div>
    </section>
  )
}

function Step({
  step,
  startedAt,
  selectedEvidence,
  onSelectEvidence,
}: {
  step: StepEvent
  startedAt: number
  selectedEvidence: string | null
  onSelectEvidence: (id: string) => void
}) {
  return (
    <li className="track-step">
      <span className="track-time">{startedAt.toFixed(1)}s</span>
      <div className="track-body">
        <p className="step-label">
          Step {step.index}
          <span className="step-time"> {step.seconds.toFixed(1)} s</span>
        </p>
        {step.reasoning && (
          <details className="reasoning">
            <summary>Model reasoning</summary>
            <p>{step.reasoning}</p>
          </details>
        )}
        {step.content && <p className="narration">{step.content}</p>}
        <ul className="calls">
          {step.tool_calls.map((call, i) => (
            <Call key={i} call={call} selectedEvidence={selectedEvidence} onSelectEvidence={onSelectEvidence} />
          ))}
        </ul>
      </div>
    </li>
  )
}

function Call({
  call,
  selectedEvidence,
  onSelectEvidence,
}: {
  call: ToolCallView
  selectedEvidence: string | null
  onSelectEvidence: (id: string) => void
}) {
  if (call.name === 'submit_findings') {
    return (
      <li className="call">
        {call.accepted ? (
          <div className="flag flag-ok">
            <CheckIcon />
            <span>Answer accepted: every claim checked against the evidence</span>
          </div>
        ) : (
          <Rejection problems={call.problems ?? []} />
        )}
      </li>
    )
  }

  const id = call.evidence_id
  const body = (
    <>
      <span>
        <span className="call-name">{toolName(call.name)}</span>
        <span className="call-args">{describeArgs(call.name, call.arguments)}</span>
      </span>
      {id && <span className="evidence-chip">{id}</span>}
    </>
  )
  return (
    <li className="call">
      {id ? (
        <button
          type="button"
          className="call-button"
          aria-pressed={selectedEvidence === id}
          onClick={() => onSelectEvidence(id)}
        >
          {body}
        </button>
      ) : (
        <div className="call-button">{body}</div>
      )}
      {call.warnings?.map((w, i) => (
        <div key={i} className="flag flag-caution flag-short" title={w}>
          <CautionIcon />
          <span>{shortWarning(w)}</span>
        </div>
      ))}
      {call.error && (
        <div className="flag flag-caution" title={call.error}>
          <CautionIcon />
          <span>The tool refused the call and told the model why: {plainToolError(call.error)}</span>
        </div>
      )}
    </li>
  )
}

/** The checker sending an answer back: what it got wrong in red, one per line; numbers
 * written without naming their source (bookkeeping) in one quiet line under them. */
function Rejection({ problems }: { problems: string[] }) {
  const real = problems.filter((p) => !isBookkeeping(p))
  const paperwork = problems.length - real.length
  const tally = paperwork === 1 ? 'one number' : `${paperwork} numbers`
  return (
    <div className={`flag ${real.length ? 'flag-alarm' : 'flag-caution'}`}>
      {real.length ? <AlarmIcon /> : <CautionIcon />}
      <span>
        {real.length
          ? `The checker sent the answer back: ${real.length === 1 ? 'one error' : `${real.length} errors`}`
          : 'The checker sent the answer back for its paperwork'}
        {real.length > 0 && (
          <ul>
            {real.map((p, i) => (
              <li key={i} title={p}>
                {plainProblem(p)}
              </li>
            ))}
          </ul>
        )}
        {paperwork > 0 && (
          <span className="flag-paperwork">
            {real.length ? 'Also ' : ': '}
            {tally} written without naming the tool result {paperwork === 1 ? 'it comes' : 'they come'} from, or not
            rounded.
          </span>
        )}
      </span>
    </div>
  )
}

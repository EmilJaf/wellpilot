import type { StepEvent, ToolCallView } from '../api'
import { AlarmIcon, CautionIcon, CheckIcon } from '../icons'
import { compact, describeArgs, toolName } from '../labels'
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

  return (
    <section aria-label="Agent log">
      <div className="panel-head">
        <h2>Agent log</h2>
        {state.model && (
          <span className="panel-meta">
            {state.model}
            {tokensIn > 0 && `, ${compact(tokensIn)} in / ${compact(tokensOut)} out tokens`}
          </span>
        )}
      </div>
      <div className="panel-body">
        {state.status === 'idle' && (
          <p className="empty">
            Ask a question about a well. The agent plans the analysis, runs it on the Volve data and
            reports, and every step it takes appears here as it happens.
          </p>
        )}
        {state.question && <p className="question">{state.question}</p>}
        {state.asOf && <p className="narration">Data after {state.asOf} is hidden from the agent.</p>}
        <ol className="track" aria-live="polite" aria-label="Agent steps">
          {state.steps.map((step, i) => (
            <Step
              key={step.index}
              step={step}
              startedAt={starts[i]}
              selectedEvidence={selectedEvidence}
              onSelectEvidence={onSelectEvidence}
            />
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
        {state.error && (
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
          <span className="chart-note"> {step.seconds.toFixed(1)}s</span>
        </p>
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
            <span>Answer submitted and every claim checked against the evidence</span>
          </div>
        ) : (
          <div className="flag flag-alarm">
            <AlarmIcon />
            <span>
              Checker sent the answer back ({call.problems?.length ?? 0})
              <ul>
                {call.problems?.map((p, i) => <li key={i}>{p}</li>)}
              </ul>
            </span>
          </div>
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
        <div key={i} className="flag flag-caution">
          <CautionIcon />
          <span>{w}</span>
        </div>
      ))}
      {call.error && (
        <div className="flag flag-alarm">
          <AlarmIcon />
          <span>Tool error, sent back to the model: {call.error}</span>
        </div>
      )}
    </li>
  )
}

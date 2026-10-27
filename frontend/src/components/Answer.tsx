import type { ReactNode } from 'react'
import type { RunSummary, Span, StepEvent } from '../api'
import { AlarmIcon, CheckIcon, RedoIcon } from '../icons'
import { plainProblem, compact, describeArgs, toolName } from '../labels'
import type { RunState } from '../useRun'
import { corrections } from '../corrections'
import { CorrectionList } from './Corrections'

/** What the reader clicked: a number (its value marked in the evidence) or a quote (marked
    on the report page it came from). */
export type Focus = { text: string; value: number; quote?: boolean }

type Props = {
  state: RunState
  recorded: RunSummary | null
  selectedFinding: number | null
  selectedEvidence: string | null
  onSelectFinding: (index: number | null) => void
  onSelectEvidence: (id: string, focus?: Focus) => void
  onReplay: (runId: string) => void
  revealed: boolean
  onReveal?: () => void // a run made with data up to a past date: show what happened next
}

/**
 * The text cut at the spans the backend checker computed: quotes as quotes, declared
 * numbers as buttons to their evidence, numbers inside a quote inside it. The checker owns
 * the rules; this only slices.
 */
function bySpans(
  text: string,
  spans: Span[] | undefined,
  open: Props['onSelectEvidence'],
  quoteSource?: string,
): ReactNode[] {
  if (!spans) return [text]
  const numbers = spans.filter((s) => s.kind === 'number')
  const plain = (from: number, to: number): ReactNode[] => {
    const parts: ReactNode[] = []
    let at = from
    for (const n of numbers) {
      if (n.start < at || n.end > to) continue
      const written = text.slice(n.start, n.end)
      parts.push(text.slice(at, n.start))
      parts.push(
        <button
          key={`n${n.start}`}
          type="button"
          className="num"
          title={`Show where ${written} comes from (${n.evidence_id})`}
          onClick={() => open(n.evidence_id, { text: written, value: n.value })}
        >
          {written}
        </button>,
      )
      at = n.end
    }
    parts.push(text.slice(at, to))
    return parts
  }
  const parts: ReactNode[] = []
  let at = 0
  for (const q of spans) {
    if (q.kind !== 'quote' || q.start < at) continue
    parts.push(...plain(at, q.start))
    // <q> draws its own quotation marks, so the text's are left out. With a report among the
    // sources, the quote opens its page with the passage marked.
    const inner = text.slice(q.start + 1, q.end - 1)
    const source = q.evidence_id ?? quoteSource  // the evidence the checker found it in
    parts.push(
      source ? (
        <q
          key={`q${q.start}`}
          className="quote-link"
          title={`Show this passage on the page it was read from (${source})`}
          onClick={(e) => {
            if ((e.target as HTMLElement).closest('button')) return
            e.stopPropagation()
            open(source, { text: inner, value: Number.NaN, quote: true })
          }}
        >
          {plain(q.start + 1, q.end - 1)}
        </q>
      ) : (
        <q key={`q${q.start}`}>{plain(q.start + 1, q.end - 1)}</q>
      ),
    )
    at = q.end
  }
  parts.push(...plain(at, text.length))
  return parts
}

/** Where the summary's first sentence ends, unless a quote or number runs across it. */
function firstSentenceEnd(text: string, spans: Span[] | undefined): number | null {
  for (const m of text.matchAll(/[.!?][”"’)]?(?=\s+[A-Z“"(])/g)) {
    const cut = (m.index ?? 0) + m[0].length
    if (cut < text.length && !(spans ?? []).some((s) => s.start < cut && s.end > cut)) return cut
  }
  return null
}

/** The summary with its first sentence, the plain answer, set apart as the lead. */
function Summary({ text, spans, open, quoteSource }: { text: string; spans?: Span[]; open: Props['onSelectEvidence']; quoteSource?: string }) {
  const cut = firstSentenceEnd(text, spans)
  if (cut === null) return <p className="summary-lead">{bySpans(text, spans, open, quoteSource)}</p>
  const rest = text.slice(cut).trimStart()
  const offset = text.length - rest.length
  const shift = (list: Span[] | undefined, from: number, to: number, by: number) =>
    list?.filter((s) => s.start >= from && s.end <= to).map((s) => ({ ...s, start: s.start - by, end: s.end - by }))
  return (
    <>
      <p className="summary-lead">{bySpans(text.slice(0, cut), shift(spans, 0, cut, 0), open, quoteSource)}</p>
      <p className="summary">{bySpans(rest, shift(spans, offset, text.length, offset), open, quoteSource)}</p>
    </>
  )
}

export function Answer(props: Props) {
  const { state } = props
  const final = state.final
  return (
    <section aria-label="Answer" className="answer-panel">
      <div className="panel-head">
        <h2>Answer</h2>
      </div>
      <div className="panel-body answer-body" aria-live="polite">
        {final ? (
          <Final {...props} />
        ) : state.status === 'idle' ? (
          <Intro recorded={props.recorded} onReplay={props.onReplay} />
        ) : state.status === 'error' ? (
          <div className="empty">
            <p>
              {state.steps.length
                ? 'No answer: the run stopped. The agent log says where.'
                : `The agent could not start: ${state.error ?? 'the server refused the run'}.`}
            </p>
            {props.recorded && (
              <button type="button" className="primary-button" onClick={() => props.onReplay(props.recorded!.id)}>
                Watch a recorded run instead
              </button>
            )}
          </div>
        ) : (
          <Live steps={state.steps} thinking={state.thinking} asOf={state.asOf} />
        )}
      </div>
    </section>
  )
}

/** First visit: what this is, how to read it, and the one thing to try. */
function Intro({ recorded, onReplay }: { recorded: RunSummary | null; onReplay: (id: string) => void }) {
  return (
    <div className="intro">
      <h3 className="intro-title">Ask why a well changed.</h3>
      <p className="intro-lead">
        An AI agent works it out on the field&rsquo;s own data and reports. Every number in its answer is
        checked against the data before you see it. Set a past date and it answers as of that day; then see
        what happened next.
      </p>
      {/* plan, check, trace: the order the agent works in */}
      <ol className="intro-steps">
        <li>
          <strong>Runs real analysis</strong>
          <span>Decline fits, event detection, SQL and the 2005 plan, a Norwegian PDF read page by page.</span>
        </li>
        <li>
          <strong>Checks every number</strong>
          <span>A deterministic checker sends a wrong answer back to the agent to fix.</span>
        </li>
        <li>
          <strong>Shows its sources</strong>
          <span>Any number opens the tool result and the code that produced it.</span>
        </li>
      </ol>
      {recorded && (
        <button type="button" className="primary-button" onClick={() => onReplay(recorded.id)}>
          {recorded.as_of ? `Watch it answer as of ${recorded.as_of}` : 'Watch a recorded run'}
        </button>
      )}
      <p className="intro-who">
        <strong>For production and reservoir engineers.</strong> Why output fell, whether a gauge or the reservoir is
        to blame, what next year brings, how the field compares with its plan: questions that take hours across
        databases, PDFs and spreadsheets, answered in about a minute.
      </p>
      <p className="intro-note">
        Runs on Muse Spark Open, Meta&rsquo;s open-weight model, so an operator can host it next to well data that
        never leaves the company. The data here is Volve, a North Sea field Equinor published openly.
      </p>
    </div>
  )
}

/** While the agent works: what it is doing now, and the checker's verdicts as they come. */
function Live({ steps, thinking, asOf }: { steps: StepEvent[]; thinking: number | null; asOf: string | null }) {
  const calls = steps.flatMap((s) => s.tool_calls)
  const results = calls.filter((c) => c.evidence_id).length
  const submissions = calls.filter((c) => c.name === 'submit_findings')
  const rejected = [...submissions].reverse().find((c) => c.accepted === false)
  const latest = [...calls].reverse().find((c) => c.name !== 'submit_findings')
  return (
    <div className="live">
      <p className="live-now">
        <span className="pulse" aria-hidden />
        {thinking !== null ? `Step ${thinking}: the model is deciding what to do next` : 'Starting the run'}
      </p>
      {asOf && <p className="live-last">The agent sees the data up to {asOf} only, as if it were that day.</p>}
      {latest && (
        <p className="live-last">
          Last tool run: <strong>{toolName(latest.name)}</strong>
          {describeArgs(latest.name, latest.arguments) && `, ${describeArgs(latest.name, latest.arguments)}`}
        </p>
      )}
      <p className="live-count">
        {results} tool {results === 1 ? 'result' : 'results'} so far, each one citable evidence.
      </p>
      {rejected && submissions[submissions.length - 1] === rejected && (
        <div className="flag flag-alarm live-rejection" role="status">
          <AlarmIcon />
          <span>
            <strong>
              The checker sent the answer back with {rejected.problems?.length ?? 0}{' '}
              {rejected.problems?.length === 1 ? 'problem' : 'problems'}. The agent is fixing them.
            </strong>
            <ul>
              {rejected.problems?.slice(0, 5).map((p, i) => <li key={i}>{plainProblem(p)}</li>)}
            </ul>
          </span>
        </div>
      )}
      <p className="empty">The answer appears here once every claim passes the checker.</p>
    </div>
  )
}

function Final({ state, selectedFinding, selectedEvidence, onSelectFinding, onSelectEvidence, revealed, onReveal }: Props) {
  const final = state.final!
  const tools = Object.fromEntries(final.evidence.map((e) => [e.id, e.tool]))
  const isReport = (id: string) => tools[id] === 'search_reports' || tools[id] === 'read_report_page'
  const anyReport = final.evidence.map((e) => e.id).find(isReport)
  const rounds = final.check_rounds
  const sentBack = rounds.filter((r) => r.length)
  const problems = sentBack.reduce((n, r) => n + r.length, 0)
  const fixes = corrections(state.steps, final.findings)
  const history = fixes.length ? <CorrectionList items={fixes} bare /> : <Rounds rounds={rounds} />
  const seconds = state.steps.reduce((t, step) => t + step.seconds, 0)
  const facts = final.findings.filter((f) => f.confidence === 'fact').length
  return (
    <>
      <div className={`status${final.verified ? '' : ' status-alarm'}`} role="status">
        <div className="status-main">
          {final.verified ? <CheckIcon /> : <AlarmIcon />}
          <strong>{final.verified ? 'Sources verified' : 'Not verified'}</strong>
          <span>
            {final.verified
              ? rounds.length > 1
                ? `on attempt ${rounds.length}, after ${problems} ${problems === 1 ? 'fix' : 'fixes'}`
                : 'on the first attempt'
              : final.finished
                ? final.findings.length
                  ? 'problems remained after the last correction round'
                  : 'no findings were submitted'
                : `stopped early: ${final.stopped ?? 'step limit'}`}
          </span>
        </div>
        <dl className="status-metrics">
          <div>
            <dt>Steps</dt>
            <dd>{final.steps}</dd>
          </div>
          <div>
            <dt>Tool results</dt>
            <dd>{final.evidence.length}</dd>
          </div>
          <div>
            <dt>Time</dt>
            <dd>{Math.round(seconds)} s</dd>
          </div>
          <div>
            <dt>Tokens</dt>
            <dd>{compact(final.tokens.input + final.tokens.output)}</dd>
          </div>
        </dl>
        <details className="status-more">
          <summary>
            {sentBack.length > 0 ? `What the checker caught (${problems})` : 'What the checker checks'}
          </summary>
          <p className="status-note">
            Every number, date and quote in the answer was found in the tool result it cites, as the well,
            quantity, direction and event the text says, and causes are marked as inferences. Not checked: whether
            the reasoning holds. That is for you to judge, and every source is one click away.
          </p>
          {rounds.length > 0 && sentBack.length > 0 && history}
        </details>
      </div>

      {state.asOf && (
        <div className="past">
          <p>
            <strong>Answered as of {state.asOf}.</strong> The agent saw no data after that day.
          </p>
          {onReveal && (
            <button type="button" className={revealed ? 'link-button' : 'primary-button'} onClick={onReveal}>
              Show what happened next
            </button>
          )}
        </div>
      )}

      {final.summary && (
        <section className="conclusion" aria-label="Conclusion">
          <h3 className="section-label">Conclusion</h3>
          <Summary text={final.summary} spans={final.summary_spans} open={onSelectEvidence} quoteSource={anyReport} />
        </section>
      )}

      <div className="findings-head">
        <h3 className="section-label">
          Findings <span>{final.findings.length}: {facts} {facts === 1 ? 'fact' : 'facts'}, {final.findings.length - facts}{' '}
          {final.findings.length - facts === 1 ? 'inference' : 'inferences'}</span>
        </h3>
        <p className="findings-key">
          <span><i className="mark mark-fact" /> fact: shown by the data</span>
          <span><i className="mark" /> inference: an interpretation</span>
          <span>underlined numbers open their source</span>
        </p>
      </div>

      <ol className="findings">
        {final.findings.map((finding, i) => {
          const fact = finding.confidence === 'fact'
          const fixed = fixes.filter((c) => c.finalIndex === i)
          return (
            <li key={i} className={`finding${selectedFinding === i ? ' is-selected' : ''}`}>
              <span className="finding-index" aria-hidden>
                {String(i + 1).padStart(2, '0')}
              </span>
              <div className="finding-body">
                <div className="finding-head">
                  {/* The label selects the finding (and marks its dates on the charts); the claim
                      cannot be one button because its numbers are buttons to evidence. */}
                  <button
                    type="button"
                    className={`pill pill-button${fact ? ' pill-fact' : ''}`}
                    aria-pressed={selectedFinding === i}
                    aria-label={`${fact ? 'Fact' : 'Inference'} ${i + 1}: show its dates on the charts`}
                    onClick={() => onSelectFinding(selectedFinding === i ? null : i)}
                  >
                    {fact ? 'Fact' : 'Inference'}
                  </button>
                </div>
                <p
                  className="claim"
                  onClick={(e) => {
                    if ((e.target as HTMLElement).closest('button')) return
                    onSelectFinding(selectedFinding === i ? null : i)
                  }}
                >
                  {bySpans(finding.claim, finding.spans, onSelectEvidence, finding.evidence_ids.find(isReport))}
                </p>
                {finding.detail && (
                  <details className="finding-detail">
                    <summary>Details for engineers</summary>
                    <p>{bySpans(finding.detail, finding.detail_spans, onSelectEvidence, finding.evidence_ids.find(isReport))}</p>
                  </details>
                )}
                {fixed.length > 0 && (
                  <p className="fix-tag">
                    <RedoIcon />
                    <span>
                      <strong>Fixed after the checker sent it back:</strong> {[...new Set(fixed.flatMap((c) => c.changes))].join(' ')}
                    </span>
                  </p>
                )}
                <p className="sources">
                  <span className="sources-label">Sources</span>
                  {finding.evidence_ids.map((id) => (
                    <button
                      key={id}
                      type="button"
                      className="evidence-chip"
                      aria-pressed={selectedEvidence === id}
                      onClick={() => {
                        onSelectFinding(i)
                        onSelectEvidence(id)
                      }}
                    >
                      {id}
                      {tools[id] && <span className="chip-tool">{toolName(tools[id])}</span>}
                    </button>
                  ))}
                </p>
              </div>
            </li>
          )
        })}
      </ol>
    </>
  )
}

function Rounds({ rounds }: { rounds: string[][] }) {
  return (
    <details className="rounds">
      <summary>What the checker sent back</summary>
      <ol>
        {rounds.map((round, i) => (
          <li key={i}>
            <span className="round-label">Attempt {i + 1}:</span>{' '}
            {round.length === 0 ? (
              'passed'
            ) : (
              <ul>
                {round.map((p, j) => <li key={j}>{p}</li>)}
              </ul>
            )}
          </li>
        ))}
      </ol>
    </details>
  )
}

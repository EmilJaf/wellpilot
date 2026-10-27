import type { ReactNode } from 'react'
import type { RunSummary, Span, StepEvent } from '../api'
import { AlarmIcon, CheckIcon, RedoIcon } from '../icons'
import { brief, compact, describeArgs, toolName } from '../labels'
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
        {final && (
          <span className="panel-meta">
            {final.steps} steps, {final.evidence.length} tool results, {compact(final.tokens.input + final.tokens.output)}{' '}
            tokens
          </span>
        )}
      </div>
      <div className="panel-body answer-body" aria-live="polite">
        {final ? (
          <Final {...props} />
        ) : state.status === 'idle' ? (
          <Intro recorded={props.recorded} onReplay={props.onReplay} />
        ) : state.status === 'error' ? (
          <p className="empty">No answer: the run stopped. The agent log says where.</p>
        ) : (
          <Live steps={state.steps} thinking={state.thinking} />
        )}
      </div>
    </section>
  )
}

/** First visit: what this is, how to read it, and the one thing to try. */
function Intro({ recorded, onReplay }: { recorded: RunSummary | null; onReplay: (id: string) => void }) {
  return (
    <div className="intro">
      <p className="intro-lead">
        Ask why a well changed. An AI agent works it out on the field's own data and reports, and
        every number in its answer is checked against the data it came from.
      </p>
      <p className="intro-who">
        <strong>For production and reservoir engineers</strong> reviewing wells: why output fell,
        whether a dead gauge or the reservoir is to blame, what next year will bring, and how it
        compares with the development plan. Today that means hours across a production database,
        PDF reports in several languages and spreadsheets. Here it is one question, answered in
        about a minute, with every number traceable to the data.
      </p>
      <ol className="how">
        <li>
          <strong>It plans and runs real analysis.</strong> Decline-curve fits, event detection on
          rates and pressures, SQL on daily production, and a search of the 2005 development
          plan, a Norwegian PDF the model read page by page.
        </li>
        <li>
          <strong>A checker verifies the answer before you see it.</strong> Every number, date and
          quote must be found in the tool result it cites. Causes must be marked as inferences.
          A wrong answer goes back to the agent to fix.
        </li>
        <li>
          <strong>You can trace any number.</strong> Click it to open the exact tool result and
          the code that produced it.
        </li>
      </ol>
      {recorded && (
        <button type="button" className="primary-button" onClick={() => onReplay(recorded.id)}>
          Watch a recorded run
        </button>
      )}
      <p className="intro-note">
        The agent runs on Muse Spark Open, Meta's open-weight model, so an operator can host it
        on their own servers, next to well data that never leaves the company. This demo uses
        Volve, an oil field in the Norwegian North Sea that Equinor operated from 2008 to 2016
        and published openly.
      </p>
    </div>
  )
}

/** While the agent works: what it is doing now, and the checker's verdicts as they come. */
function Live({ steps, thinking }: { steps: StepEvent[]; thinking: number | null }) {
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
              {rejected.problems?.slice(0, 5).map((p, i) => <li key={i}>{brief(p)}</li>)}
            </ul>
          </span>
        </div>
      )}
      <p className="empty">The answer appears here once every claim passes the checker.</p>
    </div>
  )
}

function Final({ state, selectedFinding, selectedEvidence, onSelectFinding, onSelectEvidence }: Props) {
  const final = state.final!
  const tools = Object.fromEntries(final.evidence.map((e) => [e.id, e.tool]))
  const isReport = (id: string) => tools[id] === 'search_reports' || tools[id] === 'read_report_page'
  const anyReport = final.evidence.map((e) => e.id).find(isReport)
  const rounds = final.check_rounds
  const sentBack = rounds.filter((r) => r.length)
  const problems = sentBack.reduce((n, r) => n + r.length, 0)
  const fixes = corrections(state.steps, final.findings)
  const history = fixes.length ? <CorrectionList items={fixes} /> : <Rounds rounds={rounds} />
  return (
    <>
      {final.summary && (
        <Summary text={final.summary} spans={final.summary_spans} open={onSelectEvidence} quoteSource={anyReport} />
      )}

      {final.verified ? (
        <div className="verdict">
          <CheckIcon />
          <div>
            <strong>
              Verified{rounds.length > 1 ? ` on attempt ${rounds.length}` : ' on the first attempt'}
            </strong>
            <p>
              Every number, date and quote below was found in the tool result it cites, and causes are
              marked as inferences. Whether the reasoning holds is for you to judge: every source is one
              click away.
              {sentBack.length > 0 &&
                ` The checker sent ${sentBack.length === 1 ? 'the first answer' : `${sentBack.length} answers`} back with ${problems} ${problems === 1 ? 'problem' : 'problems'}, and the agent fixed ${problems === 1 ? 'it' : 'them'}.`}
            </p>
            {sentBack.length > 0 && history}
          </div>
        </div>
      ) : (
        <div className="verdict verdict-alarm">
          <AlarmIcon />
          <div>
            <strong>Not verified</strong>
            <p>
              {final.finished
                ? final.findings.length
                  ? 'The checker still found problems after the last correction round.'
                  : 'The agent answered in text without submitting findings to check.'
                : `The run stopped early (${final.stopped ?? 'step limit'})${
                    final.findings.length
                      ? '. Below is its last submitted answer, which did not pass the checker.'
                      : ' before the agent submitted an answer.'
                  }`}
            </p>
            {rounds.length > 0 && history}
          </div>
        </div>
      )}

      <p className="answer-hint">
        Click an underlined number to see where it comes from. <span className="pill pill-fact">Fact</span>{' '}
        is shown directly by the data; <span className="pill">Inference</span> is an interpretation, such as a
        likely cause.
      </p>

      <ol className="findings">
        {final.findings.map((finding, i) => {
          const fact = finding.confidence === 'fact'
          const fixed = fixes.filter((c) => c.finalIndex === i)
          return (
            <li key={i} className={`finding${selectedFinding === i ? ' is-selected' : ''}`}>
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

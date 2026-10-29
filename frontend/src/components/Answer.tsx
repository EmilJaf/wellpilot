import type { ReactNode } from 'react'
import type { MetaModels, RunSummary, Span, StepEvent } from '../api'
import { AlarmIcon, CheckIcon, RedoIcon } from '../icons'
import { BARRELS_NOTE, barrels, plainProblem, compact, describeArgs, modelName, toolName, volumeAt } from '../labels'
import type { RunState } from '../useRun'
import { corrections } from '../corrections'
import { CaughtList, CorrectionList } from './Corrections'
import { FoolChecker } from './FoolChecker'

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
  wells: string[] // the field's wells, for the editor's "name another well" trick
  onAsk: () => void // open the ask bar for a question of one's own
  examples: ReactNode // the other recorded answers, for the first screen
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
  // Inside a quote the words are the report's own, so they stay as written.
  const plain = (from: number, to: number, quoted = false): ReactNode[] => {
    const parts: ReactNode[] = []
    let at = from
    let previous: (typeof numbers)[number] | null = null
    for (const n of numbers) {
      if (n.start < at || n.end > to) continue
      const written = text.slice(n.start, n.end)
      // "324 192 to 469 712 Sm3": the unit after the second number is the first one's too.
      const first =
        previous && /^\s*(?:to|-|–)\s*$/.test(text.slice(previous.end, n.start))
          ? text.slice(previous.start, previous.end)
          : null
      previous = n
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
      // An oil volume in Sm³ also in barrels, set apart as the page's own conversion.
      const oil = quoted ? null : volumeAt(written, text.slice(at, to))
      if (oil) {
        const low = first ? volumeAt(first, text.slice(at, to)) : null
        parts.push(text.slice(at, at + oil.length))
        parts.push(
          <span key={`b${n.start}`} className="bbl" title={BARRELS_NOTE}>
            {' '}
            (about {low && barrels(low.sm3) !== barrels(oil.sm3) ? `${barrels(low.sm3)} to ` : ''}
            {barrels(oil.sm3)} {oil.perDay ? 'barrels a day' : 'barrels'})
          </span>,
        )
        at += oil.length
      }
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
          {plain(q.start + 1, q.end - 1, true)}
        </q>
      ) : (
        <q key={`q${q.start}`}>{plain(q.start + 1, q.end - 1, true)}</q>
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

/** Where the headline ends: the summary's first clause (before a dash, semicolon, colon or
    comma) when it is a statement of 3 to 24 words, else its first sentence. A model's first
    sentence runs to 40 words; its first clause is usually the answer ("F-12 needs attention
    most"). The cut never falls inside a quote or a number. */
function headlineEnd(text: string, spans: Span[] | undefined): number | null {
  const sentence = firstSentenceEnd(text, spans)
  const inside = (cut: number) => (spans ?? []).some((s) => s.start < cut && s.end > cut)
  for (const m of text.slice(0, sentence ?? text.length).matchAll(/\s[—–]\s|[;:,]\s/g)) {
    const cut = m.index ?? 0
    const words = text.slice(0, cut).trim().split(/\s+/).length
    if (words > 24) break
    if (words >= 3 && !inside(cut)) return cut
  }
  return sentence
}

/** The summary with its first clause, the plain answer, as the headline; the rest of it reads
    on beneath. The words are the agent's own, checked like every finding: the page only cuts
    them and starts the rest with a capital. */
function Summary({ text, spans, open, quoteSource }: { text: string; spans?: Span[]; open: Props['onSelectEvidence']; quoteSource?: string }) {
  const cut = headlineEnd(text, spans)
  if (cut === null) return <p className="summary-lead">{bySpans(text, spans, open, quoteSource)}</p>
  const rest = text.slice(cut).replace(/^[\s—–;:,]+/, '')
  const offset = text.length - rest.length
  const shift = (list: Span[] | undefined, from: number, to: number, by: number) =>
    list?.filter((s) => s.start >= from && s.end <= to).map((s) => ({ ...s, start: s.start - by, end: s.end - by }))
  const head = text.slice(0, cut)
  return (
    <>
      <p className={`summary-lead${head.split(/\s+/).length <= 10 ? ' is-short' : ''}`}>
        {bySpans(head, shift(spans, 0, cut, 0), open, quoteSource)}
      </p>
      {rest && (
        <p className="summary">
          {bySpans(rest.charAt(0).toUpperCase() + rest.slice(1), shift(spans, offset, text.length, offset), open, quoteSource)}
        </p>
      )}
    </>
  )
}

export function Answer(props: Props) {
  const { state } = props
  const final = state.final
  return (
    <section id="answer" aria-label="Answer" className="answer-panel">
      <div className="panel-body answer-body" aria-live="polite">
        {final ? (
          <Final {...props} />
        ) : state.status === 'idle' ? (
          <Intro recorded={props.recorded} onReplay={props.onReplay} onAsk={props.onAsk} examples={props.examples} />
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

/** First visit: what this is and one thing to do, watch it work; asking your own question and
    the other recorded answers come after it. */
function Intro({
  recorded,
  onReplay,
  onAsk,
  examples,
}: {
  recorded: RunSummary | null
  onReplay: (id: string) => void
  onAsk: () => void
  examples: ReactNode
}) {
  return (
    <div className="intro">
      <h3 className="intro-title">Ask why a well changed.</h3>
      <p className="intro-lead">
        An AI agent on Meta&rsquo;s open-weight Muse Spark Open works it out from the field&rsquo;s own data and
        reports, and a checker verifies every number before you see it.
      </p>
      {recorded && (
        <div className="intro-case">
          <p className="intro-case-label">
            {recorded.as_of
              ? `It is ${recorded.as_of}. The agent sees no data after that day, and is asked:`
              : 'The agent is asked:'}
          </p>
          <p className="intro-question">{recorded.question}</p>
          <button type="button" className="primary-button intro-cta" onClick={() => onReplay(recorded.id)}>
            Watch it work
          </button>
          <p className="intro-after">
            A recording of a real run, so no wait and no cost.{recorded.as_of && ' Afterwards, see whether its forecast came true.'}
          </p>
        </div>
      )}
      <div className="intro-more">
        <button type="button" className="link-button" onClick={onAsk}>
          Ask your own question
        </button>
        {examples}
      </div>
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
      <p className="intro-who">
        <strong>For production and reservoir engineers.</strong> Why output fell, whether a gauge or the reservoir is
        to blame, what next year brings: questions that mean working across databases, PDF reports and spreadsheets,
        answered in about a minute. The model is open-weight, so an operator can run it next to data that never leaves
        the company. The data here is Volve, a North Sea field Equinor published openly.
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

/** One line on what each Meta model did for this answer, its evidence a click away. */
function MetaLine({ m, open }: { m: MetaModels; open: (id: string) => void }) {
  const chip = (id: string) => (
    <button key={id} type="button" className="evidence-chip meta-chip" onClick={() => open(id)}>
      {id}
    </button>
  )
  const parts: ReactNode[] = [
    <span key="muse">
      <strong>{m.muse.model ? modelName(m.muse.model) : 'The agent'}</strong> planned and ran {m.muse.tool_calls} tool calls
      over {m.muse.steps} steps{m.muse.reasoning_shown && ', its reasoning in the log'}
      {m.muse.pages.length > 0 && <>, looked at a report page {m.muse.pages.map(chip)}</>}
      {m.muse.second_opinion && <>, and gave its answer a second opinion</>}
    </span>,
  ]
  const sam: ReactNode[] = []
  for (const f of m.sam.figures)
    sam.push(
      <span key={f.evidence_id}>
        measured the chart on that page and agreed with Muse on {f.agreed} of the {f.compared} figure readings it could
        compare{f.not_compared ? ` (${f.not_compared} could not be compared)` : ''} {chip(f.evidence_id)}
      </span>,
    )
  if (m.sam.logs.length)
    sam.push(<span key="logs">read F-12&rsquo;s 2009 production log off its plots {m.sam.logs.map(chip)}</span>)
  if (sam.length)
    parts.push(
      <span key="sam">
        ; <strong>SAM 3.1</strong>{' '}
        {sam.map((s, i) => (
          <span key={i}>
            {i > 0 && ', and '}
            {s}
          </span>
        ))}
      </span>,
    )
  return (
    <p className="meta-line">
      <span className="meta-line-label">Meta models in this answer</span>
      {parts}.
    </p>
  )
}

function Final({ state, selectedFinding, selectedEvidence, onSelectFinding, onSelectEvidence, revealed, onReveal, wells }: Props) {
  const final = state.final!
  const tools = Object.fromEntries(final.evidence.map((e) => [e.id, e.tool]))
  const isReport = (id: string) => tools[id] === 'search_reports' || tools[id] === 'read_report_page'
  const anyReport = final.evidence.map((e) => e.id).find(isReport)
  // Rounds up to the first accepted answer; after a second opinion the agent may have
  // submitted again, which is told apart below.
  const review = final.review ?? state.review
  const firstAccepted = final.check_rounds.findIndex((r) => r.length === 0)
  const rounds = review && firstAccepted >= 0 ? final.check_rounds.slice(0, firstAccepted + 1) : final.check_rounds
  const afterReview = review && firstAccepted >= 0 ? final.check_rounds.slice(firstAccepted + 1) : []
  const sentBack = rounds.filter((r) => r.length)
  const problems = sentBack.reduce((n, r) => n + r.length, 0)
  const fixes = corrections(state.steps, final.findings)
  const history = fixes.length ? <CorrectionList items={fixes} bare /> : <Rounds rounds={rounds} />
  const seconds = state.steps.reduce((t, step) => t + step.seconds, 0)
  const sam = final.meta_models && (final.meta_models.sam.figures.length > 0 || final.meta_models.sam.logs.length > 0)

  // The answer first, then one line on how it was checked; the checker's rounds, the models'
  // parts and the second opinion fold under it. An answer that did not pass says so first.
  const status = (
    <div className={`status${final.verified ? '' : ' status-alarm'}`} role="status">
      <div className="status-main">
        {final.verified ? <CheckIcon /> : <AlarmIcon />}
        <p className="status-line">
          <strong>
            {final.verified ? 'Every number, date and quote matches its source.' : 'Not every number could be checked.'}
          </strong>{' '}
          {final.verified
            ? sentBack.length > 0 &&
              `The checker sent the answer back ${sentBack.length === 1 ? 'once' : sentBack.length === 2 ? 'twice' : `${sentBack.length} times`}, with ${problems} ${problems === 1 ? 'issue' : 'issues'}, and the agent fixed ${problems === 1 ? 'it' : 'them'}.`
            : final.finished
              ? final.findings.length
                ? 'Problems remained after the last correction round.'
                : 'No findings were submitted.'
              : `Stopped early: ${final.stopped ?? 'step limit'}.`}
          {final.verified && review?.outcome === 'revised' &&
            ` A second opinion then questioned the reasoning, and the agent revised its answer${afterReview.length > 1 ? `, checked again on attempt ${afterReview.length}` : ''}.`}
          {final.verified && review?.outcome === 'kept' &&
            ' A second opinion then questioned the reasoning; the revision did not pass the checker, so this is the checked answer.'}
        </p>
      </div>
      {sam && final.meta_models && <MetaLine m={final.meta_models} open={onSelectEvidence} />}
      <details className="status-more">
        <summary>How this answer was checked</summary>
        <p className="status-facts">
          {final.steps} steps and {final.evidence.length} tool results in {Math.round(seconds)} s,{' '}
          {compact(final.tokens.input + final.tokens.output)} tokens.
        </p>
        <p className="status-note">
          Every number, date and quote in the answer was found in the tool result it cites, as the well, quantity,
          direction and event the text says, and causes are marked as inferences. Not checked: whether the reasoning
          holds. That is for you to judge, and every source is one click away.
        </p>
        {sentBack.length > 0 && fixes.length > 0 && <CaughtList items={fixes} />}
        {rounds.length > 0 && sentBack.length > 0 && history}
        {!sam && final.meta_models && <MetaLine m={final.meta_models} open={onSelectEvidence} />}
        {review && <SecondOpinion review={review} />}
      </details>
    </div>
  )

  return (
    <>
      {/* The headline is the agent's own summary, checked like every finding: no line of
          the answer is written by the page. */}
      {final.summary && (
        <section className="conclusion" aria-label="Conclusion">
          <h3 className="sr-only">Conclusion</h3>
          <Summary text={final.summary} spans={final.summary_spans} open={onSelectEvidence} quoteSource={anyReport} />
        </section>
      )}

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

      {!final.verified && status}

      <div className="findings-head">
        <h3 className="section-label">Findings</h3>
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
                <p className="sources">
                  <span className="sources-label">Sources</span>
                  {finding.evidence_ids.map((id) => (
                    <button
                      key={id}
                      type="button"
                      className="evidence-chip"
                      aria-pressed={selectedEvidence === id}
                      title={`Open tool result ${id}`}
                      onClick={() => {
                        onSelectFinding(i)
                        onSelectEvidence(id)
                      }}
                    >
                      {tools[id] ? toolName(tools[id]) : id}
                    </button>
                  ))}
                </p>
                {/* The fit statistics and the checker's corrections: for engineers, folded. */}
                {(finding.detail || fixed.length > 0) && (
                  <details className="finding-detail">
                    <summary>Details for engineers</summary>
                    {finding.detail && (
                      <p>{bySpans(finding.detail, finding.detail_spans, onSelectEvidence, finding.evidence_ids.find(isReport))}</p>
                    )}
                    {fixed.length > 0 && (
                      <p className="fix-tag">
                        <RedoIcon />
                        <span>
                          <strong>Fixed after the checker sent it back:</strong>{' '}
                          {[...new Set(fixed.flatMap((c) => c.changes))].join(' ')}
                        </span>
                      </p>
                    )}
                    <p className="finding-ids">Tool results: {finding.evidence_ids.join(', ')}</p>
                  </details>
                )}
              </div>
            </li>
          )
        })}
      </ol>

      {final.verified && status}

      {state.id && final.findings.length > 0 && (
        <FoolChecker key={state.id} runId={state.id} findings={final.findings} selectedFinding={selectedFinding} wells={wells} />
      )}
    </>
  )
}

/** A second reading of the answer's reasoning by the same model, as a senior reservoir
    engineer: what the checker does not check. */
function SecondOpinion({ review }: { review: NonNullable<RunState['review']> }) {
  return (
    <section className={`second-opinion${review.points.length ? '' : ' is-sound'}`} aria-label="Second opinion">
      <h3 className="section-label">Second opinion</h3>
      {review.points.length === 0 ? (
        <p>
          <CheckIcon /> A reviewer (the same model, reading as a senior reservoir engineer) found the reasoning sound. It
          checks what the checker does not: whether the conclusion follows.
        </p>
      ) : (
        <>
          <p>
            A reviewer (the same model, reading as a senior reservoir engineer) questioned the reasoning, which the
            checker does not check:
          </p>
          <ol>
            {review.points.map((p, i) => (
              <li key={i}>
                {p.point} <span className="second-opinion-check">Check: {p.check}</span>
              </li>
            ))}
          </ol>
          <p className="second-opinion-outcome">
            {review.outcome === 'revised'
              ? 'The agent ran the checks and revised its answer; the findings above are the revision, verified again.'
              : review.outcome === 'kept'
                ? 'The agent’s revision did not pass the checker, so the findings above are its checked answer from before the review.'
                : 'The agent is answering these points.'}
          </p>
        </>
      )}
    </section>
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

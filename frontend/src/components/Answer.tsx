import type { ReactNode } from 'react'
import { AlarmIcon, CheckIcon } from '../icons'
import type { DeclaredNumber } from '../api'
import { compact } from '../labels'
import type { RunState } from '../useRun'

type Props = {
  state: RunState
  selectedFinding: number | null
  selectedEvidence: string | null
  onSelectFinding: (index: number | null) => void
  onSelectEvidence: (id: string) => void
}

// A quoted passage opens after a space or "(" and closes before a space or punctuation, so
// an apostrophe inside a word does not end it (same rule as the backend checker).
const QUOTED = /(^|[\s(])['"‘“«]([^]{6,}?)['"’”»](?=[\s.,;:)/]|$)/g

// Dates and well names hold digits that are not declared numbers (same as the checker).
const NOT_NUMBERS = /\b\d{4}-\d{2}(?:-\d{2})?\b|\b(?:15\/9-)?F-\d+(?:\s?[A-Z]\b)?/g

/** Where a declared text first stands as a whole number: "2" in "2 years", not in "2010-07-18" or "2 100". */
function standaloneAt(text: string, written: string): number {
  if (!written) return -1
  const escaped = written.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const pattern = new RegExp(
    `(?<![\\w.,])(?<!\\d[ \\u00a0\\u202f])${escaped}(?!\\d|[.,]\\d|[ \\u00a0\\u202f]\\d{3}\\b)`,
    'g',
  )
  const masked = [...text.matchAll(NOT_NUMBERS)].map((m) => [m.index, m.index + m[0].length])
  for (const match of text.matchAll(pattern)) {
    if (!masked.some(([from, to]) => match.index < to && match.index + written.length > from)) {
      return match.index
    }
  }
  return -1
}

/**
 * The text with each declared number as a button to the evidence it came from, and
 * quoted passages set as quotes.
 */
function withMarks(text: string, numbers: DeclaredNumber[] | null | undefined, open: (id: string) => void): ReactNode[] {
  // Earliest occurrence first; a longer number wins when two start at the same place.
  const spans = (numbers ?? [])
    .map((n) => ({ n, at: standaloneAt(text, n.text) }))
    .filter((s) => s.at >= 0)
    .sort((a, b) => a.at - b.at || b.n.text.length - a.n.text.length)
  const parts: ReactNode[] = []
  let last = 0
  for (const { n, at } of spans) {
    if (at < last) continue
    parts.push(...withQuotes(text.slice(last, at)))
    parts.push(
      <button
        key={`n${at}`}
        type="button"
        className="num"
        title={`${n.value} from ${n.evidence_id}`}
        onClick={() => open(n.evidence_id)}
      >
        {n.text}
      </button>,
    )
    last = at + n.text.length
  }
  parts.push(...withQuotes(text.slice(last)))
  return parts
}

function withQuotes(text: string): ReactNode[] {
  const parts: ReactNode[] = []
  let last = 0
  for (const match of text.matchAll(QUOTED)) {
    const start = match.index + match[1].length
    parts.push(text.slice(last, start))
    parts.push(<q key={start}>{match[2]}</q>)
    last = match.index + match[0].length
  }
  parts.push(text.slice(last))
  return parts
}

export function Answer({ state, selectedFinding, selectedEvidence, onSelectFinding, onSelectEvidence }: Props) {
  const final = state.final
  return (
    <section aria-label="Answer">
      <div className="panel-head">
        <h2>Answer</h2>
        {final && (
          <span className="panel-meta">
            {final.steps} steps, {final.evidence.length} evidence items,{' '}
            {compact(final.tokens.input + final.tokens.output)} tokens
          </span>
        )}
      </div>
      <div className="panel-body" aria-live="polite">
        {!final && (
          <p className="empty">
            {state.status === 'running'
              ? 'The answer appears once the agent submits it and every claim passes the checker.'
              : 'Findings appear here, each one linked to the evidence it rests on.'}
          </p>
        )}
        {final && (
          <>
            {final.summary && (
              <p className="summary">{withMarks(final.summary, final.summary_numbers, onSelectEvidence)}</p>
            )}
            {final.verified ? (
              <div className="verdict">
                <CheckIcon />
                <span>
                  <strong>Verified</strong>
                  Every number, date and quote below was found in the evidence it cites.
                </span>
              </div>
            ) : (
              <div className="verdict verdict-alarm">
                <AlarmIcon />
                <span>
                  <strong>Not verified</strong>
                  {final.finished
                    ? 'The checker still found problems after the last correction round.'
                    : 'The agent ran out of steps before submitting an answer.'}
                </span>
              </div>
            )}
            <ul className="findings">
              {final.findings.map((finding, i) => (
                <li key={i} className={`finding${selectedFinding === i ? ' is-selected' : ''}`}>
                  {/* The marker selects the finding (keyboard); the claim does too on click.
                      The claim cannot be a button: its numbers are buttons to evidence. */}
                  <button
                    type="button"
                    className={`marker ${finding.confidence === 'fact' ? 'marker-fact' : ''}`}
                    aria-pressed={selectedFinding === i}
                    aria-label={`${finding.confidence === 'fact' ? 'Fact' : 'Inference'} ${i + 1}: show its dates on the charts`}
                    onClick={() => onSelectFinding(selectedFinding === i ? null : i)}
                  />
                  <span>
                    <p
                      className="claim"
                      onClick={(e) => {
                        if ((e.target as HTMLElement).closest('button')) return
                        onSelectFinding(selectedFinding === i ? null : i)
                      }}
                    >
                      {withMarks(finding.claim, finding.numbers, onSelectEvidence)}
                    </p>
                    <span className="chips">
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
                        </button>
                      ))}
                    </span>
                  </span>
                </li>
              ))}
            </ul>
            <div className="marker-key">
              <span>
                <span className="marker marker-fact" /> Fact shown directly by the evidence
              </span>
              <span>
                <span className="marker" /> Inference, such as a likely cause
              </span>
            </div>
            {final.check_rounds.length > 0 && (
              <details className="rounds">
                <summary>
                  Checked {final.check_rounds.length}{' '}
                  {final.check_rounds.length === 1 ? 'time' : 'times'}
                  {final.check_rounds.some((r) => r.length) &&
                    `, ${final.check_rounds.reduce((n, r) => n + r.length, 0)} problems sent back to the agent`}
                </summary>
                <ol>
                  {final.check_rounds.map((round, i) => (
                    <li key={i}>
                      {round.length === 0 ? 'Passed' : (
                        <ul>
                          {round.map((p, j) => <li key={j}>{p}</li>)}
                        </ul>
                      )}
                    </li>
                  ))}
                </ol>
              </details>
            )}
          </>
        )}
      </div>
    </section>
  )
}

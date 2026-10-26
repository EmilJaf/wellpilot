import type { ReactNode } from 'react'
import { AlarmIcon, CheckIcon } from '../icons'
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
      <div className="panel-body">
        {!final && (
          <p className="empty">
            {state.status === 'running'
              ? 'The answer appears once the agent submits it and every claim passes the checker.'
              : 'Findings appear here, each one linked to the evidence it rests on.'}
          </p>
        )}
        {final && (
          <>
            {final.summary && <p className="summary">{withQuotes(final.summary)}</p>}
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
                  <span
                    className={`marker ${finding.confidence === 'fact' ? 'marker-fact' : ''}`}
                    role="img"
                    aria-label={finding.confidence === 'fact' ? 'Fact' : 'Inference'}
                  />
                  <span>
                    <button
                      type="button"
                      className="claim"
                      aria-pressed={selectedFinding === i}
                      onClick={() => onSelectFinding(selectedFinding === i ? null : i)}
                    >
                      {withQuotes(finding.claim)}
                    </button>
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

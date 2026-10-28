import { type Correction, type Piece } from '../corrections'
import { AlarmIcon } from '../icons'
import { plainProblem } from '../labels'

/** The problem as one plain sentence, without the "Finding 3" the list already shows. */
function problemText(c: Correction): string {
  return plainProblem(c.problem)
    .replace(/^(Finding \d+|Summary)( is|:)?\s*/, '')
    .replace(/^./, (s) => s.toUpperCase())
}

/** Long unchanged runs shortened to a few words of context on each side of a change. */
function elide(text: string, first: boolean, last: boolean): string {
  const w = text.split(' ')
  if (w.length <= 10) return text
  const head = first ? [] : w.slice(0, 4)
  const tail = last ? [] : w.slice(-4)
  return [...head, '…', ...tail].join(' ')
}

export function Diff({ pieces }: { pieces: Piece[] }) {
  return (
    <p className="fix-diff">
      {/* the space between words stays outside the marks, so a strike-through starts at the word */}
      {pieces.map((p, i) => (
        <span key={i}>
          {i > 0 && ' '}
          {p.kind === 'del' ? (
            <del>{p.text}</del>
          ) : p.kind === 'ins' ? (
            <ins>{p.text}</ins>
          ) : (
            elide(p.text, i === 0, i === pieces.length - 1)
          )}
        </span>
      ))}
    </p>
  )
}

/**
 * What the checker caught before the reader saw the answer, one plain line each, in sight:
 * the proof that the checker is not a formality. How each was fixed sits behind a disclosure.
 */
// Errors of meaning first (a limit said to hold, a number given another name, unit,
// direction, date or well, a cause stated as fact), then the bookkeeping (a number not
// declared, not in the cited result).
const MEANING = /stayed within|is named|is written in|is written as|belongs to|wrong way|only for|dead gauge|states a cause|ties it|date|called the change|model's P50|none of its evidence|share below|survey of|calls the decline|is the fit from|rests on evidence with warnings/

const BOOKKEEPING = /undeclared|round '|three significant|claim holds \d+ numbers/i

export function CaughtList({ items: all, max = 4 }: { items: Correction[]; max?: number }) {
  // The same problem raised twice in one answer (a number written twice) is listed once.
  const items = all.filter(
    (c, i) => all.findIndex((d) => d.target === c.target && problemText(d) === problemText(c)) === i,
  )
  // Bookkeeping (a number written without naming its source, a rounding) is one line; what
  // the answer got wrong is listed, so a real catch is not lost among the paperwork.
  const bookkeeping = items.filter((c) => BOOKKEEPING.test(c.problem) && !MEANING.test(c.problem))
  const ranked = items
    .filter((c) => !bookkeeping.includes(c))
    .sort((a, b) => Number(!MEANING.test(a.problem)) - Number(!MEANING.test(b.problem)))
  const shown = ranked.slice(0, max)
  return (
    <div className="caught">
      <p className="caught-title">
        {ranked.length === 0
          ? 'The checker sent the answer back for its paperwork:'
          : `The checker caught ${ranked.length === 1 ? 'this' : `these ${ranked.length}`} before you saw the answer:`}
      </p>
      {bookkeeping.length > 0 && (
        <p className="caught-paperwork">
          {bookkeeping.length === 1 ? 'One number was' : `${bookkeeping.length} numbers were`} written without naming
          the result it comes from, or not rounded; the agent added the sources.
        </p>
      )}
      <ul>
        {shown.map((c, i) => {
          const fixed = c.changes.length > 0 && !c.changes[0].startsWith('The run ended')
          return (
            <li key={i}>
              <AlarmIcon />
              <span>
                {c.target && <strong>{c.target}: </strong>}
                {problemText(c)}
                {fixed && <span className="caught-fixed"> Fixed.</span>}
              </span>
            </li>
          )
        })}
      </ul>
      {ranked.length > max && <p className="caught-more">and {ranked.length - max} more, below.</p>}
    </div>
  )
}

/** What the checker caught, attempt by attempt, and what the agent changed each time. */
export function CorrectionList({ items, bare = false }: { items: Correction[]; bare?: boolean }) {
  const attempts = [...new Set(items.map((c) => c.attempt))]
  // One entry per finding: its problems together, what changed once.
  const byTarget = (attempt: number) => {
    const groups = new Map<string, Correction[]>()
    items
      .filter((c) => c.attempt === attempt)
      .forEach((c, i) => groups.set(c.target || `#${i}`, [...(groups.get(c.target || `#${i}`) ?? []), c]))
    return [...groups.values()]
  }
  const rounds = attempts.map((attempt) => (
    <div key={attempt} className="fix-round">
      <p className="round-label">Attempt {attempt} was sent back:</p>
      <ol>
        {byTarget(attempt).map((group, i) => {
          const changes = [...new Set(group.flatMap((c) => c.changes))]
          const diff = group.find((c) => c.diff)?.diff
          return (
            <li key={i} title={group.map((c) => c.problem).join('\n')}>
              {group[0].target && <p className="fix-target">{group[0].target}</p>}
              {group.map((c, j) => (
                <p key={j} className="fix-problem">
                  {problemText(c)}
                </p>
              ))}
              {changes.length > 0 && <p className="fix-change">{changes.join(' ')}</p>}
              {diff && <Diff pieces={diff} />}
            </li>
          )
        })}
      </ol>
    </div>
  ))
  // Inside another disclosure (the answer's status) the list shows as it is.
  if (bare) return <div className="fixes">{rounds}</div>
  return (
    <details className="fixes">
      <summary>
        What the checker caught, and how the agent fixed {items.length === 1 ? 'it' : `all ${items.length}`}
      </summary>
      {rounds}
    </details>
  )
}

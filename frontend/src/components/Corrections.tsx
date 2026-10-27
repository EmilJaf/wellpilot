import { type Correction, type Piece } from '../corrections'
import { brief } from '../labels'

/** The problem as one plain sentence, without the "Finding 3" the list already shows. */
function problemText(c: Correction): string {
  const warning = c.problem.match(/\((E\d+): ([^.;:]+)/)
  return brief(c.problem)
    .replace(/^(Finding \d+|Summary)( is|:)?\s*/, '')
    .replace(/^undeclared number\(s\) (.+?)\.$/i, '$1 written without naming the tool result it comes from.')
    .replace(/evidence with warnings\.$/, warning ? `a tool result with a warning (${warning[1]}: ${warning[2]}).` : '$&')
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

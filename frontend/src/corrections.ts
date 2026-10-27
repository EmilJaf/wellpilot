// What the checker sent back, paired with what the agent changed in its next attempt.

import type { DeclaredNumber, Finding, StepEvent } from './api'

/** One submit_findings call as the agent sent it, with the checker's verdict. */
type Submission = {
  summary: string
  findings: Finding[]
  summaryNumbers: DeclaredNumber[]
  problems: string[]
}

export type Piece = { kind: 'same' | 'del' | 'ins'; text: string }

/** A problem the checker sent back, and what the agent changed in its next submission. */
export type Correction = {
  attempt: number
  problem: string
  target: string // "Finding 3", "Summary", or "" for the whole answer
  changes: string[]
  diff: Piece[] | null
  finalIndex: number | null // the final finding it became, for marking it there
}

function submissions(steps: StepEvent[]): Submission[] {
  return steps
    .flatMap((s) => s.tool_calls)
    .filter((c) => c.name === 'submit_findings' && c.accepted !== undefined)
    .map((c) => {
      const args = c.arguments as { summary?: string; findings?: Finding[]; summary_numbers?: DeclaredNumber[] }
      return {
        summary: args.summary ?? '',
        findings: Array.isArray(args.findings) ? args.findings : [],
        summaryNumbers: args.summary_numbers ?? [],
        problems: c.problems ?? [],
      }
    })
}

const words = (text: string) => new Set(text.toLowerCase().split(/[^\p{L}\p{N}.,%-]+/u).filter(Boolean))

function similarity(a: string, b: string): number {
  const x = words(a)
  const y = words(b)
  const shared = [...x].filter((w) => y.has(w)).length
  return shared / Math.max(1, new Set([...x, ...y]).size)
}

/** The finding in `list` that `claim` became: the same position if it still reads alike, else the closest. */
function counterpart(claim: string, list: Finding[], at: number): number | null {
  if (list[at] && similarity(claim, list[at].claim) >= 0.4) return at
  let best: number | null = null
  let score = 0.4
  for (let i = 0; i < list.length; i++) {
    const s = similarity(claim, list[i].claim)
    if (s >= score) [best, score] = [i, s]
  }
  return best
}

/** Word-level diff (longest common subsequence), for showing a rewording. */
export function wordDiff(before: string, after: string): Piece[] {
  const a = before.split(/\s+/).filter(Boolean)
  const b = after.split(/\s+/).filter(Boolean)
  const lcs = a.map(() => new Array<number>(b.length + 1).fill(0))
  lcs.push(new Array<number>(b.length + 1).fill(0))
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1])
    }
  }
  const pieces: Piece[] = []
  const push = (kind: Piece['kind'], text: string) => {
    const last = pieces[pieces.length - 1]
    if (last?.kind === kind) last.text += ` ${text}`
    else pieces.push({ kind, text })
  }
  let i = 0
  let j = 0
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) {
      push('same', a[i++])
      j++
    } else if (i < a.length && (j >= b.length || lcs[i + 1][j] >= lcs[i][j + 1])) {
      push('del', a[i++])
    } else {
      push('ins', b[j++])
    }
  }
  return pieces
}

const list = (ids: string[]) => ids.join(', ')
const plural = (n: number, word: string) => (n === 1 ? word : `${word}s`)

function describe(
  before: { text: string; numbers: DeclaredNumber[]; confidence?: string; evidence?: string[] },
  after: { text: string; numbers: DeclaredNumber[]; confidence?: string; evidence?: string[] } | null,
): { changes: string[]; diff: Piece[] | null } {
  if (!after) return { changes: ['The agent removed it.'], diff: null }
  const changes: string[] = []
  if (before.confidence && after.confidence && before.confidence !== after.confidence) {
    changes.push(`Relabelled it ${after.confidence === 'fact' ? 'Fact' : 'Inference'}.`)
  }
  const dropped = (before.evidence ?? []).filter((id) => !(after.evidence ?? []).includes(id))
  const added = (after.evidence ?? []).filter((id) => !(before.evidence ?? []).includes(id))
  if (dropped.length) changes.push(`Dropped ${plural(dropped.length, 'source')} ${list(dropped)}.`)
  if (added.length) changes.push(`Added ${plural(added.length, 'source')} ${list(added)}.`)
  const declared = after.numbers.filter((n) => !before.numbers.some((m) => m.text === n.text))
  if (declared.length) {
    changes.push(`Named the source of ${declared.map((n) => `${n.text} (${n.evidence_id})`).join(', ')}.`)
  }
  const reworded = before.text !== after.text
  if (reworded) changes.push('Reworded it.')
  if (!changes.length) changes.push('Resubmitted it unchanged.')
  return { changes, diff: reworded ? wordDiff(before.text, after.text) : null }
}

/** Every problem the checker sent back, paired with what changed in the next attempt. */
export function corrections(steps: StepEvent[], final: Finding[]): Correction[] {
  const subs = submissions(steps)
  return subs.flatMap((sub, k) =>
    sub.problems.map((problem): Correction => {
      const next = subs[k + 1]
      const finding = problem.match(/^Finding (\d+)/)
      const target = finding ? `Finding ${finding[1]}` : /^Summary/.test(problem) ? 'Summary' : ''
      const base = { attempt: k + 1, problem, target, finalIndex: null }
      if (!next) return { ...base, changes: ['The run ended before the agent fixed it.'], diff: null }
      if (target === 'Summary') {
        return {
          ...base,
          ...describe(
            { text: sub.summary, numbers: sub.summaryNumbers },
            { text: next.summary, numbers: next.summaryNumbers },
          ),
        }
      }
      const index = finding ? Number(finding[1]) - 1 : -1
      const before = sub.findings[index]
      if (!before) return { ...base, changes: [], diff: null }
      const at = counterpart(before.claim, next.findings, index)
      const after = at !== null ? next.findings[at] : null
      const view = (f: Finding) => ({
        text: f.claim,
        numbers: f.numbers ?? [],
        confidence: f.confidence,
        evidence: f.evidence_ids,
      })
      return {
        ...base,
        ...describe(view(before), after ? view(after) : null),
        finalIndex: after ? counterpart(after.claim, final, at ?? 0) : null,
      }
    }),
  )
}

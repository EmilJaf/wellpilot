// Plain-language names for the agent's tools, as an engineer would describe the step.

export const TOOL_NAMES: Record<string, string> = {
  list_wells: 'List wells',
  production_monthly: 'Monthly production',
  detect_events: 'Detect events',
  fit_decline: 'Fit decline curve',
  forecast_production: 'Forecast oil',
  run_sql: 'Query data',
  calculate: 'Calculate',
  search_reports: 'Search reports',
  read_report_page: 'Read report page',
  plan_vs_actual: 'Compare with the 2005 plan',
  field_vs_design: 'Field against design limits',
  field_screen: 'Screen every well',
  waterflood: 'Waterflood diagnostics',
  look_at_page: 'Look at a report page',
  submit_findings: 'Submit answer',
}

export const toolName = (name: string) => TOOL_NAMES[name] ?? name

/** One line describing what a call was asked to do. */
export function describeArgs(name: string, args: Record<string, unknown>): string {
  const parts: string[] = []
  if (typeof args.well === 'string') parts.push(args.well)
  if (args.start || args.end) parts.push(`${args.start ?? 'start'} to ${args.end ?? 'end'}`)
  if (typeof args.model === 'string' && args.model !== 'auto') parts.push(`${args.model} model`)
  if (typeof args.horizon_days === 'number') parts.push(`${args.horizon_days} days ahead`)
  if (typeof args.query === 'string') parts.push(`“${args.query}”`)
  if (typeof args.topic === 'string') parts.push(`topic ${args.topic.replace(/_/g, ' ')}`)
  if (typeof args.expression === 'string') parts.push(args.expression)
  if (typeof args.sql === 'string') parts.push(args.sql.replace(/\s+/g, ' ').slice(0, 120))
  if (typeof args.doc_id === 'string') parts.push(`${args.doc_id} page ${args.page}`)
  if (typeof args.question === 'string') parts.push(`“${args.question}”`)
  if (name === 'submit_findings' && Array.isArray(args.findings)) {
    parts.push(`${args.findings.length} findings`)
  }
  return parts.join(', ')
}

/** ISO dates and months mentioned in a text, for marking them on the charts. */
export function datesIn(text: string): string[] {
  return [...new Set(text.match(/\b\d{4}-\d{2}(?:-\d{2})?\b/g) ?? [])]
}

/** Wells named in a text, e.g. "F-12", "F-15 D", "P-3". */
export function wellsIn(text: string): string[] {
  return [...new Set(text.match(/\b[A-Z]{1,3}-\d+(?: [A-D]\b)?/g) ?? [])]
}

export const compact = (n: number) =>
  n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n)

/**
 * A checker problem in one short sentence: the first sentence, without long parentheticals
 * (the full fit warning). The full text stays in the answer's record of the checks.
 */
export function brief(problem: string): string {
  const shorter = problem.replace(/\s*\([^()]{40,}\)/g, '')
  const first = shorter.match(/^.*?[.!?](?=\s|$)/)
  return (first ? first[0] : shorter).trim()
}

/**
 * A checker problem in plain words, for readers rather than the model: its first sentence,
 * with the checker's terms ("undeclared", "evidence with warnings") said as what they mean.
 * The model still gets the full text, with its instructions.
 */
export function plainProblem(problem: string): string {
  const warning = problem.match(/\((E\d+): ([^.;:]+)/)
  return brief(problem)
    .replace(/undeclared number\(s\) (.+?)\.$/i, '$1 written without naming the tool result it comes from.')
    .replace(/'(.+?)' = \S+ is not in (E\d+)\.$/, "'$1' is not in its source, $2.")
    .replace(/'(.+?)' is named (.+?), but in (E\d+) it is the (.+?)\.$/, "'$1' is called $2, but $3 has it as $4.")
    .replace(/'(.+?)' is in (E\d+) only for (.+?), not for the well its sentence names\.$/, "'$1' belongs to $3 in $2, not to the well named.")
    .replace(/'(.+?)' is written as a rise, but (E\d+) has it as a fall.*$/, "'$1' is written as a rise; $2 shows a fall.")
    .replace(/'(.+?)' is in (E\d+), but the text ties it to (\S+), and .*$/, "'$1' is dated $3, but that date is another event's.")
    .replace(/states a cause \('(.+?)'\) but is marked fact\.$/, "states a cause ('$1') as a fact; causes are inferences.")
    .replace(/is marked fact but rests on evidence with warnings\.?$/, warning ? `rests on a tool result with a warning (${warning[1]}: ${warning[2]}).` : 'rests on a tool result with a warning.')
    .replace(/'(.+?)' comes from (E\d+), which has warnings.*$/, "'$1' comes from an unreliable result ($2) but reads as certain.")
    .replace(/quoted text not found in .*$/, 'quotes something its sources do not say word for word.')
    .replace(/: round (.+?) in the (?:claim|summary) to three significant digits.*$/, ': asked to round $1 so it reads at a glance.')
    .replace(/: (.+?) (?:is a label|are labels) a reader may not know.*$/, ': asked to say $1 in plain words.')
}

/** Letters and digits only, lowercased: how quotes are compared (as the backend does). */
export const squash = (text: string) => text.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '')

/** Does a quote on a page match a passage the reader is looking at? */
export function sameQuote(a: string, b: string): boolean {
  const x = squash(a)
  const y = squash(b)
  return x.length >= 6 && y.length >= 6 && (x.includes(y) || y.includes(x))
}

/** Forecast methods by their short names, as the backtest and the time machine show them. */
export const SHORT: Record<string, string> = {
  agent: 'agent',
  classic: 'classic',
  recent_3m: 'last 3 months',
  recent_6m: 'last 6 months',
  recent_12m: 'last 12 months',
  recent_24m: 'last 24 months',
}

/** A relative forecast error as a signed percent, e.g. +7% or −52%. */
export const signed = (e: number) => `${e >= 0 ? '+' : '−'}${Math.abs(Math.round(e * 100))}%`

/** An oil volume, in k Sm³ or MSm³. */
export const volume = (sm3: number) => (sm3 >= 1e6 ? `${(sm3 / 1e6).toFixed(2)} MSm³` : `${Math.round(sm3 / 1000)}k Sm³`)

/** A tool warning as a short label for the log ("poor fit: r2_log 0.688 < 0.7"); the full
    text is in its title and in the evidence. */
export function shortWarning(warning: string): string {
  const sentence = warning.match(/^.*?\.(?=\s|$)/)?.[0] ?? warning
  if (sentence.length <= 70) return sentence.replace(/\.$/, '')
  const clause = sentence.match(/^[^:;]+/)?.[0] ?? sentence
  return clause.length <= 70 ? clause : `${clause.slice(0, 67)}…`
}

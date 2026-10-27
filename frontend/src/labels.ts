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
  if (name === 'submit_findings' && Array.isArray(args.findings)) {
    parts.push(`${args.findings.length} findings`)
  }
  return parts.join(', ')
}

/** ISO dates and months mentioned in a text, for marking them on the charts. */
export function datesIn(text: string): string[] {
  return [...new Set(text.match(/\b\d{4}-\d{2}(?:-\d{2})?\b/g) ?? [])]
}

/** Wells named in a text, e.g. "F-12", "F-15 D". */
export function wellsIn(text: string): string[] {
  return [...new Set(text.match(/\bF-\d+(?: [A-D]\b)?/g) ?? [])]
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

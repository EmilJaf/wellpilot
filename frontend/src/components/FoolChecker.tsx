import { useEffect, useState } from 'react'
import { api, type AttackResult, type CheckResult, type Finding } from '../api'
import { AlarmIcon, CautionIcon, CheckIcon } from '../icons'
import { plainProblem } from '../labels'
import { wordDiff } from '../corrections'
import { Diff } from './Corrections'

const KINDS: Record<string, string> = {
  number: 'a changed number',
  well: 'another well',
  date: 'a moved date',
  direction: 'a reversed direction',
  quote: 'a misquote',
  cause: 'a cause called a fact',
  scale: 'a wrong scale',
  unsupported: 'a value from elsewhere',
}

/**
 * "Try to fool the checker": the reader edits one finding of a finished answer and the
 * deterministic checker runs again on the run's saved tool results, with no model call.
 * The quick tricks apply the classic ways an answer goes wrong; the textarea takes any edit.
 * Or the model tries: it writes false versions of the finding, and each goes through the
 * same checker.
 */
export function FoolChecker({
  runId,
  findings,
  selectedFinding,
  wells,
}: {
  runId: string
  findings: Finding[]
  selectedFinding: number | null
  wells: string[]
}) {
  const [index, setIndex] = useState(selectedFinding ?? 0)
  const original = findings[index] ?? findings[0]
  const [claim, setClaim] = useState(original.claim)
  const [confidence, setConfidence] = useState(original.confidence)
  const [result, setResult] = useState<{ value?: CheckResult; error?: string; claim: string } | null>(null)
  const [checking, setChecking] = useState(false)
  const [attacks, setAttacks] = useState<{ index: number; value?: AttackResult; error?: string } | null>(null)
  const [attacking, setAttacking] = useState(false)
  const [modelOn, setModelOn] = useState(false)

  useEffect(() => {
    api
      .health()
      .then((h) => setModelOn(h.llm_configured))
      .catch(() => setModelOn(false))
  }, [])

  // Following the selection: picking a finding in the answer brings it into the editor
  // (state adjusted during render, so no effect runs after the paint).
  const [followed, setFollowed] = useState(selectedFinding)
  if (selectedFinding !== followed) {
    setFollowed(selectedFinding)
    if (selectedFinding !== null && selectedFinding < findings.length) pick(selectedFinding)
  }

  function pick(i: number) {
    setIndex(i)
    setClaim(findings[i].claim)
    setConfidence(findings[i].confidence)
    setResult(null)
  }

  function reset() {
    setClaim(original.claim)
    setConfidence(original.confidence)
    setResult(null)
  }

  const numberSpans = (original.spans ?? []).filter((s) => s.kind === 'number')
  const quoteSpan = (original.spans ?? []).find((s) => s.kind === 'quote')
  const wellName = claim.match(/\b(?:15\/9-)?[A-Z]{1,3}-\d+(?:\s?[A-Z]\b)?/)?.[0]
  const otherWell = wellName && [...wells, 'F-14', 'F-12'].find((w) => w !== wellName && w !== wellName.replace(/^15\/9-/, ''))

  const tricks: { label: string; apply: () => void; show: boolean }[] = [
    {
      label: 'Change a number',
      show: numberSpans.length > 0 && claim === original.claim,
      apply: () => {
        const span = numberSpans[0]
        const text = original.claim.slice(span.start, span.end)
        setClaim(original.claim.slice(0, span.start) + bumped(text) + original.claim.slice(span.end))
      },
    },
    {
      label: 'Add a cause',
      show: !/\bbecause\b/.test(claim),
      apply: () => setClaim(claim.replace(/\.?\s*$/, '') + ' because the injectors pushed water into the well.'),
    },
    { label: 'Call it a fact', show: confidence === 'inference', apply: () => setConfidence('fact') },
    {
      label: 'Name another well',
      show: Boolean(wellName && otherWell),
      apply: () => wellName && otherWell && setClaim(claim.replace(wellName, otherWell)),
    },
    {
      label: 'Misquote',
      show: Boolean(quoteSpan) && claim === original.claim,
      apply: () => {
        if (!quoteSpan) return
        const quote = original.claim.slice(quoteSpan.start, quoteSpan.end)
        // Drop the quote's last word, keeping its closing mark.
        const misquoted = quote.replace(/\s+\S+(\s*[”"’'»])$/u, '$1')
        setClaim(original.claim.slice(0, quoteSpan.start) + misquoted + original.claim.slice(quoteSpan.end))
      },
    },
  ]

  async function letTheModelTry() {
    setAttacking(true)
    const at = index
    try {
      setAttacks({ index: at, value: await api.attack(runId, at + 1) })
    } catch (e) {
      setAttacks({ index: at, error: (e as Error).message })
    } finally {
      setAttacking(false)
    }
  }

  async function check() {
    setChecking(true)
    const edited = claim
    try {
      const value = await api.check(runId, { findings: [{ index: index + 1, claim: edited, confidence }] })
      setResult({ value, claim: edited })
    } catch (e) {
      setResult({ error: (e as Error).message, claim: edited })
    } finally {
      setChecking(false)
    }
  }

  const changed = claim !== original.claim || confidence !== original.confidence
  const onlyLabel = claim === original.claim && confidence === 'fact' && original.confidence === 'inference'

  return (
    <section id="fool" className="fool" aria-label="Try to fool the checker">
      <h3 className="fool-title">Try to fool the checker</h3>
      <p className="fool-lead">
        Edit a finding below, or pick a trick, then check it. The checker is plain code, no AI: it compares the text
        with this run&rsquo;s saved results, so it catches wrong numbers, units, dates, wells, directions, quotes and
        causes stated as facts. It does not judge meaning in words (&ldquo;most&rdquo; for &ldquo;least&rdquo; passes).
      </p>
      <div className="fool-pick">
        <label>
          Finding{' '}
          <select value={index} onChange={(e) => pick(Number(e.target.value))}>
            {findings.map((_, i) => (
              <option key={i} value={i}>
                {String(i + 1).padStart(2, '0')}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className={`pill pill-button${confidence === 'fact' ? ' pill-fact' : ''}`}
          onClick={() => setConfidence(confidence === 'fact' ? 'inference' : 'fact')}
          title="Switch between fact and inference"
        >
          {confidence === 'fact' ? 'Fact' : 'Inference'}
        </button>
      </div>
      <textarea
        aria-label="The finding, to edit"
        value={claim}
        rows={3}
        onChange={(e) => setClaim(e.target.value)}
      />
      <p className="fool-tricks">
        <span className="suggestions-label">Try</span>
        {tricks
          .filter((t) => t.show)
          .map((t) => (
            <button key={t.label} type="button" className="link-button" onClick={t.apply}>
              {t.label}
            </button>
          ))}
        {changed && (
          <button type="button" className="link-button" onClick={reset}>
            Reset
          </button>
        )}
      </p>
      <div className="fool-actions">
        <button type="button" className="primary-button" onClick={check} disabled={checking || !claim.trim()}>
          {checking ? 'Checking' : 'Check this version'}
        </button>
        {modelOn && (
          <button type="button" className="secondary-button" onClick={letTheModelTry} disabled={attacking}>
            {attacking ? 'The model is trying' : 'Let the model try'}
          </button>
        )}
      </div>
      {result && result.claim === claim && (
        <div className="fool-result" aria-live="polite">
          {result.error ? (
            <div className="flag flag-alarm">
              <AlarmIcon />
              <span>The check could not run: {result.error}</span>
            </div>
          ) : result.value?.accepted ? (
            <div className="flag flag-ok">
              <CheckIcon />
              <span>
                Accepted: every number, date and quote in this version is in its sources.
                {onlyLabel &&
                  ' Fact or inference is judged by warnings on the evidence and causal wording; a plain claim on clean evidence may carry either label.'}
              </span>
            </div>
          ) : (
            <div className="flag flag-alarm">
              <AlarmIcon />
              <div>
                <strong>Caught: sent back with {result.value!.problems.length === 1 ? 'one problem' : `${result.value!.problems.length} problems`}.</strong>
                <ul>
                  {result.value!.problems.map((p, i) => (
                    <li key={i}>
                      {sentence(plainProblem(p).replace(/^Finding \d+:?\s*/, ''))}
                      {closest(p) && <span className="fool-closest"> {closest(p)}</span>}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
          {claim !== original.claim && <Diff pieces={wordDiff(original.claim, claim)} />}
        </div>
      )}
      {attacks && attacks.index === index && <Attempts result={attacks} />}
    </section>
  )
}

/** The model's attempts on one finding and what the checker made of each. */
function Attempts({ result }: { result: { value?: AttackResult; error?: string } }) {
  if (result.error)
    return (
      <div className="flag flag-alarm fool-attempts">
        <AlarmIcon />
        <span>The model could not try: {result.error}</span>
      </div>
    )
  const attempts = result.value!.attempts.filter((a) => !a.unchanged)
  const caught = attempts.filter((a) => a.caught).length
  const all = caught === attempts.length
  return (
    <div className="fool-attempts" aria-live="polite">
      <p className="fool-attempts-head">
        <strong>
          The model ({result.value!.model}) wrote {attempts.length === 1 ? 'one false version' : `${attempts.length} false versions`}{' '}
          of this finding; the checker caught {all ? (attempts.length === 1 ? 'it' : 'all of them') : `${caught} of them`}.
        </strong>
      </p>
      <ol>
        {attempts.map((a, i) => (
          <li key={i} className={a.caught ? 'is-caught' : 'is-through'}>
            <p className="fool-attempt-kind">
              {a.caught ? <CheckIcon /> : <CautionIcon />} {KINDS[a.kind] ?? a.kind}
              {a.confidence === 'fact' ? ', labelled a fact' : ', labelled an inference'}
            </p>
            <Diff pieces={wordDiff(a.original, a.claim)} />
            <p className="fool-attempt-note">What the model says is false: {a.what_is_false}</p>
            <p className="fool-attempt-verdict">
              {a.caught
                ? `Caught: ${plainProblem(a.problems[0]).replace(/^Finding \d+:?\s*/, '')}`
                : 'Got through. Read it: the checker tests sources, not meaning, and the model may have written something still true.'}
            </p>
          </li>
        ))}
      </ol>
    </div>
  )
}

/** A problem as a sentence: a capital first, and a cause the checker names in plain words. */
function sentence(problem: string): string {
  const plain = problem.replace(/^states a cause \('([^']+)'\) as a fact; causes are inferences\.?$/, (_, word) =>
    `"${word}" states a cause, and a cause is an interpretation: the finding must be marked Inference.`,
  )
  return plain.charAt(0).toUpperCase() + plain.slice(1)
}

/** The checker's hint on a number that is not in its source: the nearest values there. */
function closest(problem: string): string | null {
  const m = problem.match(/Closest there: ([^]*?)\.(?:\s|$)/)
  return m ? `The closest values in that result are ${m[1]}.` : null
}

/** The number as written, about 17% higher, in the same style: "1 551" -> "1 815", "69%" -> "81%", "0.45" -> "0.53". */
function bumped(text: string): string {
  const m = text.match(/^(\d[\d   ,]*)(?:([.,])(\d+))?(.*)$/)
  if (!m) return text
  const [, whole, point, fraction, rest] = m
  const sep = whole.match(/[   ,]/)?.[0]
  const value = Number(whole.replace(/[   ,]/g, '') + (fraction ? `.${fraction}` : ''))
  const raised = value * 1.17
  const decimals = fraction?.length ?? 0
  let out = raised.toFixed(decimals)
  if (point && fraction) out = out.replace('.', point)
  if (sep) out = out.replace(/\B(?=(\d{3})+(?!\d))/g, sep)
  return out + rest
}

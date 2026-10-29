import { useEffect, useRef } from 'react'
import { CheckIcon } from '../icons'
import { STEPS, type Step } from '../story'

/**
 * The demo as five steps across the top. Any step opens on its own (a judge coming back
 * goes straight to the one they want), and each has an address, ?step=N, so a screen can
 * be opened directly for a video take.
 */
export function StoryBar({ current, done, onStep }: { current: Step; done: (s: Step) => boolean; onStep: (s: Step) => void }) {
  // On a phone the bar scrolls sideways: keep the current step in it, without moving the page.
  const bar = useRef<HTMLElement>(null)
  useEffect(() => {
    const nav = bar.current
    const el = nav?.querySelector<HTMLElement>('.is-current')
    if (!nav || !el) return
    if (el.offsetLeft < nav.scrollLeft || el.offsetLeft + el.offsetWidth > nav.scrollLeft + nav.clientWidth) {
      nav.scrollTo({ left: el.offsetLeft - 16, behavior: 'smooth' })
    }
  }, [current])
  // A phone shows the current step only, with arrows to the ones beside it.
  const at = STEPS.findIndex((s) => s.step === current)
  const prev = STEPS[at - 1]
  const next = STEPS[at + 1]
  return (
    <nav className="story" aria-label="The demo in five steps" ref={bar}>
      <button
        type="button"
        className="story-arrow story-prev"
        disabled={!prev}
        onClick={() => prev && onStep(prev.step)}
        aria-label={prev ? `Back to step ${prev.step}: ${prev.label}` : 'First step'}
      >
        &lsaquo;
      </button>
      <span className="story-count">
        Step {current} of {STEPS.length}
      </span>
      <ol>
        {STEPS.map(({ step, label, hint }) => (
          <li key={step}>
            <a
              href={`?step=${step}`}
              className={`story-step${step === current ? ' is-current' : ''}${done(step) && step !== current ? ' is-done' : ''}`}
              aria-current={step === current ? 'step' : undefined}
              title={hint}
              onClick={(e) => {
                if (e.metaKey || e.ctrlKey || e.shiftKey) return // a new tab opens the address
                e.preventDefault()
                onStep(step)
              }}
            >
              <span className="story-number" aria-hidden="true">
                {done(step) && step !== current ? <CheckIcon /> : step}
              </span>
              {label}
            </a>
          </li>
        ))}
      </ol>
      <button
        type="button"
        className="story-arrow story-next"
        disabled={!next}
        onClick={() => next && onStep(next.step)}
        aria-label={next ? `On to step ${next.step}: ${next.label}` : 'Last step'}
      >
        &rsaquo;
      </button>
    </nav>
  )
}

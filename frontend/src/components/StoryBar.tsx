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
  return (
    <nav className="story" aria-label="The demo in five steps" ref={bar}>
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
    </nav>
  )
}

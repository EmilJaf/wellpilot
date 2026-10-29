// The demo's five steps, in the story bar and in the address (?step=N).
export type Step = 1 | 2 | 3 | 4 | 5

export const STEPS: { step: Step; label: string; hint: string }[] = [
  { step: 1, label: 'The question', hint: 'The end of 2010: the agent sees no data after that day.' },
  { step: 2, label: 'Watch it work', hint: 'The agent plans, runs real analysis and checks itself, step by step.' },
  { step: 3, label: 'The answer', hint: 'Every number in it was checked against the data before you see it.' },
  { step: 4, label: 'What happened next', hint: 'Open the data it could not see and compare its forecast.' },
  { step: 5, label: 'Try to fool it', hint: 'Change a number, add a cause: the checker catches it.' },
]

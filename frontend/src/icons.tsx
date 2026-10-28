// Small status icons. Status always pairs an icon with text, never colour alone.

const base = {
  width: 16,
  height: 16,
  viewBox: '0 0 16 16',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.6,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
}

export const CheckIcon = () => (
  <svg {...base}>
    <path d="M3 8.5 6.5 12 13 4.5" />
  </svg>
)

export const CautionIcon = () => (
  <svg {...base}>
    <path d="M8 2.2 14.5 13.5H1.5Z" />
    <path d="M8 6.5v3.2M8 11.6v.1" />
  </svg>
)

export const AlarmIcon = () => (
  <svg {...base}>
    <path d="M5.2 1.8h5.6l4 4v5.6l-4 4H5.2l-4-4V5.8Z" />
    <path d="M5.8 5.8l4.4 4.4M10.2 5.8l-4.4 4.4" />
  </svg>
)

export const RedoIcon = () => (
  <svg {...base}>
    <path d="M13.5 3.5v3.5H10" />
    <path d="M13.2 7A5.5 5.5 0 1 0 12 11.6" />
  </svg>
)

export const MicIcon = () => (
  <svg {...base}>
    <rect x="5.5" y="1.5" width="5" height="8" rx="2.5" />
    <path d="M3 7.5a5 5 0 0 0 10 0M8 12.5v2M5.5 14.5h5" />
  </svg>
)

export const StopIcon = () => (
  <svg {...base}>
    <rect x="3.5" y="3.5" width="9" height="9" rx="1" fill="currentColor" />
  </svg>
)

export const StepsIcon = () => (
  <svg {...base}>
    <circle cx="3.5" cy="3.5" r="1.5" />
    <circle cx="12.5" cy="8" r="1.5" />
    <circle cx="3.5" cy="12.5" r="1.5" />
    <path d="M5 3.5h4.5a3 3 0 0 1 0 6H6.5a3 3 0 0 0 0 6" transform="translate(0 -1.5) scale(1 .85)" />
  </svg>
)

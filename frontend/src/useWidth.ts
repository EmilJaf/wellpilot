import { useEffect, useRef, useState } from 'react'

/** A container's width, kept current as it resizes, for charts drawn in SVG. */
export function useWidth(initial: number, min = 300) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(initial)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new ResizeObserver(() => setWidth(Math.max(min, el.clientWidth)))
    observer.observe(el)
    return () => observer.disconnect()
  }, [min])
  return [ref, width] as const
}

import { useLayoutEffect, useRef, useState } from 'react'

/** A container's width, kept current as it resizes, for charts drawn in SVG. It is measured
    before the first paint: the observer reports a few frames late, and for those frames an
    initial width wider than a phone pushed the whole page sideways. */
export function useWidth(initial: number, min = 300) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(initial)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const measure = () => setWidth(Math.max(min, el.clientWidth))
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    return () => observer.disconnect()
  }, [min])
  return [ref, width] as const
}

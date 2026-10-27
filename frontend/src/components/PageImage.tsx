import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { sameQuote } from '../labels'

type Located = { quote: string; boxes: number[][] }

/**
 * A report page as Muse Spark read it, with the quoted passages marked where they stand on
 * the page (positions from the PDF's text layer, computed at ingest).
 */
export function PageImage({ doc, page, alt, marks }: { doc: string; page: number; alt: string; marks: string[] }) {
  const [located, setLocated] = useState<{ key: string; quotes: Located[] } | null>(null)
  const key = `${doc}/${page}`
  const host = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let current = true
    api
      .pageQuotes(doc, page)
      .then((quotes) => current && setLocated({ key, quotes }))
      .catch(() => current && setLocated({ key, quotes: [] }))
    return () => {
      current = false
    }
  }, [doc, page, key])

  const boxes = (located?.key === key ? located.quotes : [])
    .filter((q) => marks.some((m) => sameQuote(q.quote, m)))
    .flatMap((q) => q.boxes)

  // Bring the first marked line into view once the image has its size.
  const scrollToMark = () => host.current?.querySelector('.page-mark')?.scrollIntoView({ block: 'center' })
  useEffect(() => {
    if (boxes.length) scrollToMark()
  }, [boxes.length])

  return (
    <>
      <div className="page-frame" ref={host}>
        <img className="page-image" src={api.pageImageUrl(doc, page)} alt={alt} onLoad={() => boxes.length && scrollToMark()} />
        {boxes.map(([x, y, w, h], i) => (
          <span
            key={i}
            className="page-mark"
            aria-hidden
            style={{
              left: `${x * 100}%`,
              top: `${y * 100}%`,
              width: `${w * 100}%`,
              height: `${h * 100}%`,
            }}
          />
        ))}
        {boxes.length > 0 && <span className="sr-only">The quoted passage is marked on the page.</span>}
      </div>
      {/* outside the frame: the marks are placed in percent of the frame's height */}
      <a className="page-open" href={api.pageImageUrl(doc, page)} target="_blank" rel="noreferrer">
        Open the page full size
      </a>
    </>
  )
}

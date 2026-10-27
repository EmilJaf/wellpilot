/**
 * Ask by voice: record a short clip in the browser and turn it into 16 kHz mono WAV, the
 * one format every audio-capable model accepts. The server has no audio library and
 * browsers record WebM or MP4, so the conversion happens here with the Web Audio API.
 */

export type Recording = {
  stop: () => void
  done: Promise<Blob>
}

export const canRecord = () =>
  typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== 'undefined'

/** Starts recording from the microphone; stops itself after maxMs. */
export async function startRecording(maxMs: number): Promise<Recording> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  const recorder = new MediaRecorder(stream)
  const chunks: BlobPart[] = []
  recorder.ondataavailable = (e) => e.data.size > 0 && chunks.push(e.data)
  const done = new Promise<Blob>((resolve, reject) => {
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop())
      resolve(new Blob(chunks, { type: recorder.mimeType }))
    }
    recorder.onerror = () => {
      stream.getTracks().forEach((t) => t.stop())
      reject(new Error('the recording failed'))
    }
  })
  recorder.start()
  const timer = window.setTimeout(() => recorder.state === 'recording' && recorder.stop(), maxMs)
  return {
    stop: () => {
      window.clearTimeout(timer)
      if (recorder.state === 'recording') recorder.stop()
    },
    done,
  }
}

const RATE = 16_000

/** The clip decoded, mixed to mono, resampled to 16 kHz and written as 16-bit PCM WAV. */
export async function toWav16k(clip: Blob): Promise<ArrayBuffer> {
  const context = new AudioContext()
  try {
    const decoded = await context.decodeAudioData(await clip.arrayBuffer())
    const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * RATE)), RATE)
    const source = offline.createBufferSource()
    source.buffer = decoded
    source.connect(offline.destination)
    source.start()
    const samples = (await offline.startRendering()).getChannelData(0)
    return wav(samples)
  } finally {
    await context.close()
  }
}

function wav(samples: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const ascii = (at: number, text: string) => [...text].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)))
  ascii(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  ascii(8, 'WAVE')
  ascii(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM chunk size
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, RATE, true)
  view.setUint32(28, RATE * 2, true) // bytes per second
  view.setUint16(32, 2, true) // bytes per sample frame
  view.setUint16(34, 16, true) // bits per sample
  ascii(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return buffer
}

export function toBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  }
  return btoa(binary)
}

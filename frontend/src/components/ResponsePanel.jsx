import { useEffect, useRef, useState, useCallback } from 'react'

export default function ResponsePanel({ text, audio, audioChunks, narrateState, latencyMs, query, streaming, onNarrate }) {
  const audioRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const playedRef = useRef(0)  // index of next chunk to play

  // Play audio chunks sequentially as they arrive
  const playNextChunk = useCallback(() => {
    if (!audioRef.current || !audioChunks?.length) return
    const nextIdx = playedRef.current
    if (nextIdx >= audioChunks.length) return

    const blob = b64ToBlob(audioChunks[nextIdx], 'audio/wav')
    audioRef.current.src = URL.createObjectURL(blob)
    audioRef.current.play().catch(() => {
      // Browser blocked autoplay — user can click Play
    })
    playedRef.current = nextIdx + 1
  }, [audioChunks])

  // Auto-play first chunk when it arrives
  useEffect(() => {
    if (audioChunks?.length === 1 && playedRef.current === 0) {
      playNextChunk()
    }
  }, [audioChunks, playNextChunk])

  // Resume playback when a new chunk arrives and we've stalled (finished playing
  // previous chunk but next hadn't synthesized yet)
  useEffect(() => {
    if (
      audioChunks &&
      audioChunks.length > 1 &&
      playedRef.current > 0 &&
      playedRef.current < audioChunks.length &&
      !playing
    ) {
      playNextChunk()
    }
  }, [audioChunks?.length, playing, playNextChunk])

  // When current chunk ends, play the next queued chunk
  function handleEnded() {
    setPlaying(false)
    if (audioChunks && playedRef.current < audioChunks.length) {
      playNextChunk()
    }
  }

  function togglePlay() {
    if (!audioRef.current) return
    if (playing) {
      audioRef.current.pause()
    } else {
      // If all chunks have been played, restart from the beginning
      if (audioChunks && playedRef.current >= audioChunks.length) {
        playedRef.current = 0
      }
      playNextChunk()
    }
  }

  // Reset playback position when new narration starts
  useEffect(() => {
    if (narrateState === 'loading') {
      playedRef.current = 0
    }
  }, [narrateState])

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
      {/* Query echo */}
      <div className="px-6 pt-5 pb-4 border-b border-slate-700/60">
        <span className="text-slate-500 text-xs uppercase tracking-wider">Query</span>
        <p className="hindi text-slate-300 text-sm mt-1 leading-relaxed">"{query}"</p>
      </div>

      {/* Answer */}
      <div className="px-6 py-5">
        <FormattedText text={text} streaming={streaming} />
      </div>

      {/* Footer */}
      <div className="px-6 py-3 border-t border-slate-700/60 flex items-center gap-3 bg-slate-800/50">
        {streaming && !audio && (
          <span className="text-slate-400 text-xs flex items-center gap-2">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-orange-400 animate-pulse" />
            Generating response &amp; audio...
          </span>
        )}
        {audio && (
          <NarrateButton
            state={narrateState}
            hasAudio={true}
            playing={playing}
            chunksLoading={streaming}
            onNarrate={onNarrate}
            onTogglePlay={togglePlay}
          />
        )}
        {!streaming && !audio && (
          <NarrateButton
            state={narrateState}
            hasAudio={false}
            playing={false}
            chunksLoading={false}
            onNarrate={onNarrate}
            onTogglePlay={togglePlay}
          />
        )}
        {latencyMs && (
          <span className="text-slate-500 text-xs ml-auto">{(latencyMs / 1000).toFixed(1)}s</span>
        )}
      </div>

      <audio
        ref={audioRef}
        hidden
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={handleEnded}
      />
    </div>
  )
}

function NarrateButton({ state, hasAudio, playing, onNarrate, onTogglePlay }) {
  if (state === 'loading') {
    return (
      <span className="text-slate-400 text-xs flex items-center gap-2">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-orange-400 animate-pulse" />
        Generating audio...
      </span>
    )
  }

  if (hasAudio) {
    return (
      <button
        onClick={onTogglePlay}
        className="flex items-center gap-2 bg-green-700 hover:bg-green-600 text-white text-sm px-4 py-1.5 rounded-lg transition-colors"
      >
        {playing ? '⏸ Pause' : '▶ सुनें / Play'}
      </button>
    )
  }

  if (state === 'error') {
    return (
      <span className="text-red-400 text-xs">Audio failed — TTS server may be busy</span>
    )
  }

  return (
    <button
      onClick={onNarrate}
      className="flex items-center gap-2 text-slate-400 hover:text-orange-400 text-xs border border-slate-600 hover:border-orange-500 px-3 py-1.5 rounded-lg transition-colors"
    >
      🔊 सुनें / Narrate
    </button>
  )
}

function FormattedText({ text, streaming }) {
  const lines = text.split('\n')

  return (
    <div className="hindi text-white text-base space-y-2" style={{ lineHeight: '1.9' }}>
      {lines.map((line, i) => {
        const isLast = i === lines.length - 1
        const trimmed = line.trim()

        if (trimmed === '') return <div key={i} className="h-2" />

        const isNumbered = /^\d+\./.test(trimmed)
        const isBullet = /^[-•]/.test(trimmed)

        return (
          <div key={i} className={isNumbered || isBullet ? 'pl-5' : ''}>
            {trimmed}
            {isLast && streaming && (
              <span className="inline-block w-0.5 h-[1.1em] bg-orange-400 ml-0.5 align-middle animate-pulse" />
            )}
          </div>
        )
      })}
    </div>
  )
}

function b64ToBlob(base64, mimeType) {
  const byteChars = atob(base64)
  const bytes = new Uint8Array(byteChars.length)
  for (let i = 0; i < byteChars.length; i++) bytes[i] = byteChars.charCodeAt(i)
  return new Blob([bytes], { type: mimeType })
}

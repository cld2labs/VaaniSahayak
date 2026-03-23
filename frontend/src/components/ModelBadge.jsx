/**
 * Visual badge showing the sovereign AI model pipeline.
 * Key demo talking point: no OpenAI, all Indian models.
 */
export default function ModelBadge() {
  const models = [
    { label: 'LLM', name: 'Param-1', org: 'IIT Madras' },
    { label: 'TTS', name: 'Indic Parler-TTS', org: 'AI4Bharat' },
    { label: 'Embed', name: 'MiniLM', org: 'SBERT' },
  ]

  return (
    <div className="hidden sm:flex items-center gap-1 text-xs">
      <span className="text-slate-500 mr-1">Powered by</span>
      {models.map((m, i) => (
        <span key={m.name} className="flex items-center gap-1">
          <span className="bg-slate-700 rounded px-2 py-1 text-slate-200">
            <span className="text-slate-400">{m.label}: </span>
            {m.name}
          </span>
          {i < models.length - 1 && <span className="text-slate-600">→</span>}
        </span>
      ))}
      <span className="ml-2 bg-orange-900/40 text-orange-400 border border-orange-800 rounded-full px-2 py-0.5 font-medium">
        🇮🇳 Sovereign AI
      </span>
    </div>
  )
}

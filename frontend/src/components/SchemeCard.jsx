export default function SchemeCard({ scheme }) {
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-xl p-4 flex flex-col gap-2">
      <p className="hindi font-semibold text-white text-sm leading-snug">{scheme.name}</p>
      <div className="flex flex-wrap gap-1">
        {scheme.category && (
          <span className="text-xs bg-blue-900/50 text-blue-300 rounded-full px-2 py-0.5">
            {scheme.category}
          </span>
        )}
        {scheme.state && (
          <span className="text-xs bg-slate-700 text-slate-300 rounded-full px-2 py-0.5">
            {scheme.state}
          </span>
        )}
      </div>
      <div className="flex items-center justify-between mt-auto">
        <span className="text-xs text-slate-500">
          Score: {(scheme.score * 100).toFixed(0)}%
        </span>
        {scheme.official_link && (
          <a
            href={scheme.official_link}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-orange-400 hover:text-orange-300 underline"
          >
            Official site →
          </a>
        )}
      </div>
    </div>
  )
}

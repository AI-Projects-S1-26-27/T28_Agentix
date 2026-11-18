export default function ErrorBanner({ message, onRetry }) {
  return (
    <div
      className="flex items-center justify-between gap-4 rounded-lg border px-4 py-3"
      style={{
        borderColor: 'var(--accent-danger)',
        background: 'color-mix(in srgb, var(--accent-danger) 8%, var(--surface))',
      }}
    >
      <div className="flex items-center gap-3">
        <span className="text-lg">&#9888;</span>
        <p className="text-sm font-medium" style={{ color: 'var(--accent-danger)' }}>
          {message}
        </p>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="shrink-0 text-xs font-semibold px-3 py-1.5 rounded-md transition-opacity hover:opacity-80"
          style={{
            background: 'var(--accent-danger)',
            color: '#fff',
          }}
        >
          Retry
        </button>
      )}
    </div>
  )
}

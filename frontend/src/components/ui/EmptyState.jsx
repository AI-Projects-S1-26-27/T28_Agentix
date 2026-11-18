export default function EmptyState({ icon: Icon, title, subtitle }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
      {Icon && (
        <div
          className="mb-4 p-4 rounded-full"
          style={{ background: 'var(--surface-elevated)' }}
        >
          <Icon className="w-8 h-8" style={{ color: 'var(--text-muted)' }} />
        </div>
      )}
      <h3 className="text-base font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
        {title}
      </h3>
      {subtitle && (
        <p className="text-sm max-w-sm" style={{ color: 'var(--text-muted)' }}>
          {subtitle}
        </p>
      )}
    </div>
  )
}

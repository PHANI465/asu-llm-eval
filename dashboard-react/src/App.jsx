import { useEffect, useState } from 'react'
import {
  CheckCircle, XCircle, AlertTriangle, Clock, DollarSign,
  Brain, Target, Zap, Activity, Github, ExternalLink,
  RefreshCw, ChevronDown, ChevronUp
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts'

// ─────────────────────────────────────────────────────────────────────────────
// Data URL
// ─────────────────────────────────────────────────────────────────────────────
const DATA_URL =
  'https://raw.githubusercontent.com/PHANI465/asu-llm-eval/main/results/latest_report.json'

// ─────────────────────────────────────────────────────────────────────────────
// Design tokens
// ─────────────────────────────────────────────────────────────────────────────
const C = {
  bg:       '#0f1117',
  card:     '#1a1f2e',
  card2:    '#1e2438',
  border:   '#2d3748',
  borderHi: '#3d4a6a',
  pass:     '#4ade80',
  passD:    '#16a34a',
  passBg:   'rgba(74,222,128,0.08)',
  fail:     '#f87171',
  failD:    '#dc2626',
  failBg:   'rgba(248,113,113,0.08)',
  warn:     '#fbbf24',
  text:     '#ffffff',
  text2:    '#c8d0e0',
  muted:    '#888888',
  gold:     '#FFC627',
  maroon:   '#8C1D40',
  blue:     '#60a5fa',
  purple:   '#c084fc',
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatters
// ─────────────────────────────────────────────────────────────────────────────
const fmtPct    = v => v == null ? 'N/A' : `${(v * 100).toFixed(1)}%`
const fmtScore  = v => v == null ? 'N/A' : v.toFixed(4)
const fmtSec    = v => v == null ? 'N/A' : `${v}s`
const fmtUsd    = v => v == null ? 'N/A' : `$${v.toFixed(4)}`
const fmtTs     = ts => {
  if (!ts) return '—'
  try {
    return new Date(ts).toLocaleString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
  } catch { return ts }
}

// ─────────────────────────────────────────────────────────────────────────────
// Metric card configuration
// gate key → { label, icon, format, direction, unit }
// ─────────────────────────────────────────────────────────────────────────────
const METRIC_CONFIG = {
  hallucination_rate: {
    label:     'Hallucination Rate',
    icon:      Brain,
    format:    fmtPct,
    direction: 'max',
    unit:      'lower is better',
  },
  answer_relevancy: {
    label:     'Answer Relevancy',
    icon:      Target,
    format:    fmtScore,
    direction: 'min',
    unit:      'higher is better',
  },
  faithfulness: {
    label:     'Faithfulness',
    icon:      CheckCircle,
    format:    fmtScore,
    direction: 'min',
    unit:      'higher is better',
  },
  context_precision: {
    label:     'Context Precision',
    icon:      Activity,
    format:    fmtScore,
    direction: 'min',
    unit:      'higher is better',
  },
  latency_p95: {
    label:     'Latency p95',
    icon:      Clock,
    format:    fmtSec,
    direction: 'max',
    unit:      'lower is better',
  },
  cost_per_query: {
    label:     'Cost per Query',
    icon:      DollarSign,
    format:    fmtUsd,
    direction: 'max',
    unit:      'lower is better',
  },
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

function LoadingScreen() {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', minHeight: '100vh', gap: 24,
    }}>
      <div className="spinner" />
      <p style={{ color: C.muted, fontSize: 15 }}>Loading evaluation data…</p>
    </div>
  )
}

function ErrorScreen({ message, onRetry }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', minHeight: '100vh', gap: 20, padding: 24,
    }}>
      <AlertTriangle size={48} color={C.warn} />
      <h2 style={{ color: C.text, fontSize: 20, fontWeight: 600 }}>
        Could not load data
      </h2>
      <p style={{
        color: C.muted, fontSize: 14, maxWidth: 480, textAlign: 'center',
        lineHeight: 1.6,
      }}>
        {message}
      </p>
      <p style={{ color: C.muted, fontSize: 13, textAlign: 'center' }}>
        If this is the first run, push a commit to trigger the CI pipeline —
        it will write <code style={{ color: C.blue }}>results/latest_report.json</code> to the repo.
      </p>
      <button
        onClick={onRetry}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          background: C.maroon, color: C.text, border: 'none',
          borderRadius: 8, padding: '10px 20px', cursor: 'pointer',
          fontSize: 14, fontWeight: 500,
        }}
      >
        <RefreshCw size={16} /> Retry
      </button>
    </div>
  )
}

function StatusBadge({ status }) {
  const isPass = status === 'PASS'
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      background: isPass ? C.passBg : C.failBg,
      border: `1.5px solid ${isPass ? C.pass : C.fail}`,
      color: isPass ? C.pass : C.fail,
      borderRadius: 999, padding: '6px 18px',
      fontSize: 15, fontWeight: 700, letterSpacing: '0.05em',
    }}>
      {isPass
        ? <CheckCircle size={16} strokeWidth={2.5} />
        : <XCircle    size={16} strokeWidth={2.5} />}
      {status}
    </span>
  )
}

function MetricCard({ gateKey, gate }) {
  const cfg   = METRIC_CONFIG[gateKey] || {}
  const Icon  = cfg.icon || Activity
  const pass  = gate?.passed ?? true
  const value = gate?.value
  const thr   = gate?.threshold

  const fmtValue = cfg.format ? cfg.format(value) : (value ?? 'N/A')
  const fmtThr   = cfg.format ? cfg.format(thr)   : (thr ?? '—')

  return (
    <div style={{
      background:    pass ? C.passBg : C.failBg,
      border:        `1px solid ${pass ? C.pass + '40' : C.fail + '40'}`,
      borderRadius:  12,
      padding:       '20px 22px',
      display:       'flex',
      flexDirection: 'column',
      gap:           12,
      transition:    'transform 0.15s, box-shadow 0.15s',
    }}
      onMouseEnter={e => {
        e.currentTarget.style.transform  = 'translateY(-2px)'
        e.currentTarget.style.boxShadow  = `0 8px 24px ${pass ? 'rgba(74,222,128,0.12)' : 'rgba(248,113,113,0.12)'}`
      }}
      onMouseLeave={e => {
        e.currentTarget.style.transform  = 'none'
        e.currentTarget.style.boxShadow  = 'none'
      }}
    >
      {/* Top row: icon + pass/fail pill */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{
          background: pass ? 'rgba(74,222,128,0.15)' : 'rgba(248,113,113,0.15)',
          borderRadius: 8, padding: 8,
        }}>
          <Icon size={18} color={pass ? C.pass : C.fail} strokeWidth={2} />
        </div>
        <span style={{
          fontSize: 11, fontWeight: 600, letterSpacing: '0.06em',
          color: pass ? C.pass : C.fail,
          textTransform: 'uppercase',
        }}>
          {pass ? '✓ PASS' : '✗ FAIL'}
        </span>
      </div>

      {/* Value */}
      <div>
        <div style={{ fontSize: 28, fontWeight: 700, color: C.text, lineHeight: 1 }}>
          {fmtValue}
        </div>
        <div style={{ fontSize: 13, color: C.muted, marginTop: 4 }}>
          {cfg.label || gateKey}
        </div>
      </div>

      {/* Threshold */}
      <div style={{
        fontSize: 12, color: C.muted,
        borderTop: `1px solid ${C.border}`,
        paddingTop: 10,
        display: 'flex', justifyContent: 'space-between',
      }}>
        <span>{cfg.direction === 'max' ? 'Max allowed' : 'Min required'}</span>
        <span style={{ color: C.text2, fontWeight: 500 }}>{fmtThr}</span>
      </div>
    </div>
  )
}

function GateTable({ gates }) {
  const rows = Object.entries(gates)

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
        <thead>
          <tr style={{ borderBottom: `1px solid ${C.border}` }}>
            {['Gate', 'Value', 'Threshold', 'Status'].map(h => (
              <th key={h} style={{
                textAlign: 'left', padding: '10px 14px',
                color: C.muted, fontWeight: 600,
                fontSize: 12, letterSpacing: '0.05em', textTransform: 'uppercase',
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([key, gate], i) => {
            const cfg  = METRIC_CONFIG[key] || {}
            const pass = gate?.passed ?? true
            const fmt  = cfg.format || (v => v)
            return (
              <tr
                key={key}
                style={{
                  background:   i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)',
                  borderBottom: `1px solid ${C.border}30`,
                }}
              >
                <td style={{ padding: '12px 14px', color: C.text2, fontWeight: 500 }}>
                  {cfg.label || key}
                </td>
                <td style={{ padding: '12px 14px', color: C.text, fontWeight: 600 }}>
                  {fmt(gate?.value)}
                </td>
                <td style={{ padding: '12px 14px', color: C.muted }}>
                  {gate?.threshold != null
                    ? `${cfg.direction === 'max' ? '≤' : '≥'} ${fmt(gate.threshold)}`
                    : '—'}
                </td>
                <td style={{ padding: '12px 14px' }}>
                  <span style={{
                    display: 'inline-flex', alignItems: 'center', gap: 5,
                    color:      pass ? C.pass : C.fail,
                    background: pass ? C.passBg : C.failBg,
                    border:     `1px solid ${pass ? C.pass + '50' : C.fail + '50'}`,
                    borderRadius: 6, padding: '3px 10px',
                    fontSize: 12, fontWeight: 600,
                  }}>
                    {pass
                      ? <CheckCircle size={12} strokeWidth={2.5} />
                      : <XCircle    size={12} strokeWidth={2.5} />}
                    {pass ? 'PASS' : 'FAIL'}
                  </span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function FailureCard({ failure }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div style={{
      background: C.card2, border: `1px solid ${C.fail}40`,
      borderRadius: 10, padding: '16px 18px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            <span style={{
              background: C.failBg, border: `1px solid ${C.fail}50`,
              color: C.fail, borderRadius: 6, padding: '2px 8px',
              fontSize: 11, fontWeight: 700,
            }}>
              faithfulness = {failure.faithfulness?.toFixed(2) ?? 'N/A'}
            </span>
            {failure.category && (
              <span style={{
                color: C.muted, fontSize: 11,
                background: 'rgba(255,255,255,0.05)',
                padding: '2px 8px', borderRadius: 6,
              }}>
                {failure.category}
              </span>
            )}
          </div>
          <p style={{ color: C.text, fontSize: 14, fontWeight: 500, lineHeight: 1.5 }}>
            {failure.question}
          </p>
        </div>
        <button
          onClick={() => setExpanded(v => !v)}
          style={{
            background: 'transparent', border: 'none', cursor: 'pointer',
            color: C.muted, padding: 4, flexShrink: 0,
          }}
        >
          {expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>
      </div>

      {expanded && (
        <div style={{
          marginTop: 12, paddingTop: 12,
          borderTop: `1px solid ${C.border}`,
        }}>
          <p style={{ color: C.muted, fontSize: 11, fontWeight: 600,
            letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 6 }}>
            Answer
          </p>
          <p style={{ color: C.text2, fontSize: 13, lineHeight: 1.6 }}>
            {failure.answer}
          </p>
          {failure.error && (
            <p style={{ color: C.warn, fontSize: 12, marginTop: 8 }}>
              Error: {failure.error}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function Section({ title, children, extra }) {
  return (
    <section style={{ marginBottom: 32 }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 16,
      }}>
        <h2 style={{
          fontSize: 16, fontWeight: 600, color: C.text,
          display: 'flex', alignItems: 'center', gap: 10,
        }}>
          <span style={{
            display: 'inline-block', width: 3, height: 18,
            background: C.maroon, borderRadius: 2,
          }} />
          {title}
        </h2>
        {extra}
      </div>
      <div style={{
        background: C.card, border: `1px solid ${C.border}`,
        borderRadius: 12, padding: 20,
      }}>
        {children}
      </div>
    </section>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)

  const load = () => {
    setLoading(true)
    setError(null)
    fetch(DATA_URL, { cache: 'no-store' })
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status} — ${r.statusText}`)
        return r.json()
      })
      .then(json => { setData(json); setLoading(false) })
      .catch(err  => { setError(err.message); setLoading(false) })
  }

  useEffect(() => { load() }, [])

  if (loading) return <LoadingScreen />
  if (error)   return <ErrorScreen message={error} onRetry={load} />

  const overall   = data.overall_result ?? 'UNKNOWN'
  const isPass    = overall === 'PASS'
  const metrics   = data.metrics        ?? {}
  const gates     = data.gate_results?.gates ?? {}
  const failures  = data.sample_failures ?? []
  const testMode  = data.test_mode

  return (
    <div className="fade-in" style={{ background: C.bg, minHeight: '100vh' }}>

      {/* ── Top bar ─────────────────────────────────────────────────────── */}
      <div style={{
        background:   C.card,
        borderBottom: `1px solid ${C.border}`,
        padding:      '0 24px',
        display:      'flex', alignItems: 'center',
        height:       52,
        position:     'sticky', top: 0, zIndex: 100,
      }}>
        <span style={{
          fontSize: 13, fontWeight: 700, color: C.gold,
          letterSpacing: '0.04em',
        }}>
          🎓 ASU LLM EVAL
        </span>
        <span style={{
          marginLeft: 12, fontSize: 12, color: C.muted,
          borderLeft: `1px solid ${C.border}`, paddingLeft: 12,
        }}>
          Quality Gate Dashboard
        </span>
        <div style={{ flex: 1 }} />
        <button
          onClick={load}
          title="Refresh data"
          style={{
            background: 'transparent', border: `1px solid ${C.border}`,
            borderRadius: 7, padding: '5px 10px', cursor: 'pointer',
            color: C.muted, display: 'flex', alignItems: 'center', gap: 5,
            fontSize: 12,
          }}
        >
          <RefreshCw size={13} /> Refresh
        </button>
      </div>

      {/* ── Page content ────────────────────────────────────────────────── */}
      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '32px 24px' }}>

        {/* ─ SECTION 1: Header ─────────────────────────────────────────── */}
        <div style={{
          background: `linear-gradient(135deg, #1a0a12 0%, ${C.card} 55%, #0f1530 100%)`,
          border:     `1px solid ${C.border}`,
          borderRadius: 16, padding: '28px 32px',
          marginBottom: 32,
          display: 'flex', flexWrap: 'wrap',
          alignItems: 'center', justifyContent: 'space-between',
          gap: 20,
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <h1 style={{ fontSize: 26, fontWeight: 800, color: C.text }}>
                ASU LLM Eval Dashboard
              </h1>
              <StatusBadge status={overall} />
              {testMode && (
                <span style={{
                  fontSize: 11, color: C.warn,
                  background: 'rgba(251,191,36,0.1)',
                  border: '1px solid rgba(251,191,36,0.3)',
                  borderRadius: 6, padding: '2px 8px',
                }}>
                  TEST MODE
                </span>
              )}
            </div>
            <p style={{ color: C.muted, fontSize: 14, marginTop: 8 }}>
              Powered by RAGAS + GPT-4o · gpt-4o-mini judge
            </p>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, textAlign: 'right' }}>
            <div style={{ fontSize: 13, color: C.muted }}>
              <span style={{ color: C.text2, fontWeight: 500 }}>Last run: </span>
              {fmtTs(data.run_timestamp)}
            </div>
            <div style={{ fontSize: 13, color: C.muted }}>
              <span style={{ color: C.text2, fontWeight: 500 }}>Commit: </span>
              <code style={{ color: C.blue, fontSize: 12 }}>{data.commit_id ?? '—'}</code>
            </div>
            <div style={{ fontSize: 13, color: C.muted }}>
              <span style={{ color: C.text2, fontWeight: 500 }}>Questions: </span>
              {data.total_questions ?? '—'}
            </div>
          </div>
        </div>

        {/* ─ SECTION 2: Metric Cards ───────────────────────────────────── */}
        <Section title="Quality Metrics">
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
            gap: 16,
          }}>
            {Object.keys(METRIC_CONFIG).map(key => (
              <MetricCard key={key} gateKey={key} gate={gates[key]} />
            ))}
          </div>
        </Section>

        {/* ─ SECTION 3: Gate Status Table ──────────────────────────────── */}
        <Section
          title="Gate Status"
          extra={
            <span style={{
              fontSize: 12, color: isPass ? C.pass : C.fail,
              background: isPass ? C.passBg : C.failBg,
              border: `1px solid ${isPass ? C.pass + '40' : C.fail + '40'}`,
              borderRadius: 6, padding: '3px 10px',
            }}>
              {data.gate_results?.passed_gates?.length ?? 0} / {Object.keys(gates).length} passed
            </span>
          }
        >
          {Object.keys(gates).length > 0
            ? <GateTable gates={gates} />
            : <p style={{ color: C.muted, fontSize: 14 }}>No gate data available.</p>
          }
        </Section>

        {/* ─ SECTION 4: Sample Failures ────────────────────────────────── */}
        <Section
          title="Sample Failures"
          extra={
            <span style={{
              fontSize: 12, color: C.muted,
              background: 'rgba(255,255,255,0.05)',
              borderRadius: 6, padding: '3px 10px',
            }}>
              {failures.length} failure{failures.length !== 1 ? 's' : ''}
            </span>
          }
        >
          {failures.length === 0
            ? (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 10,
                color: C.pass, padding: '8px 0',
              }}>
                <CheckCircle size={18} strokeWidth={2.5} />
                <span style={{ fontSize: 14, fontWeight: 500 }}>
                  No failures in this run
                </span>
              </div>
            )
            : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {failures.map((f, i) => <FailureCard key={i} failure={f} />)}
              </div>
            )
          }
        </Section>

        {/* ─ SECTION 5: Footer ─────────────────────────────────────────── */}
        <footer style={{
          borderTop:   `1px solid ${C.border}`,
          paddingTop:  24, marginTop: 8,
          display:     'flex', flexWrap: 'wrap',
          alignItems:  'center', justifyContent: 'space-between', gap: 12,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <a
              href="https://github.com/PHANI465/asu-llm-eval"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                color: C.muted, fontSize: 13, textDecoration: 'none',
              }}
              onMouseEnter={e => e.currentTarget.style.color = C.text2}
              onMouseLeave={e => e.currentTarget.style.color = C.muted}
            >
              <Github size={15} /> PHANI465/asu-llm-eval
              <ExternalLink size={12} />
            </a>
          </div>
          <p style={{ color: C.muted, fontSize: 12 }}>
            Powered by{' '}
            <span style={{ color: C.text2 }}>RAGAS</span>
            {' + '}
            <span style={{ color: C.text2 }}>GPT-4o</span>
            {' · '}
            <span style={{ color: C.text2 }}>Pinecone</span>
            {' · '}
            <span style={{ color: C.gold }}>Arizona State University</span>
          </p>
        </footer>
      </div>
    </div>
  )
}

import { useEffect, useState, useCallback, useMemo } from 'react'
import {
  CheckCircle, XCircle, AlertTriangle, Clock, DollarSign,
  Brain, Target, Activity, Github, ExternalLink,
  RefreshCw, ChevronDown, ChevronUp, Search, TrendingUp,
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts'

// ─────────────────────────────────────────────────────────────────────────────
// Data URLs
// ─────────────────────────────────────────────────────────────────────────────
const REPORT_URL  = 'https://raw.githubusercontent.com/PHANI465/asu-llm-eval/main/results/latest_report.json'
const HISTORY_URL = 'https://raw.githubusercontent.com/PHANI465/asu-llm-eval/main/results/eval_history.json'
const REFRESH_INTERVAL = 30

// ─────────────────────────────────────────────────────────────────────────────
// Design tokens
// ─────────────────────────────────────────────────────────────────────────────
const C = {
  bg:       '#0f1117',
  card:     '#1a1f2e',
  card2:    '#1e2438',
  card3:    '#141824',
  border:   '#2d3748',
  pass:     '#4ade80',
  passD:    '#16a34a',
  passBg:   'rgba(74,222,128,0.08)',
  fail:     '#f87171',
  failD:    '#dc2626',
  failBg:   'rgba(248,113,113,0.08)',
  warn:     '#fbbf24',
  warnBg:   'rgba(251,191,36,0.08)',
  text:     '#ffffff',
  text2:    '#c8d0e0',
  muted:    '#888888',
  gold:     '#FFC627',
  maroon:   '#8C1D40',
  blue:     '#60a5fa',
  purple:   '#c084fc',
  orange:   '#fb923c',
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatters
// ─────────────────────────────────────────────────────────────────────────────
const fmtPct   = v => v == null ? 'N/A' : `${(v * 100).toFixed(1)}%`
const fmtScore = v => v == null ? 'N/A' : v.toFixed(4)
const fmtSec   = v => v == null ? 'N/A' : `${v}s`
const fmtUsd   = v => v == null ? 'N/A' : `$${v.toFixed(4)}`
const fmtTs    = ts => {
  if (!ts) return '—'
  try {
    return new Date(ts).toLocaleString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
  } catch { return ts }
}
const shortTs = ts => {
  if (!ts) return ''
  try {
    const d  = new Date(ts)
    const mo = String(d.getMonth() + 1).padStart(2, '0')
    const dy = String(d.getDate()).padStart(2, '0')
    const hr = String(d.getHours()).padStart(2, '0')
    const mn = String(d.getMinutes()).padStart(2, '0')
    return `${mo}/${dy} ${hr}:${mn}`
  } catch { return ts.substring(0, 10) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Metric card configuration
// ─────────────────────────────────────────────────────────────────────────────
const METRIC_CONFIG = {
  hallucination_rate: {
    label: 'Hallucination Rate', icon: Brain,
    format: fmtPct, direction: 'max', unit: 'lower is better',
  },
  answer_relevancy: {
    label: 'Answer Relevancy', icon: Target,
    format: fmtScore, direction: 'min', unit: 'higher is better',
  },
  faithfulness: {
    label: 'Faithfulness', icon: CheckCircle,
    format: fmtScore, direction: 'min', unit: 'higher is better',
  },
  context_precision: {
    label: 'Context Precision', icon: Activity,
    format: fmtScore, direction: 'min', unit: 'higher is better',
  },
  latency_p95: {
    label: 'Latency p95', icon: Clock,
    format: fmtSec, direction: 'max', unit: 'lower is better',
  },
  cost_per_query: {
    label: 'Cost per Query', icon: DollarSign,
    format: fmtUsd, direction: 'max', unit: 'lower is better',
  },
}

// ─────────────────────────────────────────────────────────────────────────────
// Score color helper (for question table cells)
// ─────────────────────────────────────────────────────────────────────────────
function scoreColor(v) {
  if (v == null) return C.muted
  if (v >= 0.8)  return C.pass
  if (v >= 0.5)  return C.warn
  return C.fail
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
      <p style={{ color: C.muted, fontSize: 14, maxWidth: 480, textAlign: 'center', lineHeight: 1.6 }}>
        {message}
      </p>
      <p style={{ color: C.muted, fontSize: 13, textAlign: 'center' }}>
        Push a commit to trigger the CI pipeline — it will write{' '}
        <code style={{ color: C.blue }}>results/latest_report.json</code> to the repo.
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
    <div
      style={{
        background: pass ? C.passBg : C.failBg,
        border: `1px solid ${pass ? C.pass + '40' : C.fail + '40'}`,
        borderRadius: 12, padding: '20px 22px',
        display: 'flex', flexDirection: 'column', gap: 12,
        transition: 'transform 0.15s, box-shadow 0.15s',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.transform = 'translateY(-2px)'
        e.currentTarget.style.boxShadow = `0 8px 24px ${pass ? 'rgba(74,222,128,0.12)' : 'rgba(248,113,113,0.12)'}`
      }}
      onMouseLeave={e => {
        e.currentTarget.style.transform = 'none'
        e.currentTarget.style.boxShadow = 'none'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{
          background: pass ? 'rgba(74,222,128,0.15)' : 'rgba(248,113,113,0.15)',
          borderRadius: 8, padding: 8,
        }}>
          <Icon size={18} color={pass ? C.pass : C.fail} strokeWidth={2} />
        </div>
        <span style={{
          fontSize: 11, fontWeight: 600, letterSpacing: '0.06em',
          color: pass ? C.pass : C.fail, textTransform: 'uppercase',
        }}>
          {pass ? '✓ PASS' : '✗ FAIL'}
        </span>
      </div>
      <div>
        <div style={{ fontSize: 28, fontWeight: 700, color: C.text, lineHeight: 1 }}>
          {fmtValue}
        </div>
        <div style={{ fontSize: 13, color: C.muted, marginTop: 4 }}>
          {cfg.label || gateKey}
        </div>
      </div>
      <div style={{
        fontSize: 12, color: C.muted,
        borderTop: `1px solid ${C.border}`, paddingTop: 10,
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
              <tr key={key} style={{
                background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)',
                borderBottom: `1px solid ${C.border}30`,
              }}>
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

// ─── Dark recharts tooltip ────────────────────────────────────────────────────
function DarkTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#1a1f2e', border: `1px solid ${C.border}`,
      borderRadius: 8, padding: '10px 14px', fontSize: 12, minWidth: 150,
    }}>
      <p style={{ color: C.muted, marginBottom: 8, fontSize: 11 }}>{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ color: entry.color, marginBottom: 3 }}>
          {entry.name}:{' '}
          <strong>
            {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
          </strong>
        </p>
      ))}
    </div>
  )
}

// ─── Trend Charts (2×2 grid) ──────────────────────────────────────────────────
function TrendCharts({ chartData }) {
  if (!chartData || chartData.length < 2) {
    return (
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', padding: '48px 0', gap: 12,
      }}>
        <TrendingUp size={36} color={C.muted} />
        <p style={{ color: C.muted, fontSize: 14 }}>
          Run the pipeline again to see trends
        </p>
        <p style={{ color: C.muted, fontSize: 12 }}>
          ({chartData?.length || 0} run{chartData?.length !== 1 ? 's' : ''} recorded — need at least 2)
        </p>
      </div>
    )
  }

  const card = {
    background: C.card2, border: `1px solid ${C.border}`,
    borderRadius: 10, padding: '20px 16px',
  }
  const ax    = { fill: C.muted, fontSize: 11 }
  const grid  = { stroke: C.border, strokeDasharray: '3 3' }
  const lText = { wrapperStyle: { fontSize: 12, color: C.muted } }

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fill, minmax(440px, 1fr))',
      gap: 16,
    }}>

      {/* Chart 1 — Faithfulness & Relevancy */}
      <div style={card}>
        <p style={{ color: C.text2, fontSize: 13, fontWeight: 600, marginBottom: 16 }}>
          Faithfulness &amp; Answer Relevancy
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData} margin={{ top: 4, right: 24, bottom: 4, left: 0 }}>
            <CartesianGrid {...grid} />
            <XAxis dataKey="label" tick={ax} />
            <YAxis domain={[0, 1]} tick={ax} tickFormatter={v => v.toFixed(1)} />
            <Tooltip content={<DarkTooltip />} />
            <Legend {...lText} />
            <ReferenceLine y={0.80} stroke={C.fail}   strokeDasharray="5 4"
              label={{ value: '0.80', fill: C.fail,   fontSize: 10, position: 'insideTopRight' }} />
            <ReferenceLine y={0.75} stroke={C.orange} strokeDasharray="5 4"
              label={{ value: '0.75', fill: C.orange, fontSize: 10, position: 'insideBottomRight' }} />
            <Line type="monotone" dataKey="faithfulness"     name="Faithfulness" stroke={C.blue}   strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
            <Line type="monotone" dataKey="answer_relevancy" name="Relevancy"    stroke={C.pass}   strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Chart 2 — Hallucination Rate */}
      <div style={card}>
        <p style={{ color: C.text2, fontSize: 13, fontWeight: 600, marginBottom: 16 }}>
          Hallucination Rate
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData} margin={{ top: 4, right: 24, bottom: 4, left: 0 }}>
            <CartesianGrid {...grid} />
            <XAxis dataKey="label" tick={ax} />
            <YAxis domain={[0, 0.2]} tick={ax} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
            <Tooltip content={<DarkTooltip />} />
            <ReferenceLine y={0.10} stroke={C.fail} strokeDasharray="5 4"
              label={{ value: '10%', fill: C.fail, fontSize: 10, position: 'insideTopRight' }} />
            <Line type="monotone" dataKey="hallucination_rate" name="Hallucination Rate" stroke={C.fail} strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Chart 3 — Latency p95 */}
      <div style={card}>
        <p style={{ color: C.text2, fontSize: 13, fontWeight: 600, marginBottom: 16 }}>
          Latency p95 (seconds)
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData} margin={{ top: 4, right: 24, bottom: 4, left: 0 }}>
            <CartesianGrid {...grid} />
            <XAxis dataKey="label" tick={ax} />
            <YAxis domain={[0, 20]} tick={ax} tickFormatter={v => `${v}s`} />
            <Tooltip content={<DarkTooltip />} />
            <ReferenceLine y={15} stroke={C.fail} strokeDasharray="5 4"
              label={{ value: '15s', fill: C.fail, fontSize: 10, position: 'insideTopRight' }} />
            <Line type="monotone" dataKey="latency_p95" name="Latency p95" stroke={C.orange} strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Chart 4 — Cost Per Query */}
      <div style={card}>
        <p style={{ color: C.text2, fontSize: 13, fontWeight: 600, marginBottom: 16 }}>
          Cost Per Query (USD)
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData} margin={{ top: 4, right: 24, bottom: 4, left: 0 }}>
            <CartesianGrid {...grid} />
            <XAxis dataKey="label" tick={ax} />
            <YAxis domain={[0, 0.025]} tick={ax} tickFormatter={v => `$${v.toFixed(3)}`} />
            <Tooltip content={<DarkTooltip />} />
            <ReferenceLine y={0.02} stroke={C.fail} strokeDasharray="5 4"
              label={{ value: '$0.02', fill: C.fail, fontSize: 10, position: 'insideTopRight' }} />
            <Line type="monotone" dataKey="cost_per_query" name="Cost/Query" stroke={C.purple} strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

    </div>
  )
}

// ─── Question Breakdown Table ─────────────────────────────────────────────────
function QuestionTable({ allResults }) {
  const [filterCat,   setFilterCat]   = useState('all')
  const [filterDiff,  setFilterDiff]  = useState('all')
  const [filterRes,   setFilterRes]   = useState('all')
  const [searchText,  setSearchText]  = useState('')

  const filtered = useMemo(() => {
    let arr = [...(allResults || [])]

    // Default sort: FAIL first, then faithfulness ascending
    arr.sort((a, b) => {
      if (a.passed !== b.passed) return a.passed ? 1 : -1
      return (a.faithfulness ?? 0) - (b.faithfulness ?? 0)
    })

    if (filterCat  !== 'all') arr = arr.filter(r => r.category  === filterCat)
    if (filterDiff !== 'all') arr = arr.filter(r => r.difficulty === filterDiff)
    if (filterRes === 'pass') arr = arr.filter(r =>  r.passed)
    if (filterRes === 'fail') arr = arr.filter(r => !r.passed)
    if (searchText.trim()) {
      const q = searchText.toLowerCase()
      arr = arr.filter(r => (r.question || '').toLowerCase().includes(q))
    }

    return arr
  }, [allResults, filterCat, filterDiff, filterRes, searchText])

  const totalCount = (allResults || []).length

  if (!totalCount) {
    return (
      <div style={{ textAlign: 'center', padding: '36px 0', color: C.muted, fontSize: 14 }}>
        No question results yet. Run the evaluation pipeline first.
      </div>
    )
  }

  const sel = {
    background: C.card2, color: C.text2, border: `1px solid ${C.border}`,
    borderRadius: 7, padding: '6px 10px', fontSize: 13, cursor: 'pointer',
    outline: 'none',
  }

  return (
    <div>
      {/* ── Filter controls ──────────────────────────────────────────────── */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: 10,
        marginBottom: 14, alignItems: 'center',
      }}>
        <select value={filterCat} onChange={e => setFilterCat(e.target.value)} style={sel}>
          <option value="all">All Categories</option>
          <option value="admissions">admissions</option>
          <option value="tuition">tuition</option>
          <option value="housing">housing</option>
          <option value="scholarships">scholarships</option>
          <option value="graduate_admissions">graduate_admissions</option>
          <option value="general_info">general_info</option>
        </select>

        <select value={filterDiff} onChange={e => setFilterDiff(e.target.value)} style={sel}>
          <option value="all">All Difficulties</option>
          <option value="easy">easy</option>
          <option value="medium">medium</option>
          <option value="hard">hard</option>
        </select>

        <select value={filterRes} onChange={e => setFilterRes(e.target.value)} style={sel}>
          <option value="all">All Results</option>
          <option value="pass">PASS only</option>
          <option value="fail">FAIL only</option>
        </select>

        <div style={{ position: 'relative', flex: 1, minWidth: 180 }}>
          <Search size={14} style={{
            position: 'absolute', left: 10, top: '50%',
            transform: 'translateY(-50%)', color: C.muted, pointerEvents: 'none',
          }} />
          <input
            type="text"
            placeholder="Search questions…"
            value={searchText}
            onChange={e => setSearchText(e.target.value)}
            style={{ ...sel, paddingLeft: 30, width: '100%', boxSizing: 'border-box' }}
          />
        </div>

        <span style={{ color: C.muted, fontSize: 12, whiteSpace: 'nowrap' }}>
          Showing <strong style={{ color: C.text2 }}>{filtered.length}</strong> of {totalCount} questions
        </span>
      </div>

      {/* ── Table ─────────────────────────────────────────────────────────── */}
      <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto', borderRadius: 8 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead style={{ position: 'sticky', top: 0, zIndex: 10 }}>
            <tr style={{ background: C.card, borderBottom: `2px solid ${C.border}` }}>
              {['#', 'Question', 'Category', 'Difficulty', 'Faithfulness', 'Relevancy', 'Latency', 'Status'].map(h => (
                <th key={h} style={{
                  textAlign: 'left', padding: '10px 12px',
                  color: C.muted, fontWeight: 600,
                  fontSize: 11, letterSpacing: '0.05em', textTransform: 'uppercase',
                  background: C.card, whiteSpace: 'nowrap',
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={8} style={{
                  textAlign: 'center', padding: '24px', color: C.muted, fontSize: 13,
                  background: C.card3,
                }}>
                  No questions match the current filters.
                </td>
              </tr>
            ) : filtered.map((row, i) => {
              const rowBg = i % 2 === 0 ? C.card : C.card3
              const pass  = row.passed
              return (
                <tr
                  key={row.id ?? i}
                  style={{ background: rowBg, borderBottom: `1px solid ${C.border}20` }}
                  onMouseEnter={e => e.currentTarget.style.background = C.border}
                  onMouseLeave={e => e.currentTarget.style.background = rowBg}
                >
                  <td style={{ padding: '10px 12px', color: C.muted, fontWeight: 600, whiteSpace: 'nowrap' }}>
                    {row.id ?? i + 1}
                  </td>
                  <td style={{ padding: '10px 12px', maxWidth: 300 }}>
                    <span
                      title={row.question}
                      style={{
                        display: 'block', color: C.text2,
                        whiteSpace: 'nowrap', overflow: 'hidden',
                        textOverflow: 'ellipsis', maxWidth: 280,
                      }}
                    >
                      {row.question}
                    </span>
                  </td>
                  <td style={{ padding: '10px 12px' }}>
                    <span style={{
                      background: 'rgba(255,255,255,0.06)', color: C.text2,
                      borderRadius: 5, padding: '2px 7px', fontSize: 11, whiteSpace: 'nowrap',
                    }}>
                      {row.category || '—'}
                    </span>
                  </td>
                  <td style={{ padding: '10px 12px' }}>
                    <span style={{
                      color: row.difficulty === 'hard'   ? C.fail
                           : row.difficulty === 'medium' ? C.warn : C.pass,
                      fontSize: 11, fontWeight: 700, textTransform: 'uppercase',
                    }}>
                      {row.difficulty || '—'}
                    </span>
                  </td>
                  <td style={{ padding: '10px 12px', fontWeight: 600, color: scoreColor(row.faithfulness), whiteSpace: 'nowrap' }}>
                    {row.faithfulness != null ? row.faithfulness.toFixed(3) : 'N/A'}
                  </td>
                  <td style={{ padding: '10px 12px', fontWeight: 600, color: scoreColor(row.answer_relevancy), whiteSpace: 'nowrap' }}>
                    {row.answer_relevancy != null ? row.answer_relevancy.toFixed(3) : 'N/A'}
                  </td>
                  <td style={{ padding: '10px 12px', color: C.text2, whiteSpace: 'nowrap' }}>
                    {row.latency_seconds != null ? `${row.latency_seconds.toFixed(1)}s` : 'N/A'}
                  </td>
                  <td style={{ padding: '10px 12px' }}>
                    <span style={{
                      display: 'inline-flex', alignItems: 'center', gap: 5,
                      color: pass ? C.pass : C.fail,
                      fontSize: 12, fontWeight: 700,
                    }}>
                      <span style={{
                        width: 7, height: 7, borderRadius: '50%',
                        background: pass ? C.pass : C.fail,
                        display: 'inline-block', flexShrink: 0,
                      }} />
                      {pass ? 'PASS' : 'FAIL'}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
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
        <div style={{ marginTop: 12, paddingTop: 12, borderTop: `1px solid ${C.border}` }}>
          <p style={{
            color: C.muted, fontSize: 11, fontWeight: 600,
            letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 6,
          }}>
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
  const [data,       setData]       = useState(null)
  const [history,    setHistory]    = useState(null)
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState(null)
  const [countdown,  setCountdown]  = useState(REFRESH_INTERVAL)
  const [refreshing, setRefreshing] = useState(false)

  // ── Fetch both URLs ──────────────────────────────────────────────────────
  const fetchData = useCallback(() => {
    setRefreshing(true)
    Promise.all([
      fetch(REPORT_URL, { cache: 'no-store' }).then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status} — ${r.statusText}`)
        return r.json()
      }),
      fetch(HISTORY_URL, { cache: 'no-store' })
        .then(r => r.ok ? r.json() : null)
        .catch(() => null),
    ])
      .then(([report, hist]) => {
        setData(report)
        setHistory(hist)
        setLoading(false)
        setRefreshing(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
        setRefreshing(false)
      })
  }, [])

  // Initial load
  useEffect(() => { fetchData() }, [fetchData])

  // ── Auto-refresh countdown ───────────────────────────────────────────────
  useEffect(() => {
    let secs = REFRESH_INTERVAL
    setCountdown(secs)
    const timer = setInterval(() => {
      secs -= 1
      if (secs <= 0) {
        secs = REFRESH_INTERVAL
        fetchData()
      }
      setCountdown(secs)
    }, 1000)
    return () => clearInterval(timer)
  }, [fetchData])

  // ── Chart data from history ──────────────────────────────────────────────
  const chartData = useMemo(() => {
    if (!history?.runs?.length) return []
    return history.runs.map(run => ({
      label:              shortTs(run.run_timestamp),
      faithfulness:       run.faithfulness,
      answer_relevancy:   run.answer_relevancy,
      hallucination_rate: run.hallucination_rate,
      latency_p95:        run.latency_p95_seconds,
      cost_per_query:     run.cost_per_query_usd,
    }))
  }, [history])

  if (loading) return <LoadingScreen />
  if (error)   return <ErrorScreen message={error} onRetry={fetchData} />

  const overall    = data.overall_result ?? 'UNKNOWN'
  const isPass     = overall === 'PASS'
  const metrics    = data.metrics        ?? {}
  const gates      = data.gate_results?.gates ?? {}
  const failures   = data.sample_failures ?? []
  const allResults = data.all_results    ?? []
  const testMode   = data.test_mode
  const totalCost  = data.total_cost_usd
  const totalTok   = data.total_tokens

  return (
    <div className="fade-in" style={{ background: C.bg, minHeight: '100vh' }}>

      {/* ── Sticky top bar ────────────────────────────────────────────────── */}
      <div style={{
        background: C.card, borderBottom: `1px solid ${C.border}`,
        padding: '0 24px', display: 'flex', alignItems: 'center',
        height: 52, position: 'sticky', top: 0, zIndex: 100,
      }}>
        <span style={{ fontSize: 13, fontWeight: 700, color: C.gold, letterSpacing: '0.04em' }}>
          🎓 ASU LLM EVAL
        </span>
        <span style={{
          marginLeft: 12, fontSize: 12, color: C.muted,
          borderLeft: `1px solid ${C.border}`, paddingLeft: 12,
        }}>
          Quality Gate Dashboard
        </span>
        <div style={{ flex: 1 }} />
        <span style={{
          fontSize: 12,
          color: refreshing ? C.blue : C.muted,
          marginRight: 16,
        }}>
          {refreshing ? '↻ Refreshing…' : `Auto-refreshing in ${countdown}s`}
        </span>
        <button
          onClick={fetchData}
          title="Refresh now"
          style={{
            background: 'transparent', border: `1px solid ${C.border}`,
            borderRadius: 7, padding: '5px 10px', cursor: 'pointer',
            color: C.muted, display: 'flex', alignItems: 'center', gap: 5, fontSize: 12,
          }}
        >
          <RefreshCw size={13} /> Refresh
        </button>
      </div>

      {/* ── Page content ─────────────────────────────────────────────────── */}
      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '32px 24px' }}>

        {/* ─── SECTION 1: Header ──────────────────────────────────────────── */}
        <div style={{
          background: `linear-gradient(135deg, #1a0a12 0%, ${C.card} 55%, #0f1530 100%)`,
          border: `1px solid ${C.border}`,
          borderRadius: 16, padding: '28px 32px',
          marginBottom: 32,
        }}>
          {/* Title row */}
          <div style={{
            display: 'flex', flexWrap: 'wrap',
            alignItems: 'flex-start', justifyContent: 'space-between', gap: 20,
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
            </div>
          </div>

          {/* Stats chips row */}
          <div style={{
            display: 'flex', flexWrap: 'wrap', gap: 12,
            marginTop: 20, paddingTop: 20,
            borderTop: `1px solid ${C.border}40`,
          }}>
            {/* Questions */}
            <div style={{
              background: 'rgba(255,255,255,0.04)', border: `1px solid ${C.border}`,
              borderRadius: 8, padding: '8px 16px',
            }}>
              <div style={{ fontSize: 11, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                Questions
              </div>
              <div style={{
                fontSize: 20, fontWeight: 700, marginTop: 2,
                color: testMode ? C.warn : C.pass,
              }}>
                {data.total_questions ?? '—'}
                <span style={{ fontSize: 13, color: C.muted, fontWeight: 400 }}> / 100</span>
              </div>
            </div>

            {/* Total Cost */}
            <div style={{
              background: 'rgba(255,255,255,0.04)', border: `1px solid ${C.border}`,
              borderRadius: 8, padding: '8px 16px',
            }}>
              <div style={{ fontSize: 11, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                Total Cost This Run
              </div>
              <div style={{ fontSize: 20, fontWeight: 700, color: C.purple, marginTop: 2 }}>
                {totalCost != null ? `$${totalCost.toFixed(3)}` : 'N/A'}
              </div>
            </div>

            {/* Total Tokens */}
            <div style={{
              background: 'rgba(255,255,255,0.04)', border: `1px solid ${C.border}`,
              borderRadius: 8, padding: '8px 16px',
            }}>
              <div style={{ fontSize: 11, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                Total Tokens
              </div>
              <div style={{ fontSize: 20, fontWeight: 700, color: C.blue, marginTop: 2 }}>
                {totalTok != null ? totalTok.toLocaleString() : 'N/A'}
              </div>
            </div>

            {/* Run history count */}
            {history?.runs?.length > 0 && (
              <div style={{
                background: 'rgba(255,255,255,0.04)', border: `1px solid ${C.border}`,
                borderRadius: 8, padding: '8px 16px',
              }}>
                <div style={{ fontSize: 11, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  Total Runs
                </div>
                <div style={{ fontSize: 20, fontWeight: 700, color: C.text2, marginTop: 2 }}>
                  {history.runs.length}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ─── SECTION 2: Quality Metrics ────────────────────────────────── */}
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

        {/* ─── SECTION 3: Gate Status ─────────────────────────────────────── */}
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
            : <p style={{ color: C.muted, fontSize: 14 }}>No gate data available.</p>}
        </Section>

        {/* ─── SECTION 4: Trend Charts ─────────────────────────────────────── */}
        <Section
          title="Metric Trends"
          extra={
            history?.runs?.length > 0 && (
              <span style={{
                fontSize: 12, color: C.muted,
                background: 'rgba(255,255,255,0.05)',
                borderRadius: 6, padding: '3px 10px',
              }}>
                {history.runs.length} run{history.runs.length !== 1 ? 's' : ''}
              </span>
            )
          }
        >
          <TrendCharts chartData={chartData} />
        </Section>

        {/* ─── SECTION 5: Question Results ────────────────────────────────── */}
        <Section
          title="Question Results"
          extra={
            allResults.length > 0 && (
              <span style={{
                fontSize: 12, color: C.muted,
                background: 'rgba(255,255,255,0.05)',
                borderRadius: 6, padding: '3px 10px',
              }}>
                {allResults.length} question{allResults.length !== 1 ? 's' : ''}
              </span>
            )
          }
        >
          <QuestionTable allResults={allResults} />
        </Section>

        {/* ─── SECTION 6: Sample Failures ─────────────────────────────────── */}
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
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: C.pass, padding: '8px 0' }}>
                <CheckCircle size={18} strokeWidth={2.5} />
                <span style={{ fontSize: 14, fontWeight: 500 }}>No failures in this run</span>
              </div>
            )
            : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {failures.map((f, i) => <FailureCard key={i} failure={f} />)}
              </div>
            )
          }
        </Section>

        {/* ─── SECTION 7: Footer ──────────────────────────────────────────── */}
        <footer style={{
          borderTop: `1px solid ${C.border}`,
          paddingTop: 24, marginTop: 8,
          display: 'flex', flexWrap: 'wrap',
          alignItems: 'center', justifyContent: 'space-between', gap: 12,
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

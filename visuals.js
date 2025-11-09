// visuals.js (ES module, no bundler)
import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'
import Regions    from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js'

// 🔧 Point these to your files (relative to index.html)
const AUDIO_URL = './Rocky.mp3'
const JSON_URL  = './mapped_emotions.json'

// UI elements
const playBtn       = document.getElementById('play')
const stopBtn       = document.getElementById('stop')
const nowEl         = document.getElementById('now')
const segmentMeta   = document.getElementById('segmentMeta')
const segmentText   = document.getElementById('segmentText')
const voiceDetails  = document.getElementById('voiceDetails')
const gladiaDetails = document.getElementById('gladiaDetails')
const segmentCard   = document.getElementById('segmentCard')
const voiceCard     = document.getElementById('voiceCard')
const gladiaCard    = document.getElementById('gladiaCard')
const segmentBadge  = document.getElementById('segmentBadge')

let segmentsData = []
let activeSegmentIndex = null

resetDetails()

const regionsPlugin = Regions.create({
  dragSelection: false,
  slop: 0,
})

const ws = WaveSurfer.create({
  container: '#waveform',
  url: AUDIO_URL,
  height: 140,
  waveColor: '#93c5fd',
  progressColor: '#3b82f6',
  cursorColor: '#0f172a',
  minPxPerSec: 50,
  plugins: [regionsPlugin],
})

// Controls
playBtn.onclick = () => ws.playPause()
stopBtn.onclick = () => ws.stop()
ws.on('audioprocess', (t) => {
  nowEl.textContent = `${t.toFixed(2)}s`
  updateSegmentForTime(t)
})
ws.on('seek', (progress) => {
  const time = progress * ws.getDuration()
  nowEl.textContent = `${time.toFixed(2)}s`
  updateSegmentForTime(time)
})
ws.on('pause', () => updateSegmentForTime(ws.getCurrentTime()))
ws.on('finish', () => updateSegmentForTime(ws.getDuration()))

// Load JSON when audio is ready
ws.on('ready', async () => {
  const raw = await fetch(JSON_URL).then(r => r.json())
  const segments = Array.isArray(raw) ? raw : (raw.segments || raw.items || [])
  segmentsData = segments.map((item, idx) => ({ ...item, __idx: idx }))
  highlightMismatchRegions(segmentsData)
  if (segmentsData.length) {
    updateSegmentForTime(ws.getCurrentTime())
  } else {
    resetDetails()
  }
})

function showSegment(item) {
  if (!item) return resetDetails()
  if (typeof item.__idx === 'number') {
    activeSegmentIndex = item.__idx
  }
  segmentMeta.textContent = `Segment #${item.index}  |  Speaker ${item.speaker}  |  ${fix(item.start)}s – ${fix(item.end)}s`
  segmentText.textContent = item.text || '–'
  applyMismatchState(isMismatch(item), 'In sync')

  const vad = item.normalized_vad || {}
  const voiceStats = [
    { label: 'Sentiment', value: fallback(item.voice_sentiment) },
    { label: 'Emotion', value: fallback(item.voice_emotion_label) },
    { label: 'Confidence', value: num(item.voice_emotion_prob) },
    { label: 'VAD (V/A/D)', value: `V:${num(vad.valence)}  A:${num(vad.arousal)}  D:${num(vad.dominance)}` },
  ]
  const gladiaStats = [
    { label: 'Sentiment', value: fallback(item.gladia_sentiment) },
    { label: 'Emotion', value: fallback(item.gladia_emotion) },
  ]

  voiceDetails.innerHTML = renderStats(voiceStats)
  gladiaDetails.innerHTML = renderStats(gladiaStats)
}

function resetDetails() {
  activeSegmentIndex = null
  segmentMeta.textContent = 'Press play or scrub the waveform to explore segment analysis.'
  segmentText.textContent = ''
  applyMismatchState(false, 'Awaiting segment')
  voiceDetails.innerHTML = renderStats([
    { label: 'Sentiment', value: '–' },
    { label: 'Emotion', value: '–' },
    { label: 'Confidence', value: '–' },
    { label: 'VAD (V/A/D)', value: '–' },
  ])
  gladiaDetails.innerHTML = renderStats([
    { label: 'Sentiment', value: '–' },
    { label: 'Emotion', value: '–' },
  ])
}

function updateSegmentForTime(time) {
  if (!segmentsData.length || Number.isNaN(time)) return
  const segment = segmentsData.find((item) => {
    const start = Number(item.start) || 0
    const end = Number(item.end) || start
    return time >= start && time <= end
  })
  if (!segment) {
    const firstStart = Number(segmentsData[0]?.start) || 0
    const lastEnd = Number(segmentsData[segmentsData.length - 1]?.end) || firstStart
    if ((time < firstStart || time > lastEnd) && activeSegmentIndex !== null) {
      resetDetails()
    }
    return
  }
  if (segment.__idx !== activeSegmentIndex) {
    showSegment(segment)
  }
}

function isMismatch(item) {
  if (!item) return false
  const gladiaSent = normalize(item.gladia_sentiment)
  const voiceSent = normalize(item.voice_sentiment)
  const gladiaEmotion = normalize(item.gladia_emotion)
  const voiceEmotion = normalize(item.voice_emotion_label)
  return (
    gladiaSent &&
    voiceSent &&
    gladiaEmotion &&
    voiceEmotion &&
    gladiaSent !== voiceSent &&
    gladiaEmotion !== voiceEmotion
  )
}

function applyMismatchState(flag, okLabel = 'In sync') {
  ;[segmentCard, voiceCard, gladiaCard].forEach((el) => {
    if (!el) return
    el.classList.toggle('mismatch', flag)
  })
  if (segmentBadge) {
    segmentBadge.classList.toggle('mismatch', flag)
    segmentBadge.textContent = flag ? 'Mismatch detected' : okLabel
  }
}

function highlightMismatchRegions(segments) {
  if (!regionsPlugin) return
  regionsPlugin.clearRegions()
  segments
    .filter(isMismatch)
    .forEach((item) => {
      const start = Number(item.start) || 0
      const end = Number(item.end) || start
      if (end <= start) return
      regionsPlugin.addRegion({
        start,
        end,
        drag: false,
        resize: false,
        content: '',
        color: 'rgba(225,29,72,0.22)',
      })
    })
}

function renderStats(list) {
  return list.map(({ label, value }) => (
    `<div class="stat"><span class="label">${escapeHtml(label)}</span><span class="value">${escapeHtml(value)}</span></div>`
  )).join('')
}

function fallback(v) {
  return v == null || v === '' ? '–' : String(v)
}
function fix(n){ return Number(n).toFixed(3) }
function num(n){
  if (n == null || Number.isNaN(+n)) return '–'
  const value = Number(n)
  return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(3)
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) => (
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]
  ))
}

function normalize(value) {
  if (value == null) return null
  return String(value).trim().toLowerCase()
}
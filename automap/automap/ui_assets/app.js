const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const audio = $('audio');
let state = null;
let busy = false;
let raf = 0;

function formatTime(value) {
  const seconds = Math.max(0, Number(value) || 0);
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
}

function setAudioTime(value) {
  try { audio.currentTime = value; } catch { /* metadata not ready yet */ }
}

function hslFor(midi, low, high) {
  const span = Math.max(1, high - low);
  const t = Math.min(1, Math.max(0, (midi - low) / span));
  // Low notes are cyan, middle notes green, high notes red.
  return `hsl(${Math.round(190 - t * 190)}, 82%, 64%)`;
}

function lyricAt(time) {
  if (!state || !Array.isArray(state.lines)) return '';
  const line = state.lines.find((item) => time >= item.start - 0.2 && time <= item.end + 0.4);
  return line ? line.text : '';
}

function resizeCanvas() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { dpr, width: rect.width, height: rect.height };
}

function draw() {
  const ctx = canvas.getContext('2d');
  const { dpr, width, height } = resizeCanvas();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0c111c';
  ctx.fillRect(0, 0, width, height);
  if (!state) return;

  const padX = 26;
  const padY = 30;
  const innerW = Math.max(1, width - padX * 2);
  const innerH = Math.max(1, height - padY * 2);
  const duration = Math.max(1, state.duration);
  const low = state.low;
  const high = state.high;
  const yFor = (midi) => padY + innerH - ((midi - low) / Math.max(1, high - low)) * innerH;
  const xFor = (time) => padX + (time / duration) * innerW;

  // Faint pitch lanes and time grid.
  ctx.lineWidth = 1;
  for (let lane = 0; lane <= 8; lane += 1) {
    const y = padY + (innerH / 8) * lane;
    ctx.strokeStyle = lane % 2 === 0 ? 'rgba(255,255,255,0.055)' : 'rgba(255,255,255,0.028)';
    ctx.beginPath();
    ctx.moveTo(padX, y);
    ctx.lineTo(width - padX, y);
    ctx.stroke();
  }
  const tick = duration > 240 ? 60 : duration > 120 ? 30 : 15;
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillStyle = 'rgba(255,255,255,0.32)';
  for (let time = 0; time <= duration; time += tick) {
    const x = xFor(time);
    ctx.strokeStyle = 'rgba(255,255,255,0.055)';
    ctx.beginPath();
    ctx.moveTo(x, padY);
    ctx.lineTo(x, height - padY);
    ctx.stroke();
    ctx.fillText(formatTime(time), x + 4, height - 8);
  }

  // Note bars.
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const noteHeight = Math.max(5, Math.min(13, innerH / Math.max(10, high - low) * 0.72));
  for (const note of state.notes || []) {
    const x = xFor(note.start);
    const w = Math.max(noteHeight, xFor(note.start + note.dur) - x);
    const y = yFor(note.midi);
    ctx.strokeStyle = hslFor(note.midi, low, high);
    ctx.lineWidth = noteHeight;
    ctx.shadowColor = ctx.strokeStyle;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + w, y);
    ctx.stroke();
  }
  ctx.shadowBlur = 0;

  // Playhead.
  const time = Math.min(duration, Math.max(0, audio.currentTime || 0));
  const playX = xFor(time);
  const gradient = ctx.createLinearGradient(0, padY, 0, height - padY);
  gradient.addColorStop(0, 'rgba(255,255,255,0.0)');
  gradient.addColorStop(0.5, 'rgba(255,255,255,0.65)');
  gradient.addColorStop(1, 'rgba(255,255,255,0.0)');
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(playX, padY);
  ctx.lineTo(playX, height - padY);
  ctx.stroke();

  $('lyric').textContent = lyricAt(time);
  $('time').textContent = `${formatTime(time)} / ${formatTime(duration)}`;
}

function tick() {
  draw();
  raf = requestAnimationFrame(tick);
}

function setBusy(value, message) {
  busy = value;
  $('down').disabled = value;
  $('up').disabled = value;
  $('next').disabled = value;
  $('loading').hidden = !value;
  if (message) $('loading-text').textContent = message;
  if (!value) draw();
}

function applyState(next) {
  const previousCandidate = state && state.candidate_id;
  state = next;
  $('title').textContent = state.title || 'Untitled';
  $('artist').textContent = state.artist || '';
  $('meta').textContent = `${state.note_count} notes · model step ${state.model_step}`;
  const reward = state.last_reward;
  if (typeof reward === 'number' && reward !== 0) {
    $('status').className = `status ${reward > 0 ? 'positive' : 'negative'}`;
    $('status').textContent = `last grade ${reward > 0 ? 'GOOD' : 'BAD'} · ${state.note_count} notes · step ${state.model_step}`;
  } else {
    $('status').className = 'status';
    $('status').textContent = `${state.note_count} notes · step ${state.model_step} · listen, then grade`;
  }
  if (!previousCandidate || previousCandidate !== state.candidate_id) {
    audio.pause();
    audio.src = state.audio_url;
    setAudioTime(0);
    audio.load();
    $('play').textContent = 'PLAY';
  }
  draw();
}

async function request(url, options = {}) {
  const response = await fetch(url, { cache: 'no-store', ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed: ${response.status}`);
  return payload;
}

async function loadState() {
  try {
    applyState(await request('/api/state'));
  } catch (error) {
    $('status').textContent = String(error.message || error);
  }
}

async function grade(value) {
  if (!state || busy) return;
  setBusy(true, value > 0 ? 'Reinforcing the good map and generating the next…' : 'Reinforcing the bad map and generating the next…');
  try {
    const payload = await request('/api/grade', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ grade: value, candidate_id: state.candidate_id })
    });
    applyState(payload.state);
  } catch (error) {
    $('status').textContent = String(error.message || error);
  } finally {
    setBusy(false);
  }
}

async function nextCandidate() {
  if (!state || busy) return;
  setBusy(true, 'Generating another candidate…');
  try {
    const payload = await request('/api/next', { method: 'POST' });
    applyState(payload.state);
  } catch (error) {
    $('status').textContent = String(error.message || error);
  } finally {
    setBusy(false);
  }
}

$('down').addEventListener('click', () => grade(1));
$('up').addEventListener('click', () => grade(5));
$('next').addEventListener('click', nextCandidate);
$('play').addEventListener('click', () => {
  if (audio.paused) {
    audio.play().then(() => { $('play').textContent = 'PAUSE'; }).catch(() => {});
  } else {
    audio.pause();
    $('play').textContent = 'PLAY';
  }
});
audio.addEventListener('play', () => { $('play').textContent = 'PAUSE'; });
audio.addEventListener('pause', () => { $('play').textContent = 'PLAY'; });
audio.addEventListener('ended', () => { $('play').textContent = 'PLAY'; });

// Click or drag the preview to scrub. The canvas is the timeline.
let scrubbing = false;
function scrub(event) {
  if (!state) return;
  const rect = canvas.getBoundingClientRect();
  const padX = 26;
  const innerW = Math.max(1, rect.width - padX * 2);
  const x = Math.min(rect.width - padX, Math.max(padX, event.clientX - rect.left));
  setAudioTime(Math.max(0, Math.min(state.duration, ((x - padX) / innerW) * state.duration)));
  draw();
}
canvas.addEventListener('pointerdown', (event) => {
  scrubbing = true;
  canvas.setPointerCapture(event.pointerId);
  scrub(event);
});
canvas.addEventListener('pointermove', (event) => { if (scrubbing) scrub(event); });
canvas.addEventListener('pointerup', () => { scrubbing = false; });
canvas.addEventListener('pointercancel', () => { scrubbing = false; });

window.addEventListener('keydown', (event) => {
  if (event.target instanceof HTMLInputElement) return;
  if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') {
    event.preventDefault();
    grade(1);
  } else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') {
    event.preventDefault();
    grade(5);
  } else if (event.key.toLowerCase() === 'n') {
    event.preventDefault();
    nextCandidate();
  } else if (event.code === 'Space') {
    event.preventDefault();
    $('play').click();
  }
});
window.addEventListener('resize', draw);
loadState();
tick();

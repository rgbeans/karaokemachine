const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const vocals = $('vocals');
const backing = $('backing');
const seek = $('seek');
let state = null;
let busy = false;
let raf = 0;
let vocalsOn = true;
let backingOn = true;
let seeking = false;

function formatTime(value) {
  const seconds = Math.max(0, Number(value) || 0);
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
}

function setElementTime(element, value) {
  try { element.currentTime = value; } catch { /* metadata not ready yet */ }
}

function trackAvailable(kind) {
  return Boolean(state && state.audio && state.audio[`${kind}_available`]);
}

function masterElement() {
  if (trackAvailable('backing')) return backing;
  if (trackAvailable('vocals')) return vocals;
  return null;
}

function currentTime() {
  const master = masterElement();
  return master ? Number(master.currentTime) || 0 : 0;
}

function songDuration() {
  return state ? Math.max(1, Number(state.duration) || 1) : 1;
}

function setTime(value) {
  const time = Math.max(0, Math.min(songDuration(), Number(value) || 0));
  if (trackAvailable('vocals')) setElementTime(vocals, time);
  if (trackAvailable('backing')) setElementTime(backing, time);
  seek.value = String(time);
  draw();
}

function updateToggleLabels() {
  const vocalsReady = trackAvailable('vocals');
  const backingReady = trackAvailable('backing');
  $('vocalsToggle').disabled = !vocalsReady;
  $('backingToggle').disabled = !backingReady;
  $('vocalsToggle').classList.toggle('on', vocalsOn && vocalsReady);
  $('backingToggle').classList.toggle('on', backingOn && backingReady);
  $('vocalsToggle').textContent = `VOCALS ${vocalsOn && vocalsReady ? 'ON' : 'OFF'}`;
  $('backingToggle').textContent = `BACKING ${backingOn && backingReady ? 'ON' : 'OFF'}`;
  $('vocalsToggle').setAttribute('aria-pressed', String(vocalsOn && vocalsReady));
  $('backingToggle').setAttribute('aria-pressed', String(backingOn && backingReady));
  vocals.muted = !vocalsOn || !vocalsReady;
  backing.muted = !backingOn || !backingReady;
}

function syncAudio() {
  const vocalsReady = trackAvailable('vocals');
  const backingReady = trackAvailable('backing');
  if (!vocalsReady || !backingReady) return;
  if (vocals.readyState < 2 || backing.readyState < 2) return;
  if (vocals.paused && backing.paused) return;
  const master = masterElement();
  if (!master) return;
  const slave = master === backing ? vocals : backing;
  if (Math.abs(slave.currentTime - master.currentTime) > 0.08) {
    setElementTime(slave, master.currentTime);
  }
}

function activeElements() {
  return [vocals, backing].filter((element) =>
    element === vocals ? trackAvailable('vocals') : trackAvailable('backing')
  );
}

async function playAll() {
  const elements = activeElements();
  if (!elements.length) return;
  await Promise.allSettled(elements.map((element) => element.play()));
  $('play').textContent = elements.some((element) => !element.paused) ? 'PAUSE' : 'PLAY';
}

function pauseAll() {
  vocals.pause();
  backing.pause();
  $('play').textContent = 'PLAY';
}

function statusFor(state) {
  if (state.last_event) {
    const event = state.last_event;
    if (event.kind === 'grade') {
      const good = Number(event.reward) > 0;
      $('status').className = `status ${good ? 'positive' : 'negative'}`;
      if (event.action === 'retry') {
        return `last BAD on ${event.title} · ${event.grades_for_song || 0} graded there · another candidate for this song`;
      }
      return `last ${good ? 'GOOD' : 'BAD'} on ${event.title} · ${event.grades_for_song || 0} graded there · next ${state.next_title}`;
    }
    $('status').className = 'status';
    return `skipped ${event.title} · next ${state.next_title}`;
  }
  $('status').className = 'status';
  return `round ${state.round} · ${state.song_index}/${state.song_total} · ${state.grades_for_song} graded here · listen, then grade`;
}

function applyState(next) {
  const hadState = Boolean(state);
  const previousSong = state && state.song_id;
  const previousCandidate = state && state.candidate_id;
  const songChanged = !hadState || previousSong !== next.song_id;
  const candidateChanged = !hadState || previousCandidate !== next.candidate_id;
  state = next;

  $('queue').textContent = `${(state.album || '').toUpperCase()} · ${state.song_index}/${state.song_total} · ROUND ${state.round}`;
  $('title').textContent = state.title || 'Untitled';
  $('artist').textContent = state.artist || '';
  $('meta').textContent = `${state.note_count} notes · model step ${state.model_step}`;
  $('status').textContent = statusFor(state);

  if (songChanged) {
    pauseAll();
    if (trackAvailable('vocals')) vocals.src = state.audio.vocals;
    else vocals.removeAttribute('src');
    if (trackAvailable('backing')) backing.src = state.audio.backing;
    else backing.removeAttribute('src');
    vocals.load();
    backing.load();
    setElementTime(vocals, 0);
    setElementTime(backing, 0);
    const resume = Number(state.resume_position) || 0;
    if (resume > 0.5) {
      const restore = () => {
        setElementTime(vocals, resume);
        setElementTime(backing, resume);
        seek.value = String(resume);
        draw();
      };
      vocals.addEventListener('loadedmetadata', restore, { once: true });
      backing.addEventListener('loadedmetadata', restore, { once: true });
    }
  } else if (candidateChanged) {
    pauseAll();
    setElementTime(vocals, 0);
    setElementTime(backing, 0);
  }

  updateToggleLabels();
  $('play').disabled = !trackAvailable('vocals') && !trackAvailable('backing');
  seek.max = String(songDuration());
  if (candidateChanged) seek.value = '0';
  draw();
}

function hslFor(midi, low, high) {
  const span = Math.max(1, high - low);
  const t = Math.min(1, Math.max(0, (midi - low) / span));
  return `hsl(${Math.round(190 - t * 190)}, 82%, 64%)`;
}

function visibleWindow() {
  const duration = songDuration();
  const span = Math.min(14, duration);
  let start = currentTime() - span * 0.28;
  start = Math.max(0, Math.min(Math.max(0, duration - span), start));
  return { start, span };
}

function roundedRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function drawPlacedWords(ctx, time, windowStart, windowSpan, xFor, padX, padY, innerW, innerH, width) {
  if (!state || !Array.isArray(state.placedWords)) return;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const word of state.placedWords) {
    if (word.start < windowStart - 1.5 || word.start > windowStart + windowSpan + 0.5) continue;
    const text = String(word.text || '');
    if (!text) continue;
    const active = Math.abs(time - word.start) <= 0.45;
    const display = text.length > 24 ? `${text.slice(0, 23)}…` : text;
    ctx.font = active
      ? '600 13px Inter, ui-sans-serif, system-ui, sans-serif'
      : '500 11px Inter, ui-sans-serif, system-ui, sans-serif';
    const pillWidth = Math.min(190, ctx.measureText(display).width + 14);
    const pillHeight = active ? 24 : 21;
    const half = pillWidth / 2 + 2;
    const centerX = Math.max(padX + half, Math.min(width - padX - half, xFor(word.start)));
    const rawY = padY + Math.max(0.02, Math.min(0.98, Number(word.y) || 0.5)) * innerH;
    const centerY = Math.max(
      padY + pillHeight / 2 + 2,
      Math.min(padY + innerH - pillHeight / 2 - 2, rawY)
    );
    roundedRect(ctx, centerX - pillWidth / 2, centerY - pillHeight / 2, pillWidth, pillHeight, 5);
    ctx.fillStyle = active ? 'rgba(5,8,15,0.90)' : 'rgba(5,8,15,0.62)';
    ctx.fill();
    ctx.strokeStyle = active ? 'rgba(255,209,102,0.65)' : 'rgba(255,255,255,0.14)';
    ctx.lineWidth = 1;
    ctx.stroke();
    if (active) {
      ctx.strokeStyle = '#ffd166';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(centerX - pillWidth / 2 + 5, centerY + pillHeight / 2 + 2);
      ctx.lineTo(centerX + pillWidth / 2 - 5, centerY + pillHeight / 2 + 2);
      ctx.stroke();
    }
    ctx.fillStyle = active ? '#ffffff' : 'rgba(244,247,251,0.78)';
    ctx.fillText(display, centerX, centerY);
  }
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
  const { start, span } = visibleWindow();
  const low = state.low;
  const high = state.high;
  const yFor = (midi) => padY + innerH - ((midi - low) / Math.max(1, high - low)) * innerH;
  const xFor = (time) => padX + ((time - start) / span) * innerW;

  // Faint pitch lanes.
  ctx.lineWidth = 1;
  for (let lane = 0; lane <= 8; lane += 1) {
    const y = padY + (innerH / 8) * lane;
    ctx.strokeStyle = lane % 2 === 0 ? 'rgba(255,255,255,0.055)' : 'rgba(255,255,255,0.028)';
    ctx.beginPath();
    ctx.moveTo(padX, y);
    ctx.lineTo(width - padX, y);
    ctx.stroke();
  }

  // Time grid and labels.
  const tick = span > 18 ? 5 : span > 9 ? 2 : 1;
  const firstTick = Math.ceil(start / tick) * tick;
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillStyle = 'rgba(255,255,255,0.34)';
  for (let time = firstTick; time <= start + span + 0.01; time += tick) {
    const x = xFor(time);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.beginPath();
    ctx.moveTo(x, padY);
    ctx.lineTo(x, height - padY);
    ctx.stroke();
    ctx.fillText(formatTime(time), x + 4, height - 8);
  }

  // Notes in the visible window.
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const noteHeight = Math.max(5, Math.min(13, innerH / Math.max(10, high - low) * 0.72));
  for (const note of state.notes || []) {
    if (note.start + note.dur < start - 0.2 || note.start > start + span + 0.2) continue;
    const leftTime = Math.max(note.start, start);
    const rightTime = Math.min(note.start + note.dur, start + span);
    const x = xFor(leftTime);
    const w = Math.max(noteHeight, xFor(rightTime) - x);
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

  // Fixed playhead in the scrolling window.
  const time = currentTime();
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

  // The mapped lyric labels sit at their actual time and vertical position.
  drawPlacedWords(ctx, time, start, span, xFor, padX, padY, innerW, innerH, width);

  $('time').textContent = `${formatTime(time)} / ${formatTime(songDuration())}`;
}

function tick() {
  syncAudio();
  if (state) {
    if (!seeking) seek.value = String(currentTime());
    draw();
  }
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
  setBusy(true, value > 0
    ? 'Reinforcing the good map and continuing the queue…'
    : 'Reinforcing the bad map and continuing the queue…');
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
  setBusy(true, 'Skipping this candidate and continuing the queue…');
  try {
    const payload = await request('/api/next', { method: 'POST' });
    applyState(payload.state);
  } catch (error) {
    $('status').textContent = String(error.message || error);
  } finally {
    setBusy(false);
  }
}

function sendPosition(beacon = false) {
  if (!state) return;
  const payload = JSON.stringify({
    candidate_id: state.candidate_id,
    seconds: currentTime()
  });
  if (beacon && navigator.sendBeacon) {
    navigator.sendBeacon('/api/position', new Blob([payload], { type: 'application/json' }));
    return;
  }
  fetch('/api/position', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: payload,
    keepalive: true
  }).catch(() => {});
}

$('down').addEventListener('click', () => grade(1));
$('up').addEventListener('click', () => grade(5));
$('next').addEventListener('click', nextCandidate);
$('vocalsToggle').addEventListener('click', () => {
  vocalsOn = !vocalsOn;
  updateToggleLabels();
});
$('backingToggle').addEventListener('click', () => {
  backingOn = !backingOn;
  updateToggleLabels();
});
$('play').addEventListener('click', () => {
  const master = masterElement();
  if (!master) return;
  if (master.paused) playAll();
  else pauseAll();
});

// Keep the play button label honest when one of the two stems ends.
for (const element of [vocals, backing]) {
  element.addEventListener('play', () => { $('play').textContent = 'PAUSE'; });
  element.addEventListener('pause', () => {
    if (vocals.paused && backing.paused) $('play').textContent = 'PLAY';
  });
  element.addEventListener('ended', () => {
    if (vocals.paused && backing.paused) $('play').textContent = 'PLAY';
  });
}

seek.addEventListener('input', () => {
  seeking = true;
  setTime(Number(seek.value));
});
seek.addEventListener('change', () => { seeking = false; });
seek.addEventListener('pointerup', () => { seeking = false; });

let scrubbing = false;
let scrubWindow = null;
function scrub(event) {
  if (!state || !scrubWindow) return;
  const rect = canvas.getBoundingClientRect();
  const padX = 26;
  const innerW = Math.max(1, rect.width - padX * 2);
  const x = Math.min(rect.width - padX, Math.max(padX, event.clientX - rect.left));
  setTime(scrubWindow.start + ((x - padX) / innerW) * scrubWindow.span);
}
canvas.addEventListener('pointerdown', (event) => {
  scrubbing = true;
  scrubWindow = visibleWindow();
  canvas.setPointerCapture(event.pointerId);
  scrub(event);
});
canvas.addEventListener('pointermove', (event) => { if (scrubbing) scrub(event); });
canvas.addEventListener('pointerup', () => { scrubbing = false; scrubWindow = null; });
canvas.addEventListener('pointercancel', () => { scrubbing = false; scrubWindow = null; });

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
window.addEventListener('pagehide', () => sendPosition(true));
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') sendPosition(true);
});
window.addEventListener('beforeunload', () => sendPosition(true));
setInterval(() => { if (!busy && state) sendPosition(false); }, 5000);

loadState();
tick();

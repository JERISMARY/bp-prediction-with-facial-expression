/* ─── State ──────────────────────────────── */
let donutChart = null, radarChart = null, modelBarChart = null, gaugeAnimId = null;
let daySodiumChart = null, dayRatingChart = null, dayCategoryChart = null;
let weekHealthChart = null, weekSodiumChart = null;
let mediaRecorder = null, audioChunks = [];
let fontScale = 1.0;
const FONT_STEPS = [0.85, 1.0, 1.1, 1.2];
let fontIdx = 1;

/* ─────────────────────────────────────────
   THEME TOGGLE (Dark / Light)
───────────────────────────────────────── */
function toggleTheme() {
  const isLight = document.body.classList.toggle('light');
  localStorage.setItem('bp_theme', isLight ? 'light' : 'dark');
  const btn = document.getElementById('themeBtn');
  btn.textContent = isLight ? '🌙 Dark' : '☀️ Light';
  showToast(isLight ? '☀️ Light mode on' : '🌙 Dark mode on');
}
function applyTheme() {
  const saved = localStorage.getItem('bp_theme');
  if (saved === 'light') {
    document.body.classList.add('light');
    document.getElementById('themeBtn').textContent = '🌙 Dark';
  }
}

/* ─────────────────────────────────────────
   FONT SIZE
───────────────────────────────────────── */
function changeFontSize(dir) {
  if (dir > 0) fontIdx = Math.min(fontIdx + 1, FONT_STEPS.length - 1);
  else if (dir < 0) fontIdx = Math.max(fontIdx - 1, 0);
  else fontIdx = 1;
  const scale = FONT_STEPS[fontIdx];
  document.documentElement.style.setProperty('--font-scale', scale);
  localStorage.setItem('bp_font', fontIdx);
  showToast(`Font size: ${Math.round(scale * 100)}%`);
}
function applyFontSize() {
  const saved = parseInt(localStorage.getItem('bp_font'));
  if (!isNaN(saved)) {
    fontIdx = Math.max(0, Math.min(saved, FONT_STEPS.length - 1));
    document.documentElement.style.setProperty('--font-scale', FONT_STEPS[fontIdx]);
  }
}

/* ─────────────────────────────────────────
   FORM PROGRESS BAR
───────────────────────────────────────── */
const REQUIRED_FIELDS = ['age_group', 'exercise_frequency', 'systolic_bp', 'diastolic_bp'];
function updateProgress() {
  let filled = 0;
  REQUIRED_FIELDS.forEach(f => { if (document.getElementById(f).value !== '') filled++; });
  const pct = Math.round((filled / REQUIRED_FIELDS.length) * 100);
  document.getElementById('progFill').style.width = pct + '%';
  document.getElementById('progPct').textContent = pct + '%';
}

/* ─────────────────────────────────────────
   SEGMENT CONTROLS
───────────────────────────────────────── */
function setSeg(segId, hiddenId, btn) {
  document.getElementById(segId).querySelectorAll('.seg-btn')
    .forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(hiddenId).value = btn.dataset.val;
}

/* ─────────────────────────────────────────
   BP COLOR FEEDBACK
───────────────────────────────────────── */
const BP_COLORS = ['#22c55e', '#84cc16', '#f59e0b', '#ef4444', '#7f1d1d'];
const BP_LABELS = ['< 120 mmHg (Normal)', '120–129 (Elevated)', '130–139 (Stage-1)', '140–159 (Stage-2)', '160+ (Crisis)'];
const DIA_COLORS = ['#22c55e', '#f59e0b', '#ef4444', '#7f1d1d'];
const DIA_LABELS = ['< 80 mmHg (Normal)', '80–89 (Stage-1)', '90–99 (Stage-2)', '100+ (Crisis)'];
function updateBPColor() {
  const sv = document.getElementById('systolic_bp').value;
  const dv = document.getElementById('diastolic_bp').value;
  const ind = document.getElementById('bpIndicator');
  if (sv === '' && dv === '') { ind.style.display = 'none'; return; }
  ind.style.display = 'block';
  let parts = [];
  if (sv !== '') { const c = BP_COLORS[+sv]; parts.push(`<span style="color:${c}">Systolic: ${BP_LABELS[+sv]}</span>`); ind.style.borderLeftColor = c; ind.style.background = c + '18'; }
  if (dv !== '') { const c = DIA_COLORS[+dv]; parts.push(`<span style="color:${c}">Diastolic: ${DIA_LABELS[+dv]}</span>`); }
  ind.innerHTML = parts.join('  •  ');
}

/* ─────────────────────────────────────────
   COLLECT FORM DATA
───────────────────────────────────────── */
function collectFormData() {
  for (const f of REQUIRED_FIELDS) {
    if (!document.getElementById(f).value)
      throw new Error(`Required field missing: ${f.replace(/_/g, ' ')}`);
  }
  return {
    gender: document.getElementById('gender').value,
    age_group: document.getElementById('age_group').value,
    family_history: document.getElementById('family_history').value,
    patient_status: document.getElementById('patient_status').value,
    take_medication: document.getElementById('take_medication').value,
    time_since_diagnosis: document.getElementById('time_since_diagnosis').value,
    symptom_severity: document.getElementById('symptom_severity').value,
    shortness_of_breath: document.getElementById('shortness_of_breath').value,
    visual_changes: document.getElementById('visual_changes').value,
    nosebleeds: document.getElementById('nosebleeds').value,
    systolic_bp: document.getElementById('systolic_bp').value,
    diastolic_bp: document.getElementById('diastolic_bp').value,
    controlled_diet: document.getElementById('controlled_diet').value,
    bmi_category: document.getElementById('bmi_category').value,
    diabetes: document.getElementById('diabetes').value,
    cholesterol_level: document.getElementById('cholesterol_level').value,
    heart_rate_category: document.getElementById('heart_rate_category').value,
    exercise_frequency: document.getElementById('exercise_frequency').value,
  };
}

/* ─────────────────────────────────────────
   RUN PREDICTION
───────────────────────────────────────── */
async function runPrediction() {
  let data;
  try { data = collectFormData(); }
  catch (e) { showToast('⚠️ ' + e.message, 'error'); return; }

  const btn = document.getElementById('predictBtn');
  const btnText = document.getElementById('btnText');
  const btnLoader = document.getElementById('btnLoader');
  btn.disabled = true; btnText.style.display = 'none'; btnLoader.style.display = 'flex';

  try {
    const res = await fetch('/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data)
    });
    const json = await res.json();
    if (json.error) throw new Error(json.error);
    window._lastPredStage = json.stage; // Save for analysis alignment
    showResult(json);
    addToHistory(json);
  } catch (e) {
    showToast('❌ Prediction failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false; btnText.style.display = 'flex'; btnLoader.style.display = 'none';
  }
}

/* ─────────────────────────────────────────
   SHOW RESULT
───────────────────────────────────────── */
function showResult(res) {
  const panel = document.getElementById('resultPanel');
  panel.style.display = 'block';
  setTimeout(() => panel.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);

  // Patient name in result
  const pname = document.getElementById('patientName').value.trim();
  const page = document.getElementById('patientAge').value.trim();
  const nameEl = document.getElementById('resultPatientName');
  if (pname || page) nameEl.textContent = `Patient: ${pname}${pname && page ? ' · ' : ''}${page}`;
  else nameEl.textContent = '';

  // Stage info
  const badge = document.getElementById('resultBadge');
  badge.textContent = res.label;
  badge.style.cssText = `background:${res.color}22;color:${res.color};border:1px solid ${res.color}55`;
  document.getElementById('resultStage').textContent = res.label;
  document.getElementById('resultStage').style.color = res.color;
  document.getElementById('resultDesc').textContent = res.description;
  const ub = document.getElementById('urgencyBox');
  ub.textContent = res.urgency;
  ub.style.color = res.urgency_color + 'dd';
  ub.style.borderColor = res.urgency_color + '55';
  ub.style.background = res.urgency_color + '15';

  // Gauge first
  buildGaugeChart(res.stage, res.confidence, res.label, res.color);

  // Confidence bar
  animateCounter('confPct', 0, res.confidence, 1000, '%');
  setTimeout(() => { document.getElementById('confBarFill').style.width = res.confidence + '%'; }, 150);

  // Recommendations
  const list = document.getElementById('recoList');
  const recoText = res.recommendations.join('. ');
  list.innerHTML = `
    <li style="text-align:center; margin-bottom:8px">
      <button class="face-btn primary" onclick="speakText('${recoText}', 'en-US')" style="font-size:0.75rem">🔊 Hear All Recommendations</button>
    </li>
  ` + res.recommendations.map((r, i) =>
    `<li style="animation-delay:${i * 0.07}s">${r.replace(/^[🚨⚠️]\s*/, '')}</li>`
  ).join('');
  if (res.stage === 3) {
    const first = list.querySelector('li:not(:first-child)');
    if (first) first.style.cssText = 'background:rgba(239,68,68,0.12);border-left-color:#ef4444;color:#fca5a5;font-weight:700';
  }

  buildDonutChart(res.all_probabilities);
  buildRadarChart(res.radar);

  // Store result for copy function
  window._lastResult = res;
  showToast('✅ Analysis complete', 'success');
  // Load stage-specific food suggestions
  loadFoodSuggestions(res.stage);
}

/* ─────────────────────────────────────────
   PRINT / EXPORT
───────────────────────────────────────── */
function printResult() {
  window.print();
}

/* ─────────────────────────────────────────
   COPY SUMMARY
───────────────────────────────────────── */
function copyResult() {
  const res = window._lastResult;
  if (!res) { showToast('No result to copy', 'error'); return; }
  const pname = document.getElementById('patientName').value.trim();
  const lines = [
    `Pulse Guard AI — Hypertension Assessment`,
    pname ? `Patient: ${pname}` : '',
    `Date: ${new Date().toLocaleString()}`,
    ``,
    `Result: ${res.label}`,
    `Confidence: ${res.confidence}%`,
    ``,
    `Stage Probabilities:`,
    `  Normal:  ${res.all_probabilities['Normal']}%`,
    `  Stage-1: ${res.all_probabilities['Stage-1']}%`,
    `  Stage-2: ${res.all_probabilities['Stage-2']}%`,
    `  Crisis:  ${res.all_probabilities['Crisis']}%`,
    ``,
    `Urgency: ${res.urgency}`,
    ``,
    `Recommendations:`,
    ...res.recommendations.map(r => `  • ${r.replace(/^[🚨⚠️]\s*/, '')}`),
    ``,
    `⚠️ For clinical support only. Always consult a healthcare provider.`,
  ].filter(l => l !== '' || l === '').join('\n');

  navigator.clipboard.writeText(lines).then(() => {
    document.getElementById('copyBtn').textContent = '✅ Copied!';
    setTimeout(() => { document.getElementById('copyBtn').textContent = '📋 Copy Summary'; }, 2000);
    showToast('📋 Summary copied to clipboard', 'success');
  }).catch(() => showToast('Copy failed — try another browser', 'error'));
}

/* ─────────────────────────────────────────
   PREDICTION HISTORY (sessionStorage)
───────────────────────────────────────── */
function addToHistory(res) {
  const pname = document.getElementById('patientName').value.trim();
  const entry = {
    label: res.label, color: res.color, confidence: res.confidence,
    patient: pname || 'Anonymous',
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  };
  let hist = JSON.parse(sessionStorage.getItem('bp_history') || '[]');
  hist.unshift(entry);
  if (hist.length > 8) hist = hist.slice(0, 8);
  sessionStorage.setItem('bp_history', JSON.stringify(hist));
  renderHistory();
}

function renderHistory() {
  const hist = JSON.parse(sessionStorage.getItem('bp_history') || '[]');
  const grid = document.getElementById('historyGrid');
  const count = document.getElementById('histCount');
  count.textContent = hist.length;
  if (!hist.length) {
    grid.innerHTML = '<div class="hist-empty">No predictions yet — run an analysis above.</div>';
    return;
  }
  grid.innerHTML = hist.map((h, i) => `
    <div class="hist-card" style="border-left:3px solid ${h.color};animation-delay:${i * 0.05}s">
      <div class="hist-stage" style="color:${h.color}">${h.label}</div>
      <div class="hist-meta">Confidence: <strong>${h.confidence}%</strong></div>
      <div class="hist-meta">Patient: ${h.patient}</div>
      <div class="hist-time">🕑 ${h.time}</div>
    </div>`).join('');
}

function clearHistory() {
  sessionStorage.removeItem('bp_history');
  renderHistory();
  showToast('History cleared', 'success');
}

/* ─────────────────────────────────────────
   GAUGE / SPEEDOMETER (Canvas 2D)
───────────────────────────────────────── */
function buildGaugeChart(stage, confidence, label, color) {
  const canvas = document.getElementById('gaugeCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H - 28;
  const R = Math.min(W, H * 1.9) * 0.42;
  const r2 = R * 0.62;
  const ZONES = [
    [180, 225, '#22c55e', 'Normal'],
    [225, 270, '#f59e0b', 'Stage-1'],
    [270, 315, '#ef4444', 'Stage-2'],
    [315, 360, '#7f1d1d', 'Crisis'],
  ];
  const targetAngleDeg = ZONES[stage][0] + 22.5;

  function draw(currentAngleDeg) {
    ctx.clearRect(0, 0, W, H);
    // Background arc
    ctx.beginPath(); ctx.arc(cx, cy, R, Math.PI, 0); ctx.arc(cx, cy, r2, 0, Math.PI, true); ctx.closePath();
    ctx.fillStyle = 'rgba(17,24,39,0.85)'; ctx.fill();
    // Zone arcs
    ZONES.forEach(([s, e, col]) => {
      const sa = (s * Math.PI) / 180, ea = (e * Math.PI) / 180;
      ctx.beginPath(); ctx.arc(cx, cy, R, sa, ea); ctx.arc(cx, cy, r2, ea, sa, true); ctx.closePath();
      ctx.fillStyle = col + 'cc'; ctx.fill();
      ctx.beginPath(); ctx.arc(cx, cy, (R + r2) / 2, sa, ea);
      ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke();
    });
    // Active zone glow
    const [as, ae, ac] = ZONES[stage];
    const asr = (as * Math.PI) / 180, aer = (ae * Math.PI) / 180;
    ctx.beginPath(); ctx.arc(cx, cy, R + 5, asr, aer); ctx.arc(cx, cy, r2 - 5, aer, asr, true); ctx.closePath();
    ctx.fillStyle = ac + '44'; ctx.fill();
    ctx.beginPath(); ctx.arc(cx, cy, R + 6, asr, aer);
    ctx.strokeStyle = ac; ctx.lineWidth = 3; ctx.shadowColor = ac; ctx.shadowBlur = 14; ctx.stroke(); ctx.shadowBlur = 0;
    // Tick marks
    for (let i = 0; i <= 8; i++) {
      const td = (180 + i * 22.5), tr = (td * Math.PI) / 180, major = i % 2 === 0;
      const ir = major ? R * 0.56 : R * 0.70;
      ctx.beginPath(); ctx.moveTo(cx + (R + 2) * Math.cos(tr), cy + (R + 2) * Math.sin(tr)); ctx.lineTo(cx + ir * Math.cos(tr), cy + ir * Math.sin(tr));
      ctx.strokeStyle = major ? '#ffffff88' : '#ffffff33'; ctx.lineWidth = major ? 2.5 : 1.5; ctx.stroke();
    }
    // Zone text
    ctx.font = `600 ${R * 0.09}px Inter,sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ZONES.forEach(([s, e, col, lbl]) => {
      const mr = ((s + e) / 2 * Math.PI) / 180, lr = (R + r2) / 2;
      ctx.fillStyle = '#ffffffcc'; ctx.fillText(lbl, cx + lr * Math.cos(mr), cy + lr * Math.sin(mr));
    });
    // Hub
    const hubR = r2 * 0.25;
    ctx.beginPath(); ctx.arc(cx, cy, hubR, 0, Math.PI * 2);
    ctx.fillStyle = '#1a2235'; ctx.fill(); ctx.strokeStyle = '#3b82f655'; ctx.lineWidth = 2; ctx.stroke();
    // Needle
    const nr = (currentAngleDeg * Math.PI) / 180, nl = R * 0.88, tl = hubR * 1.1;
    ctx.shadowColor = color; ctx.shadowBlur = 16;
    ctx.beginPath(); ctx.moveTo(cx - tl * Math.cos(nr), cy - tl * Math.sin(nr)); ctx.lineTo(cx + nl * Math.cos(nr), cy + nl * Math.sin(nr));
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.lineCap = 'round'; ctx.stroke(); ctx.shadowBlur = 0;
    ctx.beginPath(); ctx.arc(cx + nl * Math.cos(nr), cy + nl * Math.sin(nr), 4, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
    // Hub dot
    ctx.beginPath(); ctx.arc(cx, cy, hubR * 0.65, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 10; ctx.fill(); ctx.shadowBlur = 0;
  }

  if (gaugeAnimId) cancelAnimationFrame(gaugeAnimId);
  const start = performance.now(), dur = 1300;
  function animate(now) {
    const t = Math.min((now - start) / dur, 1);
    const ease = t === 1 ? 1 : 1 - Math.pow(2, -10 * t) * Math.cos((t * 10 - 0.75) * (2 * Math.PI) / 3);
    draw(180 + (targetAngleDeg - 180) * ease);
    if (t < 1) gaugeAnimId = requestAnimationFrame(animate);
  }
  gaugeAnimId = requestAnimationFrame(animate);
  document.getElementById('gaugeStageName').textContent = label;
  document.getElementById('gaugeStageName').style.color = color;
  document.getElementById('gaugeConf').textContent = `Confidence: ${confidence}%`;
}

/* ─────────────────────────────────────────
   DONUT CHART
───────────────────────────────────────── */
function buildDonutChart(probs) {
  const ctx = document.getElementById('donutChart').getContext('2d');
  if (donutChart) donutChart.destroy();
  const labels = ['Normal', 'Stage-1', 'Stage-2', 'Crisis'];
  const values = [probs['Normal'], probs['Stage-1'], probs['Stage-2'], probs['Crisis']];
  const colors = ['#22c55e', '#f59e0b', '#ef4444', '#7f1d1d'];
  donutChart = new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data: values, backgroundColor: colors.map(c => c + 'cc'), borderColor: colors, borderWidth: 2, hoverOffset: 8 }] },
    options: {
      responsive: true, cutout: '68%',
      animation: { animateRotate: true, duration: 900, easing: 'easeInOutCubic' },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => ` ${c.label}: ${c.parsed}%` }, backgroundColor: '#1a2235', borderColor: '#1e3a5f', borderWidth: 1, titleColor: '#e2e8f0', bodyColor: '#94a3b8' }
      }
    }
  });
  document.getElementById('donutLegend').innerHTML = labels.map((l, i) =>
    `<div class="donut-legend-item"><div class="donut-dot" style="background:${colors[i]}"></div><span>${l}</span><strong style="margin-left:auto;color:${colors[i]}">${values[i]}%</strong></div>`
  ).join('');
}

/* ─────────────────────────────────────────
   RADAR CHART
───────────────────────────────────────── */
function buildRadarChart(radar) {
  const ctx = document.getElementById('radarChart').getContext('2d');
  if (radarChart) radarChart.destroy();
  const idx = [1, 2, 3, 4, 6, 10, 11, 13, 14, 15, 16, 17];
  radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: idx.map(i => radar.labels[i]),
      datasets: [
        { label: 'Patient Risk', data: idx.map(i => radar.values[i]), backgroundColor: 'rgba(239,68,68,0.15)', borderColor: '#ef4444', borderWidth: 2, pointBackgroundColor: '#ef4444', pointRadius: 4, fill: true },
        { label: 'Healthy Baseline', data: idx.map(() => 0.2), backgroundColor: 'rgba(34,197,94,0.08)', borderColor: '#22c55e', borderWidth: 1.5, borderDash: [5, 4], pointBackgroundColor: '#22c55e', pointRadius: 3, fill: true }
      ]
    },
    options: {
      responsive: true, animation: { duration: 900 },
      scales: { r: { min: 0, max: 1, ticks: { display: false }, grid: { color: 'rgba(255,255,255,0.07)' }, angleLines: { color: 'rgba(255,255,255,0.07)' }, pointLabels: { color: '#94a3b8', font: { size: 10, family: 'Inter' } } } },
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 }, boxWidth: 12 } },
        tooltip: { backgroundColor: '#1a2235', borderColor: '#1e3a5f', borderWidth: 1, titleColor: '#e2e8f0', bodyColor: '#94a3b8', callbacks: { label: c => ` ${c.dataset.label}: ${(c.parsed.r * 100).toFixed(0)}%` } }
      }
    }
  });
}

/* ─────────────────────────────────────────
   MODEL BAR CHART
───────────────────────────────────────── */
async function loadModelBarChart() {
  try {
    const data = await (await fetch('/model_stats')).json();
    const names = Object.keys(data.models);
    const colors = names.map(n => n === data.best ? '#3b82f6' : 'rgba(96,165,250,0.35)');
    const ctx = document.getElementById('modelBarChart').getContext('2d');
    if (modelBarChart) modelBarChart.destroy();
    modelBarChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: names,
        datasets: [
          { label: 'CV Accuracy (%)', data: names.map(n => data.models[n].cv_acc), backgroundColor: colors, borderColor: colors.map(c => c.replace('0.35', '0.8')), borderWidth: 1.5, borderRadius: 6 },
          { label: 'Test Accuracy (%)', data: names.map(n => data.models[n].test_acc), backgroundColor: 'rgba(6,182,212,0.3)', borderColor: '#06b6d4', borderWidth: 1.5, borderRadius: 6 }
        ]
      },
      options: {
        responsive: true, animation: { duration: 1000, easing: 'easeInOutCubic' },
        scales: {
          x: { ticks: { color: '#94a3b8', font: { size: 11, family: 'Inter' } }, grid: { color: 'rgba(255,255,255,0.04)' } },
          y: { min: 50, max: 105, ticks: { color: '#94a3b8', callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,0.06)' } }
        },
        plugins: {
          legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 }, boxWidth: 12 } },
          tooltip: { backgroundColor: '#1a2235', borderColor: '#1e3a5f', borderWidth: 1, titleColor: '#e2e8f0', bodyColor: '#94a3b8', callbacks: { label: c => ` ${c.dataset.label}: ${c.parsed.y}%` } }
        }
      }
    });
  } catch (e) { console.warn('Model stats:', e.message); }
}

/* ─────────────────────────────────────────
   ANIMATED COUNTER
───────────────────────────────────────── */
function animateCounter(id, from, to, dur, sfx = '') {
  const el = document.getElementById(id), t0 = performance.now();
  (function step(now) {
    const t = Math.min((now - t0) / dur, 1), e = 1 - Math.pow(1 - t, 3);
    el.textContent = (from + (to - from) * e).toFixed(1) + sfx;
    if (t < 1) requestAnimationFrame(step);
  })(t0);
}

/* ─────────────────────────────────────────
   RESET FORM
───────────────────────────────────────── */
function resetForm() {
  ['age_group', 'systolic_bp', 'diastolic_bp'].forEach(id => { document.getElementById(id).value = ''; });
  document.getElementById('time_since_diagnosis').value = '0';
  document.getElementById('exercise_frequency').value = '';
  document.querySelectorAll('.seg-control').forEach(seg => {
    seg.querySelectorAll('.seg-btn').forEach((b, i) => b.classList.toggle('active', i === 0));
  });
  const defs = { gender: '0', family_history: '0', patient_status: '0', take_medication: '0', symptom_severity: '0', shortness_of_breath: '0', visual_changes: '0', nosebleeds: '0', controlled_diet: '0', bmi_category: '0', diabetes: '0', cholesterol_level: '0', heart_rate_category: '1' };
  Object.entries(defs).forEach(([id, v]) => { const el = document.getElementById(id); if (el) el.value = v; });
  document.getElementById('hrSeg').querySelectorAll('.seg-btn').forEach((b, i) => b.classList.toggle('active', i === 1));
  document.getElementById('resultPanel').style.display = 'none';
  document.getElementById('bpIndicator').style.display = 'none';
  document.getElementById('confBarFill').style.width = '0%';
  if (gaugeAnimId) { cancelAnimationFrame(gaugeAnimId); gaugeAnimId = null; }
  if (donutChart) { donutChart.destroy(); donutChart = null; }
  if (radarChart) { radarChart.destroy(); radarChart = null; }
  const gc = document.getElementById('gaugeCanvas');
  gc.getContext('2d').clearRect(0, 0, gc.width, gc.height);
  document.getElementById('gaugeStageName').textContent = '';
  document.getElementById('gaugeConf').textContent = '';
  updateProgress();
  showToast('✅ Form cleared', 'success');
}

/* ─────────────────────────────────────────
   TOAST
───────────────────────────────────────── */
function showToast(msg, type = '') {
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = 'toast show' + (type ? ' ' + type : '');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.className = 'toast'; }, 3200);
}

/* ─────────────────────────────────────────
   INIT
───────────────────────────────────────── */
window.addEventListener('load', () => {
  applyTheme();
  applyFontSize();
  updateProgress();
  renderHistory();
  loadModelBarChart();
  loadTodayLog();
  // Keyboard shortcut: Ctrl+Enter to predict
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      runPrediction();
    }
  });
});

/* ─────────────────────────────────────────
   FOOD TRACKER — TOGGLE PANEL
───────────────────────────────────────── */
function toggleFoodTracker() {
  const panel = document.getElementById('ftPanel');
  const chevron = document.getElementById('ftChevron');
  const isOpen = panel.classList.contains('open');
  panel.classList.toggle('open', !isOpen);
  chevron.classList.toggle('open', !isOpen);
  if (!isOpen) {
    // Load fresh data when opening
    loadTodayLog();
    setTimeout(() =>
      panel.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120);
  }
}

function openFoodTracker() {
  const panel = document.getElementById('ftPanel');
  const chevron = document.getElementById('ftChevron');
  if (!panel || !chevron) return;
  if (!panel.classList.contains('open')) {
    panel.classList.add('open');
    chevron.classList.add('open');
    loadTodayLog();
  }
}

/* ─────────────────────────────────────────
   FOOD TRACKER — LOG MEAL
───────────────────────────────────────── */
async function logMeal() {
  const name = document.getElementById('mealName').value.trim();
  if (!name) { showToast('⚠️ Please enter a meal name', 'error'); return; }

  const btn = document.getElementById('mealLogBtn');
  const txt = document.getElementById('mealBtnText');
  const ldr = document.getElementById('mealBtnLoader');
  btn.disabled = true;
  txt.style.display = 'none';
  ldr.style.display = 'flex';
  ldr.innerHTML = '<span class="spinner"></span> Gemini is rating…';

  try {
    const payload = {
      meal_name: name,
      category: document.getElementById('mealCategory').value,
      sodium: document.getElementById('mealSodium').value,
      portion: document.getElementById('mealPortion').value,
      stage: window._lastResult ? window._lastResult.stage : 0,
    };
    const res = await fetch('/food-log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const json = await res.json();
    if (json.error) throw new Error(json.error);
    document.getElementById('mealName').value = '';

    // Show Gemini rating toast
    const icons = { Excellent: '✅', Good: '👍', Caution: '⚠️', Dangerous: '🚨' };
    const icon = icons[json.rating] || '📋';
    const ttype = json.rating === 'Dangerous' ? 'error' : json.rating === 'Excellent' ? 'success' : '';
    showToast(`${icon} ${json.rating || 'Meal logged'}: ${json.reason || ''}`, ttype);

    // Refresh the active tab
    const activeTabBtn = document.querySelector('.food-tab-btn.active');
    if (activeTabBtn) {
      const tabText = activeTabBtn.textContent.toLowerCase();
      if (tabText.includes('day')) loadDayAnalysis();
      else if (tabText.includes('week')) loadWeekAnalysis();
      else loadTodayLog();
    } else {
      loadTodayLog();
    }
  } catch (e) {
    showToast('❌ Failed to log meal: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    txt.style.display = '';
    ldr.style.display = 'none';
    ldr.innerHTML = '<span class="spinner"></span>';
  }
}

/* ─────────────────────────────────────────
   FOOD TRACKER — LOAD TODAY'S LOG
───────────────────────────────────────── */
const RATING_STYLE = {
  Excellent: { icon: '✅', color: '#22c55e' },
  Good: { icon: '👍', color: '#84cc16' },
  Caution: { icon: '⚠️', color: '#f59e0b' },
  Dangerous: { icon: '🚨', color: '#ef4444' },
};

async function loadTodayLog() {
  const wrap = document.getElementById('foodLogTable');
  try {
    const rows = await (await fetch('/food-log')).json();
    const badge = document.getElementById('mealCountBadge');
    badge.textContent = rows.length + ' today';

    if (!rows.length) {
      wrap.innerHTML = '<div class="hist-empty">No meals logged today — use the form above to start tracking.</div>';
      return;
    }
    wrap.innerHTML = `
      <table class="food-log-table">
        <thead>
          <tr>
            <th>Time</th><th>Meal</th><th>Category</th><th>Sodium</th><th>Portion</th><th>AI Rating</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(r => {
      const rs = RATING_STYLE[r.rating] || { icon: '📋', color: 'var(--muted2)' };
      const ratingHtml = r.rating
        ? `<span class="rating-chip" title="${r.rating_reason || ''}" style="background:${rs.color}22;color:${rs.color};border:1px solid ${rs.color}55">${rs.icon} ${r.rating}</span>`
        : `<span style="color:var(--muted)">—</span>`;
      return `
            <tr>
              <td style="color:var(--muted2);white-space:nowrap">🕑 ${r.logged_at}</td>
              <td><strong>${r.meal_name}</strong>${r.rating_reason ? `<br><small style="color:var(--muted);font-size:0.7rem">${r.rating_reason}</small>` : ''}</td>
              <td><span class="category-chip">${r.category}</span></td>
              <td><span class="sodium-chip ${r.sodium}">${r.sodium}</span></td>
              <td style="color:var(--muted2)">${r.portion}</td>
              <td>${ratingHtml}</td>
            </tr>`;
    }).join('')}
        </tbody>
      </table>`;
  } catch (e) {
    wrap.innerHTML = '<div class="hist-empty">Could not load food log.</div>';
  }
}

/* ─────────────────────────────────────────
   FOOD TRACKER — STAGE SUGGESTIONS
───────────────────────────────────────── */
async function loadFoodSuggestions(stage) {
  const panel = document.getElementById('foodSuggestPanel');
  const eatList = document.getElementById('suggEatList');
  const avList = document.getElementById('suggAvoidList');
  const tip = document.getElementById('suggTip');

  // Open the food tracker panel (in case user closed it)
  openFoodTracker();

  // Show suggestions panel with loading state
  panel.style.display = 'block';
  eatList.innerHTML = '<li style="color:var(--muted2)"><span class="spinner" style="display:inline-block;width:13px;height:13px;border-width:2px;vertical-align:middle;margin-right:6px"></span>Gemini AI is generating personalised suggestions…</li>';
  avList.innerHTML = '';
  tip.textContent = '';

  // Update badge safely (may have been re-created by last call)
  const getBadge = () => document.getElementById('sugg-stage-badge');
  if (getBadge()) {
    getBadge().textContent = 'Generating…';
    getBadge().style.backgroundColor = 'rgba(59,130,246,0.18)';
    getBadge().style.color = '#60a5fa';
  }

  // Collect extra context from the form for richer personalisation
  const diabetes = document.getElementById('diabetes')?.value || '0';
  const bmi = document.getElementById('bmi_category')?.value || '1';

  try {
    const url = `/food-suggestions?stage=${stage}&diabetes=${diabetes}&bmi=${bmi}`;
    const data = await (await fetch(url)).json();

    tip.textContent = data.tip;

    eatList.innerHTML = data.eat.map((item, i) =>
      `<li style="animation-delay:${i * 0.06}s">${item}</li>`).join('');
    avList.innerHTML = data.avoid.map((item, i) =>
      `<li style="animation-delay:${i * 0.06}s">${item}</li>`).join('');

    // Show source badge
    const srcBadge = data.source === 'gemini'
      ? `<span style="font-size:0.65rem;padding:2px 8px;border-radius:10px;background:rgba(59,130,246,0.15);color:#60a5fa;margin-left:8px;font-weight:600">✨ Gemini AI</span>`
      : `<span title="${data.error_msg || 'Using curated suggestions'}" style="font-size:0.65rem;padding:2px 8px;border-radius:10px;background:var(--surface3);color:var(--muted2);margin-left:8px;cursor:help">📋 Curated${data.error_msg ? ' ⓘ' : ''}</span>`;

    document.querySelector('#foodSuggestPanel .chart-title').innerHTML =
      `🎯 Personalized Food Suggestions ${srcBadge} <span id="sugg-stage-badge" style="background-color:${data.color}33;color:${data.color}">${data.stage}</span>`;

    // Scroll to show the suggestions after data has rendered
    setTimeout(() =>
      panel.scrollIntoView({ behavior: 'smooth', block: 'start' }), 200);

  } catch (e) {
    eatList.innerHTML = '<li style="color:var(--muted)">Could not load suggestions. Please try again.</li>';
    console.warn('Food suggestions error:', e.message);
  }
}

/* ─────────────────────────────────────────
   FOOD TRACKER — ANALYSIS TABS
   ───────────────────────────────────────── */
function showFoodTab(tab) {
  // Update buttons
  document.querySelectorAll('.food-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('onclick').includes(tab));
  });
  // Update panels
  const panels = {
    'log': 'foodTabLog',
    'day': 'foodTabDay',
    'week': 'foodTabWeek'
  };
  Object.values(panels).forEach(id => {
    document.getElementById(id).classList.remove('active');
  });
  const activeId = panels[tab];
  document.getElementById(activeId).classList.add('active');

  // Load data
  if (tab === 'day') loadDayAnalysis();
  if (tab === 'week') loadWeekAnalysis();
  if (tab === 'log') loadTodayLog();
}

async function loadDayAnalysis() {
  const stage = window._lastPredStage !== undefined ? window._lastPredStage : 0;
  try {
    const res = await fetch(`/food-analysis/day?stage=${stage}`);
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);

    // 1. Health Score Ring
    renderHealthRing(data.health_score);
    document.getElementById('dayHealthScore').textContent = data.health_score;

    // Status Text
    const advice = document.getElementById('dayHealthAdvice');
    if (data.meal_count === 0) {
      advice.textContent = "No meals logged today yet.";
    } else if (data.health_score > 80) {
      advice.innerHTML = "<span class='text-success'>Excellent!</span> Your meals perfectly align with heart-healthy goals.";
    } else if (data.health_score > 60) {
      advice.innerHTML = "<span class='text-warning'>Good job.</span> Try reduced sodium for a better score.";
    } else {
      advice.innerHTML = "<span class='text-danger'>Correction needed.</span> High sodium or risky meals detected.";
    }

    // 2. Prediction Context Badge
    const badge = document.getElementById('predCorrelationBadge');
    if (window._lastPredStage === undefined) {
      badge.textContent = "Run Prediction first";
      badge.className = "pred-corr-badge";
    } else {
      const match = (data.health_score > 60);
      badge.textContent = match ? "✅ Aligning with stage needs" : "⚠️ High dietary risk for your stage";
      badge.style.color = match ? "var(--success)" : "var(--danger)";
      badge.style.borderColor = match ? "var(--success)" : "var(--danger)";
    }

    // 3. Dataset Benchmark
    const benchmark = document.getElementById('datasetBenchmarkInfo');
    benchmark.innerHTML = `Patients in <strong>${data.stage_context}</strong> stage have an average health-rate of <strong>${data.baseline_controlled_diet}%</strong>. You are currently at <strong>${data.health_score}%</strong>.`;

    // 4. Charts
    renderDaySodiumChart(data.sodium_dist);
    renderDayRatingChart(data.rating_dist);
    renderDayCategoryChart(data.category_dist);

  } catch (e) {
    console.warn("Day analysis error:", e);
    showToast("Could not load day analysis", "error");
  }
}

function renderHealthRing(score) {
  const canvas = document.getElementById('healthRingCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) * 0.4;

  ctx.clearRect(0, 0, W, H);

  // Outer glass track
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 12;
  ctx.stroke();

  // Progress Arc
  const angle = (score / 100) * Math.PI * 2;
  const grd = ctx.createLinearGradient(0, 0, W, 0);
  grd.addColorStop(0, '#3b82f6');
  grd.addColorStop(1, '#06b6d4');

  ctx.beginPath();
  ctx.arc(cx, cy, R, -Math.PI / 2, angle - Math.PI / 2);
  ctx.strokeStyle = grd;
  ctx.lineWidth = 12;
  ctx.lineCap = 'round';
  ctx.shadowColor = score > 50 ? 'rgba(59, 130, 246, 0.4)' : 'rgba(239, 68, 68, 0.4)';
  ctx.shadowBlur = 10;
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function renderDaySodiumChart(dist) {
  const ctx = document.getElementById('daySodiumChart').getContext('2d');
  if (daySodiumChart) daySodiumChart.destroy();
  daySodiumChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Low', 'Med', 'High'],
      datasets: [{
        data: [dist.Low, dist.Medium, dist.High],
        backgroundColor: ['rgba(34, 197, 94, 0.4)', 'rgba(245, 158, 11, 0.4)', 'rgba(239, 68, 68, 0.4)'],
        borderColor: ['#22c55e', '#f59e0b', '#ef4444'],
        borderWidth: 1.5,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false, grid: { display: false } },
        y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } }
      }
    }
  });
}

function renderDayRatingChart(dist) {
  const ctx = document.getElementById('dayRatingChart').getContext('2d');
  if (dayRatingChart) dayRatingChart.destroy();
  dayRatingChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Exc', 'Good', 'Caut', 'Dang'],
      datasets: [{
        data: [dist.Excellent, dist.Good, dist.Caution, dist.Dangerous],
        backgroundColor: ['#22c55e', '#84cc16', '#f59e0b', '#ef4444'],
        borderWidth: 0,
        hoverOffset: 4
      }]
    },
    options: {
      cutout: '65%',
      plugins: { legend: { display: false } }
    }
  });
}

function renderDayCategoryChart(dist) {
  const ctx = document.getElementById('dayCategoryChart').getContext('2d');
  if (dayCategoryChart) dayCategoryChart.destroy();
  const labels = Object.keys(dist);
  const values = Object.values(dist);
  dayCategoryChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: 'rgba(59, 130, 246, 0.3)',
        borderColor: 'var(--accent)',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, ticks: { display: false }, grid: { display: false } },
        x: { ticks: { color: '#94a3b8', font: { size: 9 } }, grid: { display: false } }
      }
    }
  });
}

async function loadWeekAnalysis() {
  try {
    const res = await fetch('/food-analysis/week');
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);

    // 1. Line Chart
    renderWeekHealthChart(data.labels, data.health_scores);

    // 2. Sodium Bar Chart
    renderWeekSodiumChart(data.labels, data.sodium_high_counts);

    // 3. Dataset Insights
    loadDatasetInsights();

  } catch (e) {
    console.warn("Week analysis error:", e);
  }
}

function renderWeekHealthChart(labels, scores) {
  const ctx = document.getElementById('weekHealthChart').getContext('2d');
  if (weekHealthChart) weekHealthChart.destroy();
  weekHealthChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Health Rate',
        data: scores,
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 3,
        pointBackgroundColor: '#fff',
        pointRadius: 4,
        tension: 0.4,
        fill: true
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 0, max: 100, ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
        x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { display: false } }
      }
    }
  });
}

function renderWeekSodiumChart(labels, highCounts) {
  const ctx = document.getElementById('weekSodiumChart').getContext('2d');
  if (weekSodiumChart) weekSodiumChart.destroy();
  weekSodiumChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'High Sodium Meals',
        data: highCounts,
        backgroundColor: highCounts.map(c => c > 0 ? 'rgba(239, 68, 68, 0.4)' : 'rgba(59, 130, 246, 0.1)'),
        borderColor: highCounts.map(c => c > 0 ? '#ef4444' : 'var(--border)'),
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, ticks: { stepSize: 1, color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
        x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { display: false } }
      }
    }
  });
}

async function loadDatasetInsights() {
  const wrap = document.getElementById('weekDatasetInsights');
  try {
    const res = await fetch('/food-analysis/dataset-insights');
    const data = await res.json();
    if (!data.ok) return;

    const stages = ['Normal', 'Stage-1', 'Stage-2', 'Crisis'];
    const colors = ['#22c55e', '#f59e0b', '#ef4444', '#7f1d1d'];

    wrap.innerHTML = stages.map((name, i) => {
      const stats = data.stage_insights[i];
      return `
        <div class="insight-bar-row">
          <div class="insight-bar-label">
            <span>${name}</span>
            <strong>${stats.controlled_diet_pct}% Controlled Diet</strong>
          </div>
          <div class="insight-bar-bg">
            <div class="insight-bar-fill" style="width:${stats.controlled_diet_pct}%; background:${colors[i]}"></div>
          </div>
        </div>
      `;
    }).join('');

  } catch (e) {
    wrap.innerHTML = '<div style="font-size:0.75rem;color:var(--muted)">Benchmark data unavailable.</div>';
  }
}

/* ─────────────────────────────────────────
   GOOGLE TRANSLATE — LANGUAGE PICKER
───────────────────────────────────────── */
// Map of lang code → display label for the trigger button
const LANG_LABELS = {
  en: 'EN', ta: 'தமி', hi: 'हिन्', te: 'తెలు', ml: 'മലയ', kn: 'ಕನ್ನ',
  bn: 'বাং', mr: 'मरा', gu: 'ગુજ', pa: 'ਪੰਜ', ur: 'اردو',
  fr: 'FR', es: 'ES', ar: 'عرب', 'zh-CN': '中文', de: 'DE',
  ja: '日本', pt: 'PT', ru: 'RU', ko: '한국', id: 'ID',
};

function toggleLangDropdown(event) {
  if (event) event.stopPropagation();
  const dropdown = document.getElementById('langDropdown');
  const trigger = document.getElementById('langTriggerBtn');
  const chevron = trigger?.querySelector('.lang-arrow');
  const isOpen = dropdown.style.display === 'block';

  if (!isOpen) {
    const rect = trigger.getBoundingClientRect();
    dropdown.style.top = (rect.bottom + 6) + 'px';
    dropdown.style.left = Math.max(4, rect.right - 230) + 'px';
    dropdown.style.display = 'block';
    if (chevron) chevron.style.transform = 'rotate(180deg)';
  } else {
    dropdown.style.display = 'none';
    if (chevron) chevron.style.transform = '';
  }
}

// Close dropdown when clicking outside
document.addEventListener('click', (e) => {
  const dropdown = document.getElementById('langDropdown');
  const trigger = document.getElementById('langTriggerBtn');
  if (dropdown && trigger &&
    !dropdown.contains(e.target) &&
    !trigger.contains(e.target)) {
    dropdown.style.display = 'none';
    const chv = trigger?.querySelector('.lang-arrow');
    if (chv) chv.style.transform = '';
  }
});

// Called by Google Translate script on load
function googleTranslateElementInit() {
  new google.translate.TranslateElement({
    pageLanguage: 'en',
    autoDisplay: false,
  }, 'google_translate_element');
}

function setLang(langCode) {
  // Update button label
  const label = document.getElementById('langCurrent');
  if (label) label.textContent = LANG_LABELS[langCode] || langCode.toUpperCase();

  // Mark active button
  document.querySelectorAll('.lang-dropdown button').forEach(btn => {
    btn.classList.toggle('active',
      btn.getAttribute('onclick') === `setLang('${langCode}')`);
  });

  // Close dropdown
  const dd = document.getElementById('langDropdown');
  const chv = document.querySelector('#langTriggerBtn .lang-arrow');
  if (dd) dd.style.display = 'none';
  if (chv) chv.style.transform = '';

  // First: try driving the Google Translate select (if widget loaded)
  const combo = document.querySelector('.goog-te-combo');
  if (combo) {
    combo.value = langCode;
    combo.dispatchEvent(new Event('change'));
    return;
  }

  // Fallback: set cookie and reload — Google Translate widget reads it on load
  if (langCode === 'en') {
    // Clear translation
    document.cookie = 'googtrans=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/';
    document.cookie = 'googtrans=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/; domain=' + location.hostname;
  } else {
    const exp = new Date(Date.now() + 365 * 864e5).toUTCString();
    document.cookie = `googtrans=/en/${langCode}; expires=${exp}; path=/`;
  }
  location.reload();
}


/* ─────────────────────────────────────────
   FACIAL HYPERTENSION SCREENING
───────────────────────────────────────── */
let _faceStream = null;
let _faceImageB64 = null;

const FACE_RISK_STYLE = {
  Low: { bg: 'rgba(34,197,94,0.15)', color: '#22c55e', label: '✅ Low Risk' },
  Moderate: { bg: 'rgba(245,158,11,0.15)', color: '#f59e0b', label: '⚠️ Moderate Risk' },
  High: { bg: 'rgba(239,68,68,0.15)', color: '#ef4444', label: '🔴 High Risk' },
  Critical: { bg: 'rgba(124,58,237,0.15)', color: '#7c3aed', label: '🚨 Critical Risk' },
};

async function openFaceModal() {
  const modal = document.getElementById('faceModal');
  modal.classList.add('open');
  document.body.style.overflow = 'hidden';

  _faceImageB64 = null;
  ['faceResult', 'faceCanvas', 'faceVideo', 'faceOverlayText', 'faceCaptureBtn', 'faceRetakeBtn', 'faceAnalyseBtn'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    if (id === 'faceResult' || id === 'faceCanvas' || id === 'faceRetakeBtn' || id === 'faceAnalyseBtn') {
      el.style.display = 'none';
    } else {
      el.style.display = 'block';
      if (id === 'faceCaptureBtn' || id === 'faceVideo') el.style.display = 'flex';
    }
  });

  const uploadInput = document.getElementById('faceUploadInput');
  if (uploadInput) uploadInput.value = '';

  try {
    _faceStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
    const video = document.getElementById('faceVideo');
    const overlay = document.getElementById('faceOverlayText');
    if (video) video.srcObject = _faceStream;
    if (overlay) overlay.textContent = '📷 Position your face in the frame';
  } catch (e) {
    const overlay = document.getElementById('faceOverlayText');
    const captureBtn = document.getElementById('faceCaptureBtn');
    if (overlay) overlay.textContent = '⚠️ Camera unavailable — upload an image instead';
    if (captureBtn) captureBtn.style.display = 'none';
  }
}

function closeFaceModal() {
  document.getElementById('faceModal').classList.remove('open');
  document.body.style.overflow = '';
  if (_faceStream) {
    _faceStream.getTracks().forEach(t => t.stop());
    _faceStream = null;
  }
}

function captureFace() {
  const video = document.getElementById('faceVideo');
  const canvas = document.getElementById('faceCanvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext('2d').drawImage(video, 0, 0);
  _faceImageB64 = canvas.toDataURL('image/jpeg', 0.85);

  const v = document.getElementById('faceVideo');
  const c = document.getElementById('faceCanvas');
  const o = document.getElementById('faceOverlayText');
  const cap = document.getElementById('faceCaptureBtn');
  const ret = document.getElementById('faceRetakeBtn');
  const ana = document.getElementById('faceAnalyseBtn');

  if (v) v.style.display = 'none';
  if (c) c.style.display = 'block';
  if (o) {
    o.style.display = 'none';
    o.textContent = '📷 Position your face in the frame'; // reset text
    o.style.color = ''; // reset color
  }
  if (cap) cap.style.display = 'none';
  if (ret) ret.style.display = 'flex';
  if (ana) ana.style.display = 'flex';
}

function loadUploadedFace(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    _faceImageB64 = e.target.result;
    const canvas = document.getElementById('faceCanvas');
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      canvas.getContext('2d').drawImage(img, 0, 0);
    };
    img.src = _faceImageB64;
    const v = document.getElementById('faceVideo');
    const c = document.getElementById('faceCanvas');
    const o = document.getElementById('faceOverlayText');
    const cap = document.getElementById('faceCaptureBtn');
    const ret = document.getElementById('faceRetakeBtn');
    const ana = document.getElementById('faceAnalyseBtn');

    if (v) v.style.display = 'none';
    if (c) c.style.display = 'block';
    if (o) {
      o.style.display = 'none';
      o.textContent = '📷 Position your face in the frame'; // reset text
      o.style.color = ''; // reset color
    }
    if (cap) cap.style.display = 'none';
    if (ret) ret.style.display = 'flex';
    if (ana) ana.style.display = 'flex';
  };
  reader.readAsDataURL(file);
}

function retakeFace() {
  _faceImageB64 = null;
  const ids = ['faceCanvas', 'faceResult', 'faceVideo', 'faceOverlayText', 'faceCaptureBtn', 'faceRetakeBtn', 'faceAnalyseBtn'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    if (id === 'faceCanvas' || id === 'faceResult' || id === 'faceRetakeBtn' || id === 'faceAnalyseBtn') {
      el.style.display = 'none';
    } else {
      el.style.display = 'block';
      if (id === 'faceCaptureBtn' || id === 'faceVideo') el.style.display = 'flex';
    }
  });
  const o = document.getElementById('faceOverlayText');
  if (o) {
    o.textContent = '📷 Position your face in the frame';
    o.style.color = ''; // reset color
  }
  const u = document.getElementById('faceUploadInput');
  if (u) u.value = '';
}

async function analyseFace() {
  console.log('analyseFace triggered');
  if (!_faceImageB64) {
    console.warn('No image data found');
    showToast('Capture or upload a photo first.', 'error');
    return;
  }

  console.log('Image data length:', _faceImageB64.length);
  showToast('🔍 Analysis started...', 'info');

  const btn = document.getElementById('faceAnalyseBtn');
  const btnText = document.getElementById('faceAnalyseBtnText');
  const btnLoader = document.getElementById('faceAnalyseLoader');

  if (btnText) btnText.style.display = 'none';
  if (btnLoader) btnLoader.style.display = 'flex';
  if (btn) btn.disabled = true;

  console.log('Sending request to /facial-predict...');

  try {
    const res = await fetch('/facial-predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: _faceImageB64 }),
    });
    const data = await res.json();
    console.log('Received response:', data);

    if (!data.ok) {
      const errorMsg = data.error || 'Analysis failed.';
      if (errorMsg.includes('Face not detected')) {
        // Show face not detected message in the overlay
        document.getElementById('faceOverlayText').textContent = '❌ Face not detected in facial recognition. Please ensure your face is clearly visible.';
        document.getElementById('faceOverlayText').style.color = '#ef4444'; // red color
        // Hide result if shown
        document.getElementById('faceResult').style.display = 'none';
        // Re-enable analyse button
        document.getElementById('faceAnalyseBtn').style.display = 'inline-block';
      } else {
        showToast(errorMsg, 'error');
      }
      return;
    }

    const risk = data.risk || 'Unknown';
    const style = FACE_RISK_STYLE[risk] || { bg: 'rgba(100,116,139,0.15)', color: '#64748b', label: risk };

    const badge = document.getElementById('faceRiskBadge');
    badge.textContent = style.label;
    badge.style.background = style.bg;
    badge.style.color = style.color;

    document.getElementById('faceConfidence').textContent =
      `Confidence: ${data.confidence || 'Moderate'}`;

    // Emotion
    const emotionIcons = {
      Neutral: '😐', Happy: '😊', Sad: '😢', Angry: '😠', Fearful: '😨',
      Anxious: '😰', Calm: '😌'
    };
    document.getElementById('faceEmotion').textContent = `${emotionIcons[data.emotion] || '👤'} ${data.emotion || 'Unknown'}`;
    document.getElementById('faceEmotionDetail').textContent = data.emotion_detail || '';

    // Stress
    const stressIcons = { Low: '🟢', Moderate: '🟡', High: '🟠', Severe: '🔴' };
    document.getElementById('faceStress').textContent = `${stressIcons[data.stress_level] || '🔘'} ${data.stress_level || 'Unknown'}`;
    const stressCues = Array.isArray(data.stress_cues) ? data.stress_cues.join(', ') : (data.stress_cues || '');
    document.getElementById('faceStressCues').textContent = stressCues;

    // BP Estimate
    if (data.bp_estimate && typeof data.bp_estimate === 'object') {
      document.getElementById('faceBPEstimate').textContent =
        `${data.bp_estimate.systolic || '--'} / ${data.bp_estimate.diastolic || '--'} mmHg`;
      document.getElementById('faceBPCategory').textContent = data.bp_estimate.category || '';
    } else {
      document.getElementById('faceBPEstimate').textContent = '-- / -- mmHg';
      document.getElementById('faceBPCategory').textContent = 'Estimate unavailable';
    }

    const signsList = document.getElementById('faceSignsList');
    signsList.innerHTML = (data.signs && data.signs.length)
      ? data.signs.map(s => `<li>${s}</li>`).join('')
      : '<li>No notable signs detected</li>';

    const adviceText = data.advice || '';
    document.getElementById('faceAdvice').innerHTML = `
      <button class="face-btn primary" onclick="speakText('${adviceText}', 'en-US')" style="float:right; padding:4px 8px; font-size:0.8rem">🔊</button>
      <strong>Advice:</strong> ${adviceText}
    `;
    document.getElementById('faceDisclaimer').textContent = data.disclaimer || '';
    document.getElementById('faceResult').style.display = 'block';
    document.getElementById('faceResult').scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Add close button
    document.getElementById('faceActions').innerHTML = `<button class="face-btn secondary" onclick="closeFaceModal()">❌ Close</button>`;

  } catch (e) {
    console.error('Facial analysis error:', e);
    showToast('Network error. Please try again.', 'error');
  } finally {
    if (btnText) btnText.style.display = 'inline';
    if (btnLoader) btnLoader.style.display = 'none';
    if (btn) btn.disabled = false;
  }
}

/* ─── WAV Recorder implementation ───────────────────────── */
function WavRecorder(source) {
  this.source = source;
  this.recording = false;
  this.buffers = [];
  this.init();
}

WavRecorder.prototype.init = function() {
  this.node = audioContext.createScriptProcessor(4096, 1, 1);
  this.node.onaudioprocess = (e) => {
    if (this.recording) {
      this.buffers.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    }
  };
  this.source.connect(this.node);
  this.node.connect(audioContext.destination);
};

WavRecorder.prototype.record = function() {
  this.recording = true;
  this.buffers = [];
};

WavRecorder.prototype.stop = function() {
  this.recording = false;
};

WavRecorder.prototype.exportWAV = function(callback) {
  const buffer = this.mergeBuffers(this.buffers, this.buffers.length);
  const wav = this.encodeWAV(buffer);
  const blob = new Blob([wav], { type: 'audio/wav' });
  callback(blob);
};

WavRecorder.prototype.mergeBuffers = function(buffers, length) {
  const result = new Float32Array(length * 4096);
  let offset = 0;
  for (let i = 0; i < length; i++) {
    result.set(buffers[i], offset);
    offset += buffers[i].length;
  }
  return result;
};

WavRecorder.prototype.encodeWAV = function(samples) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, audioContext.sampleRate, true);
  view.setUint32(28, audioContext.sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }
  return buffer;
};

/* ─── VOICE STRESS SCREENING ───────────────────────── */
let audioContext, recorder, recording = false;

function openVoiceModal() {
  const modal = document.getElementById('voiceModal');
  if (modal) modal.classList.add('open');
  document.body.style.overflow = 'hidden';
  resetVoiceUI();
}
function closeVoiceModal() {
  const modal = document.getElementById('voiceModal');
  if (modal) modal.classList.remove('open');
  document.body.style.overflow = '';
  stopVoiceRecording(); // ensure it's stopped
}

async function startVoiceRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const input = audioContext.createMediaStreamSource(stream);
    recorder = new WavRecorder(input);
    recorder.record();
    recording = true;

    document.getElementById('voiceRecordCard').classList.add('recording');
    document.getElementById('voiceStatus').textContent = 'Recording...';
    document.getElementById('startRecordingBtn').style.display = 'none';
    document.getElementById('stopRecordingBtn').style.display = 'inline-block';

    // Auto stop after 6 seconds
    setTimeout(() => {
      if (recording) {
        stopVoiceRecording();
      }
    }, 6000);

  } catch (err) {
    console.error('Mic access error:', err);
    showToast('Microphone access denied or error.', 'error');
  }
}

function stopVoiceRecording() {
  const wasRecording = recording;
  if (wasRecording) {
    recorder.stop();
    recorder.exportWAV((blob) => {
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onloadend = () => {
        const base64Audio = reader.result.split(',')[1];
        analyseVoice(base64Audio);
      };
    });
    recording = false;
  }
  const card = document.getElementById('voiceRecordCard');
  const statusEl = document.getElementById('voiceStatus');
  const startBtn = document.getElementById('startRecordingBtn');
  const stopBtn = document.getElementById('stopRecordingBtn');
  const loader = document.getElementById('voiceLoader');
  if (card) card.classList.remove('recording');
  if (wasRecording) {
    if (statusEl) statusEl.textContent = 'Processing...';
    if (startBtn) startBtn.style.display = 'none';
    if (stopBtn) stopBtn.style.display = 'none';
    if (loader) loader.style.display = 'flex';
  }
}

async function analyseVoice(base64Audio) {
  // Show analyzing status
  showToast('🎙️ Analyzing voice stress...', 'info');

  try {
    const res = await fetch('/voice-predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio: base64Audio })
    });

    const data = await res.json();
    document.getElementById('voiceLoader').style.display = 'none';

    if (data.ok) {
      showToast('✅ Voice analysis complete!', 'success');
      const container = document.getElementById('voiceResult');
      const f = data.features || {};
      const stressIcons = { Low: '🟢', Moderate: '🟡', High: '🟠', Severe: '🔴' };
      const stressColors = { Low: '#22c55e', Moderate: '#f59e0b', High: '#f97316', Severe: '#ef4444' };
      const sl = data.stress_level || 'Low';
      const sColor = stressColors[sl] || '#64748b';
      // Score bar: backend max is ~10, mapped to 0-100%
      const scoreBarPct = Math.min(((data.stress_score || 0) / 10) * 100, 100).toFixed(0);
      const safeAdvice = (data.advice || '').replace(/'/g, "&#39;");

      // Helper: interpret a feature value
      const interp = (key, val) => {
        const v = parseFloat(val);
        if (key === 'pitch') return v > 250 ? 'High arousal' : v < 80 && v > 0 ? 'Tense/low' : 'Normal';
        if (key === 'jitter') return v > 0.10 ? 'High instability' : v > 0.05 ? 'Mild tremor' : 'Stable';
        if (key === 'energy') return v > 0.08 ? 'High tension' : v > 0.04 ? 'Moderate' : 'Relaxed';
        if (key === 'shimmer') return v > 0.15 ? 'High irregularity' : v > 0.08 ? 'Mild' : 'Smooth';
        if (key === 'zcr') return v > 0.12 ? 'Breathiness/tremor' : v > 0.08 ? 'Mild shakiness' : 'Steady';
        if (key === 'speech_rate') return v > 6 ? 'Very fast/rushed' : v < 1.5 ? 'Very slow/halted' : v > 4.5 ? 'Slightly fast' : 'Normal';
        if (key === 'pause_ratio') return v > 0.5 ? 'Very choppy/hesitant' : v < 0.05 ? 'No pauses (tense)' : 'Natural rhythm';
        return '';
      };

      const featureRow = (label, val, unit, interpKey) => `
        <div class="face-result-item">
          <span class="face-item-label">${label}</span>
          <span class="face-item-val">${val}${unit}</span>
          <span class="face-item-detail">${interp(interpKey, val)}</span>
        </div>`;

      container.innerHTML = `
        <div class="voice-result-card">
          <!-- Header: stress level + score bar -->
          <div style="margin-bottom:14px;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
              <span class="face-risk-badge" style="background:${sColor}22; color:${sColor}; border:1px solid ${sColor}44; font-size:0.9rem; padding:4px 10px;">
                ${stressIcons[sl] || '🔘'} ${sl} Stress
              </span>
              <span style="font-size:0.8rem; font-weight:700; color:${sColor}; margin-left:auto;">
                ${data.stress_score || 0}/10
              </span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.6rem; color:var(--muted); margin-bottom:3px;">
              <span>Calm</span><span>Moderate</span><span>High</span><span>Severe</span>
            </div>
            <div style="height:8px; background:var(--surface3); border-radius:8px; overflow:hidden;">
              <div style="height:100%; width:${scoreBarPct}%; background:linear-gradient(90deg, #22c55e, ${sColor}); border-radius:8px; transition:width 1.2s ease; box-shadow:0 0 8px ${sColor}55;"></div>
            </div>
            ${data.duration ? `<div style="font-size:0.68rem; color:var(--muted); margin-top:4px; text-align:right;">🎙️ ${data.duration}s recorded</div>` : ''}
          </div>

          <!-- Section 1: Acoustic Core -->
          <div style="font-size:0.68rem; font-weight:700; color:var(--muted); letter-spacing:0.05em; text-transform:uppercase; margin-bottom:6px;">🎚 Acoustic Features</div>
          <div class="face-extra-results" style="margin-bottom:12px;">
            ${featureRow('🎵 Pitch (F0)', Math.round(f.pitch || 0), ' Hz', 'pitch')}
            ${featureRow('〰 Jitter (F0 variation)', f.jitter || 0, '', 'jitter')}
            ${featureRow('⚡ Energy (RMS)', f.energy || 0, '', 'energy')}
            ${featureRow('〜 Shimmer (amplitude var.)', f.shimmer || 0, '', 'shimmer')}
            ${featureRow('〽 ZCR (breathiness)', f.zcr || 0, '', 'zcr')}
          </div>

          <!-- Section 2: Prosodic -->
          <div style="font-size:0.68rem; font-weight:700; color:var(--muted); letter-spacing:0.05em; text-transform:uppercase; margin-bottom:6px;">🗣 Prosodic Features</div>
          <div class="face-extra-results" style="margin-bottom:12px;">
            ${featureRow('⏱ Speech Rate', f.speech_rate || 0, ' onset/s', 'speech_rate')}
            ${featureRow('⏸ Pause Ratio', ((f.pause_ratio || 0) * 100).toFixed(1), '% silence', 'pause_ratio')}
          </div>

          <!-- Section 3: Spectral + MFCC -->
          <div style="font-size:0.68rem; font-weight:700; color:var(--muted); letter-spacing:0.05em; text-transform:uppercase; margin-bottom:6px;">📊 Spectral / MFCC</div>
          <div class="face-extra-results" style="margin-bottom:14px;">
            ${featureRow('🔆 Spectral Centroid', Math.round(f.spectral_centroid || 0), ' Hz', '')}
            ${featureRow('📉 Spectral Rolloff', Math.round(f.spectral_rolloff || 0), ' Hz', '')}
            ${featureRow('🎛 MFCC₁ (energy)', f.mfcc1 || 0, '', '')}
            ${featureRow('🎛 MFCC₂ (shape)', f.mfcc2 || 0, '', '')}
            ${featureRow('📈 MFCC variability', f.mfcc_variability || 0, '', '')}
          </div>

          <!-- Advice -->
          <div class="face-advice" style="margin-top:4px;">
            <strong>💡 Personalised Advice:</strong> ${data.advice}
          </div>
          <p class="face-disclaimer" style="margin-top:8px;">⚠️ Based on 8-factor weighted acoustic scoring. For awareness only — not a clinical diagnosis.</p>

          <!-- Actions -->
          <div style="display:flex; gap:8px; margin-top:14px;">
            <button class="face-btn primary" onclick="speakText('${safeAdvice}', 'en-US')" style="flex:1;">🔊 Hear Advice</button>
            <button class="face-btn secondary" onclick="resetVoiceUI()" style="flex:1;">🔄 Record Again</button>
            <button class="face-btn secondary" onclick="closeVoiceModal()" style="flex:1;">❌ Close</button>
          </div>
        </div>
      `;
      container.style.display = 'block';
      document.getElementById('voiceRecordCard').style.display = 'none';

      // Load stage-specific food suggestions
      if (typeof loadFoodSuggestions === 'function') {
        loadFoodSuggestions(data.recommended_stage !== undefined ? data.recommended_stage : 0);
      }
    } else {
      showToast('❌ ' + (data.error || 'Voice analysis failed.'), 'error');
      resetVoiceUI();
    }
  } catch (e) {
    console.error('Voice analysis failed:', e);
    showToast('❌ Network error.', 'error');
    resetVoiceUI();
  }
}

function resetVoiceUI() {
  document.getElementById('voiceResult').style.display = 'none';
  document.getElementById('voiceRecordCard').style.display = 'block';
  document.getElementById('voiceStatus').textContent = 'Ready to record';
  document.getElementById('startRecordingBtn').style.display = 'inline-block';
  document.getElementById('stopRecordingBtn').style.display = 'none';
  document.getElementById('voiceLoader').style.display = 'none';
}

// ---------------- Speech-to-Text (STT) / Text-to-Speech (TTS) helpers ----------------
let recognition = null;
function startSpeechRecognition(lang = 'en-US') {
  if (!('SpeechRecognition' in window) && !('webkitSpeechRecognition' in window)) {
    showToast('Speech recognition not supported in this browser.', 'error');
    return;
  }
  // stop any ongoing media recorder to avoid conflicts
  try { if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop(); } catch (e) { }

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.lang = lang;
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    document.getElementById('voiceStatus').textContent = 'Listening...';
    document.getElementById('voiceLoader').style.display = 'none';
  };

  recognition.onresult = (evt) => {
    const transcript = (evt.results && evt.results[0] && evt.results[0][0]) ? evt.results[0][0].transcript : '';
    displayTranscription(transcript, lang);
  };

  recognition.onerror = (ev) => {
    console.error('STT error:', ev);
    showToast('Transcription error. Try again.', 'error');
    document.getElementById('voiceStatus').textContent = 'Ready to record';
  };

  recognition.onend = () => {
    document.getElementById('voiceStatus').textContent = 'Ready to record';
  };

  recognition.start();
}

function stopSpeechRecognition() {
  if (recognition) recognition.stop();
}

function displayTranscription(text, lang) {
  const container = document.getElementById('voiceResult');
  container.innerHTML = `
    <div class="voice-result-card">
      <h4 style="margin-top:0">Transcription (${lang}):</h4>
      <p id="transcriptText" style="white-space:pre-wrap">${text}</p>
      <div style="display:flex; gap:8px; margin-top:10px">
        <button class="face-btn primary" onclick="speakText(document.getElementById('transcriptText').textContent,'en-US')">▶ Play EN</button>
        <button class="face-btn secondary" onclick="speakText(document.getElementById('transcriptText').textContent,'ta-IN')">▶ Play TA</button>
      </div>
      <button class="face-btn secondary" style="width:100%; margin-top:10px" onclick="resetVoiceUI()">Close</button>
    </div>
  `;
  container.style.display = 'block';
  document.getElementById('voiceRecordCard').style.display = 'none';
}

function speakText(text, lang = 'en-US') {
  if (!('speechSynthesis' in window)) {
    showToast('Text-to-speech not supported in this browser.', 'error');
    return;
  }
  if (!text || text.trim().length === 0) return;

  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = lang;

  // choose a matching voice if available
  const voices = window.speechSynthesis.getVoices();
  if (voices && voices.length) {
    // try exact match first, then prefix match
    let chosen = voices.find(v => v.lang === lang) || voices.find(v => v.lang && v.lang.startsWith(lang.split('-')[0]));
    if (!chosen) chosen = voices[0];
    utter.voice = chosen;
  }

  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
}

// Polyfill: ensure voices are loaded (some browsers load asynchronously)
if ('speechSynthesis' in window) {
  window.speechSynthesis.onvoiceschanged = () => { window.speechSynthesis.getVoices(); };
}

// ---------------- Chat UI & API ----------------
let _chatHistory = [];   // conversation memory for Gemini multi-turn

function openChatModal() {
  const modal = document.getElementById('chatModal');
  if (modal) modal.style.display = 'block';
  const body = document.getElementById('chatBody');
  if (body) body.scrollTop = body.scrollHeight;
  // Focus input
  setTimeout(() => document.getElementById('chatInput')?.focus(), 100);
}

function closeChatModal() {
  const modal = document.getElementById('chatModal');
  if (modal) modal.style.display = 'none';
}

// Generic modal close function
function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (!modal) return;
  if (modal.classList.contains('face-modal')) {
    modal.classList.remove('open');
    document.body.style.overflow = '';
  } else {
    modal.style.display = 'none';
  }
}

// Toggle chat modal (alternative to openChatModal)
function toggleChat() {
  const modal = document.getElementById('chatModal');
  if (!modal) return;
  if (modal.style.display === 'none' || !modal.style.display) {
    openChatModal();
  } else {
    closeChatModal();
  }
}

function _chatNow() {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function appendChatMessage(text, who = 'bot') {
  const body = document.getElementById('chatBody');
  if (!body) return;
  const el = document.createElement('div');
  el.className = 'chat-msg ' + (who === 'user' ? 'user' : 'bot');
  el.innerHTML = `${text}<span class="chat-msg-time">${_chatNow()}</span>`;
  body.appendChild(el);
  body.scrollTop = body.scrollHeight;
}

function _showTyping() {
  const body = document.getElementById('chatBody');
  if (!body) return null;
  const el = document.createElement('div');
  el.className = 'chat-typing';
  el.id = '_chatTyping';
  el.innerHTML = '<span></span><span></span><span></span>';
  body.appendChild(el);
  body.scrollTop = body.scrollHeight;
  return el;
}

function _removeTyping() {
  document.getElementById('_chatTyping')?.remove();
}

// For quick reply chips
function sendQuick(text) {
  // Hide quick replies after first use
  const qr = document.getElementById('chatQuickReplies');
  if (qr) qr.style.display = 'none';
  const input = document.getElementById('chatInput');
  if (input) input.value = text;
  sendChatMessage();
}

// ─── Input STT and Result TTS ─────────────────────
let inputRecognition = null;
function startInputSTT(fieldId) {
  if (!('SpeechRecognition' in window) && !('webkitSpeechRecognition' in window)) {
    showToast('Speech recognition not supported in this browser.', 'error');
    return;
  }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  inputRecognition = new SR();
  inputRecognition.lang = 'en-US';
  inputRecognition.interimResults = false;
  inputRecognition.maxAlternatives = 1;

  inputRecognition.onstart = () => {
    showToast('🎤 Listening...', 'info');
  };
  inputRecognition.onresult = (evt) => {
    const transcript = (evt.results && evt.results[0] && evt.results[0][0]) ? evt.results[0][0].transcript : '';
    document.getElementById(fieldId).value = transcript;
  };
  inputRecognition.onerror = (ev) => {
    console.error('STT error:', ev);
    showToast('Transcription failed. Try again.', 'error');
  };
  inputRecognition.start();
}

// TTS for result summary
function speakResultSummary() {
  const stage = document.getElementById('resultStage')?.textContent || 'Unknown Stage';
  const desc = document.getElementById('resultDesc')?.textContent || '';
  const text = `${stage}. ${desc}`;
  if (text.trim().length > 0) {
    speakText(text, 'en-US');
  }
}

async function sendChatMessage() {
  const input = document.getElementById('chatInput');
  const lang = document.getElementById('chatLang')?.value || 'en';
  const text = input?.value.trim();
  if (!text) return;

  console.log('🔄 [CHAT] Starting chat request for:', text);

  // Hide quick reply chips once user starts typing
  const qr = document.getElementById('chatQuickReplies');
  if (qr) qr.style.display = 'none';

  appendChatMessage(text, 'user');
  input.value = '';
  input.focus();

  // Disable send button during request
  const btn = document.getElementById('chatSendBtn');
  if (btn) btn.disabled = true;

  // Show typing indicator
  _showTyping();

  // Track history for Gemini multi-turn
  _chatHistory.push({ role: 'user', content: text });

  const requestPayload = { message: text, lang, history: _chatHistory };
  console.log('📤 [CHAT] Sending request:', requestPayload);

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestPayload)
    });

    console.log('📥 [CHAT] Response status:', res.status, res.statusText);

    const data = await res.json();
    console.log('📥 [CHAT] Response data:', data);

    _removeTyping();

    if (data && data.ok) {
      const reply = data.reply || 'I have no answer.';
      const source = data.source || 'unknown';
      console.log('✅ [CHAT] Success - Reply:', reply, 'Source:', source);

      appendChatMessage(reply, 'bot');
      _chatHistory.push({ role: 'model', content: reply });
      // keep history to last 20 messages
      if (_chatHistory.length > 20) _chatHistory = _chatHistory.slice(-20);
      // Speak reply in background
      try { speakText(reply, lang === 'ta' ? 'ta-IN' : 'en-US'); } catch (e) { }
    } else {
      const errorMsg = data.error || 'Chat failed. Please try again.';
      console.error('❌ [CHAT] API returned error:', errorMsg);
      appendChatMessage(errorMsg, 'bot');
    }
  } catch (e) {
    _removeTyping();
    console.error('❌ [CHAT] Network/fetch error:', e);
    appendChatMessage('⚠️ Network error. Please try again.', 'bot');
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ─── Bio-Report Predict Logic ─────────────────────────

function openReportModal() {
  const modal = document.getElementById('reportModal');
  if (modal) modal.classList.add('open');
  document.body.style.overflow = 'hidden';
  resetReportUI();
}

function resetReportUI() {
  document.getElementById('reportUploadCard').style.display = 'block';
  document.getElementById('reportLoader').style.display = 'none';
  document.getElementById('reportResult').style.display = 'none';
  document.getElementById('reportInput').value = '';
}

async function handleReportUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  // Immediate visual feedback
  showToast('📄 Analyzing report...', 'info');
  document.getElementById('reportUploadCard').style.display = 'none';
  document.getElementById('reportLoader').style.display = 'block';

  const reader = new FileReader();
  reader.onload = async function (e) {
    const base64Image = e.target.result.split(',')[1];

    // Set preview image
    const previewImg = document.getElementById('reportPreviewImg');
    if (previewImg) previewImg.src = e.target.result;
    document.getElementById('reportResult').style.display = 'none';

    try {
      // Show analyzing status with spinner
      const loaderText = document.querySelector('#reportLoader p') ||
        document.querySelector('#reportLoader');
      if (loaderText && loaderText.textContent) {
        loaderText.textContent = 'Gemini is parsing report data...';
      }

      const res = await fetch('/report-analyse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Image })
      });
      const data = await res.json();

      document.getElementById('reportLoader').style.display = 'none';

      if (data.ok) {
        showToast('✅ Report analyzed successfully!', 'success');
        showReportResults(data.extracted_data);
      } else {
        showToast('❌ ' + (data.error || 'Report analysis failed.'), 'error');
        resetReportUI();
      }
    } catch (err) {
      showToast('❌ Error connecting to server.', 'error');
      console.error(err);
      resetReportUI();
    }
  };
  reader.readAsDataURL(file);
}

function showReportResults(extractedData) {
  const container = document.getElementById('reportResult');
  const mapping = {
    "gender": { id: "gender", seg: "genderSeg" },
    "age_group": { id: "age_group" },
    "family_history": { id: "family_history", seg: "famHistSeg" },
    "patient_status": { id: "patient_status", seg: "patStatusSeg" },
    "take_medication": { id: "take_medication", seg: "medSeg" },
    "time_since_diagnosis": { id: "time_since_diagnosis" },
    "symptom_severity": { id: "symptom_severity", seg: "sevSeg" },
    "shortness_of_breath": { id: "shortness_of_breath", seg: "sobSeg" },
    "visual_changes": { id: "visual_changes", seg: "visSeg" },
    "nosebleeds": { id: "nosebleeds", seg: "noseSeg" },
    "systolic_bp": { id: "systolic_bp" },
    "diastolic_bp": { id: "diastolic_bp" },
    "controlled_diet": { id: "controlled_diet", seg: "dietSeg" },
    "bmi_category": { id: "bmi_category", seg: "bmiSeg" },
    "diabetes": { id: "diabetes", seg: "diabetesSeg" },
    "cholesterol_level": { id: "cholesterol_level", seg: "cholSeg" },
    "heart_rate_category": { id: "heart_rate_category", seg: "hrSeg" },
    "exercise_frequency": { id: "exercise_frequency" }
  };

  let html = `
    <div class="voice-result-card" style="margin-top:0">
      <div class="face-result-header">
        <span class="face-risk-badge" style="background:#6366f1; color:#fff">REPORT EXTRACTED</span>
      </div>
      <div class="face-extra-results" style="grid-template-columns: 1fr; gap: 5px; margin-top: 10px;">
  `;

  for (const [key, val] of Object.entries(extractedData)) {
    const config = mapping[key];
    if (!config) continue;

    const fieldElem = document.getElementById(config.id);
    let matched = false;

    if (fieldElem) {
      if (fieldElem.tagName === 'SELECT') {
        for (const opt of fieldElem.options) {
          const optText = opt.text.toLowerCase().replace(/[^a-z0-9]/g, '');
          const valText = val.toLowerCase().replace(/[^a-z0-9]/g, '');
          if (optText === valText || opt.value === val) {
            fieldElem.value = opt.value;
            matched = true;
            break;
          }
        }
      } else if (config.seg) {
        const segDiv = document.getElementById(config.seg);
        if (segDiv) {
          const btns = segDiv.querySelectorAll('.seg-btn');
          for (const btn of btns) {
            const btnText = btn.textContent.toLowerCase().replace(/[^a-z0-9]/g, '');
            const valText = val.toLowerCase().replace(/[^a-z0-9]/g, '');
            if (btnText.includes(valText) ||
              btn.dataset.val === val ||
              (val.toLowerCase() === 'yes' && btn.dataset.val === '1') ||
              (val.toLowerCase() === 'no' && btn.dataset.val === '0')) {
              setSeg(config.seg, config.id, btn);
              matched = true;
              break;
            }
          }
        }
      }
    }

    html += `
      <div class="face-result-item" style="padding: 4px 0;">
        <span class="face-item-label" style="font-size: 0.75rem;">${key.replace(/_/g, ' ')}</span>
        <span class="face-item-val" style="font-size: 0.8rem; color:${matched ? '#10b981' : '#f59e0b'}">
          ${val} ${matched ? '✓' : '?'}
        </span>
      </div>
    `;
  }

  html += `
      </div>
      <p style="font-size: 0.75rem; color: var(--muted2); margin-top: 15px; text-align: center;">
        Fields marked with ✓ have been auto-filled in the main form.
      </p>
      <button class="face-btn primary" onclick="closeModal('reportModal'); setTimeout(() => runPrediction(), 300)" style="width:100%; margin-top:10px">🚀 Auto-Analyze & Predict</button>
      <button class="face-btn secondary" onclick="resetReportUI()" style="width:100%; margin-top:10px">Upload Another</button>
    </div>
  `;

  const dataContainer = document.getElementById('reportDataFields');
  if (dataContainer) {
    dataContainer.innerHTML = html;
  } else {
    container.innerHTML = html;
  }
  container.style.display = 'block';

  if (typeof updateProgress === 'function') updateProgress();
  if (typeof updateBPColor === 'function') updateBPColor();

  // Auto-trigger prediction after 2 seconds if all critical fields are filled
  setTimeout(() => {
    const requiredFilled = REQUIRED_FIELDS.every(f => document.getElementById(f).value !== '');
    if (requiredFilled) {
      showToast('✅ Report analyzed! Running prediction...', 'success');
      setTimeout(() => {
        closeModal('reportModal');
        runPrediction();
      }, 800);
    }
  }, 1500);
}

console.log('>>> DEBUG: Bio Report Button in DOM:', !!document.getElementById('fabReportBtn'));


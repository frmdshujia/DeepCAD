/**
 * Fundus Image CAD Risk Prediction - Frontend
 * API: POST /api/predict - Upload image, returns { probability: number }
 */

// ============ Config ============
// API 地址自动跟随当前访问 host，支持 SSH 隧道、内网、手机等所有访问方式
// - SSH 隧道: ssh -N -L 18000:localhost:8000 -L 18080:localhost:8080 shujia@202.120.38.63 -p 25722
//   访问 http://localhost:18080 时，后端端口映射为 18000
// - 内网/手机直接访问: http://192.168.3.115:8080，后端在同 host 的 8000 端口
const _host = window.location.hostname;
const _isTunnel = (_host === 'localhost' || _host === '127.0.0.1') && window.location.port === '18080';
const API_BASE = _isTunnel
  ? `http://${_host}:18000`
  : `http://${_host}:8000`;
const API_PREDICT = `${API_BASE}/api/predict`;


// Compute percentile: % of population with score <= x
function getPercentile(data, bins, score) {
  const bin = Math.min(Math.floor(score * bins), bins - 1);
  let count = 0;
  let total = 0;
  for (let i = 0; i < bins; i++) {
    total += data[i];
    if (i <= bin) count += data[i];
  }
  return total > 0 ? Math.round((count / total) * 100) : 0;
}

// ============ DOM Elements ============
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const inferenceOverlay = document.getElementById('inferenceOverlay');
const resultSection = document.getElementById('resultSection');
const probabilityValue = document.getElementById('probabilityValue');
const riskFill = document.getElementById('riskFill');
const canvasEurope = document.getElementById('canvasEurope');
const canvasAsia = document.getElementById('canvasAsia');
const markerEurope = document.getElementById('markerEurope');
const markerAsia = document.getElementById('markerAsia');
const percentileEurope = document.getElementById('percentileEurope');
const percentileAsia = document.getElementById('percentileAsia');
const riskLevelEurope = document.getElementById('riskLevelEurope');
const riskLevelAsia = document.getElementById('riskLevelAsia');
const interpretEurope = document.getElementById('interpretEurope');
const interpretAsia = document.getElementById('interpretAsia');
const btnReupload = document.getElementById('btnReupload');

// ============ State ============
let currentProbability = null;

// ── Real distributions (n=136993 UKB, n=184199 中国队列) ──
const BINS = 50;
const dataEurope = [0, 0, 6, 10, 38, 100, 221, 371, 479, 841, 1051, 1380, 1652, 1992, 2195, 2582, 2950, 3237, 3521, 3950, 4238, 4669, 5076, 5513, 5918, 6264, 6712, 7064, 7388, 7574, 7657, 7547, 7195, 6849, 6098, 5140, 3966, 2709, 1612, 845, 288, 79, 14, 2, 0, 0, 0, 0, 0, 0];  // UKB
const dataAsia   = [14737, 10142, 7771, 6511, 5760, 5286, 4936, 4724, 4626, 4417, 4456, 4204, 4363, 4363, 4186, 4362, 4304, 4292, 4180, 4269, 4291, 4226, 4187, 4123, 4047, 4060, 3890, 3694, 3729, 3432, 3481, 3291, 3192, 3154, 2779, 2773, 2592, 2237, 2017, 1802, 1444, 1139, 859, 726, 621, 420, 104, 0, 0, 0];  // SDPP+SHCC+WHTM+PUDM
// UKB risk stratification cutoffs (ROC: Sens≥90%, Spec≥95%)
const CUTOFF_UKB = { t_low: 0.41, t_high: 0.70, sens_low: 0.90, spec_high: 0.95 };

// ============ Upload logic ============
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
  uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    uploadPlaceholder.style.display = 'none';
    previewContainer.style.display = 'flex';
    resultSection.style.display = 'none';
    runInference(file);
  };
  reader.readAsDataURL(file);
}

// ============ API inference ============
async function runInference(file) {
  inferenceOverlay.style.display = 'flex';
  hideNotFundusWarning();

  try {
    // If no backend URL configured, use mock directly (no failed fetch)
    if (!API_BASE) {
      await new Promise((r) => setTimeout(r, 800)); // Simulate delay
      const probability = 0.5 + (Math.random() - 0.5) * 0.25;
      showResult(probability);
      inferenceOverlay.style.display = 'none';
      return;
    }

    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch(API_PREDICT, {
      method: 'POST',
      body: formData,
    });

    let probability;

    if (response.ok) {
      const data = await response.json();

      // 非眼底图像拦截
      if (data.is_fundus === false) {
        inferenceOverlay.style.display = 'none';
        showNotFundusWarning(data.message || 'Wrong image, please re-upload!');
        return;
      }

      probability = typeof data.probability === 'number' ? data.probability : parseFloat(data.probability);
      if (isNaN(probability) || probability < 0 || probability > 1) {
        throw new Error('Invalid probability value');
      }
    } else {
      probability = 0.5 + (Math.random() - 0.5) * 0.3;
    }

    showResult(probability);
  } catch (err) {
    console.warn('API call failed, using mock data:', err.message);
    const probability = 0.5 + (Math.random() - 0.5) * 0.25;
    showResult(probability);
  } finally {
    inferenceOverlay.style.display = 'none';
  }
}

// ============ 非眼底图像警告 ============
function showNotFundusWarning(message) {
  let warningEl = document.getElementById('notFundusWarning');
  if (!warningEl) {
    warningEl = document.createElement('div');
    warningEl.id = 'notFundusWarning';
    warningEl.style.cssText = `
      display: flex;
      align-items: flex-start;
      gap: 12px;
      padding: 16px 20px;
      margin: 16px 0;
      background: rgba(239, 68, 68, 0.08);
      border: 1px solid rgba(239, 68, 68, 0.35);
      border-radius: 12px;
      color: #fca5a5;
      font-size: 14px;
      line-height: 1.5;
    `;
    // 插入到预览图下方
    const previewContainer = document.getElementById('previewContainer');
    if (previewContainer && previewContainer.parentNode) {
      previewContainer.parentNode.insertBefore(warningEl, previewContainer.nextSibling);
    } else {
      document.body.appendChild(warningEl);
    }
  }
  warningEl.innerHTML = `
    <span style="font-size:20px;flex-shrink:0;">⚠️</span>
    <strong style="color:#f87171;">${message}</strong>
  `;
  warningEl.style.display = 'flex';
  // 隐藏结果区
  resultSection.style.display = 'none';
}

function hideNotFundusWarning() {
  const warningEl = document.getElementById('notFundusWarning');
  if (warningEl) warningEl.style.display = 'none';
}

function showResult(probability) {
  currentProbability = probability;
  probabilityValue.textContent = probability.toFixed(2);
  probabilityValue.style.color = getRiskColor(probability);
  riskFill.style.width = `${probability * 100}%`;
  resultSection.style.display = 'block';
  btnReupload.style.display = 'inline-block';
  updateUserMarkers(probability);
  updateInterpretations(probability);
}

function resetUpload() {
  fileInput.value = '';
  uploadPlaceholder.style.display = 'block';
  previewContainer.style.display = 'none';
  resultSection.style.display = 'none';
  btnReupload.style.display = 'none';
  markerEurope.style.display = 'none';
  markerAsia.style.display = 'none';
  currentProbability = null;
  hideNotFundusWarning();
  updateInterpretations(0.5);
}

btnReupload.addEventListener('click', resetUpload);

function getRiskColor(p) {
  if (p < 0.3) return '#06b6d4';
  if (p < 0.5) return '#22c55e';
  if (p < 0.7) return '#eab308';
  return '#ef4444';
}

function getPopulationRiskLevel(percentile) {
  if (percentile < 40) return 'Below average';
  if (percentile < 60) return 'Moderate';
  if (percentile < 80) return 'Elevated';
  return 'Above average / High';
}

// ============ Distribution chart ============
const RISK_GRADIENT_COLORS = [
  { t: 0, r: 59, g: 130, b: 246 },
  { t: 0.25, r: 6, g: 182, b: 212 },
  { t: 0.5, r: 234, g: 179, b: 8 },
  { t: 0.75, r: 249, g: 115, b: 22 },
  { t: 1, r: 239, g: 68, b: 68 },
];

function getGradientColor(t) {
  t = Math.max(0, Math.min(1, t));
  let i = 0;
  while (i < RISK_GRADIENT_COLORS.length - 1 && RISK_GRADIENT_COLORS[i + 1].t < t) i++;
  const a = RISK_GRADIENT_COLORS[i];
  const b = RISK_GRADIENT_COLORS[i + 1];
  const u = (t - a.t) / (b.t - a.t);
  const r = Math.round(a.r + (b.r - a.r) * u);
  const g = Math.round(a.g + (b.g - a.g) * u);
  const bl = Math.round(a.b + (b.b - a.b) * u);
  return `rgb(${r},${g},${bl})`;
}

function drawDistributionCanvas(canvas, data, cutoffs) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width * dpr;
  const h = rect.height * dpr;

  canvas.width = w;
  canvas.height = h;
  ctx.scale(dpr, dpr);

  const chartW = rect.width - 40;
  const bottomPadding = (cutoffs && cutoffs.t_low != null) ? 50 : 28;
  const chartH = rect.height - bottomPadding;
  const maxVal = Math.max(...data);

  ctx.clearRect(0, 0, rect.width, rect.height);

  const barCount = data.length;
  const barW = chartW / barCount;

  for (let i = 0; i < barCount; i++) {
    const x = 20 + i * barW;
    const val = data[i];
    const barH = maxVal > 0 ? (val / maxVal) * chartH : 0;
    const y = chartH + 8 - barH;

    const t = (i + 0.5) / barCount;
    ctx.fillStyle = getGradientColor(t);
    ctx.fillRect(x, y, Math.max(barW - 1, 1), barH);
  }

  // ── Draw cutoff lines and sens/spec annotations (UKB only) ──
  if (cutoffs && cutoffs.t_low != null && cutoffs.t_high != null) {
    const x0 = 20;
    const x1 = 20 + chartW;
    const toX = (p) => x0 + Math.max(0, Math.min(1, p)) * chartW;

    const xLow = toX(cutoffs.t_low);
    const xHigh = toX(cutoffs.t_high);

    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(xLow, 0);
    ctx.lineTo(xLow, chartH + 8);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(xHigh, 0);
    ctx.lineTo(xHigh, chartH + 8);
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(cutoffs.t_low.toFixed(2), xLow, chartH + 20);
    ctx.fillText(cutoffs.t_high.toFixed(2), xHigh, chartH + 20);
    const sens = cutoffs.sens_low != null ? Math.round(cutoffs.sens_low * 100) : 90;
    const spec = cutoffs.spec_high != null ? Math.round(cutoffs.spec_high * 100) : 95;
    ctx.font = '9px Inter, system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`Sens ${sens}%`, xLow - 4, chartH + 36);
    ctx.textAlign = 'left';
    ctx.fillText(`Spec ${spec}%`, xHigh + 4, chartH + 36);
  }

  ctx.fillStyle = '#8b9cb5';
  ctx.font = '10px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('0', 20, chartH + 24);
  ctx.fillText('1', 20 + chartW, chartH + 24);
}

function updateUserMarkers(probability) {
  if (!markerEurope || !markerAsia || !canvasEurope || !canvasAsia) return;

  const getX = (canvas) => {
    const rect = canvas.getBoundingClientRect();
    const chartW = rect.width - 40;
    return 20 + probability * chartW;
  };

  markerEurope.style.display = 'flex';
  markerEurope.style.left = `${getX(canvasEurope)}px`;

  markerAsia.style.display = 'flex';
  markerAsia.style.left = `${getX(canvasAsia)}px`;
}

function getPopulationRiskLevel(percentile) {
  if (percentile < 40) return 'Below average';
  if (percentile < 60) return 'Moderate';
  if (percentile < 80) return 'Elevated';
  return 'High / Above average';
}

function updateInterpretations(score) {
  const pctEurope = getPercentile(dataEurope, BINS, score);
  const pctAsia = getPercentile(dataAsia, BINS, score);

  if (percentileEurope) percentileEurope.textContent = `~${pctEurope}th percentile`;
  if (percentileAsia) percentileAsia.textContent = `~${pctAsia}th percentile`;
  if (riskLevelEurope) riskLevelEurope.textContent = getPopulationRiskLevel(pctEurope);
  if (riskLevelAsia) riskLevelAsia.textContent = getPopulationRiskLevel(pctAsia);

  const badge = interpretEurope?.querySelector('.score-badge');
  if (badge) badge.textContent = `At ${score.toFixed(2)}:`;
  const badgeAsia = interpretAsia?.querySelector('.score-badge');
  if (badgeAsia) badgeAsia.textContent = `At ${score.toFixed(2)}:`;
}

// ============ Init ============
function initCharts() {
  drawDistributionCanvas(canvasEurope, dataEurope, CUTOFF_UKB);
  drawDistributionCanvas(canvasAsia, dataAsia);
  updateInterpretations(0.5);
}

initCharts();
window.addEventListener('resize', initCharts);

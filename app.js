'use strict';
/* =====================================================================
   ระบบหาปริมาณความเสียหายโครงสร้างจากภาพถ่าย + ตัวช่วย AI (YOLO)
   ===================================================================== */

// ---------------------------------------------------------------- utils
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const sum = (a) => a.reduce((s, v) => s + v, 0);
const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '-');
function icons() { try { lucide.createIcons(); } catch (e) { /* offline icons missing: ignore */ } }

const LS = {
  get(k, d) { try { const v = localStorage.getItem('crackai.' + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('crackai.' + k, JSON.stringify(v)); } catch { /* quota/private mode */ } },
  del(k) { try { localStorage.removeItem('crackai.' + k); } catch { } },
};

function toast(msg, type = 'info', ms = 4200) {
  const box = $('toasts');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  box.appendChild(t);
  setTimeout(() => t.remove(), ms);
}
const showToast = toast; // ชื่อเดิม

// ---------------------------------------------------------------- dialogs
function openDlg(id) { $(id).classList.add('open'); if (id === 'dlg-train') ensureTrainPolling(); }
function closeDlg(id) { $(id).classList.remove('open'); if (id === 'dlg-train') ensureTrainPolling(); }
function topDialog() { return [...document.querySelectorAll('.dlg-bg.open')].sort((a, b) => getComputedStyle(b).zIndex - getComputedStyle(a).zIndex)[0]; }

/** ถามยืนยัน: คืนค่า true (ตกลง) / false (ปุ่มรอง) / null (ปิด) */
function askConfirm(title, html, { yes = 'ตกลง', no = 'ยกเลิก', danger = false } = {}) {
  return new Promise((resolve) => {
    $('cf-title').textContent = title;
    $('cf-body').innerHTML = html;
    $('cf-yes').textContent = yes;
    $('cf-no').textContent = no;
    $('cf-yes').className = 'btn ' + (danger ? 'btn-dark' : 'btn-primary');
    const done = (v) => { closeDlg('dlg-confirm'); $('dlg-confirm').onclick = null; resolve(v); };
    $('cf-yes').onclick = (e) => { e.stopPropagation(); done(true); };
    $('cf-no').onclick = (e) => { e.stopPropagation(); done(false); };
    $('dlg-confirm').onclick = (e) => { if (e.target.id === 'dlg-confirm') done(null); };
    $('dlg-confirm')._dismiss = () => done(null);
    openDlg('dlg-confirm');
    setTimeout(() => $('cf-yes').focus(), 50);
  });
}

// ---------------------------------------------------------------- state
const TYPES = {
  crack: { name: 'รอยร้าว', prefix: 'C', color: '#D9363E' },
  area: { name: 'พื้นที่เสียหาย', prefix: 'A', color: '#E8730C' },
  rebar: { name: 'เหล็กเสริม', prefix: 'R', color: '#2F8F5B' },
};
const REF_COLOR = '#2563EB';
const State = {
  image: null, imageName: '', imageUrl: null, imageBlob: null,
  refMode: 'planar', calibrated: false, calibType: null, pxPerCm: 1, H_Matrix: null,
  mode: 'idle', tool: null, refPoints: [], currentPath: [],
  hovered: -1, dragging: -1, drawingDrag: false,
  records: { crack: [], area: [], rebar: [] },
  showAi: true, aiModel: null, dirty: false, libraryId: null, libSavedAt: 0,
  server: { ok: false, info: null },
  classes: null,
};
let nextId = Date.now();
const newId = () => ++nextId;

// ---------------------------------------------------------------- geometry
function solveGaussJordan(A, B) {
  const n = A.length;
  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) maxRow = k;
    [A[i], A[maxRow]] = [A[maxRow], A[i]];
    [B[i], B[maxRow]] = [B[maxRow], B[i]];
    if (Math.abs(A[i][i]) < 1e-12) throw new Error('singular');
    for (let k = i + 1; k < n; k++) {
      const c = -A[k][i] / A[i][i];
      for (let j = i; j < n; j++) A[k][j] = i === j ? 0 : A[k][j] + c * A[i][j];
      B[k] += c * B[i];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = B[i] / A[i][i];
    for (let k = i - 1; k >= 0; k--) B[k] -= A[k][i] * x[i];
  }
  return x;
}

/** เรียงมุม 4 จุดเป็น บนซ้าย, บนขวา, ล่างขวา, ล่างซ้าย (ทนต่อการหมุนภาพ) */
function sortRectCorners(pts) {
  const cx = sum(pts.map((p) => p.x)) / 4, cy = sum(pts.map((p) => p.y)) / 4;
  const byAngle = [...pts].sort((a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx));
  let start = 0;
  byAngle.forEach((p, i) => { if (p.x + p.y < byAngle[start].x + byAngle[start].y) start = i; });
  return [0, 1, 2, 3].map((k) => byAngle[(start + k) % 4]);
}

function isConvexQuad(q) {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4];
    const cr = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (Math.abs(cr) < 1e-6) return false;
    if (sign === 0) sign = Math.sign(cr); else if (Math.sign(cr) !== sign) return false;
  }
  return true;
}

function calculateHomography(srcPixels, realW, realH) {
  const s = sortRectCorners(srcPixels);
  if (!isConvexQuad(s)) throw new Error('not convex');
  const dst = [{ x: 0, y: 0 }, { x: realW, y: 0 }, { x: realW, y: realH }, { x: 0, y: realH }];
  const A = [], B = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = s[i], u = dst[i].x, v = dst[i].y;
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); B.push(v);
  }
  const h = solveGaussJordan(A, B);
  if (!h.every(Number.isFinite)) throw new Error('nan');
  return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}

function pixelToReal(px, py) {
  if (!State.calibrated) return { x: 0, y: 0 };
  if (State.calibType === 'planar') return { x: px / State.pxPerCm, y: py / State.pxPerCm };
  if (State.calibType === 'perspective' && State.H_Matrix) {
    const H = State.H_Matrix, w = H[2][0] * px + H[2][1] * py + H[2][2];
    return { x: (H[0][0] * px + H[0][1] * py + H[0][2]) / w, y: (H[1][0] * px + H[1][1] * py + H[1][2]) / w };
  }
  return { x: 0, y: 0 };
}
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const realDist = (a, b) => dist(pixelToReal(a.x, a.y), pixelToReal(b.x, b.y));
function calculateRealLength(pts) {
  let len = 0;
  for (let i = 0; i < pts.length - 1; i++) len += realDist(pts[i], pts[i + 1]);
  return len;
}
function shoelace(pts) {
  let a = 0;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) a += (pts[j].x + pts[i].x) * (pts[j].y - pts[i].y);
  return Math.abs(a / 2);
}
function calculateRealArea(pts) { return pts.length < 3 ? 0 : shoelace(pts.map((p) => pixelToReal(p.x, p.y))); }
function realPerimeter(pts) { return calculateRealLength(pts) + (pts.length > 2 ? realDist(pts[pts.length - 1], pts[0]) : 0); }
function labelAnchor(r) {
  if (r.polys && r.polys.length) {
    const big = r.polys.reduce((a, b) => (shoelace(b) > shoelace(a) ? b : a));
    const xs = big.map((p) => p.x), ys = big.map((p) => p.y);
    return { x: (Math.min(...xs) + Math.max(...xs)) / 2, y: Math.min(...ys) };
  }
  const p = r.points;
  return p[Math.floor(p.length / 2)];
}

/** ขยายเส้น polyline เป็น polygon หนา w px (ใช้ทำป้ายกำกับจากเส้นที่วาดด้วยมือ) */
function bufferPolyline(points, w) {
  const p = points.filter((q, i) => i === 0 || dist(q, points[i - 1]) > 0.5);
  if (p.length < 2) return null;
  const h = w / 2, L = [], R = [];
  const norm = (v) => { const l = Math.hypot(v.x, v.y) || 1; return { x: v.x / l, y: v.y / l }; };
  for (let i = 0; i < p.length; i++) {
    const dPrev = i > 0 ? norm({ x: p[i].x - p[i - 1].x, y: p[i].y - p[i - 1].y }) : null;
    const dNext = i < p.length - 1 ? norm({ x: p[i + 1].x - p[i].x, y: p[i + 1].y - p[i].y }) : null;
    let d = dPrev && dNext ? norm({ x: dPrev.x + dNext.x, y: dPrev.y + dNext.y }) : (dPrev || dNext);
    if (dPrev && dNext && Math.hypot(dPrev.x + dNext.x, dPrev.y + dNext.y) < 1e-3) d = dPrev; // ย้อนกลับ 180°
    const n = { x: -d.y, y: d.x };
    let k = 1;
    if (dPrev && dNext) { const np = { x: -dPrev.y, y: dPrev.x }; k = 1 / Math.max(0.35, n.x * np.x + n.y * np.y); }
    L.push({ x: p[i].x + n.x * h * k, y: p[i].y + n.y * h * k });
    R.push({ x: p[i].x - n.x * h * k, y: p[i].y - n.y * h * k });
  }
  return L.concat(R.reverse());
}

/** คำนวณค่าจริงของแต่ละรายการ (รองรับทั้งเส้นวาดมือ และรูปร่างจาก AI) */
function computeRecord(type, r) {
  if (r.polys && r.polys.length) {
    const areas = r.polys.map(calculateRealArea);
    if (type === 'area') { r.realArea = sum(areas); return; }
    // ความยาวของแถบบางๆ ≈ เส้นรอบรูป / 2 ; ถ้าเป็นกรอบสี่เหลี่ยม (detection) ใช้เส้นทแยงมุม
    const len = r.isBox ? sum(r.polys.map((p) => realDist(p[0], p[2]))) : sum(r.polys.map((p) => realPerimeter(p) / 2));
    r.realLen = len;
    if (type === 'crack') r.widthEst = !r.isBox && len > 0 ? (sum(areas) / len) * 10 : null; // cm -> mm
  } else if (type === 'area') r.realArea = calculateRealArea(r.points);
  else r.realLen = calculateRealLength(r.points);
}
function recalculateAllMeasurements() {
  for (const t of Object.keys(TYPES)) State.records[t].forEach((r) => computeRecord(t, r));
  updateDashboard();
}

// ---------------------------------------------------------------- canvas + image
const canvas = $('main-canvas');
const ctx = canvas.getContext('2d');

function setImage(img, name) {
  State.image = img; State.imageName = name || ''; State.imageBlob = null;
  canvas.width = img.naturalWidth || img.width; canvas.height = img.naturalHeight || img.height;
  $('canvas-empty').classList.add('hidden');
  canvas.classList.remove('hidden');
  $('img-name').textContent = name ? name : 'ภาพจากโปรเจกต์';
}

function resetInteraction() {
  State.mode = 'idle'; State.tool = null; State.currentPath = []; State.dragging = -1; State.hovered = -1; State.drawingDrag = false;
  document.querySelectorAll('.tool').forEach((b) => b.classList.remove('on'));
  $('drawing-controls').classList.add('hidden');
}

$('imageUpload').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  if (State.dirty && hasRecords()) {
    const ok = await askConfirm('เริ่มภาพใหม่', 'ผลการวัดของภาพปัจจุบันยังไม่ได้บันทึก ต้องการเริ่มภาพใหม่หรือไม่', { yes: 'เริ่มภาพใหม่', danger: true });
    if (!ok) return;
  }
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    if (State.imageUrl) URL.revokeObjectURL(State.imageUrl);
    State.imageUrl = url;
    setImage(img, file.name);
    State.calibrated = false; State.calibType = null; State.H_Matrix = null; State.refPoints = [];
    State.records = { crack: [], area: [], rebar: [] };
    State.dirty = false; State.aiModel = null; State.libraryId = null; State.libSavedAt = 0;
    if (window.Library) Library.statusLine();
    resetInteraction();
    updateCalStatus();
    updateInstruction('วางจุดอ้างอิงขนาดในขั้นที่ 2 เพื่อให้วัดค่าได้ถูกต้อง');
    render(); updateDashboard(); updateUi();
  };
  img.onerror = () => { URL.revokeObjectURL(url); toast('เปิดไฟล์ภาพนี้ไม่ได้ กรุณาใช้ JPG หรือ PNG', 'error'); };
  img.src = url;
});

function setRefMode(mode) {
  State.refMode = mode;
  $('cfg-planar').classList.toggle('hidden', mode !== 'planar');
  $('cfg-persp').classList.toggle('hidden', mode !== 'perspective');
  $('tab-planar').classList.toggle('on', mode === 'planar');
  $('tab-persp').classList.toggle('on', mode === 'perspective');
  if (State.image && State.mode === 'calibrating') startCalibration();
}

function setMode(text, color) {
  $('mode-text').textContent = text;
  $('mode-sw').style.background = color || 'var(--ink-3)';
}
function updateInstruction(t) { $('instruction-text').textContent = t; }

function startCalibration() {
  if (!State.image) return toast('กรุณาเลือกภาพถ่ายในขั้นที่ 1 ก่อน', 'error');
  const again = State.mode === 'calibrating'; // กดซ้ำระหว่างวางจุด = เริ่มวางใหม่
  resetInteraction();
  State.mode = 'calibrating';
  const req = State.refMode === 'planar' ? 2 : 4;
  if (again || State.refPoints.length !== req || State.calibType !== State.refMode) {
    State.refPoints = []; State.calibrated = false; State.calibType = null;
    updateCalStatus(); recalculateAllMeasurements();
  }
  setMode(`วางจุดอ้างอิง ${State.refPoints.length}/${req}`, REF_COLOR);
  updateInstruction(req === 2
    ? 'คลิกที่ปลายทั้งสองของวัตถุอ้างอิง คลิกค้างที่จุดเพื่อลากปรับตำแหน่งได้'
    : 'คลิกที่มุมทั้ง 4 ของแผ่นอ้างอิง ลำดับใดก็ได้ คลิกค้างที่จุดเพื่อลากปรับตำแหน่งได้');
  updateUi(); render();
}

function selectTool(tool) {
  if (!State.image) return toast('กรุณาเลือกภาพถ่ายในขั้นที่ 1 ก่อน', 'error');
  if (!State.calibrated) return toast('กรุณาวางจุดอ้างอิงขนาดในขั้นที่ 2 ให้เสร็จก่อน', 'error');
  resetInteraction();
  State.mode = 'drawing'; State.tool = tool;
  $(`tool-${tool}`).classList.add('on');
  $('drawing-controls').classList.remove('hidden');
  setMode(`เครื่องมือ: ${TYPES[tool].name}`, TYPES[tool].color);
  updateInstruction(tool === 'area'
    ? 'คลิกทีละจุดรอบขอบพื้นที่เสียหาย (อย่างน้อย 3 จุด) แล้วกด "บันทึกเส้นนี้" หรือ Enter'
    : 'คลิกค้างแล้วลากตามแนว หรือคลิกทีละจุด แล้วกด "บันทึกเส้นนี้" หรือ Enter');
  updateUi(); render();
}

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  return { x: (e.clientX - rect.left) * (canvas.width / rect.width), y: (e.clientY - rect.top) * (canvas.height / rect.height) };
}
const screenScale = () => canvas.width / Math.max(1, canvas.getBoundingClientRect().width);
function hitRef(pos) {
  const r = 18 * screenScale();
  for (let i = 0; i < State.refPoints.length; i++) if (dist(pos, State.refPoints[i]) < r) return i;
  return -1;
}

canvas.addEventListener('pointerdown', (e) => {
  if (!State.image || State.mode === 'idle') return;
  if (e.pointerType === 'mouse' && e.button !== 0) return;
  e.preventDefault();
  try { canvas.setPointerCapture(e.pointerId); } catch { }
  const pos = getPos(e);
  if (State.mode === 'calibrating') {
    const req = State.refMode === 'planar' ? 2 : 4;
    const hit = hitRef(pos);
    if (hit !== -1) State.dragging = hit;
    else if (State.refPoints.length < req) {
      State.refPoints.push(pos);
      setMode(`วางจุดอ้างอิง ${State.refPoints.length}/${req}`, REF_COLOR);
      if (State.refPoints.length === req) processCalibration();
    } else toast('วางจุดครบแล้ว ลากจุดเดิมเพื่อปรับตำแหน่ง หรือกด "วางจุดอ้างอิง" อีกครั้งเพื่อเริ่มวางใหม่', 'info', 3500);
    render();
  } else if (State.mode === 'drawing') {
    State.drawingDrag = true;
    State.currentPath.push(pos);
    render();
  }
});
canvas.addEventListener('pointermove', (e) => {
  if (!State.image) return;
  const pos = getPos(e);
  if (State.mode === 'calibrating') {
    const hit = hitRef(pos);
    if (hit !== State.hovered) { State.hovered = hit; render(); }
    canvas.className = State.dragging !== -1 ? 'c-grabbing' : hit !== -1 ? 'c-grab' : 'c-cross';
    if (State.dragging !== -1) {
      State.refPoints[State.dragging] = pos;
      if (State.refPoints.length === (State.refMode === 'planar' ? 2 : 4)) processCalibration(true);
      render();
    }
  } else if (State.mode === 'drawing') {
    canvas.className = 'c-cross';
    if (State.drawingDrag && e.buttons) {
      const last = State.currentPath[State.currentPath.length - 1];
      if (last && dist(last, pos) > 8 * screenScale()) { State.currentPath.push(pos); render(); }
    }
  } else canvas.className = 'c-default';
});
const endPointer = () => {
  if (State.dragging !== -1) { State.dragging = -1; if (State.calibrated) toast('ปรับจุดอ้างอิงแล้ว ค่าทั้งหมดถูกคำนวณใหม่', 'success', 2000); render(); }
  State.drawingDrag = false;
};
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);

function processCalibration(silent = false) {
  try {
    if (State.refMode === 'planar') {
      const px = dist(State.refPoints[0], State.refPoints[1]);
      const real = parseFloat($('ref-len').value);
      if (!(real > 0)) throw new Error('len');
      if (px < 3) throw new Error('short');
      State.pxPerCm = px / real; State.calibType = 'planar';
    } else {
      const w = parseFloat($('ref-w').value), h = parseFloat($('ref-h').value);
      if (!(w > 0 && h > 0)) throw new Error('len');
      State.H_Matrix = calculateHomography(State.refPoints, w, h); State.calibType = 'perspective';
    }
  } catch (err) {
    if (!silent) toast(err.message === 'len' ? 'กรุณาใส่ขนาดจริงของวัตถุอ้างอิงให้ถูกต้อง' : 'จุดอ้างอิงไม่เป็นรูปสี่เหลี่ยมที่ถูกต้อง กรุณาลากจุดให้อยู่ที่มุมของแผ่นอ้างอิง', 'error');
    State.calibrated = false; updateCalStatus(); recalculateAllMeasurements(); return;
  }
  State.calibrated = true; markDirty();
  recalculateAllMeasurements(); updateCalStatus();
  if (!silent) {
    setMode('อ้างอิงขนาดสำเร็จ', 'var(--ok)');
    toast('กำหนดขนาดอ้างอิงแล้ว เลือกเครื่องมือวัดหรือให้ AI ตรวจจับได้เลย', 'success');
    updateInstruction('เลือกเครื่องมือในขั้นที่ 3 เพื่อวัดด้วยมือ หรือกดตรวจจับด้วย AI ในขั้นที่ 4');
  }
  updateUi(); render();
}
['ref-len', 'ref-w', 'ref-h'].forEach((id) => $(id).addEventListener('change', () => {
  if (State.refPoints.length === (State.refMode === 'planar' ? 2 : 4) && State.calibType === State.refMode) processCalibration(true);
}));

function updateCalStatus() {
  const el = $('cal-status');
  if (State.calibrated) {
    el.className = 'cal-status ok';
    el.innerHTML = `<i data-lucide="circle-check"></i>อ้างอิงแล้ว: ${State.calibType === 'planar' ? 'ภาพตรง 2 จุด' : 'ภาพเอียง 4 จุด (Homography)'}`;
  } else {
    el.className = 'cal-status';
    el.innerHTML = '<i data-lucide="circle-dashed"></i>ยังไม่ได้กำหนดขนาด';
  }
  icons();
}

// ---------------------------------------------------------------- drawing actions
function undoPoint() { if (State.currentPath.length) { State.currentPath.pop(); render(); } }
function cancelDrawing() { State.currentPath = []; render(); }
function finishDrawing() {
  const n = State.currentPath.length;
  if (State.tool !== 'area' && n < 2) return toast('เส้นสั้นเกินไป ต้องมีอย่างน้อย 2 จุด', 'error');
  if (State.tool === 'area' && n < 3) return toast('พื้นที่ต้องมีอย่างน้อย 3 จุด', 'error');
  const rec = { id: newId(), source: 'manual', points: [...State.currentPath] };
  if (State.tool === 'crack') { rec.width = 0; rec.depth = 0; } else rec.desc = '-';
  openRecordDialog(State.tool, rec, true);
}

function openRecordDialog(type, rec, isNew) {
  const T = TYPES[type];
  $('rec-title').textContent = (isNew ? 'บันทึกข้อมูล' : 'แก้ไขข้อมูล') + T.name;
  let body = '';
  if (type === 'crack') {
    const est = rec.widthEst ? `<div class="hint">AI ประมาณจากภาพ ≈ ${fmt(rec.widthEst, 2)} mm ควรวัดจริงด้วย crack gauge แล้วกรอกค่าที่วัดได้</div>` : '';
    body = `<div class="field"><label for="mod-crack-w">ความกว้างรอยร้าว (mm)</label><input type="number" class="input" id="mod-crack-w" step="0.05" min="0" placeholder="เช่น 0.2" value="${rec.width || ''}">${est}</div>
            <div class="field"><label for="mod-crack-d">ความลึก (mm) ถ้ามี</label><input type="number" class="input" id="mod-crack-d" step="0.1" min="0" placeholder="เช่น 5" value="${rec.depth || ''}"></div>`;
  } else {
    const ph = type === 'area' ? 'เช่น คอนกรีตหลุดร่อนถึงเหล็กเสริม' : 'เช่น เป็นสนิมขุม หน้าตัดลดลง';
    body = `<div class="field"><label for="mod-desc">${type === 'area' ? 'รายละเอียด / สาเหตุ' : 'รายละเอียดความเสียหาย'}</label><input class="input" id="mod-desc" maxlength="300" placeholder="${ph}" value="${esc(rec.desc && rec.desc !== '-' ? rec.desc : '')}"></div>`;
  }
  $('rec-body').innerHTML = body;
  $('rec-cancel').textContent = isNew ? 'ยกเลิก (ลบเส้น)' : 'ยกเลิก';
  const close = () => closeDlg('dlg-record');
  $('rec-cancel').onclick = () => { close(); if (isNew) { State.currentPath = []; render(); } };
  $('dlg-record')._dismiss = $('rec-cancel').onclick;
  const save = () => {
    if (type === 'crack') {
      rec.width = Math.max(0, parseFloat($('mod-crack-w').value) || 0);
      rec.depth = Math.max(0, parseFloat($('mod-crack-d').value) || 0);
    } else rec.desc = $('mod-desc').value.trim() || '-';
    if (!isNew && rec.source === 'ai') rec.reviewed = true;
    computeRecord(type, rec);
    if (isNew) { State.records[type].push(rec); State.currentPath = []; }
    markDirty();
    close(); render(); updateDashboard();
    toast(isNew ? 'บันทึกข้อมูลแล้ว' : 'แก้ไขข้อมูลแล้ว', 'success', 2000);
  };
  $('rec-ok').onclick = save;
  openDlg('dlg-record');
  setTimeout(() => { const i = $('rec-body').querySelector('input'); if (i) i.focus(); }, 60);
}

function editRecord(type, id) { const r = State.records[type].find((x) => x.id === id); if (r) openRecordDialog(type, r, false); }
function deleteRecord(type, id) {
  State.records[type] = State.records[type].filter((x) => x.id !== id);
  markDirty(); render(); updateDashboard(); toast('ลบรายการแล้ว', 'info', 2000);
}
function markDirty() { State.dirty = true; if (window.Library) Library.schedule(); }
const hasRecords = () => Object.values(State.records).some((a) => a.length);
const aiCount = () => sum(Object.values(State.records).map((a) => a.filter((r) => r.source === 'ai').length));

document.addEventListener('keydown', (e) => {
  const dlg = topDialog();
  if (e.key === 'Escape' && dlg) {
    if (dlg._dismiss) dlg._dismiss(); else closeDlg(dlg.id);
    return;
  }
  if (e.key === 'Enter' && dlg && dlg.id === 'dlg-record' && !(e.target.tagName === 'BUTTON' && dlg.contains(e.target))) { e.preventDefault(); $('rec-ok').click(); return; }
  if (dlg || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  if (State.mode !== 'drawing') return;
  if (e.key === 'Enter' && State.currentPath.length) { e.preventDefault(); finishDrawing(); }
  else if ((e.key === 'Backspace' || e.key === 'Delete' || (e.key === 'z' && (e.ctrlKey || e.metaKey))) && State.currentPath.length) { e.preventDefault(); undoPoint(); }
  else if (e.key === 'Escape') cancelDrawing();
});

// ---------------------------------------------------------------- render
function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
function render() {
  if (!State.image) return;
  const W = canvas.width, H = canvas.height;
  const u = Math.max(1, Math.min(W, H) / 700); // หน่วยปรับขนาดเส้น/ตัวอักษรตามขนาดภาพ
  ctx.setLineDash([]);
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(State.image, 0, 0, W, H);

  const path = (pts, close) => {
    ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    if (close) ctx.closePath();
  };
  const strokeShadow = (color, lw, dash) => {
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.setLineDash(dash || []);
    ctx.strokeStyle = 'rgba(255,255,255,.85)'; ctx.lineWidth = lw + 2.5 * u; ctx.stroke();
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke();
    ctx.setLineDash([]);
  };
  const label = (text, x, y, color) => {
    ctx.font = `600 ${Math.round(14 * u)}px "IBM Plex Sans Thai", "Leelawadee UI", Tahoma, sans-serif`;
    const pad = 5 * u, w = ctx.measureText(text).width + pad * 2, h = 20 * u;
    const bx = Math.min(Math.max(x - w / 2, 2), W - w - 2), by = Math.min(Math.max(y - h - 6 * u, 2), H - h - 2);
    ctx.fillStyle = 'rgba(255,255,255,.95)'; ctx.fillRect(bx, by, w, h);
    ctx.fillStyle = color; ctx.fillRect(bx, by, 3 * u, h);
    ctx.fillStyle = '#1E2228'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(text, bx + pad + 1.5 * u, by + h / 2 + u);
  };

  for (const [type, T] of Object.entries(TYPES)) {
    State.records[type].forEach((r, idx) => {
      if (r.source === 'ai' && !State.showAi) return;
      if (r.polys) {
        r.polys.forEach((poly) => {
          if (poly.length < 3) return;
          path(poly, true);
          ctx.fillStyle = rgba(T.color, 0.28); ctx.fill();
          strokeShadow(T.color, 1.6 * u, r.isBox ? [6 * u, 4 * u] : null);
        });
      } else if (r.points.length >= 2) {
        path(r.points, type === 'area');
        if (type === 'area') { ctx.fillStyle = rgba(T.color, 0.25); ctx.fill(); }
        strokeShadow(T.color, 3 * u);
      }
      const a = labelAnchor(r);
      if (a) label(`${T.prefix}${idx + 1}${r.source === 'ai' ? ` ${Math.round((r.conf || 0) * 100)}%` : ''}`, a.x, a.y, T.color);
    });
  }

  if (State.currentPath.length) {
    const c = State.tool ? TYPES[State.tool].color : '#ffffff';
    if (State.currentPath.length >= 2) {
      path(State.currentPath, State.tool === 'area');
      if (State.tool === 'area' && State.currentPath.length > 2) { ctx.fillStyle = rgba(c, 0.18); ctx.fill(); }
      strokeShadow(c, 3 * u, [8 * u, 5 * u]);
    }
    State.currentPath.forEach((p) => { ctx.beginPath(); ctx.arc(p.x, p.y, 3.5 * u, 0, Math.PI * 2); ctx.fillStyle = c; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2 * u; ctx.stroke(); });
  }

  // จุดอ้างอิง + ป้ายขนาดอ้างอิง (คงอยู่ในภาพที่ส่งออก)
  const rp = State.refPoints;
  if (rp.length) {
    if (State.refMode === 'perspective' && rp.length === 4) {
      const s = sortRectCorners(rp);
      path(s, true); strokeShadow(REF_COLOR, 2.5 * u, State.mode === 'calibrating' ? [7 * u, 5 * u] : null);
      if (State.calibrated && State.calibType === 'perspective') {
        label(`Ref ${$('ref-w').value}×${$('ref-h').value} cm`, sum(s.map((p) => p.x)) / 4, sum(s.map((p) => p.y)) / 4, REF_COLOR);
      }
    } else if (State.refMode === 'planar' && rp.length === 2) {
      path(rp, false); strokeShadow(REF_COLOR, 2.5 * u);
      if (State.calibrated && State.calibType === 'planar') label(`Ref ${$('ref-len').value} cm`, (rp[0].x + rp[1].x) / 2, (rp[0].y + rp[1].y) / 2, REF_COLOR);
    }
    if (State.mode === 'calibrating') {
      rp.forEach((p, i) => {
        const hot = i === State.hovered || i === State.dragging;
        ctx.beginPath(); ctx.arc(p.x, p.y, (hot ? 13 : 9) * u, 0, Math.PI * 2);
        ctx.fillStyle = hot ? 'rgba(37,99,235,.25)' : 'rgba(0,0,0,.18)'; ctx.fill();
        ctx.beginPath(); ctx.arc(p.x, p.y, 7 * u, 0, Math.PI * 2);
        ctx.fillStyle = REF_COLOR; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2 * u; ctx.stroke();
        ctx.fillStyle = '#fff'; ctx.font = `700 ${Math.round(9 * u)}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(String(i + 1), p.x, p.y + 0.5 * u);
      });
    }
  }
}

// ---------------------------------------------------------------- dashboard
function widthKey(c) {
  if (c.width > 0) return { key: `${c.width} mm`, n: c.width };
  if (c.widthEst > 0) { const v = Math.round(c.widthEst * 10) / 10; return { key: `≈ ${v.toFixed(1)} mm (AI ประมาณ)`, n: v + 0.001 }; }
  return { key: 'ไม่ระบุ', n: Infinity };
}
function srcTag(r) { return r.source === 'ai' ? `<span class="tag ai">AI ${Math.round((r.conf || 0) * 100)}%</span>` : '<span class="tag">วัดมือ</span>'; }
function actBtns(type, r) {
  return `<button class="icon-btn" title="แก้ไข" aria-label="แก้ไข" onclick="editRecord('${type}',${r.id})"><i data-lucide="pencil"></i></button>` +
    `<button class="icon-btn" title="ลบ" aria-label="ลบ" onclick="deleteRecord('${type}',${r.id})" style="color:var(--err)"><i data-lucide="trash-2"></i></button>`;
}
function updateDashboard() {
  const R = State.records;
  const empty = (n) => `<tr class="empty-row"><td colspan="${n}">ยังไม่มีข้อมูล</td></tr>`;
  const groups = {};
  let tC = 0, tA = 0, tR = 0;
  $('tb-raw-crack').innerHTML = R.crack.length ? R.crack.map((c, i) => {
    tC += c.realLen || 0;
    const k = widthKey(c); (groups[k.key] = groups[k.key] || { n: k.n, v: 0 }).v += c.realLen || 0;
    const w = c.width > 0 ? fmt(c.width, 2).replace(/\.?0+$/, '') : c.widthEst > 0 ? `<span class="est" title="ประมาณจากภาพด้วย AI">≈ ${fmt(c.widthEst, 2)}</span>` : '-';
    return `<tr><td><b>C${i + 1}</b></td><td>${srcTag(c)}</td><td class="num">${w}</td><td class="num">${c.depth > 0 ? c.depth : '-'}</td><td class="r num"><b>${fmt(c.realLen)}</b></td><td class="c">${actBtns('crack', c)}</td></tr>`;
  }).join('') : empty(6);
  $('tb-raw-area').innerHTML = R.area.length ? R.area.map((a, i) => {
    tA += a.realArea || 0;
    return `<tr><td><b>A${i + 1}</b></td><td>${srcTag(a)}</td><td class="desc-cell" title="${esc(a.desc)}">${esc(a.desc)}</td><td class="r num"><b>${fmt(a.realArea)}</b></td><td class="c">${actBtns('area', a)}</td></tr>`;
  }).join('') : empty(5);
  $('tb-raw-rebar').innerHTML = R.rebar.length ? R.rebar.map((r, i) => {
    tR += r.realLen || 0;
    return `<tr><td><b>R${i + 1}</b></td><td>${srcTag(r)}</td><td class="desc-cell" title="${esc(r.desc)}">${esc(r.desc)}</td><td class="r num"><b>${fmt(r.realLen)}</b></td><td class="c">${actBtns('rebar', r)}</td></tr>`;
  }).join('') : empty(5);
  $('cnt-crack').textContent = R.crack.length ? `${R.crack.length} รายการ` : '';
  $('cnt-area').textContent = R.area.length ? `${R.area.length} รายการ` : '';
  $('cnt-rebar').textContent = R.rebar.length ? `${R.rebar.length} รายการ` : '';
  $('total-crack-len').textContent = tC.toFixed(2);
  $('total-area').textContent = tA.toFixed(2);
  $('total-rebar').textContent = tR.toFixed(2);
  const keys = Object.keys(groups).sort((a, b) => groups[a].n - groups[b].n);
  $('tb-summary-crack').innerHTML = keys.length
    ? keys.map((k) => `<tr><td>${esc(k)}</td><td class="r num"><b>${groups[k].v.toFixed(2)}</b></td></tr>`).join('') +
      `<tr><td class="r" style="color:var(--ink-2)">รวม</td><td class="r num" style="color:var(--crack)"><b>${tC.toFixed(2)}</b></td></tr>`
    : empty(2);
  $('ai-controls').classList.toggle('hidden', aiCount() === 0);
  icons();
  updateUi();
}
function crackGroups() {
  const g = {};
  State.records.crack.forEach((c) => { const k = widthKey(c); (g[k.key] = g[k.key] || { n: k.n, v: 0 }).v += c.realLen || 0; });
  return Object.keys(g).sort((a, b) => g[a].n - g[b].n).map((k) => [k, g[k].v]);
}

function updateUi() {
  const img = !!State.image, cal = State.calibrated, ok = State.server.ok;
  const info = State.server.info || {};
  const steps = [1, 2, 3, 4].map((n) => $(`step-${n}`));
  steps.forEach((s) => s.classList.remove('current', 'done'));
  if (img) steps[0].classList.add('done');
  if (cal) steps[1].classList.add('done');
  const cur = !img ? 0 : State.mode === 'calibrating' || !cal ? 1 : State.tool ? 2 : ok && info.active_model ? 3 : 2;
  steps[cur].classList.add('current');
  if (State.mode === 'idle') setMode(img ? (cal ? 'พร้อมวัด' : 'รอวางจุดอ้างอิง') : 'รอภาพถ่าย', img && cal ? 'var(--ok)' : null);
  $('btn-detect').disabled = !(ok && info.active_model && img);
  $('btn-add-ds').disabled = !img;
  $('btn-train').disabled = !ok;
  const b = $('btn-ai-layer');
  if (b._show !== State.showAi) {
    b._show = State.showAi;
    b.innerHTML = State.showAi ? '<i data-lucide="eye-off"></i><span>ซ่อนผล AI</span>' : '<i data-lucide="eye"></i><span>แสดงผล AI</span>';
    icons();
  }
}

// ---------------------------------------------------------------- project save / load
function imageDataUrl(type = 'image/jpeg', q = 0.92) {
  const c = document.createElement('canvas');
  c.width = canvas.width; c.height = canvas.height;
  c.getContext('2d').drawImage(State.image, 0, 0, c.width, c.height);
  return c.toDataURL(type, q);
}
function projectSnapshot() {
  return {
    metadata: { bldg: $('meta-bldg').value, floor: $('meta-floor').value, loc: $('meta-loc').value, imageName: State.imageName, aiModel: State.aiModel },
    config: { refMode: State.refMode, planarLen: $('ref-len').value, perspW: $('ref-w').value, perspH: $('ref-h').value },
    state: { calibrated: State.calibrated, calibType: State.calibType, pxPerCm: State.pxPerCm, H_Matrix: State.H_Matrix, refPoints: State.refPoints, records: State.records },
  };
}
function saveProject() {
  if (!State.image) return toast('ยังไม่มีภาพหรือโปรเจกต์ให้บันทึก', 'error');
  const data = { app: 'crack-ai', version: 2, ...projectSnapshot(), imageData: imageDataUrl() };
  const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
  saveAs(blob, `Inspection_${safeName($('meta-bldg').value || 'project')}_${new Date().toISOString().slice(0, 10)}.json`);
  if (window.Library && LS.get('autoLib', true)) Library.saveCurrent();
  State.dirty = false;
  toast('บันทึกโปรเจกต์แล้ว', 'success');
}
/** คืนค่าโปรเจกต์ (จากไฟล์ .json หรือจากคลัง) ลงหน้าจอ */
function restoreProject(data, img, name) {
  const m = data.metadata || {}, c = data.config || {}, s = data.state || {};
  $('meta-bldg').value = m.bldg || ''; $('meta-floor').value = m.floor || ''; $('meta-loc').value = m.loc || '';
  resetInteraction();
  setImage(img, name || m.imageName);
  $('ref-len').value = c.planarLen || 10; $('ref-w').value = c.perspW || 21; $('ref-h').value = c.perspH || 29.7;
  setRefMode(c.refMode === 'perspective' ? 'perspective' : 'planar');
  State.calibrated = !!s.calibrated; State.calibType = s.calibType || null; State.pxPerCm = s.pxPerCm || 1;
  State.H_Matrix = s.H_Matrix || null; State.refPoints = Array.isArray(s.refPoints) ? s.refPoints : [];
  const recs = s.records || {};
  State.records = { crack: [], area: [], rebar: [] };
  for (const t of Object.keys(TYPES)) {
    State.records[t] = (Array.isArray(recs[t]) ? recs[t] : []).filter((r) => (r.points && r.points.length) || (r.polys && r.polys.length))
      .map((r) => ({ source: 'manual', ...r, id: r.id || newId() }));
  }
  State.aiModel = m.aiModel || null; State.dirty = false; State.libraryId = null; State.libSavedAt = 0;
  updateCalStatus(); recalculateAllMeasurements(); render();
}
$('projectUpload').addEventListener('change', (event) => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    let data;
    try { data = JSON.parse(e.target.result); if (!data.imageData || !data.state) throw 0; } catch { return toast('ไฟล์โปรเจกต์ไม่ถูกต้อง', 'error'); }
    const img = new Image();
    img.onload = () => {
      if (State.imageUrl) { URL.revokeObjectURL(State.imageUrl); State.imageUrl = null; }
      restoreProject(data, img, (data.metadata || {}).imageName || file.name);
      if (window.Library) { Library.statusLine(); if (hasRecords()) Library.schedule(); } // ภาพเดียวกันจะรวมกับโปรเจกต์เดิมในคลัง
      toast('เปิดโปรเจกต์แล้ว', 'success');
    };
    img.onerror = () => toast('ภาพในไฟล์โปรเจกต์เสียหาย', 'error');
    img.src = data.imageData;
  };
  reader.readAsText(file);
});
window.addEventListener('beforeunload', (e) => { if (State.dirty && hasRecords()) { e.preventDefault(); e.returnValue = ''; } });
const safeName = (s) => String(s).replace(/[\\/:*?"<>|\s]+/g, '_').slice(0, 60) || 'x';

// ---------------------------------------------------------------- Excel
async function logoDataUrl() {
  const img = $('logo-img');
  if (img.classList.contains('hidden') || !img.naturalWidth) return null;
  try {
    const c = document.createElement('canvas');
    const s = Math.min(1, 240 / Math.max(img.naturalWidth, img.naturalHeight));
    c.width = Math.round(img.naturalWidth * s); c.height = Math.round(img.naturalHeight * s);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    return { data: c.toDataURL('image/png'), w: c.width, h: c.height };
  } catch { return null; } // ภาพจาก file:// อาจถูกเบราว์เซอร์บล็อก
}
async function exportExcel() {
  if (!hasRecords()) return toast('ยังไม่มีข้อมูลสำหรับส่งออก', 'error');
  if (typeof ExcelJS === 'undefined') return toast('โหลดไลบรารี Excel ไม่สำเร็จ ตรวจสอบโฟลเดอร์ vendor', 'error');
  toast('กำลังสร้างไฟล์ Excel', 'info', 2000);
  const bldg = $('meta-bldg').value || '-', flr = $('meta-floor').value || '-', loc = $('meta-loc').value || '-';
  const wb = new ExcelJS.Workbook();
  wb.creator = 'Structural Inspection System';
  const ws = wb.addWorksheet('Summary');
  ws.columns = [{ width: 34 }, { width: 22 }, { width: 14 }, { width: 22 }];
  const logo = await logoDataUrl();
  let r0 = 1;
  if (logo) {
    const id = wb.addImage({ base64: logo.data, extension: 'png' });
    const h = 48, w = (logo.w / logo.h) * h;
    ws.addImage(id, { tl: { col: 0, row: 0 }, ext: { width: w, height: h } });
    ws.getRow(1).height = 40; ws.getRow(2).height = 4; r0 = 3;
    for (let i = 1; i < r0; i++) ws.addRow([]);
  }
  ws.addRow(['รายงานหาปริมาณความเสียหายโครงสร้างจากภาพถ่าย']);
  ws.mergeCells(`A${r0}:D${r0}`);
  ws.getCell(`A${r0}`).font = { size: 14, bold: true };
  ws.addRow(['อาคาร:', bldg, 'ชั้น:', flr]);
  ws.addRow(['ตำแหน่ง:', loc, 'วันที่:', new Date().toLocaleDateString('th-TH')]);
  ws.addRow(['การอ้างอิงขนาด:', State.calibrated ? (State.calibType === 'planar' ? `ภาพตรง 2 จุด (${$('ref-len').value} cm)` : `ภาพเอียง 4 จุด (${$('ref-w').value}×${$('ref-h').value} cm)`) : 'ยังไม่ได้กำหนด']);
  if (aiCount()) ws.addRow(['โมเดล AI ที่ใช้:', State.aiModel || '-', 'จำนวนผล AI:', aiCount()]);
  ws.addRow([]);
  ws.addRow(['สรุปรวมทั้งหมด']).font = { bold: true };
  ws.addRow(['รวมความยาวรอยร้าว (cm)', Number($('total-crack-len').textContent)]);
  ws.addRow(['รวมพื้นที่เสียหาย (ตร.ซม.)', Number($('total-area').textContent)]);
  ws.addRow(['รวมความยาวเหล็กเสริม (cm)', Number($('total-rebar').textContent)]);
  ws.addRow([]);
  ws.addRow(['ความยาวรอยร้าว แยกตามความกว้าง']).font = { bold: true };
  ws.addRow(['ความกว้าง', 'ความยาวรวม (cm)']).font = { bold: true };
  crackGroups().forEach(([k, v]) => ws.addRow([k, Number(v.toFixed(2))]));
  ws.addRow([]);
  ws.addRow(['ภาพถ่ายประกอบการตรวจสอบ']).font = { bold: true, size: 12 };
  const imgRow = ws.rowCount; // index 0-based ของแถวถัดไป

  const prevMode = State.mode, prevShow = State.showAi;
  State.mode = 'idle'; State.showAi = true; render();
  const imgData = canvas.toDataURL('image/jpeg', 0.88);
  State.mode = prevMode; State.showAi = prevShow; render();
  const imageId = wb.addImage({ base64: imgData, extension: 'jpeg' });
  const sc = Math.min(1, 720 / canvas.width);
  ws.addImage(imageId, { tl: { col: 0, row: imgRow }, ext: { width: canvas.width * sc, height: canvas.height * sc } });

  const src = (r) => (r.source === 'ai' ? 'AI' : 'วัดมือ');
  const conf = (r) => (r.source === 'ai' ? Math.round((r.conf || 0) * 100) / 100 : '');
  if (State.records.crack.length) {
    const s = wb.addWorksheet('Cracks');
    s.columns = [{ header: 'ID', key: 'id', width: 8 }, { header: 'ที่มา', key: 's', width: 10 }, { header: 'ความมั่นใจ AI', key: 'c', width: 13 },
      { header: 'กว้าง (mm)', key: 'w', width: 12 }, { header: 'กว้างประมาณจาก AI (mm)', key: 'we', width: 22 }, { header: 'ลึก (mm)', key: 'd', width: 10 }, { header: 'ยาว (cm)', key: 'l', width: 12 }];
    s.getRow(1).font = { bold: true };
    State.records.crack.forEach((c, i) => s.addRow({ id: `C${i + 1}`, s: src(c), c: conf(c), w: c.width || '', we: c.widthEst ? Number(c.widthEst.toFixed(2)) : '', d: c.depth || '', l: Number((c.realLen || 0).toFixed(2)) }));
  }
  if (State.records.area.length) {
    const s = wb.addWorksheet('Damaged_Areas');
    s.columns = [{ header: 'ID', key: 'id', width: 8 }, { header: 'ที่มา', key: 's', width: 10 }, { header: 'ความมั่นใจ AI', key: 'c', width: 13 }, { header: 'รายละเอียด', key: 'desc', width: 40 }, { header: 'พื้นที่ (ตร.ซม.)', key: 'a', width: 15 }];
    s.getRow(1).font = { bold: true };
    State.records.area.forEach((a, i) => s.addRow({ id: `A${i + 1}`, s: src(a), c: conf(a), desc: a.desc, a: Number((a.realArea || 0).toFixed(2)) }));
  }
  if (State.records.rebar.length) {
    const s = wb.addWorksheet('Rebar');
    s.columns = [{ header: 'ID', key: 'id', width: 8 }, { header: 'ที่มา', key: 's', width: 10 }, { header: 'ความมั่นใจ AI', key: 'c', width: 13 }, { header: 'รายละเอียด', key: 'desc', width: 40 }, { header: 'ยาว (cm)', key: 'l', width: 12 }];
    s.getRow(1).font = { bold: true };
    State.records.rebar.forEach((r, i) => s.addRow({ id: `R${i + 1}`, s: src(r), c: conf(r), desc: r.desc, l: Number((r.realLen || 0).toFixed(2)) }));
  }
  const buf = await wb.xlsx.writeBuffer();
  saveAs(new Blob([buf], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }), `Inspection_Report_${safeName(bldg)}_${safeName(loc)}.xlsx`);
  toast('ส่งออก Excel แล้ว', 'success');
}

// ---------------------------------------------------------------- logo
function showLogo(src) {
  const img = $('logo-img');
  img.onload = () => { img.classList.remove('hidden'); $('logo-mark').classList.add('hidden'); $('logo-slot').classList.add('has-logo'); };
  img.onerror = () => { img.classList.add('hidden'); $('logo-mark').classList.remove('hidden'); $('logo-slot').classList.remove('has-logo'); if (img._fallback) { const f = img._fallback; img._fallback = null; img.src = f; } };
  img.src = src;
}
function initLogo() {
  const saved = LS.get('logo', null);
  const img = $('logo-img');
  if (saved) showLogo(saved);
  else { img._fallback = 'assets/logo.svg'; showLogo('assets/logo.png'); }
}
$('logo-slot').addEventListener('click', async () => {
  if (LS.get('logo', null)) {
    const a = await askConfirm('โลโก้หน่วยงาน', 'ต้องการเปลี่ยนไฟล์โลโก้ หรือลบโลโก้ที่ตั้งไว้', { yes: 'เลือกไฟล์ใหม่', no: 'ลบโลโก้' });
    if (a === false) { LS.del('logo'); $('logo-img').classList.add('hidden'); $('logo-mark').classList.remove('hidden'); $('logo-slot').classList.remove('has-logo'); initLogo(); return; }
    if (a !== true) return;
  }
  $('logo-input').click();
});
$('logo-input').addEventListener('change', (e) => {
  const f = e.target.files[0]; e.target.value = '';
  if (!f) return;
  if (f.size > 1.5 * 1024 * 1024) return toast('ไฟล์โลโก้ใหญ่เกิน 1.5 MB', 'error');
  const rd = new FileReader();
  rd.onload = () => { LS.set('logo', rd.result); showLogo(rd.result); toast('ตั้งโลโก้แล้ว', 'success', 2000); };
  rd.readAsDataURL(f);
});

// ================================================================= AI SERVER
function apiBase() {
  const u = ($('api-url').value || '').trim();
  if (u) return u.replace(/\/+$/, '');
  // เปิดจากเซิร์ฟเวอร์ในเครื่อง -> origin เดียวกัน, เปิดจาก GitHub Pages หรือไฟล์ -> เรียก 127.0.0.1:8765
  const local = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);
  return location.protocol === 'http:' && local ? '' : 'http://127.0.0.1:8765';
}
function fmtErr(d) {
  if (!d) return '';
  if (typeof d.detail === 'string') return d.detail;
  if (Array.isArray(d.detail)) return d.detail.map((x) => `${(x.loc || []).slice(-1)[0] || ''}: ${x.msg}`).join(', ');
  return '';
}
async function api(path, { method = 'GET', body, json, timeout = 0 } = {}) {
  const opt = { method, headers: { 'X-CrackAI': '1' } };
  if (json !== undefined) { opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(json); } else if (body) opt.body = body;
  let timer;
  if (timeout) { const c = new AbortController(); opt.signal = c.signal; timer = setTimeout(() => c.abort(), timeout); }
  let res;
  try { res = await fetch(apiBase() + path, opt); } catch { throw new Error('เชื่อมต่อเซิร์ฟเวอร์ AI ไม่ได้'); } finally { clearTimeout(timer); }
  const txt = await res.text();
  let data; try { data = JSON.parse(txt); } catch { data = { detail: txt.slice(0, 300) }; }
  if (!res.ok) throw new Error(fmtErr(data) || `เซิร์ฟเวอร์ตอบกลับผิดพลาด (${res.status})`);
  return data;
}
function apiUpload(path, fd, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open('POST', apiBase() + path);
    x.setRequestHeader('X-CrackAI', '1');
    x.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
    x.onload = () => { let d; try { d = JSON.parse(x.responseText); } catch { d = { detail: x.responseText.slice(0, 300) }; } x.status < 300 ? resolve(d) : reject(new Error(fmtErr(d) || `ผิดพลาด (${x.status})`)); };
    x.onerror = () => reject(new Error('เชื่อมต่อเซิร์ฟเวอร์ AI ไม่ได้'));
    x.send(fd);
  });
}

let lastAiHtml = '';
async function pollHealth() {
  try {
    const h = await api('/api/health', { timeout: 5000 });
    State.server = { ok: true, info: h };
  } catch (e) {
    State.server = { ok: false, info: null, err: e.message };
  }
  renderServerState();
  ensureTrainPolling();
}
function deviceText(info) {
  if (info.cuda && info.devices && info.devices.length) return info.devices.map((d) => `GPU ${d.id} ${d.name}`).join(', ');
  if (info.mps) return 'Apple GPU (MPS)';
  return 'CPU';
}
function renderServerState() {
  const s = State.server, info = s.info || {};
  const tr = lastTrainStatus && lastTrainStatus.state === 'running' ? lastTrainStatus : null;
  $('srv-dot').className = 'dot ' + (!s.ok ? 'err' : info.training ? 'busy' : 'ok');
  $('srv-text').textContent = !s.ok ? 'AI ไม่ได้เชื่อมต่อ' : info.training ? (tr ? `กำลังเทรน ${tr.epoch}/${tr.epochs}` : 'กำลังเทรน') : info.active_model ? 'AI พร้อม' : 'AI ยังไม่มีโมเดล';
  let html;
  if (!s.ok) {
    const remote = location.protocol === 'https:';
    html = `<i data-lucide="server-off"></i><div><b>ยังไม่ได้เชื่อมต่อเซิร์ฟเวอร์ AI</b>บนเครื่องที่มีการ์ดจอ ดับเบิลคลิก <code>start_windows.bat</code> หรือรัน <code>./start_mac_linux.sh</code>` +
      (remote ? ` และเพิ่ม <code>${esc(location.origin)}</code> ในไฟล์ <code>server/config.json</code> แนะนำ Chrome หรือ Edge และกดอนุญาตเมื่อเบราว์เซอร์ถามสิทธิ์เข้าถึงเครือข่ายในเครื่อง` : ' แล้วเปิดหน้านี้ที่ <code>http://127.0.0.1:8765</code>') +
      ' การวัดด้วยมือและคลังงานใช้ได้ตามปกติ</div>';
  } else if (!info.active_model) {
    html = `<i data-lucide="circle-alert"></i><div><b>เชื่อมต่อแล้ว แต่ยังไม่มีโมเดล</b>กด "เทรนโมเดล" เพื่อสร้างโมเดลจากชุดข้อมูลของคุณ หรือนำเข้าไฟล์ .pt ที่มีอยู่ ใช้ ${esc(deviceText(info))}</div>`;
  } else {
    const m = info.active_model;
    html = `<i data-lucide="sparkles"></i><div><b>พร้อมตรวจจับ</b>โมเดล ${esc(m.label)} (${esc((m.names || []).join(', '))}) ใช้ ${esc(deviceText(info))}</div>`;
  }
  if (html !== lastAiHtml) { $('ai-state').innerHTML = html; lastAiHtml = html; icons(); }
  $('ai-model-name').textContent = s.ok && info.active_model ? info.active_model.label : '';
  updateUi();
}
$('srv-chip').addEventListener('click', () => {
  if (State.server.ok) { openTrain(State.server.info.training ? 't-prog' : null); }
  else toast('ยังไม่ได้เปิดเซิร์ฟเวอร์ AI ดูวิธีเปิดในขั้นที่ 4', 'info');
});

// ---- detection
async function imageBlob() {
  if (State.imageBlob) return State.imageBlob;
  const c = document.createElement('canvas');
  c.width = canvas.width; c.height = canvas.height;
  c.getContext('2d').drawImage(State.image, 0, 0, c.width, c.height);
  State.imageBlob = await new Promise((r) => c.toBlob(r, 'image/jpeg', 0.95));
  return State.imageBlob;
}
function setBusy(on, text) { $('canvas-busy').classList.toggle('hidden', !on); if (text) $('canvas-busy-text').textContent = text; }

async function runDetect() {
  if (!State.image) return toast('กรุณาเลือกภาพก่อน', 'error');
  if (aiCount()) {
    const ok = await askConfirm('ตรวจจับใหม่', `ผล AI เดิม ${aiCount()} รายการจะถูกแทนที่ด้วยผลใหม่ รายการที่วัดด้วยมือยังอยู่ครบ`, { yes: 'ตรวจจับใหม่' });
    if (!ok) return;
  }
  const btn = $('btn-detect');
  btn.disabled = true; setBusy(true, 'กำลังตรวจจับความเสียหาย');
  try {
    const fd = new FormData();
    fd.append('image', await imageBlob(), 'image.jpg');
    fd.append('conf', $('ai-conf').value);
    fd.append('tile', $('ai-tile').value);
    fd.append('imgsz', String(parseInt($('ai-imgsz').value, 10) || 0));
    const r = await api('/api/detect', { method: 'POST', body: fd });
    const sx = canvas.width / r.image_size[0], sy = canvas.height / r.image_size[1];
    for (const t of Object.keys(TYPES)) State.records[t] = State.records[t].filter((x) => x.source !== 'ai');
    const cnt = { crack: 0, area: 0, rebar: 0 };
    for (const d of r.detections) {
      const type = TYPES[d.measure] ? d.measure : 'area';
      const rec = {
        id: newId(), source: 'ai', conf: d.conf, clsName: d.name, isBox: !!d.is_box,
        polys: d.polygons.map((poly) => poly.map(([x, y]) => ({ x: x * sx, y: y * sy }))),
      };
      if (type === 'crack') { rec.width = 0; rec.depth = 0; } else rec.desc = d.label || d.name;
      computeRecord(type, rec);
      State.records[type].push(rec); cnt[type]++;
    }
    State.aiModel = r.model; State.showAi = true; markDirty();
    render(); updateDashboard();
    const n = r.detections.length, sec = (r.time_ms / 1000).toFixed(1);
    if (!n) toast(`ไม่พบความเสียหายที่ความมั่นใจ ${$('ai-conf').value} ขึ้นไป ลองลดค่าความมั่นใจ (${sec} วินาที)`, 'info', 6000);
    else toast(`พบ ${n} รายการ: รอยร้าว ${cnt.crack}, พื้นที่ ${cnt.area}, เหล็กเสริม ${cnt.rebar} (${sec} วินาที${r.tiles > 1 ? `, ${r.tiles} ช่อง` : ''})`, 'success', 6000);
    if (n && !State.calibrated) toast('ยังไม่ได้วางจุดอ้างอิงขนาด ค่าความยาวและพื้นที่จะเป็น 0 จนกว่าจะทำขั้นที่ 2', 'info', 7000);
  } catch (e) {
    toast('ตรวจจับไม่สำเร็จ: ' + e.message, 'error', 7000);
  } finally { setBusy(false); updateUi(); }
}
function toggleAiLayer() { State.showAi = !State.showAi; render(); updateUi(); }
async function clearAi() {
  const ok = await askConfirm('ลบผล AI', `ลบผลตรวจจับจาก AI ทั้งหมด ${aiCount()} รายการ รายการที่วัดด้วยมือยังอยู่`, { yes: 'ลบผล AI', danger: true });
  if (!ok) return;
  for (const t of Object.keys(TYPES)) State.records[t] = State.records[t].filter((x) => x.source !== 'ai');
  markDirty(); render(); updateDashboard();
}
$('ai-conf').addEventListener('input', () => { $('ai-conf-out').textContent = Number($('ai-conf').value).toFixed(2); LS.set('conf', $('ai-conf').value); });
$('ai-tile').addEventListener('change', () => LS.set('tile', $('ai-tile').value));
$('ai-imgsz').addEventListener('change', () => LS.set('imgsz', $('ai-imgsz').value));
$('api-url').addEventListener('change', () => { LS.set('api', $('api-url').value.trim()); pollHealth(); });

// ---- add current image to dataset
async function loadClasses() { State.classes = await api('/api/classes'); LS.set('classes', State.classes); return State.classes; }
function classIndexFor(type, r, classes) {
  if (r.clsName) { const i = classes.findIndex((c) => c.name === r.clsName); if (i >= 0) return i; }
  return classes.findIndex((c) => c.measure === type);
}
async function openAddToDataset() {
  if (!State.image) return toast('กรุณาเลือกภาพก่อน', 'error');
  let classes = LS.get('classes', null) || [{ name: 'crack', measure: 'crack' }, { name: 'spalling', measure: 'area' }, { name: 'exposed_rebar', measure: 'rebar' }];
  if (State.server.ok) { try { classes = await loadClasses(); LS.set('classes', classes); } catch { } }
  State.classes = classes;
  const auto = Math.max(3, Math.round(Math.min(canvas.width, canvas.height) * 0.004));
  $('ds-crack-w').value = LS.get('dsCrackW', auto);
  $('ds-rebar-w').value = LS.get('dsRebarW', auto * 3);
  const R = State.records, parts = [];
  const miss = [];
  for (const t of Object.keys(TYPES)) {
    if (!R[t].length) continue;
    const ai = R[t].filter((x) => x.source === 'ai').length;
    parts.push(`${TYPES[t].name} ${R[t].length}${ai ? ` (AI ${ai})` : ''}`);
    if (R[t].some((r) => classIndexFor(t, r, classes) < 0)) miss.push(TYPES[t].name);
  }
  $('ds-summary').className = parts.length ? 'msg ok' : 'msg info';
  $('ds-summary').innerHTML = parts.length
    ? `<i data-lucide="circle-check"></i><span>ภาพนี้จะถูกเพิ่มพร้อมป้ายกำกับ ${esc(parts.join(', '))}</span>`
    : '<i data-lucide="info"></i><span>ภาพนี้ยังไม่มีรายการใด จะถูกเพิ่มเป็น "ภาพพื้นหลัง" (ไม่มีความเสียหาย) ซึ่งช่วยลดการตรวจจับผิด</span>';
  $('ds-warn').classList.toggle('hidden', !miss.length);
  $('ds-warn').innerHTML = miss.length ? `<i data-lucide="triangle-alert"></i><span>ยังไม่มีคลาสสำหรับ ${esc(miss.join(', '))} ในหน้าตั้งค่าคลาส รายการเหล่านี้จะไม่ถูกเพิ่ม</span>` : '';
  icons(); openDlg('dlg-ds');
}
/** แปลงรายการวัดเป็นป้ายกำกับ YOLO-seg (พิกัด 0-1) */
function buildLabels(records, W, H, classes, cw, rw) {
  const labels = [];
  const norm = (poly) => poly.map((p) => [Math.min(1, Math.max(0, p.x / W)), Math.min(1, Math.max(0, p.y / H))]);
  for (const t of Object.keys(TYPES)) {
    for (const r of records[t] || []) {
      const cls = classIndexFor(t, r, classes);
      if (cls < 0) continue;
      if (r.polys) r.polys.forEach((p) => { if (p.length >= 3) labels.push({ cls, points: norm(p) }); });
      else if (t === 'area') { if (r.points.length >= 3) labels.push({ cls, points: norm(r.points) }); }
      else { const poly = bufferPolyline(r.points, t === 'crack' ? cw : rw); if (poly) labels.push({ cls, points: norm(poly) }); }
    }
  }
  return labels;
}
async function submitAnnotation() {
  const cw = Math.max(2, parseFloat($('ds-crack-w').value) || 4), rw = Math.max(2, parseFloat($('ds-rebar-w').value) || 12);
  LS.set('dsCrackW', cw); LS.set('dsRebarW', rw);
  const btn = $('ds-ok'); btn.disabled = true;
  try {
    const e = await Library.saveCurrent({ useForTraining: true, labelWidths: { crack: cw, rebar: rw } });
    if (!e) return;
    closeDlg('dlg-ds');
    if (State.server.ok) await Library.syncServer();
    else toast('เก็บเข้าคลังและเลือก "ใช้เทรน" แล้ว จะส่งเข้าชุดเทรนเมื่อเชื่อมต่อเซิร์ฟเวอร์ AI', 'success', 6000);
  } finally { btn.disabled = false; }
}

// ================================================================= TRAINING DIALOG
const HYP_DEFAULT = {
  family: 'yolo26', size: 's', task: 'segment', epochs: 150, imgsz: 1024, batch: -1, device: 'auto', val: 20, patience: 50, base: '',
  lr0: 0.01, optimizer: 'auto', workers: 4, cache: '', seed: 0, close_mosaic: 10,
  fliplr: 0.5, flipud: 0.5, degrees: 10, translate: 0.1, scale: 0.5, mosaic: 1.0, hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4,
  amp: true, cos_lr: false, plots: true,
};
const PRESETS = {
  quick: { size: 'n', epochs: 50, imgsz: 640, batch: 16, patience: 20 },
  standard: { size: 's', epochs: 150, imgsz: 1024, batch: -1, patience: 50 },
  fine: { size: 'm', epochs: 300, imgsz: 1280, batch: -1, patience: 80 },
};
const HYP_KEYS = Object.keys(HYP_DEFAULT);
const MEASURE_LABEL = { crack: 'รอยร้าว: วัดความยาว', area: 'พื้นที่เสียหาย: วัดพื้นที่', rebar: 'เหล็กเสริม: วัดความยาว' };
const PALETTE = ['#D9363E', '#E8730C', '#2F8F5B', '#2563EB', '#7C3AED', '#0891B2', '#B45309', '#DB2777'];
let dsStats = null, modelsCache = null, classRows = [], classesDirty = false;

function setHypForm(h) {
  for (const k of HYP_KEYS) {
    const el = $('h-' + k); if (!el) continue;
    if (el.type === 'checkbox') el.checked = !!h[k];
    else {
      if (el.tagName === 'SELECT' && ![...el.options].some((o) => o.value === String(h[k]))) continue;
      el.value = h[k];
    }
  }
  markPreset();
}
function readHypForm() {
  const h = {};
  for (const k of HYP_KEYS) {
    const el = $('h-' + k); if (!el) continue;
    if (el.type === 'checkbox') h[k] = el.checked;
    else if (el.type === 'number' || k === 'imgsz') h[k] = el.value === '' ? HYP_DEFAULT[k] : Number(el.value);
    else h[k] = el.value;
  }
  return h;
}
function markPreset() {
  const h = readHypForm();
  document.querySelectorAll('#presets button').forEach((b) => {
    const p = PRESETS[b.dataset.p];
    b.classList.toggle('on', Object.keys(p).every((k) => String(h[k]) === String(p[k])));
  });
}
function resetHyp() { setHypForm(HYP_DEFAULT); LS.set('hyp', readHypForm()); cfgCheck(); }
document.querySelectorAll('#presets button').forEach((b) => b.addEventListener('click', () => {
  setHypForm({ ...readHypForm(), ...PRESETS[b.dataset.p] }); LS.set('hyp', readHypForm()); cfgCheck();
}));
document.querySelector('#t-cfg').addEventListener('change', () => { LS.set('hyp', readHypForm()); markPreset(); cfgCheck(); });

document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab)));
function switchTab(id) {
  document.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === id));
  document.querySelectorAll('.tab-pane').forEach((p) => p.classList.toggle('on', p.id === id));
  if (id === 't-cfg') cfgCheck();
  if (id === 't-prog') { pollTrain(); loadModels(); }
}

async function openTrain(tab) {
  if (!State.server.ok) return toast('ยังไม่ได้เชื่อมต่อเซิร์ฟเวอร์ AI', 'error');
  openDlg('dlg-train');
  const info = State.server.info || {};
  const dev = $('h-device');
  dev.innerHTML = '<option value="auto">อัตโนมัติ</option>' +
    (info.devices || []).map((d) => `<option value="${esc(d.id)}">GPU ${esc(d.id)}: ${esc(d.name)} (${d.mem_gb} GB)</option>`).join('') +
    (info.mps ? '<option value="mps">Apple GPU (MPS)</option>' : '') + '<option value="cpu">CPU (ช้ามาก)</option>';
  setHypForm({ ...HYP_DEFAULT, ...LS.get('hyp', {}) });
  try {
    await Promise.all([loadClasses().then(() => { classRows = State.classes.map((c) => ({ ...c })); classesDirty = false; renderClassRows(); }), refreshDataset(), loadModels()]);
  } catch (e) { toast(e.message, 'error'); }
  cfgCheck(); if (window.Library) Library.trainCard();
  switchTab(tab || (info.training ? 't-prog' : document.querySelector('.tabs button.on').dataset.tab));
}

// ---- dataset tab
async function refreshDataset() { dsStats = await api('/api/dataset'); renderDataset(); }
function renderDataset() {
  const s = dsStats; if (!s) return;
  $('ds-images').textContent = s.images; $('ds-labeled').textContent = s.labeled; $('ds-bg').textContent = s.background;
  $('ds-type').textContent = s.seg_lines && s.box_lines ? `Polygon ${s.seg_lines}, กรอบ ${s.box_lines}` : s.seg_lines ? `Polygon (${s.seg_lines})` : s.box_lines ? `กรอบ (${s.box_lines})` : '-';
  const cls = classRows.length ? classRows : State.classes || [];
  const ids = new Set([...cls.map((_, i) => String(i)), ...Object.keys(s.instances)]);
  $('ds-inst').innerHTML = [...ids].sort((a, b) => a - b).map((i) => {
    const c = cls[i];
    const n = s.instances[i] || 0;
    return `<tr><td class="num">${i}</td><td>${c ? `<span class="tag" style="background:${rgba(c.color, 0.14)};color:var(--ink)">${esc(c.label || c.name)}</span> <span class="hint">${esc(c.name)}</span>` : '<span style="color:var(--err)">ไม่มีคลาสนี้ในตั้งค่า</span>'}</td><td class="r num">${n}${n === 0 && s.labeled ? ' <span class="hint">ยังไม่มีตัวอย่าง</span>' : ''}</td></tr>`;
  }).join('') || '<tr class="empty-row"><td colspan="3">ยังไม่มีข้อมูล</td></tr>';
  $('ds-warns').innerHTML = s.warnings.map((w) => `<div class="msg warn"><i data-lucide="triangle-alert"></i><span>${esc(w)}</span></div>`).join('');
  icons();
}
const drop = $('drop');
drop.addEventListener('click', () => $('ds-files').click());
drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('ds-files').click(); } });
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('over'); uploadFiles([...e.dataTransfer.files]); });
$('ds-files').addEventListener('change', (e) => { const f = [...e.target.files]; e.target.value = ''; uploadFiles(f); });

async function uploadFiles(files) {
  if (!files.length) return;
  const total = sum(files.map((f) => f.size));
  if (total > 8 * 1024 ** 3) return toast('ไฟล์รวมใหญ่เกิน 8 GB กรุณาแบ่งอัปโหลด', 'error');
  const fd = new FormData();
  files.forEach((f) => fd.append('files', f, f.name));
  $('up-progress').classList.remove('hidden'); $('up-result').innerHTML = '';
  $('up-text').textContent = `กำลังอัปโหลด ${files.length} ไฟล์ (${(total / 1024 ** 2).toFixed(1)} MB)`;
  const setP = (p) => { $('up-bar').style.width = (p * 100).toFixed(0) + '%'; $('up-pct').textContent = (p * 100).toFixed(0) + '%'; };
  setP(0);
  try {
    const r = await apiUpload('/api/dataset/upload', fd, (p) => { setP(p); if (p >= 1) $('up-text').textContent = 'อัปโหลดเสร็จ กำลังตรวจสอบไฟล์'; });
    dsStats = r.stats; renderDataset();
    const lines = [`เพิ่ม ${r.added} ภาพ (มีป้ายกำกับ ${r.with_label}, ไม่มี ${r.without_label})`];
    let html = `<div class="msg ok"><i data-lucide="circle-check"></i><span>${esc(lines.join(''))}</span></div>`;
    if (r.issue_count) html += `<div class="msg warn" style="margin-top:8px"><i data-lucide="triangle-alert"></i><div>พบปัญหา ${r.issue_count} จุด (บรรทัดที่ผิดถูกตัดออก)<ul>${r.issues.slice(0, 8).map((x) => `<li>${esc(x)}</li>`).join('')}</ul></div></div>`;
    $('up-result').innerHTML = html; icons();
    const cur = (State.classes || []).map((c) => c.name);
    if (r.yaml_names && r.yaml_names.join('|') !== cur.join('|')) {
      const ok = await askConfirm('พบรายชื่อคลาสในชุดข้อมูล', `ไฟล์ที่อัปโหลดกำหนดคลาสไว้ ${r.yaml_names.length} คลาส: <b>${esc(r.yaml_names.join(', '))}</b><br><br>ต้องการใช้รายชื่อนี้แทนคลาสปัจจุบันหรือไม่ ลำดับคลาสต้องตรงกับไฟล์ป้ายกำกับ`, { yes: 'ใช้รายชื่อนี้' });
      if (ok) {
        classRows = r.yaml_names.map((n, i) => {
          const old = (State.classes || []).find((c) => c.name.toLowerCase() === String(n).toLowerCase());
          const name = String(n).trim().replace(/\s+/g, '_').replace(/[^A-Za-z0-9_-]/g, '') || `class${i}`;
          return old ? { ...old, name } : { name, label: String(n), measure: guessMeasure(String(n)), color: PALETTE[i % PALETTE.length] };
        });
        renderClassRows(); await saveClasses(); switchTab('t-class');
      }
    }
  } catch (e) {
    $('up-result').innerHTML = `<div class="msg err"><i data-lucide="circle-alert"></i><span>${esc(e.message)}</span></div>`; icons();
  } finally { setTimeout(() => $('up-progress').classList.add('hidden'), 800); }
}
function guessMeasure(n) { n = n.toLowerCase(); return /crack|ร้าว/.test(n) ? 'crack' : /rebar|steel|เหล็ก/.test(n) ? 'rebar' : 'area'; }
async function clearDataset() {
  const ok = await askConfirm('ล้างชุดข้อมูล', `ภาพและป้ายกำกับทั้งหมด ${dsStats ? dsStats.images : ''} ภาพในชุดข้อมูลจะถูกลบถาวร โมเดลที่เทรนแล้วจะยังอยู่`, { yes: 'ลบทั้งหมด', danger: true });
  if (!ok) return;
  try { dsStats = await api('/api/dataset', { method: 'DELETE' }); renderDataset(); $('up-result').innerHTML = ''; toast('ล้างชุดข้อมูลแล้ว', 'info'); } catch (e) { toast(e.message, 'error'); }
}

// ---- class tab
function renderClassRows() {
  $('cls-body').innerHTML = classRows.map((c, i) => `
    <tr>
      <td class="num"><b>${i}</b></td>
      <td><input class="input sm" data-i="${i}" data-k="name" value="${esc(c.name)}" placeholder="crack" aria-label="ชื่อคลาส ${i}"></td>
      <td><input class="input sm" data-i="${i}" data-k="label" value="${esc(c.label)}" placeholder="รอยร้าว" aria-label="ชื่อที่แสดง ${i}"></td>
      <td><select class="input sm" data-i="${i}" data-k="measure" aria-label="ประเภทการวัด ${i}">${Object.entries(MEASURE_LABEL).map(([k, v]) => `<option value="${k}" ${c.measure === k ? 'selected' : ''}>${v}</option>`).join('')}</select></td>
      <td><input type="color" class="input sm" data-i="${i}" data-k="color" value="${esc(c.color)}" aria-label="สี ${i}"></td>
      <td class="c"><button class="icon-btn" onclick="moveClass(${i},-1)" ${i === 0 ? 'disabled' : ''} aria-label="เลื่อนขึ้น"><i data-lucide="arrow-up"></i></button><button class="icon-btn" onclick="moveClass(${i},1)" ${i === classRows.length - 1 ? 'disabled' : ''} aria-label="เลื่อนลง"><i data-lucide="arrow-down"></i></button></td>
      <td class="c"><button class="icon-btn" onclick="removeClass(${i})" ${classRows.length <= 1 ? 'disabled' : ''} aria-label="ลบคลาส" style="color:var(--err)"><i data-lucide="trash-2"></i></button></td>
    </tr>`).join('');
  icons();
  renderDataset();
}
$('cls-body').addEventListener('input', (e) => {
  const el = e.target; if (!el.dataset.k) return;
  classRows[+el.dataset.i][el.dataset.k] = el.value; classesDirty = true;
});
function addClassRow() { classRows.push({ name: `class${classRows.length}`, label: '', measure: 'area', color: PALETTE[classRows.length % PALETTE.length] }); classesDirty = true; renderClassRows(); }
function moveClass(i, d) { const j = i + d; if (j < 0 || j >= classRows.length) return; [classRows[i], classRows[j]] = [classRows[j], classRows[i]]; classesDirty = true; renderClassRows(); }
function removeClass(i) { if (classRows.length <= 1) return; classRows.splice(i, 1); classesDirty = true; renderClassRows(); }
function validateClasses() {
  const seen = new Set();
  for (const c of classRows) {
    const n = String(c.name || '').trim();
    if (!/^[A-Za-z0-9_-]+$/.test(n)) return `ชื่อคลาส "${n}" ใช้ได้เฉพาะ a-z, 0-9, _ และ -`;
    if (seen.has(n.toLowerCase())) return `ชื่อคลาส "${n}" ซ้ำกัน`;
    seen.add(n.toLowerCase());
  }
  return null;
}
async function saveClasses(silent) {
  const err = validateClasses();
  if (err) { toast(err, 'error'); return false; }
  try {
    State.classes = await api('/api/classes', { method: 'PUT', json: classRows.map((c) => ({ ...c, name: c.name.trim(), label: (c.label || '').trim() })) });
    classRows = State.classes.map((c) => ({ ...c })); classesDirty = false;
    renderClassRows(); await refreshDataset(); cfgCheck();
    if (!silent) toast('บันทึกคลาสแล้ว', 'success', 2000);
    return true;
  } catch (e) { toast(e.message, 'error'); return false; }
}

// ---- config check
function cfgCheck() {
  const h = readHypForm(), s = dsStats, info = State.server.info || {};
  const out = [];
  const add = (t, m) => out.push({ t, m });
  const training = info.training || (lastTrainStatus && lastTrainStatus.state === 'running');
  if (training) add('err', 'มีการเทรนกำลังทำงานอยู่ รอให้เสร็จหรือกดหยุดในแท็บความคืบหน้า');
  if (!s || s.images < 2) add('err', 'ต้องมีภาพในชุดข้อมูลอย่างน้อย 2 ภาพ (แท็บ "ข้อมูลภาพ")');
  else if (!s.labeled) add('err', 'ยังไม่มีภาพที่มีป้ายกำกับ');
  if (s && s.max_class_id >= classRows.length) add('err', `ป้ายกำกับใช้คลาสหมายเลข ${s.max_class_id} แต่ตั้งค่าไว้ ${classRows.length} คลาส เพิ่มคลาสในแท็บ "คลาส"`);
  if (s && h.task === 'segment' && s.box_lines) add('err', `มีป้ายกำกับแบบกรอบ ${s.box_lines} รายการ ใช้กับ Segmentation ไม่ได้ เปลี่ยนชนิดงานเป็น Detection`);
  if (s && s.images >= 2 && s.images < 50) add('warn', `มีเพียง ${s.images} ภาพ โมเดลจะยังไม่แม่น แนะนำอย่างน้อย 100-300 ภาพต่อคลาส`);
  if (classesDirty) add('warn', 'มีการแก้ไขคลาสที่ยังไม่ได้บันทึก ระบบจะบันทึกให้อัตโนมัติเมื่อกดเริ่มเทรน');
  const cpu = h.device === 'cpu' || (h.device === 'auto' && !info.cuda && !info.mps);
  if (cpu) add('warn', 'เทรนด้วย CPU จะช้ามาก (หลายชั่วโมงถึงหลายวัน) แนะนำใช้เครื่องที่มีการ์ดจอ NVIDIA');
  if (h.batch === -1 && (cpu || h.device === 'mps' || (!info.cuda && h.device === 'auto'))) add('info', 'Batch อัตโนมัติ (-1) ใช้ได้เฉพาะ GPU NVIDIA อุปกรณ์นี้จะใช้ค่า 16 แทน ถ้าหน่วยความจำไม่พอให้ลดเป็น 4-8');
  if (h.imgsz >= 1280 && ['l', 'x'].includes(h.size)) add('warn', 'รุ่น l/x กับภาพ 1280 px ขึ้นไปใช้หน่วยความจำ GPU มาก ถ้าเกิด out of memory ให้ลด batch');
  if (h.base && !(modelsCache || []).some((m) => m.path === h.base)) add('warn', 'ไม่พบโมเดลตั้งต้นที่เลือก');
  if (h.base) {
    const bm = (modelsCache || []).find((m) => m.path === h.base);
    if (bm && bm.classes && bm.classes.map((c) => c.name).join('|') !== classRows.map((c) => c.name).join('|')) add('warn', 'รายการคลาสต่างจากโมเดลตั้งต้น การเทรนต่อจะรีเซ็ตส่วนหัวการจำแนก (ใช้ได้ แต่ได้ประโยชน์น้อยลง)');
  }
  const ic = { err: 'circle-alert', warn: 'triangle-alert', info: 'info' };
  $('cfg-check').innerHTML = out.map((o) => `<div class="msg ${o.t}"><i data-lucide="${ic[o.t]}"></i><span>${esc(o.m)}</span></div>`).join('');
  icons();
  const hasErr = out.some((o) => o.t === 'err');
  $('btn-start-train').disabled = hasErr;
  $('train-foot').textContent = hasErr ? 'แก้รายการสีแดงในแท็บตั้งค่าก่อนเริ่มเทรน' : s ? `ชุดข้อมูล ${s.images} ภาพ, ${classRows.length} คลาส` : '';
  return !hasErr;
}

async function startTrain() {
  if (classesDirty && !(await saveClasses(true))) return;
  if (!cfgCheck()) { switchTab('t-cfg'); return toast('ยังเริ่มเทรนไม่ได้ ดูรายการที่ต้องแก้ในแท็บตั้งค่า', 'error'); }
  const h = readHypForm();
  const hyp = {};
  ['epochs', 'imgsz', 'batch', 'patience', 'lr0', 'optimizer', 'workers', 'cache', 'seed', 'close_mosaic', 'fliplr', 'flipud', 'degrees', 'translate', 'scale', 'mosaic', 'hsv_h', 'hsv_s', 'hsv_v', 'amp', 'cos_lr', 'plots', 'device'].forEach((k) => { hyp[k] = h[k]; });
  const btn = $('btn-start-train'); btn.disabled = true;
  try {
    const r = await api('/api/train', { method: 'POST', json: { family: h.family, size: h.size, task: h.task, val_split: h.val / 100, seed: h.seed, hyp, base_weights: h.base || null } });
    toast(`เริ่มเทรนแล้ว (ใช้สอน ${r.split.train} ภาพ, ตรวจสอบ ${r.split.val} ภาพ)`, 'success', 5000);
    lastTrainStatus = { state: 'starting' };
    await pollHealth(); switchTab('t-prog');
  } catch (e) { toast('เริ่มเทรนไม่สำเร็จ: ' + e.message, 'error', 8000); cfgCheck(); }
}

// ---- progress
let trainTimer = null, lastTrainStatus = null;
function ensureTrainPolling() {
  const need = $('dlg-train').classList.contains('open') || (State.server.info && State.server.info.training);
  if (need && !trainTimer) { trainTimer = setInterval(pollTrain, 2000); pollTrain(); }
  else if (!need && trainTimer) { clearInterval(trainTimer); trainTimer = null; }
}
async function pollTrain() {
  if (!State.server.ok) return;
  let s;
  try { s = await api('/api/train/status'); } catch { return; }
  const prev = lastTrainStatus ? lastTrainStatus.state : null;
  lastTrainStatus = s;
  if (prev === 'running' || prev === 'starting') {
    if (s.state === 'done') { toast('เทรนเสร็จแล้ว ไปที่แท็บความคืบหน้าเพื่อเลือกใช้โมเดล', 'success', 8000); loadModels(); pollHealth(); }
    else if (s.state === 'failed') { toast('เทรนไม่สำเร็จ ดูรายละเอียดในแท็บความคืบหน้า', 'error', 8000); pollHealth(); }
    else if (s.state === 'stopped') { loadModels(); pollHealth(); }
  }
  renderProgress(s);
  renderServerState();
}
function fmtDur(sec) {
  if (!Number.isFinite(sec) || sec < 0) return '-';
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
  return h ? `${h} ชม. ${m} นาที` : m ? `${m} นาที` : `${Math.max(1, Math.round(sec))} วินาที`;
}
function renderProgress(s) {
  if (!$('dlg-train').classList.contains('open')) return;
  const L = { idle: 'ยังไม่มีการเทรน', starting: 'กำลังเริ่ม', running: 'กำลังเทรน', done: 'เทรนเสร็จแล้ว', failed: 'เทรนไม่สำเร็จ', stopped: 'หยุดแล้ว', unknown: 'ไม่ทราบสถานะ' };
  $('p-state').textContent = L[s.state] || s.state;
  $('p-sub').textContent = s.run ? `${s.run}, ${String(s.model || '').split(/[\\/]/).pop()}${s.task ? ', ' + s.task : ''}${s.started ? ', เริ่ม ' + new Date(s.started * 1000).toLocaleString('th-TH') : ''}` : '';
  const running = s.state === 'running' || s.state === 'starting';
  $('p-stop').disabled = !running;
  const ep = s.epoch || 0, eps = s.epochs || 0;
  $('p-bar').style.width = eps ? `${Math.min(100, (ep / eps) * 100)}%` : running ? '2%' : '0%';
  $('p-epoch').textContent = eps ? `${ep}/${eps}` : '-';
  const hist = s.history || [], last = hist[hist.length - 1] || {};
  const pick = (k) => (last[`${k}(M)`] != null ? last[`${k}(M)`] : last[`${k}(B)`]);
  $('p-map50').textContent = pick('mAP50') != null ? pick('mAP50').toFixed(3) : '-';
  $('p-pr').textContent = pick('precision') != null ? `${pick('precision').toFixed(2)} / ${pick('recall').toFixed(2)}` : '-';
  $('p-eta').textContent = running && ep && last.time ? fmtDur((last.time / ep) * (eps - ep)) : s.state === 'done' && last.time ? `ใช้เวลา ${fmtDur(last.time)}` : '-';
  drawChart(hist);
  let msg = '';
  if (s.state === 'failed') msg = `<div class="msg err"><i data-lucide="circle-alert"></i><span>${esc(s.error || s.message || 'ไม่ทราบสาเหตุ ดู log ด้านล่าง')}</span></div>`;
  else if ((s.state === 'done' || s.state === 'stopped') && s.best) {
    const active = State.server.info && State.server.info.active_model && State.server.info.active_model.path === s.best;
    msg = `<div class="msg ${s.state === 'done' ? 'ok' : 'info'}"><i data-lucide="${s.state === 'done' ? 'circle-check' : 'info'}"></i><div style="flex:1">${s.state === 'done' ? 'ได้โมเดลที่ดีที่สุดแล้ว' : 'หยุดก่อนครบรอบ แต่ยังใช้โมเดลที่ดีที่สุดจนถึงตอนหยุดได้'}</div>${active ? '<span class="tag ai">กำลังใช้งาน</span>' : `<button class="btn btn-sm btn-primary" onclick="activateModel(${esc(JSON.stringify(s.best))})"><i data-lucide="check"></i>ใช้โมเดลนี้ตรวจจับ</button>`}</div>`;
  } else if (s.state === 'starting' || (running && !ep)) msg = '<div class="msg info"><span class="spin"></span><span>กำลังเตรียมข้อมูลและโหลดโมเดลตั้งต้น ครั้งแรกอาจต้องดาวน์โหลดไฟล์ จะใช้เวลาสักครู่</span></div>';
  if ($('p-msg')._h !== msg) { $('p-msg').innerHTML = msg; $('p-msg')._h = msg; icons(); }
  const log = $('p-log'), atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 10;
  if (log.textContent !== (s.log || '')) { log.textContent = s.log || ''; if (atBottom) log.scrollTop = log.scrollHeight; }
}
function drawChart(hist) {
  const svg = $('p-chart'), W = 600, H = 160, P = 8;
  $('p-chart-empty').classList.toggle('hidden', hist.length > 0);
  if (!hist.length) { svg.innerHTML = ''; return; }
  const n = Math.max(2, hist.length);
  const x = (i) => P + (i / (n - 1)) * (W - 2 * P);
  const y = (v) => H - P - v * (H - 2 * P);
  const line = (vals, color, w, op = 1) => {
    const pts = vals.map((v, i) => (v == null ? null : `${x(i).toFixed(1)},${y(Math.max(0, Math.min(1, v))).toFixed(1)}`)).filter(Boolean);
    return pts.length ? `<polyline fill="none" stroke="${color}" stroke-width="${w}" stroke-opacity="${op}" stroke-linejoin="round" vector-effect="non-scaling-stroke" points="${pts.join(' ')}"/>` : '';
  };
  const g = (k) => hist.map((h) => (h[`${k}(M)`] != null ? h[`${k}(M)`] : h[`${k}(B)`]));
  const lossKey = hist.some((h) => h['val/seg_loss'] != null) ? 'val/seg_loss' : 'val/box_loss';
  const loss = hist.map((h) => h[lossKey]);
  const lmax = Math.max(...loss.filter((v) => v != null), 1e-9);
  let grid = '';
  for (let i = 0; i <= 4; i++) grid += `<line x1="0" x2="${W}" y1="${y(i / 4)}" y2="${y(i / 4)}" stroke="#EFEEE9" vector-effect="non-scaling-stroke"/>`;
  svg.innerHTML = grid + line(loss.map((v) => (v == null ? null : v / lmax)), '#D9363E', 1.5, 0.55) + line(g('mAP50-95'), '#1E2228', 2) + line(g('mAP50'), '#E0A800', 2.5);
}
async function stopTrain() {
  const ok = await askConfirm('หยุดเทรน', 'ต้องการหยุดการเทรนตอนนี้หรือไม่ โมเดลที่ดีที่สุดจนถึงตอนหยุดจะยังใช้งานได้', { yes: 'หยุดเทรน', danger: true });
  if (!ok) return;
  try { await api('/api/train/stop', { method: 'POST' }); toast('หยุดการเทรนแล้ว', 'info'); await pollHealth(); pollTrain(); loadModels(); } catch (e) { toast(e.message, 'error'); }
}

// ---- models
async function loadModels() {
  let r;
  try { r = await api('/api/models'); } catch { return; }
  modelsCache = r.models;
  const activePath = r.active && r.active.path;
  const st = { done: 'เสร็จ', stopped: 'หยุดก่อนครบ', failed: 'ไม่สำเร็จ', running: 'กำลังเทรน', uploaded: 'นำเข้า', unknown: '-' };
  $('m-body').innerHTML = r.models.length ? r.models.map((m) => {
    const map = m.metrics['mAP50(M)'] != null ? m.metrics['mAP50(M)'] : m.metrics['mAP50(B)'];
    const act = m.path === activePath;
    return `<tr class="${act ? 'active-row' : ''}"><td><b>${esc(m.run)}</b><div class="hint">${new Date(m.mtime * 1000).toLocaleString('th-TH')}</div></td><td>${esc(m.model)}${m.task ? `<div class="hint">${esc(m.task)}</div>` : ''}</td><td>${st[m.state] || esc(m.state)}</td><td class="r num">${map != null ? map.toFixed(3) : '-'}</td><td class="desc-cell" title="${esc((m.classes || []).map((c) => c.name).join(', '))}">${esc((m.classes || []).map((c) => c.label || c.name).join(', '))}</td><td class="c">${act ? '<span class="tag ai">ใช้งานอยู่</span>' : `<button class="btn btn-sm" onclick="activateModel(${esc(JSON.stringify(m.path))})">ใช้งาน</button>`}</td></tr>`;
  }).join('') : '<tr class="empty-row"><td colspan="6">ยังไม่มีโมเดล เทรนครั้งแรกหรือนำเข้าไฟล์ .pt</td></tr>';
  const base = $('h-base'), cur = base.value;
  base.innerHTML = '<option value="">โมเดลตั้งต้นของ Ultralytics (ดาวน์โหลดอัตโนมัติครั้งแรก)</option>' + r.models.map((m) => `<option value="${esc(m.path)}">${esc(m.run)} (${esc(m.model)})</option>`).join('');
  if ([...base.options].some((o) => o.value === cur)) base.value = cur;
}
async function activateModel(path) {
  try {
    const m = await api('/api/models/activate', { method: 'POST', json: { path } });
    toast(`ใช้โมเดล ${m.label} สำหรับตรวจจับแล้ว`, 'success');
    await pollHealth(); loadModels(); if (lastTrainStatus) renderProgress(lastTrainStatus);
  } catch (e) { toast(e.message, 'error'); }
}
$('pt-input').addEventListener('change', async (e) => {
  const f = e.target.files[0]; e.target.value = '';
  if (!f) return;
  const fd = new FormData(); fd.append('file', f, f.name);
  toast('กำลังนำเข้าโมเดล', 'info', 2000);
  try { const m = await apiUpload('/api/models/upload', fd); toast(`นำเข้าและเลือกใช้ ${m.label} แล้ว คลาส: ${m.names.join(', ')}`, 'success', 6000); await pollHealth(); loadModels(); }
  catch (err) { toast('นำเข้าไม่สำเร็จ: ' + err.message, 'error', 7000); }
});

// ---- dialog backdrop click closes (except record/confirm handled separately)
['dlg-ds', 'dlg-train'].forEach((id) => $(id).addEventListener('mousedown', (e) => { if (e.target.id === id) closeDlg(id); }));

// ================================================================= init
(function init() {
  $('api-url').value = LS.get('api', '');
  const conf = LS.get('conf', null); if (conf) { $('ai-conf').value = conf; $('ai-conf-out').textContent = Number(conf).toFixed(2); }
  const tile = LS.get('tile', null); if (tile) $('ai-tile').value = tile;
  const imgsz = LS.get('imgsz', null); if (imgsz != null) $('ai-imgsz').value = imgsz;
  initLogo();
  updateCalStatus(); updateDashboard(); icons();
  if (typeof initWhatsNew === 'function') initWhatsNew();
  pollHealth();
  setInterval(() => { if (!document.hidden) pollHealth(); }, 5000);
})();

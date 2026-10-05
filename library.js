'use strict';
/* =====================================================================
   คลังงานตรวจสอบ (Inspection Library)
   - เก็บทุกโปรเจกต์ที่มีการวัด/ตรวจทานโดยผู้ใช้ ไว้ในเบราว์เซอร์ (IndexedDB) อัตโนมัติ
   - แต่ละโปรเจกต์เลือกได้ว่า "ใช้เทรน" หรือไม่
   - ส่งออกเป็นชุดข้อมูล YOLO (.zip) หรือส่งเข้าเซิร์ฟเวอร์เทรนโดยตรง
   - สำรอง/กู้คืนทั้งคลังเป็นไฟล์ .zip (ข้อมูลอยู่ในเบราว์เซอร์เครื่องนั้น ควรสำรองเป็นระยะ)
   ===================================================================== */
const Library = (() => {
  const DB_NAME = 'crackai-library', DB_VER = 1;
  let dbp = null, timer = null, saving = false, pending = false, lastSaved = 0, cache = [];

  // ------------------------------------------------------------- IndexedDB helpers
  function db() {
    if (!dbp) {
      dbp = new Promise((res, rej) => {
        if (!('indexedDB' in window)) return rej(new Error('เบราว์เซอร์นี้ไม่รองรับการเก็บข้อมูลในเครื่อง'));
        const r = indexedDB.open(DB_NAME, DB_VER);
        r.onupgradeneeded = () => {
          const d = r.result;
          if (!d.objectStoreNames.contains('projects')) {
            const s = d.createObjectStore('projects', { keyPath: 'id' });
            s.createIndex('imageHash', 'imageHash');
          }
          if (!d.objectStoreNames.contains('images')) d.createObjectStore('images', { keyPath: 'id' });
        };
        r.onsuccess = () => res(r.result);
        r.onerror = () => rej(r.error);
      });
    }
    return dbp;
  }
  const reqP = (r) => new Promise((res, rej) => { r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
  const done = (t) => new Promise((res, rej) => { t.oncomplete = () => res(); t.onerror = () => rej(t.error); t.onabort = () => rej(t.error || new Error('aborted')); });
  async function get(store, id) { const d = await db(); return reqP(d.transaction(store).objectStore(store).get(id)); }
  async function getAll() { const d = await db(); return reqP(d.transaction('projects').objectStore('projects').getAll()); }
  async function put(store, val) { const d = await db(); const t = d.transaction(store, 'readwrite'); t.objectStore(store).put(val); return done(t); }
  async function putBoth(entry, imgRec) {
    const d = await db(); const t = d.transaction(['projects', 'images'], 'readwrite');
    t.objectStore('projects').put(entry); if (imgRec) t.objectStore('images').put(imgRec);
    return done(t);
  }
  async function remove(id) {
    const d = await db(); const t = d.transaction(['projects', 'images'], 'readwrite');
    t.objectStore('projects').delete(id); t.objectStore('images').delete(id); return done(t);
  }
  async function findByHash(h) { if (!h) return null; const d = await db(); return reqP(d.transaction('projects').objectStore('projects').index('imageHash').get(h)); }

  const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`);
  async function sha1(blob) {
    try {
      const h = await crypto.subtle.digest('SHA-1', await blob.arrayBuffer());
      return [...new Uint8Array(h)].map((x) => x.toString(16).padStart(2, '0')).join('');
    } catch { return null; } // crypto.subtle ใช้ไม่ได้ในบาง context -> ไม่ตรวจภาพซ้ำ
  }
  const blobToDataUrl = (b) => new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = () => rej(r.error); r.readAsDataURL(b); });
  function quotaMsg(e) {
    return e && (e.name === 'QuotaExceededError' || /quota/i.test(e.message || ''))
      ? 'พื้นที่จัดเก็บในเบราว์เซอร์เต็ม กรุณาสำรองคลังแล้วลบโปรเจกต์เก่าออก'
      : `บันทึกเข้าคลังไม่สำเร็จ: ${e && e.message ? e.message : e}`;
  }

  function counts(records) {
    const c = { crack: 0, area: 0, rebar: 0, manual: 0, ai: 0 };
    for (const t of Object.keys(TYPES)) for (const r of records[t] || []) { c[t]++; r.source === 'ai' ? c.ai++ : c.manual++; }
    return c;
  }
  function thumb() {
    const prevMode = State.mode, prevShow = State.showAi;
    State.mode = 'idle'; State.showAi = true; render();
    const s = Math.min(1, 360 / canvas.width);
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(canvas.width * s)); c.height = Math.max(1, Math.round(canvas.height * s));
    c.getContext('2d').drawImage(canvas, 0, 0, c.width, c.height);
    State.mode = prevMode; State.showAi = prevShow; render();
    return c.toDataURL('image/jpeg', 0.72);
  }
  const autoOn = () => LS.get('autoLib', true);
  function titleOf(e) {
    const m = e.metadata || {};
    return [m.bldg, m.floor && `ชั้น ${m.floor}`, m.loc].filter(Boolean).join(' / ') || e.imageName || 'ไม่ระบุชื่อ';
  }
  function autoWidths(W, H) { const a = Math.max(3, Math.round(Math.min(W, H) * 0.004)); return { crack: a, rebar: a * 3 }; }

  // ------------------------------------------------------------- save current project
  function schedule() {
    if (!autoOn() || !State.image) return;
    clearTimeout(timer);
    timer = setTimeout(() => { if (hasRecords() || State.libraryId) saveCurrent(); }, 1200);
  }
  async function saveCurrent(opts = {}) {
    if (!State.image) return null;
    if (saving) { pending = true; return null; }
    saving = true; clearTimeout(timer);
    try {
      const blob = await imageBlob();
      let id = State.libraryId, existing = id ? await get('projects', id) : null, hash = existing ? existing.imageHash : null;
      if (!existing) {
        hash = await sha1(blob);
        existing = await findByHash(hash); // ภาพเดิม = โปรเจกต์เดิม ไม่สร้างซ้ำ
        id = existing ? existing.id : uid();
      }
      State.libraryId = id;
      const c = counts(State.records);
      const entry = {
        ...(existing || {}),
        id, app: 'crack-ai', format: 2,
        createdAt: existing ? existing.createdAt : Date.now(), updatedAt: Date.now(),
        imageHash: hash, imageName: State.imageName, imageW: canvas.width, imageH: canvas.height,
        ...projectSnapshot(), counts: c, thumb: thumb(),
      };
      if (opts.useForTraining !== undefined) { entry.useForTraining = !!opts.useForTraining; entry.userSet = true; }
      else if (!entry.userSet) entry.useForTraining = c.manual > 0 && c.ai === 0; // ค่าเริ่มต้น: เฉพาะงานที่วัดด้วยมือล้วน
      if (opts.labelWidths) entry.labelWidths = opts.labelWidths;
      const hasImg = existing && (await get('images', id));
      await putBoth(entry, hasImg ? null : { id, blob });
      lastSaved = Date.now(); State.libSavedAt = lastSaved;
      statusLine();
      if ($('dlg-lib').classList.contains('open')) refresh();
      return entry;
    } catch (e) {
      toast(quotaMsg(e), 'error', 7000); return null;
    } finally {
      saving = false;
      if (pending) { pending = false; schedule(); }
    }
  }
  function statusLine() {
    const el = $('lib-status'); if (!el) return;
    if (!autoOn()) { el.textContent = 'ปิดการบันทึกเข้าคลังอัตโนมัติอยู่'; return; }
    el.textContent = State.libraryId && State.libSavedAt
      ? `บันทึกเข้าคลังแล้ว ${new Date(State.libSavedAt).toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit' })} น.`
      : 'งานที่วัดแล้วจะถูกเก็บเข้าคลังให้อัตโนมัติ';
  }

  // ------------------------------------------------------------- open / delete
  async function open(id) {
    if (State.libraryId && State.libraryId !== id && State.dirty && hasRecords()) await saveCurrent();
    const e = await get('projects', id), im = await get('images', id);
    if (!e || !im) return toast('ไม่พบข้อมูลภาพของโปรเจกต์นี้', 'error');
    const url = URL.createObjectURL(im.blob);
    const img = new Image();
    img.onload = () => {
      if (State.imageUrl) URL.revokeObjectURL(State.imageUrl);
      State.imageUrl = url;
      restoreProject(e, img, e.imageName);
      State.libraryId = id; State.imageBlob = im.blob; State.libSavedAt = e.updatedAt; State.dirty = false;
      statusLine(); closeDlg('dlg-lib');
      toast(`เปิด "${titleOf(e)}" จากคลังแล้ว`, 'success');
    };
    img.onerror = () => toast('ภาพในคลังเสียหาย', 'error');
    img.src = url;
  }
  async function del(id) {
    const e = await get('projects', id); if (!e) return;
    const srv = e.serverName ? (State.server.ok ? '<br><br>ภาพนี้จะถูกลบออกจากชุดข้อมูลเทรนบนเซิร์ฟเวอร์ด้วย' : '<br><br>ภาพนี้เคยส่งเข้าเซิร์ฟเวอร์เทรนแล้ว ขณะนี้ไม่ได้เชื่อมต่อเซิร์ฟเวอร์ สำเนาบนเซิร์ฟเวอร์จะยังอยู่') : '';
    const ok = await askConfirm('ลบออกจากคลัง', `ลบ "${esc(titleOf(e))}" ออกจากคลังถาวร${srv}`, { yes: 'ลบ', danger: true });
    if (!ok) return;
    if (e.serverName && State.server.ok) { try { await api(`/api/dataset/item/${encodeURIComponent(e.serverName)}`, { method: 'DELETE' }); } catch { } }
    await remove(id);
    if (State.libraryId === id) { State.libraryId = null; State.libSavedAt = 0; statusLine(); }
    refresh(); toast('ลบออกจากคลังแล้ว', 'info');
  }
  async function setTrain(id, on, el) {
    const e = await get('projects', id); if (!e) return;
    if (on && e.counts && e.counts.ai) {
      const ok = await askConfirm('ใช้ผล AI เป็นข้อมูลเทรน', `ภาพนี้มีผลจาก AI ${e.counts.ai} รายการ ซึ่งจะถูกนับว่า "ตรวจทานแล้ว" และใช้สอนโมเดลด้วย<br><br>ควรเปิดภาพแล้วลบรายการที่ AI ตรวจผิดก่อน ไม่เช่นนั้นโมเดลจะเรียนรู้สิ่งที่ผิด`, { yes: 'ตรวจทานแล้ว ใช้เทรน' });
      if (!ok) { if (el) el.checked = false; return; }
    }
    e.useForTraining = on; e.userSet = true;
    await put('projects', e); refresh();
  }
  async function downloadJson(id) {
    const e = await get('projects', id), im = await get('images', id);
    if (!e || !im) return;
    const data = { app: 'crack-ai', version: 2, metadata: { ...e.metadata, imageName: e.imageName }, config: e.config, state: e.state, imageData: await blobToDataUrl(im.blob) };
    saveAs(new Blob([JSON.stringify(data)], { type: 'application/json' }), `Inspection_${safeName(titleOf(e))}.json`);
  }

  // ------------------------------------------------------------- dataset export / server sync
  async function classesForExport() {
    if (State.server.ok) { try { const c = await loadClasses(); LS.set('classes', c); return c; } catch { } }
    return LS.get('classes', null) || [
      { name: 'crack', label: 'รอยร้าว', measure: 'crack' },
      { name: 'spalling', label: 'คอนกรีตกะเทาะ/หลุดร่อน', measure: 'area' },
      { name: 'exposed_rebar', label: 'เหล็กเสริมโผล่/เป็นสนิม', measure: 'rebar' },
    ];
  }
  function labelsOf(e, classes) {
    const w = e.labelWidths || autoWidths(e.imageW, e.imageH);
    return buildLabels(e.state.records, e.imageW, e.imageH, classes, w.crack, w.rebar);
  }
  const fileStem = (e) => `${safeName((e.metadata && e.metadata.bldg) || 'project')}_${String(e.id).replace(/[^A-Za-z0-9]/g, '').slice(0, 10)}`;

  async function exportYolo() {
    if (typeof JSZip === 'undefined') return toast('โหลดไลบรารี zip ไม่สำเร็จ ตรวจสอบโฟลเดอร์ vendor', 'error');
    const list = (await getAll()).filter((e) => e.useForTraining);
    if (!list.length) return toast('ยังไม่มีโปรเจกต์ที่เลือก "ใช้เทรน"', 'error');
    toast(`กำลังสร้างชุดข้อมูลจาก ${list.length} โปรเจกต์`, 'info', 2500);
    const classes = await classesForExport();
    const zip = new JSZip();
    let n = 0;
    for (const e of list) {
      const im = await get('images', e.id); if (!im) continue;
      const stem = fileStem(e);
      zip.file(`images/${stem}.jpg`, im.blob);
      zip.file(`labels/${stem}.txt`, labelsOf(e, classes).map((l) => [l.cls, ...l.points.flat().map((v) => v.toFixed(6))].join(' ')).join('\n') + '\n');
      n++;
    }
    zip.file('data.yaml', `# ชุดข้อมูลจากคลังงานตรวจสอบ (${new Date().toISOString().slice(0, 10)})\npath: .\ntrain: images\nval: images\nnames:\n${classes.map((c, i) => `  ${i}: ${c.name}`).join('\n')}\n`);
    zip.file('classes.json', JSON.stringify(classes, null, 2));
    const blob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE', compressionOptions: { level: 3 } });
    saveAs(blob, `crackai_dataset_${new Date().toISOString().slice(0, 10)}_${n}img.zip`);
    toast(`ส่งออกชุดข้อมูล ${n} ภาพแล้ว นำไฟล์นี้ไปอัปโหลดในหน้า "เทรนโมเดล" ได้ทันที`, 'success', 6000);
  }

  async function syncServer() {
    if (!State.server.ok) return toast('ยังไม่ได้เชื่อมต่อเซิร์ฟเวอร์ AI ใช้ปุ่ม "ดาวน์โหลดชุดข้อมูล" แทนได้', 'error');
    const all = await getAll();
    const on = all.filter((e) => e.useForTraining), off = all.filter((e) => !e.useForTraining && e.serverName);
    if (!on.length && !off.length) return toast('ยังไม่มีโปรเจกต์ที่เลือก "ใช้เทรน"', 'error');
    const classes = await classesForExport();
    let sent = 0, skipped = 0, removed = 0, failed = 0;
    const btn = $('lib-sync'); if (btn) btn.disabled = true;
    try {
      for (const e of on) {
        if (e.serverName && e.syncedAt >= e.updatedAt && e.syncedClasses === classes.map((c) => c.name).join('|')) { skipped++; continue; }
        const im = await get('images', e.id); if (!im) { failed++; continue; }
        try {
          const r = await api('/api/dataset/annotation', { method: 'POST', json: { image: await blobToDataUrl(im.blob), filename: fileStem(e), labels: labelsOf(e, classes) } });
          e.serverName = r.name; e.syncedAt = Date.now(); e.syncedClasses = classes.map((c) => c.name).join('|');
          await put('projects', e); sent++;
        } catch (err) { failed++; if (failed === 1) toast('ส่งบางรายการไม่สำเร็จ: ' + err.message, 'error', 6000); }
      }
      for (const e of off) {
        try { await api(`/api/dataset/item/${encodeURIComponent(e.serverName)}`, { method: 'DELETE' }); delete e.serverName; delete e.syncedAt; await put('projects', e); removed++; } catch { failed++; }
      }
    } finally { if (btn) btn.disabled = false; }
    toast(`ส่งเข้าชุดเทรน ${sent} ภาพ${skipped ? `, ไม่เปลี่ยนแปลง ${skipped}` : ''}${removed ? `, นำออก ${removed}` : ''}${failed ? `, ไม่สำเร็จ ${failed}` : ''}`, failed ? 'error' : 'success', 6000);
    refresh();
    if ($('dlg-train').classList.contains('open')) { try { await refreshDataset(); cfgCheck(); } catch { } }
    trainCard();
  }

  // ------------------------------------------------------------- backup / restore
  async function backup() {
    if (typeof JSZip === 'undefined') return toast('โหลดไลบรารี zip ไม่สำเร็จ', 'error');
    const all = await getAll();
    if (!all.length) return toast('คลังยังว่างอยู่', 'error');
    toast('กำลังสร้างไฟล์สำรอง', 'info', 2000);
    const zip = new JSZip();
    zip.file('library.json', JSON.stringify({ app: 'crack-ai-library', format: 2, version: APP_VERSION, exportedAt: Date.now(), projects: all }));
    for (const e of all) { const im = await get('images', e.id); if (im) zip.file(`images/${e.id}.jpg`, im.blob); }
    const blob = await zip.generateAsync({ type: 'blob' });
    saveAs(blob, `crackai_library_backup_${new Date().toISOString().slice(0, 10)}.zip`);
    LS.set('lastBackup', Date.now()); refresh();
  }
  async function restore(file) {
    try {
      const zip = await JSZip.loadAsync(file);
      const meta = JSON.parse(await zip.file('library.json').async('string'));
      if (meta.app !== 'crack-ai-library') throw new Error('ไม่ใช่ไฟล์สำรองคลัง');
      let added = 0, kept = 0;
      for (const e of meta.projects) {
        const cur = await get('projects', e.id);
        if (cur && cur.updatedAt >= e.updatedAt) { kept++; continue; }
        const f = zip.file(`images/${e.id}.jpg`); if (!f) continue;
        const blob = new Blob([await f.async('arraybuffer')], { type: 'image/jpeg' });
        await putBoth(e, { id: e.id, blob }); added++;
      }
      toast(`กู้คืน ${added} โปรเจกต์${kept ? ` (ข้าม ${kept} ที่ในเครื่องใหม่กว่า)` : ''}`, 'success', 5000);
      refresh();
    } catch (e) { toast('กู้คืนไม่สำเร็จ: ' + (e.message || e), 'error', 6000); }
  }

  // ------------------------------------------------------------- UI
  async function storageText() {
    try {
      const est = await navigator.storage.estimate();
      return `ใช้พื้นที่ประมาณ ${(est.usage / 1024 ** 2).toFixed(1)} MB`;
    } catch { return ''; }
  }
  async function refresh() {
    try { cache = (await getAll()).sort((a, b) => b.updatedAt - a.updatedAt); } catch (e) { $('lib-grid').innerHTML = `<div class="msg err">${esc(e.message)}</div>`; return; }
    const q = ($('lib-search').value || '').trim().toLowerCase(), f = $('lib-filter').value;
    const list = cache.filter((e) => (!q || [titleOf(e), e.imageName].join(' ').toLowerCase().includes(q)) && (f === 'all' || (f === 'on') === !!e.useForTraining));
    const nOn = cache.filter((e) => e.useForTraining).length;
    $('lib-info').textContent = `เก็บในเบราว์เซอร์นี้ ${cache.length} โปรเจกต์, เลือกใช้เทรน ${nOn} ${await storageText()}`.trim();
    const lb = LS.get('lastBackup', 0);
    $('lib-backup-hint').textContent = cache.length ? (lb ? `สำรองล่าสุด ${new Date(lb).toLocaleDateString('th-TH')}` : 'ยังไม่เคยสำรอง') : '';
    $('lib-foot').textContent = nOn ? `${nOn} โปรเจกต์พร้อมใช้เทรน` : 'เปิดสวิตช์ "ใช้เทรน" ในโปรเจกต์ที่ตรวจทานแล้ว';
    $('lib-export').disabled = !nOn;
    $('lib-sync').disabled = !State.server.ok || (!nOn && !cache.some((e) => e.serverName));
    $('lib-grid').innerHTML = list.length ? list.map(card).join('') : `<div class="empty" style="grid-column:1/-1"><i data-lucide="library"></i><b>${cache.length ? 'ไม่พบโปรเจกต์ตามเงื่อนไข' : 'คลังยังว่างอยู่'}</b><span>${cache.length ? 'ลองเปลี่ยนคำค้นหรือตัวกรอง' : 'เมื่อวัดความเสียหายในภาพใด ระบบจะเก็บงานนั้นเข้าคลังให้อัตโนมัติ'}</span></div>`;
    icons(); trainCard();
  }
  function card(e) {
    const c = e.counts || {}, cur = e.id === State.libraryId;
    const synced = e.serverName && e.syncedAt >= e.updatedAt;
    const chip = (t, n) => (n ? `<span class="chip"><span class="sw" style="background:${TYPES[t].color}"></span>${TYPES[t].name} ${n}</span>` : '');
    return `<article class="lib-card${cur ? ' cur' : ''}">
      <button class="lib-thumb" onclick="Library.open('${e.id}')" title="เปิดโปรเจกต์นี้"><img src="${e.thumb || ''}" alt="" loading="lazy"></button>
      <div class="lib-body">
        <div class="lib-title" title="${esc(titleOf(e))}">${esc(titleOf(e))}</div>
        <div class="hint">${cur ? '<b style="color:var(--yellow-ink)">กำลังเปิดอยู่</b>, ' : ''}แก้ไข ${new Date(e.updatedAt).toLocaleString('th-TH', { dateStyle: 'short', timeStyle: 'short' })}</div>
        <div class="chips">${chip('crack', c.crack)}${chip('area', c.area)}${chip('rebar', c.rebar)}${!(c.crack || c.area || c.rebar) ? '<span class="chip">ไม่มีรายการ</span>' : ''}</div>
        <div class="hint">วัดมือ ${c.manual || 0}, ผล AI ${c.ai || 0}${synced ? ', ส่งเข้าเซิร์ฟเวอร์แล้ว' : e.serverName ? ', มีการแก้ไขหลังส่ง' : ''}</div>
        <div class="lib-act">
          <label class="switch" title="ใช้ภาพนี้เป็นข้อมูลสอนโมเดล"><input type="checkbox" ${e.useForTraining ? 'checked' : ''} onchange="Library.setTrain('${e.id}', this.checked, this)"><span class="track"></span>ใช้เทรน</label>
          <span style="flex:1"></span>
          <button class="icon-btn" title="ดาวน์โหลดเป็นไฟล์โปรเจกต์ .json" aria-label="ดาวน์โหลด" onclick="Library.downloadJson('${e.id}')"><i data-lucide="download"></i></button>
          <button class="icon-btn" title="ลบออกจากคลัง" aria-label="ลบ" style="color:var(--err)" onclick="Library.del('${e.id}')"><i data-lucide="trash-2"></i></button>
          <button class="btn btn-sm" onclick="Library.open('${e.id}')">เปิด</button>
        </div>
      </div>
    </article>`;
  }
  async function trainCard() {
    const el = $('lib-train-card'); if (!el) return;
    try {
      const all = cache.length ? cache : await getAll();
      const on = all.filter((e) => e.useForTraining), pend = on.filter((e) => !(e.serverName && e.syncedAt >= e.updatedAt));
      el.innerHTML = `<i data-lucide="library"></i><div style="flex:1"><b>คลังงานตรวจสอบ</b> ${on.length} โปรเจกต์ที่เลือกใช้เทรน${pend.length ? `, ยังไม่ได้ส่ง ${pend.length}` : on.length ? ' ส่งครบแล้ว' : ''}</div>
        <button class="btn btn-sm" onclick="Library.show()">เปิดคลัง</button><button class="btn btn-sm btn-primary" onclick="Library.syncServer()" ${on.length || all.some((e) => e.serverName) ? '' : 'disabled'}>ส่งเข้าชุดข้อมูล</button>`;
      icons();
    } catch { }
  }
  function show() { openDlg('dlg-lib'); $('lib-auto').checked = autoOn(); refresh(); }

  // ------------------------------------------------------------- wiring
  function init() {
    $('lib-search').addEventListener('input', refresh);
    $('lib-filter').addEventListener('change', refresh);
    $('lib-auto').addEventListener('change', () => { LS.set('autoLib', $('lib-auto').checked); statusLine(); if ($('lib-auto').checked) schedule(); });
    $('lib-restore-input').addEventListener('change', (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) restore(f); });
    $('dlg-lib').addEventListener('mousedown', (e) => { if (e.target.id === 'dlg-lib') closeDlg('dlg-lib'); });
    if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => { }); // ขอไม่ให้เบราว์เซอร์ลบข้อมูลเอง
    statusLine();
    window.addEventListener('pagehide', () => { if (State.dirty && hasRecords() && autoOn()) saveCurrent(); });
  }
  return { init, schedule, saveCurrent, open, del, setTrain, downloadJson, exportYolo, syncServer, backup, restore, show, refresh, trainCard, statusLine };
})();
window.Library = Library;
Library.init();

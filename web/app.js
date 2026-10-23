'use strict';
// The browser UI: plain JavaScript over the same /api/v1 the desktop UI's services offer. No build step, no libraries.
// Text from the server is only ever put into the page with textContent, never as HTML.

const $ = (id) => document.getElementById(id);
const state = { token: localStorage.getItem('gh.token') || '', settings: null, job: null, results: [], galleryOffset: 0, galleryTotal: 0, imageCache: new Map() };

// ---- helpers ----------------------------------------------------------------------------------------------------

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'class') el.className = value;
    else if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
    else if (value === true) el.setAttribute(key, '');
    else if (value !== false && value != null) el.setAttribute(key, value);
  }
  for (const child of children.flat()) if (child != null) el.append(child.nodeType ? child : document.createTextNode(String(child)));
  return el;
}

function humanSize(bytes) {
  if (bytes >= 2 ** 30) return (bytes / 2 ** 30).toFixed(2) + ' GiB';
  if (bytes >= 2 ** 20) return (bytes / 2 ** 20).toFixed(0) + ' MiB';
  return Math.round(bytes / 1024) + ' KiB';
}

class ApiError extends Error {
  constructor(status, message, code) { super(message); this.status = status; this.code = code; }
}

function authHeaders() { return state.token ? { Authorization: 'Bearer ' + state.token } : {}; }

async function api(path, { method = 'GET', body, retry = true } = {}) {
  const headers = authHeaders();
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const res = await fetch(path, { method, headers, body: body !== undefined ? JSON.stringify(body) : undefined });
  if (res.status === 401 && retry) {
    await askToken();
    return api(path, { method, body, retry: false });
  }
  if (!res.ok) {
    let message = res.statusText, code;
    try { const j = await res.json(); message = j.error.message; code = j.error.code; } catch { /* not JSON */ }
    throw new ApiError(res.status, message, code);
  }
  return res.status === 204 ? null : res.json();
}

// <img src> cannot send an Authorization header, so with a token the picture is fetched and shown from a blob.
async function imageSrc(url) {
  if (!state.token) return url;
  if (state.imageCache.has(url)) return state.imageCache.get(url);
  const res = await fetch(url, { headers: authHeaders() });
  if (!res.ok) throw new ApiError(res.status, 'cannot load the image');
  const objectUrl = URL.createObjectURL(await res.blob());
  state.imageCache.set(url, objectUrl);
  return objectUrl;
}

async function img(url, attrs = {}) {
  const el = h('img', { alt: '', ...attrs });
  try { el.src = await imageSrc(url); } catch { el.alt = 'unavailable'; }
  return el;
}

const dialog = $('dlg');
function showDialog(...content) {
  const body = $('dlg-body');
  body.replaceChildren(...content);
  if (!dialog.open) dialog.showModal();
}
function closeDialog() { if (dialog.open) dialog.close(); }
dialog.addEventListener('click', (e) => { if (e.target === dialog) closeDialog(); });

function askToken() {
  return new Promise((resolve) => {
    const input = h('input', { type: 'password', placeholder: 'gh_…', style: 'width:100%' });
    showDialog(
      h('h2', {}, 'Access token needed'),
      h('p', {}, 'This server asks for an access token. Create one in the desktop app (Settings) or with "graphicshost token create <name>".'),
      input,
      h('div', { class: 'buttons' }, h('button', { class: 'primary', onclick: () => { state.token = input.value.trim(); localStorage.setItem('gh.token', state.token); state.imageCache.clear(); closeDialog(); resolve(); } }, 'Use token')));
    dialog.addEventListener('close', () => resolve(), { once: true });
  });
}

function say(el, text, kind = '') { el.textContent = text; el.className = kind || 'muted'; }
function fail(el, e) { say(el, e.message || String(e), 'error'); }

// ---- tabs --------------------------------------------------------------------------------------------------------

const loaders = {};
function showTab(name) {
  for (const b of document.querySelectorAll('#tabs button')) b.classList.toggle('active', b.dataset.tab === name);
  for (const s of document.querySelectorAll('.tab')) s.classList.toggle('active', s.id === 'tab-' + name);
  if (location.hash !== '#' + name) history.replaceState(null, '', '#' + name);
  if (loaders[name]) loaders[name]();
}
$('tabs').addEventListener('click', (e) => { if (e.target.dataset.tab) showTab(e.target.dataset.tab); });
const visible = (name) => $('tab-' + name).classList.contains('active');

// ---- status bar --------------------------------------------------------------------------------------------------

async function refreshStatus() {
  try {
    const s = await api('/api/v1/status');
    const model = s.model ? 'Model: ' + s.model.path.split(/[\\/]/).pop() : 'No model loaded';
    const queue = s.queue.active ? ` | ${s.queue.active} job(s) queued or running` : '';
    $('status').textContent = `${s.name} ${s.version} | Runtime: ${s.runtime ? s.runtime.id : 'none'} | ${model}${queue}`;
  } catch (e) {
    $('status').textContent = 'Not connected: ' + e.message;
  }
}

async function loadSettings() {
  state.settings = await api('/api/v1/settings');
  const advanced = state.settings.interfaceMode !== 'user';
  document.querySelector('#tab-generate .advanced').classList.toggle('hidden', !advanced);
}

// ---- Generate ----------------------------------------------------------------------------------------------------

async function loadGenerate() {
  try {
    await loadSettings();
    const [models, presets] = await Promise.all([api('/api/v1/models'), api('/api/v1/presets')]);
    const keep = $('g-model').value;
    const checkpoints = models.models.filter((m) => m.kind === 'checkpoint' && m.supported);
    $('g-model').replaceChildren(...(checkpoints.length ? checkpoints.map((m) => h('option', { value: m.file }, `${m.file}  (${m.family})`))
      : [h('option', { value: '' }, 'No models yet: see My Models or Discover')]));
    if (checkpoints.some((m) => m.file === keep)) $('g-model').value = keep;
    else if (models.loaded) { const f = models.loaded.path.split(/[\\/]/).pop(); if (checkpoints.some((m) => m.file === f)) $('g-model').value = f; }
    const keepPreset = $('g-preset').value;
    $('g-preset').replaceChildren(h('option', { value: '' }, '(none)'), ...presets.presets.map((p) => h('option', { value: p.name }, p.name)));
    $('g-preset').value = keepPreset;
  } catch (e) { say($('g-status'), e.message, 'error'); }
}

async function fillSamplers() {
  // the names the engine accepts, from the server
  try {
    const p = await api('/api/v1/parameters');
    $('g-sampler').replaceChildren(h('option', { value: '' }, 'Default'), ...p.samplers.map((s) => h('option', { value: s }, s)));
    $('g-scheduler').replaceChildren(h('option', { value: '' }, 'Default'), ...p.schedulers.map((s) => h('option', { value: s }, s)));
  } catch (e) { /* the form still works with the defaults */ }
}

async function readForm() {
  const body = { model: $('g-model').value, prompt: $('g-prompt').value.trim() };
  if ($('g-preset').value) body.preset = $('g-preset').value;
  if ($('g-negative').value.trim()) body.negativePrompt = $('g-negative').value.trim();
  const num = (id, key, int = true) => { const v = $(id).value; if (v !== '') body[key] = int ? parseInt(v, 10) : parseFloat(v); };
  num('g-width', 'width'); num('g-height', 'height'); num('g-seed', 'seed');
  if (parseInt($('g-count').value, 10) > 1) body.batchCount = parseInt($('g-count').value, 10);
  if (state.settings && state.settings.interfaceMode !== 'user') {
    num('g-steps', 'steps'); num('g-cfg', 'cfgScale', false);
    if ($('g-sampler').value) body.sampler = $('g-sampler').value;
    if ($('g-scheduler').value) body.scheduler = $('g-scheduler').value;
  }
  const file = $('g-source').files[0];
  if (file) {
    body.init_image = await new Promise((resolve, reject) => { const r = new FileReader(); r.onload = () => resolve(r.result); r.onerror = () => reject(new Error("cannot read the source image")); r.readAsDataURL(file); });
  } else if (state.sourceGalleryId) body.init_image_gallery_id = state.sourceGalleryId;
  if (body.init_image || body.init_image_gallery_id) body.strength = parseFloat($('g-strength').value) || 0.75;
  return body;
}

function fillForm(request, model) {
  if (model) $('g-model').value = model;
  $('g-prompt').value = request.prompt || '';
  $('g-negative').value = request.negativePrompt || '';
  for (const [id, key] of [['g-width', 'width'], ['g-height', 'height'], ['g-seed', 'seed'], ['g-steps', 'steps'], ['g-cfg', 'cfgScale']]) $(id).value = request[key] ?? '';
  $('g-count').value = request.batchCount || 1;
  $('g-sampler').value = request.sampler || '';
  $('g-scheduler').value = request.scheduler || '';
}

async function showResult(index) {
  state.selected = index;
  const r = state.results[index];
  $('g-view').replaceChildren(r ? await img(r.url, { onclick: () => openImage(r.galleryId) }) : h('p', { class: 'muted' }, 'Your images appear here.'));
  [...$('g-thumbs').children].forEach((t, i) => t.classList.toggle('selected', i === index));
}

async function pollJob(id) {
  const job = await api('/api/v1/jobs/' + id);
  state.job = job;
  const p = $('g-progress');
  if (job.state === 'queued' || job.state === 'running') {
    if (job.steps > 0) { p.max = job.steps; p.value = job.step; say($('g-status'), `Step ${job.step} of ${job.steps}` + (job.secondsPerStep > 0 ? `  (${job.secondsPerStep.toFixed(2)} s/step)` : '')); }
    else { p.removeAttribute('value'); say($('g-status'), job.state === 'queued' ? 'Waiting in the queue…' : 'Starting…'); }
    setTimeout(() => pollJob(id).catch((e) => finishGenerate(e.message, true)), 250);
    return;
  }
  p.max = 1; p.value = job.state === 'succeeded' ? 1 : 0;
  if (job.state === 'succeeded') {
    state.results = job.images.filter((i) => i.url).map((i) => ({ url: i.url, galleryId: i.galleryId, seed: i.seed }));
    const thumbs = await Promise.all(state.results.map(async (r, i) => img(r.url, { title: 'seed ' + r.seed, onclick: () => showResult(i) })));
    $('g-thumbs').replaceChildren(...thumbs);
    await showResult(0);
    finishGenerate(`Done in ${job.seconds.toFixed(1)} s: ${job.images.length} image(s).` + (job.warnings.length ? '  ' + job.warnings[0] : ''));
  } else if (job.state === 'cancelled') finishGenerate('Cancelled.');
  else finishGenerate('Failed: ' + (job.error ? job.error.message : 'unknown error'), true);
}

function finishGenerate(text, isError = false) {
  state.job = null;
  $('g-go').disabled = false; $('g-cancel').disabled = true; $('g-model').disabled = false;
  $('g-progress').max = 1; $('g-progress').value = 0;
  say($('g-status'), text, isError ? 'error' : 'muted');
  refreshStatus();
}

$('gen-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (state.job) return;
  let body;
  try { body = await readForm(); } catch (err) { return say($('g-status'), err.message, 'error'); }
  if (!body.model) return say($('g-status'), 'There is no model to generate with yet.', 'error');
  if (!body.prompt) return say($('g-status'), 'Write a prompt first.', 'error');
  $('g-go').disabled = true; $('g-cancel').disabled = false; $('g-model').disabled = true;
  $('g-progress').removeAttribute('value');
  say($('g-status'), 'Loading the model and queueing…');
  try {
    const r = await api('/api/v1/generate', { method: 'POST', body });
    state.job = r.job;
    if (r.notes.length) say($('g-status'), r.notes.join('; '));
    pollJob(r.job.id).catch((err) => finishGenerate(err.message, true));
  } catch (err) { finishGenerate('Could not start: ' + err.message, true); }
});

$('g-cancel').addEventListener('click', async () => {
  if (!state.job) return;
  say($('g-status'), 'Cancelling…');
  try { await api(`/api/v1/jobs/${state.job.id}/cancel`, { method: 'POST' }); } catch (e) { say($('g-status'), e.message, 'error'); }
});

loaders.generate = loadGenerate;

// ---- image to image: the source picture -----------------------------------------------------------------------
function setSourceNote() {
  const file = $('g-source').files[0];
  $('g-source-note').textContent = file ? file.name : state.sourceGalleryId ? `gallery image #${state.sourceGalleryId}` : '';
  $('g-source-clear').hidden = !(file || state.sourceGalleryId);
}
$('g-source').addEventListener('change', () => { state.sourceGalleryId = 0; setSourceNote(); });
$('g-source-clear').addEventListener('click', () => { $('g-source').value = ''; state.sourceGalleryId = 0; setSourceNote(); });
function useAsSource(id) { $('g-source').value = ''; state.sourceGalleryId = id; setSourceNote(); showTab('generate'); }

// ---- Gallery -----------------------------------------------------------------------------------------------------

async function loadGallery(append = false) {
  const text = $('gal-search').value.trim();
  if (!append) { state.galleryOffset = 0; $('gal-grid').replaceChildren(); }
  try {
    const r = await api(`/api/v1/gallery?limit=60&offset=${state.galleryOffset}&text=${encodeURIComponent(text)}`);
    state.galleryTotal = r.total;
    const cards = await Promise.all(r.items.map(async (item) => {
      const fig = h('figure', { onclick: () => openImage(item.id) }, await img(item.thumbnailUrl + '?size=256', { loading: 'lazy' }), h('figcaption', { title: item.prompt }, item.prompt || '(no prompt)'));
      return fig;
    }));
    $('gal-grid').append(...cards);
    state.galleryOffset += r.items.length;
    $('gal-count').textContent = `${state.galleryOffset} of ${r.total} image(s)`;
    $('gal-more').hidden = state.galleryOffset >= r.total;
  } catch (e) { $('gal-count').textContent = e.message; }
}

async function openImage(id) {
  try {
    const item = await api('/api/v1/gallery/' + id);
    const params = item.parameters || {};
    const pre = h('pre', {}, JSON.stringify(params.request || params, null, 2));
    const download = h('button', { onclick: async () => {
      const blob = await (await fetch(item.url, { headers: authHeaders() })).blob();
      const a = h('a', { href: URL.createObjectURL(blob), download: item.path.split('/').pop() });
      a.click(); URL.revokeObjectURL(a.href);
    } }, 'Download');
    showDialog(
      await img(item.url),
      h('p', {}, h('strong', {}, item.prompt || '(no prompt)')),
      h('p', { class: 'muted' }, `${item.model} · ${item.width}×${item.height} · seed ${item.seed} · ${item.created}`),
      pre,
      h('div', { class: 'buttons' },
        h('button', { class: 'primary', onclick: () => { closeDialog(); showTab('generate'); fillForm(params.request || {}, item.model); } }, 'Reuse parameters'),
        h('button', { onclick: () => { closeDialog(); useAsSource(id); } }, 'Use as source'),
        download,
        h('button', { class: 'danger', onclick: async () => {
          if (!confirm('Delete this image?')) return;
          try { await api('/api/v1/gallery/' + id, { method: 'DELETE' }); closeDialog(); loadGallery(); } catch (e) { alert(e.message); }
        } }, 'Delete'),
        h('button', { onclick: closeDialog }, 'Close')));
  } catch (e) { alert(e.message); }
}

let gallerySearchTimer;
$('gal-search').addEventListener('input', () => { clearTimeout(gallerySearchTimer); gallerySearchTimer = setTimeout(() => loadGallery(), 300); });
$('gal-more').addEventListener('click', () => loadGallery(true));
loaders.gallery = () => loadGallery();

// ---- My Models ---------------------------------------------------------------------------------------------------

async function loadModels() {
  try {
    const r = await api('/api/v1/models');
    const rows = r.models.map((m) => {
      const state_ = [m.loaded ? 'loaded' : '', !m.supported ? (m.note || 'not supported') : '', m.hasDefaults ? 'has defaults' : ''].filter(Boolean).join(', ');
      const actions = [];
      if (m.kind === 'checkpoint' && m.supported) {
        actions.push(h('button', { class: 'small', onclick: () => modelAction('/api/v1/models/load', { model: m.file }, `Loading ${m.file}…`) }, 'Load'));
        actions.push(h('button', { class: 'small', onclick: () => editDefaults(m.file) }, 'Defaults…'));
      }
      return h('tr', {}, h('td', { title: m.path }, m.file), h('td', {}, m.kind), h('td', {}, m.kind === 'checkpoint' ? m.family : ''),
        h('td', { class: 'num' }, humanSize(m.sizeBytes)), h('td', {}, m.vae || ''), h('td', {}, state_), h('td', {}, actions));
    });
    $('m-table').tBodies[0].replaceChildren(...rows);
    $('m-summary').textContent = `${r.models.length} file(s) in ${r.modelsDir}` + (r.loaded ? '' : '  ·  no model loaded');
    if (r.loaded) $('m-summary').append(' ', h('button', { class: 'small', onclick: () => modelAction('/api/v1/models/unload', {}, 'Unloading…') }, 'Unload'));
  } catch (e) { fail($('m-summary'), e); }
}

async function modelAction(path, body, text) {
  say($('m-status'), text);
  try { const r = await api(path, { method: 'POST', body }); say($('m-status'), r.notes && r.notes.length ? r.notes.join('; ') : 'Done.'); }
  catch (e) { fail($('m-status'), e); }
  loadModels(); refreshStatus();
}

async function editDefaults(file) {
  let doc = { request: {}, load: {} };
  try { doc = await api('/api/v1/defaults/' + encodeURIComponent(file)); } catch (e) { if (e.status !== 404) return alert(e.message); }
  const text = h('textarea', { rows: 12, style: 'width:100%;font-family:monospace' }, JSON.stringify({ request: doc.request, load: doc.load }, null, 2));
  const error = h('p', { class: 'error' });
  showDialog(
    h('h2', {}, 'Defaults for ' + file),
    h('p', { class: 'muted' }, 'Only the parameters listed here change; the rest keep the model\'s own defaults.'),
    text, error,
    h('div', { class: 'buttons' },
      h('button', { class: 'primary', onclick: async () => {
        let parsed;
        try { parsed = JSON.parse(text.value); } catch { error.textContent = 'This is not valid JSON.'; return; }
        try { await api('/api/v1/defaults/' + encodeURIComponent(file), { method: 'PUT', body: parsed }); closeDialog(); loadModels(); } catch (e) { error.textContent = e.message; }
      } }, 'Save'),
      h('button', { class: 'danger', onclick: async () => { try { await api('/api/v1/defaults/' + encodeURIComponent(file), { method: 'DELETE' }); } catch { /* none */ } closeDialog(); loadModels(); } }, 'Remove defaults'),
      h('button', { onclick: closeDialog }, 'Cancel')));
}
loaders.models = loadModels;

// ---- Discover ----------------------------------------------------------------------------------------------------

async function loadCatalog() {
  try {
    const r = await api(`/api/v1/catalog?text=${encodeURIComponent($('d-filter').value)}&family=${encodeURIComponent($('d-family').value)}`);
    $('d-catalog').tBodies[0].replaceChildren(...r.entries.map((e) => {
      const st = [e.installed ? 'installed' : '', e.gated ? 'needs token' : '', e.recommended && !e.installed ? 'recommended' : ''].filter(Boolean).join(', ');
      return h('tr', {}, h('td', { title: e.notes }, e.name), h('td', {}, e.kind), h('td', {}, e.family), h('td', { class: 'num' }, humanSize(e.sizeBytes)),
        h('td', { title: e.fit.detail }, e.fit.label), h('td', {}, st), h('td', {}, h('button', { class: 'small', onclick: () => getModel({ target: e.key }, e.name) }, 'Get…')));
    }));
  } catch (e) { fail($('d-status'), e); }
}

async function getModel(request, title) {
  say($('d-status'), `Preparing ${title}…`);
  try {
    const dry = await api('/api/v1/models/get', { method: 'POST', body: { ...request, dryRun: true } });
    const lines = dry.plan.items.map((i) => `• ${i.file}  (${i.role}, ${humanSize(i.sizeBytes)})${i.present ? '  already here' : ''}`);
    const text = `${lines.join('\n')}\n\n${(dry.plan.notes || []).join('\n')}\n\nTo download: ${humanSize(dry.plan.bytesToDownload)}`;
    if (!confirm(`Get ${title}?\n\n${text}`)) return say($('d-status'), 'Cancelled.');
    await api('/api/v1/models/get', { method: 'POST', body: request });
    say($('d-status'), `Downloading ${title}…`);
    loadDownloads();
  } catch (e) { fail($('d-status'), e); }
}

async function loadDownloads() {
  try {
    const r = await api('/api/v1/downloads');
    $('d-downloads').tBodies[0].replaceChildren(...r.downloads.map((d) => {
      const bar = h('progress', { max: d.bytesTotal || 1, value: d.bytesDone });
      const act = (action) => async () => { try { await api(`/api/v1/downloads/${d.id}/${action}`, { method: 'POST' }); } catch (e) { fail($('d-status'), e); } loadDownloads(); };
      return h('tr', {}, h('td', {}, d.label), h('td', {}, d.state + (d.error ? ': ' + d.error.message : '')), h('td', {}, bar, ' ', `${humanSize(d.bytesDone)} / ${d.bytesTotal ? humanSize(d.bytesTotal) : '?'}`),
        h('td', {}, d.state === 'running' ? (d.bytesPerSecond / 1e6).toFixed(1) + ' MB/s' : ''),
        h('td', {}, h('button', { class: 'small', onclick: act('pause') }, 'Pause'), ' ', h('button', { class: 'small', onclick: act('resume') }, 'Resume'), ' ', h('button', { class: 'small', onclick: act('cancel') }, 'Cancel')));
    }));
  } catch (e) { /* the next tick tries again */ }
}

$('d-search-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = $('d-query').value.trim();
  if (!q) return;
  say($('d-hub-status'), 'Searching…');
  try {
    const r = await api('/api/v1/hub/search?limit=30&q=' + encodeURIComponent(q));
    $('d-files').tBodies[0].replaceChildren();
    $('d-repos').tBodies[0].replaceChildren(...r.models.map((m) => h('tr', { class: 'clickable', onclick: () => showRepo(m.id) },
      h('td', {}, m.id), h('td', { class: 'num' }, m.downloads.toLocaleString()), h('td', {}, m.license), h('td', {}, m.gated ? 'gated' : 'open'))));
    say($('d-hub-status'), r.models.length ? 'Select a repository to see its files.' : 'Nothing found.');
  } catch (err) { fail($('d-hub-status'), err); }
});

async function showRepo(repo) {
  say($('d-hub-status'), `Reading ${repo}…`);
  try {
    const r = await api('/api/v1/hub/files?repo=' + encodeURIComponent(repo));
    $('d-files').tBodies[0].replaceChildren(...r.files.map((f) => h('tr', {}, h('td', {}, f.path), h('td', {}, f.kind), h('td', { class: 'num' }, humanSize(f.sizeBytes)), h('td', { title: f.fit.detail }, f.fit.label),
      h('td', {}, h('button', { class: 'small', onclick: () => getModel({ target: repo, file: f.path }, f.path) }, 'Get…')))));
    say($('d-hub-status'), r.files.length ? `${repo}: ${r.files.length} file(s)` + (r.gated ? ' (gated: needs a token with access)' : '') : `${repo} has no checkpoint files GraphicsHost can use.`);
  } catch (err) { fail($('d-hub-status'), err); }
}

$('d-filter').addEventListener('input', () => loadCatalog());
$('d-family').addEventListener('change', () => loadCatalog());
loaders.discover = () => { loadCatalog(); loadDownloads(); };

// ---- Settings ----------------------------------------------------------------------------------------------------

async function loadSettingsTab() {
  try {
    const s = await api('/api/v1/settings');
    state.settings = s;
    $('s-backend').value = s.backend; $('s-threads').value = s.threads; $('s-modelsdir').value = s.modelsDir; $('s-mode').value = s.interfaceMode;
    $('s-lan').checked = s.server.lan; $('s-require').checked = s.server.requireToken; $('s-port').value = s.server.port;
    $('s-log').value = s.logging.level; $('s-safety').checked = s.safety.enabled;
  } catch (e) { fail($('s-status'), e); }
  loadTokens();
}

$('s-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const patch = { backend: $('s-backend').value, threads: parseInt($('s-threads').value || '0', 10), modelsDir: $('s-modelsdir').value.trim(), interfaceMode: $('s-mode').value,
    server: { lan: $('s-lan').checked, requireToken: $('s-require').checked, port: parseInt($('s-port').value, 10) }, logging: { level: $('s-log').value }, safety: { enabled: $('s-safety').checked } };
  try {
    const r = await api('/api/v1/settings', { method: 'PUT', body: patch });
    state.settings = r.settings;
    say($('s-status'), r.restartRequired ? 'Saved. The server port or network setting applies after GraphicsHost is restarted.' : 'Saved.');
  } catch (err) { fail($('s-status'), err); }
});

async function loadTokens() {
  try {
    const r = await api('/api/v1/tokens');
    $('s-tokens').tBodies[0].replaceChildren(...r.tokens.map((t) => h('tr', {}, h('td', {}, t.name), h('td', {}, t.permissions.join(', ')), h('td', {}, t.lastUsed || 'never'),
      h('td', {}, h('button', { class: 'small danger', onclick: async () => { if (!confirm(`Revoke "${t.name}"?`)) return; try { await api('/api/v1/tokens/' + encodeURIComponent(t.id), { method: 'DELETE' }); } catch (e) { fail($('s-token-status'), e); } loadTokens(); } }, 'Revoke')))));
    say($('s-token-status'), r.tokens.length ? '' : 'No tokens yet.');
  } catch (e) { $('s-tokens').tBodies[0].replaceChildren(); say($('s-token-status'), e.status === 403 ? 'Managing tokens needs the admin permission.' : e.message, 'muted'); }
}

$('s-token-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    const r = await api('/api/v1/tokens', { method: 'POST', body: { name: $('s-token-name').value.trim(), permissions: $('s-token-perm').value.split(',') } });
    $('s-token-name').value = '';
    showDialog(h('h2', {}, 'Token created'), h('p', {}, 'Copy it now: it is not shown again.'),
      h('input', { type: 'text', readonly: true, value: r.secret, style: 'width:100%', onfocus: (ev) => ev.target.select() }),
      h('div', { class: 'buttons' }, h('button', { onclick: () => navigator.clipboard && navigator.clipboard.writeText(r.secret) }, 'Copy'), h('button', { onclick: closeDialog }, 'Close')));
    loadTokens();
  } catch (err) { fail($('s-token-status'), err); }
});
loaders.settings = loadSettingsTab;

// ---- Logs --------------------------------------------------------------------------------------------------------

async function loadLogs() {
  try {
    const r = await api('/api/v1/logs?limit=300&level=' + $('l-level').value);
    const view = $('l-view');
    const atBottom = view.scrollTop + view.clientHeight >= view.scrollHeight - 20;
    view.replaceChildren(...r.entries.map((e) => h('div', { class: e.level }, `${e.time}  ${e.level.padEnd(5)}  ${e.message}`)));
    $('l-count').textContent = r.entries.length + ' line(s)';
    if (atBottom) view.scrollTop = view.scrollHeight;
  } catch (e) { $('l-count').textContent = e.message; }
}
$('l-level').addEventListener('change', loadLogs);
loaders.logs = loadLogs;

// ---- start -------------------------------------------------------------------------------------------------------

fillSamplers();
setInterval(() => {
  if (document.hidden) return;
  refreshStatus();
  if (visible('discover')) loadDownloads();
  if (visible('logs')) loadLogs();
  if (visible('models') && !dialog.open) loadModels();
}, 2000);
refreshStatus();
const first = location.hash.slice(1);
showTab(document.getElementById('tab-' + first) ? first : 'generate');

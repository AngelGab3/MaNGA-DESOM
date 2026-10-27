/**
 * DESOM Dashboard Client-side JavaScript.
 * Handles interactive tabs, Plotly charts, demographic statistics, and pipeline triggers.
 */

let allGalaxiesData = [];
let processPollInterval = null;
let cachedRunsList = [];

// ============================================================================
// Initialization
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initHardwareInfo();
  loadExperiments();
  loadFullConfig();
  loadDemographics();
  loadDiagnostics();
  loadPipelineState();
  loadDESOM1Figures();
  loadDESOM2Figures();
  loadFiguresGallery();
});

function refreshAll() {
  loadExperiments();
  loadFullConfig();
  loadDemographics();
  loadDiagnostics();
  loadPipelineState();
  loadDESOM1Figures();
  loadDESOM2Figures();
  loadFiguresGallery();
}

// ============================================================================
// Navigation Tabs
// ============================================================================
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active', 'text-white', 'bg-space-700', 'border-cyan-500/30');
    btn.classList.add('text-slate-400');
  });

  const activeContent = document.getElementById(`tab-${tabId}`);
  const activeBtn = document.getElementById(`tab-btn-${tabId}`);

  if (activeContent) activeContent.classList.remove('hidden');
  if (activeBtn) {
    activeBtn.classList.add('active', 'text-white', 'bg-space-700', 'border-cyan-500/30');
    activeBtn.classList.remove('text-slate-400');
  }

  // Relayout Plotly charts on tab change to prevent rendering issues
  if (tabId === 'dataset') {
    Plotly.relayout('chart-redshift-hist', {});
    Plotly.relayout('chart-splits-pie', {});
  } else if (tabId === 'desom1') {
    loadSampleSpectra();
    loadDESOM1Figures();
  } else if (tabId === 'desom2') {
    loadDESOM2Figures();
  } else if (tabId === 'gallery') {
    loadFiguresGallery();
  }
}

// ============================================================================
// Hardware & Experiments
// ============================================================================
async function initHardwareInfo() {
  try {
    const res = await fetch('/api/hardware');
    const data = await res.json();
    const badge = document.getElementById('hw-text');
    if (data.backend === 'gpu') {
      badge.textContent = `GPU: ${data.devices[0] || 'NVIDIA'} (Mem Growth: Activo)`;
    } else if (data.backend === 'apple_silicon') {
      badge.textContent = `Apple Silicon (Metal MPS)`;
    } else {
      badge.textContent = `CPU Multi-hilo`;
    }
  } catch (e) {
    console.error('Error fetching hardware info:', e);
  }
}

async function loadExperiments() {
  try {
    const res = await fetch('/api/experiments');
    const runs = await res.json();
    cachedRunsList = runs;

    const select = document.getElementById('run-select');
    select.innerHTML = '';

    const countBadge = document.getElementById('modal-runs-count');
    if (countBadge) countBadge.textContent = runs.length;

    if (runs.length === 0) {
      select.innerHTML = '<option value="">Sin corridas</option>';
      return;
    }

    let activeRun = null;
    runs.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.run_id;
      const countLabel = (r.n_galaxies > 0) ? `${r.n_galaxies} galaxias` : `${r.n_target} obj`;
      opt.textContent = `${r.run_id} (${countLabel})`;
      if (r.is_active) {
        opt.selected = true;
        activeRun = r;
      }
      select.appendChild(opt);
    });

    if (activeRun) {
      const nameBanner = document.getElementById('active-run-name-banner');
      const statsBanner = document.getElementById('active-run-stats-banner');
      if (nameBanner) nameBanner.textContent = activeRun.run_id;
      if (statsBanner) {
        const galText = activeRun.n_galaxies > 0 ? `${activeRun.n_galaxies} galaxias procesadas` : `${activeRun.n_target} galaxias objetivo`;
        statsBanner.textContent = galText;
      }
    }

    select.onchange = async () => {
      await fetch(`/api/switch_run?run_id=${encodeURIComponent(select.value)}`, { method: 'POST' });
      refreshAll();
    };
  } catch (e) {
    console.error('Error loading experiments:', e);
  }
}

// ============================================================================
// Modal & Runs Management
// ============================================================================
function openNewRunModal() {
  const modal = document.getElementById('modal-runs');
  if (modal) modal.classList.remove('hidden');
  switchModalTab('create');
  const sampleEl = document.getElementById('cfg-sample-target');
  const s = sampleEl ? sampleEl.value : '10';
  const sampleInput = document.getElementById('new-run-sample-input');
  const idInput = document.getElementById('new-run-id-input');
  if (sampleInput) sampleInput.value = s;
  if (idInput) {
    const existingIds = cachedRunsList.map(r => r.run_id);
    let candidate = `run_${s}g`;
    let counter = 2;
    while (existingIds.includes(candidate)) {
      candidate = `run_${s}g_${counter}`;
      counter++;
    }
    idInput.value = candidate;
  }
}

function openManageRunsModal() {
  const modal = document.getElementById('modal-runs');
  if (modal) modal.classList.remove('hidden');
  switchModalTab('history');
}

function closeExperimentsModal() {
  const modal = document.getElementById('modal-runs');
  if (modal) modal.classList.add('hidden');
}

function switchModalTab(tab) {
  const createSec = document.getElementById('modal-sec-create');
  const historySec = document.getElementById('modal-sec-history');
  const createBtn = document.getElementById('modal-tab-create-btn');
  const historyBtn = document.getElementById('modal-tab-history-btn');

  if (tab === 'create') {
    if (createSec) createSec.classList.remove('hidden');
    if (historySec) historySec.classList.add('hidden');
    if (createBtn) createBtn.className = 'pb-3 text-xs font-semibold text-white border-b-2 border-cyan-500 flex items-center space-x-1.5 transition';
    if (historyBtn) historyBtn.className = 'pb-3 text-xs font-semibold text-slate-400 hover:text-white border-b-2 border-transparent flex items-center space-x-1.5 transition';
  } else {
    if (createSec) createSec.classList.add('hidden');
    if (historySec) historySec.classList.remove('hidden');
    if (createBtn) createBtn.className = 'pb-3 text-xs font-semibold text-slate-400 hover:text-white border-b-2 border-transparent flex items-center space-x-1.5 transition';
    if (historyBtn) historyBtn.className = 'pb-3 text-xs font-semibold text-white border-b-2 border-purple-500 flex items-center space-x-1.5 transition';
    renderRunsManagerList();
  }
}

function renderRunsManagerList() {
  const container = document.getElementById('modal-runs-list');
  if (!container) return;

  if (cachedRunsList.length === 0) {
    container.innerHTML = '<p class="text-xs text-slate-500 text-center py-6">No hay corridas guardadas aún.</p>';
    return;
  }

  container.innerHTML = '';
  cachedRunsList.forEach(r => {
    const card = document.createElement('div');
    const isActive = r.is_active;
    card.className = `p-3.5 rounded-xl border transition flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 ${
      isActive ? 'bg-space-700/60 border-cyan-500/50 shadow-lg shadow-cyan-950/20' : 'bg-space-900/60 border-space-700 hover:border-space-600'
    }`;

    const stages = r.stages || {};
    const s1 = stages.stage_01_spectra?.status === 'complete';
    const s2 = stages.stage_02_dataset?.status === 'complete';
    const s3 = stages.stage_03_desom1?.status === 'complete';
    const s4 = stages.stage_05_desom2?.status === 'complete';

    const pill = (done, label) => done 
      ? `<span class="px-1.5 py-0.5 rounded text-[9px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">E${label} ✓</span>`
      : `<span class="px-1.5 py-0.5 rounded text-[9px] font-medium bg-space-800 text-slate-500 border border-space-700">E${label}</span>`;

    const galLabel = r.n_galaxies > 0 ? `${r.n_galaxies} galaxias` : `${r.n_target} objetivo`;

    card.innerHTML = `
      <div class="space-y-1">
        <div class="flex items-center space-x-2">
          <span class="text-xs font-bold text-white font-mono">${r.run_id}</span>
          ${isActive ? '<span class="text-[9px] px-2 py-0.2 rounded-full bg-cyan-500/10 text-astro-cyan border border-cyan-500/30 font-semibold">ACTIVA</span>' : ''}
          <span class="text-[10px] text-slate-400">(${galLabel})</span>
        </div>
        <div class="flex items-center space-x-1.5 pt-0.5">
          ${pill(s1, '1')}
          ${pill(s2, '2')}
          ${pill(s3, '3')}
          ${pill(s4, '4')}
          <span class="text-[10px] text-slate-500 ml-2">Actualizado: ${r.updated_at ? r.updated_at.split('T')[0] : 'N/A'}</span>
        </div>
      </div>
      <div>
        ${isActive 
          ? '<button disabled class="px-3 py-1.5 bg-cyan-600/20 border border-cyan-500/30 text-astro-cyan rounded-lg text-xs font-semibold cursor-default">Corrida Activa</button>'
          : `<button onclick="selectAndContinueRun('${r.run_id}')" class="px-3 py-1.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg text-xs font-semibold shadow transition flex items-center space-x-1.5">
               <i class="fa-solid fa-play text-[9px]"></i>
               <span>Cargar y Continuar</span>
             </button>`
        }
      </div>
    `;
    container.appendChild(card);
  });
}

async function selectAndContinueRun(runId) {
  try {
    const res = await fetch(`/api/switch_run?run_id=${encodeURIComponent(runId)}`, { method: 'POST' });
    if (res.ok) {
      closeExperimentsModal();
      refreshAll();
      switchTab('pipeline');
    }
  } catch (e) {
    console.error('Error switching run:', e);
  }
}

async function submitCreateRun() {
  const idInput = document.getElementById('new-run-id-input');
  const sampleInput = document.getElementById('new-run-sample-input');
  const cloneRadio = document.querySelector('input[name="new-run-clone-opt"]:checked');

  const runId = idInput ? idInput.value.trim() : '';
  const sample = sampleInput ? (parseInt(sampleInput.value) || 10) : 10;
  const cloneCurrent = cloneRadio ? (cloneRadio.value === 'current') : true;

  if (!runId) {
    alert('Por favor escribe un nombre o ID para la nueva corrida.');
    return;
  }

  try {
    const res = await fetch('/api/runs/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        run_id: runId,
        n_target: sample,
        clone_current: cloneCurrent,
      })
    });
    const data = await res.json();
    if (res.ok && data.status === 'created') {
      closeExperimentsModal();
      refreshAll();
      switchTab('pipeline');
      alert(`✓ Corrida '${runId}' creada exitosamente con muestra de ${sample} galaxias.`);
    } else {
      alert(`Error creando corrida: ${data.message || 'No se pudo crear'}`);
    }
  } catch (e) {
    alert(`Error de red al crear corrida: ${e}`);
  }
}

// ============================================================================
// Demographic Statistics & Charts
// ============================================================================
async function loadDemographics() {
  try {
    const res = await fetch('/api/dataset/demographics');
    const data = await res.json();

    if (data.error) {
      document.getElementById('kpi-total-galaxies').textContent = '0';
      document.getElementById('kpi-split-summary').textContent = data.error;
      return;
    }

    // 1. KPIs
    document.getElementById('kpi-total-galaxies').textContent = data.n_total.toLocaleString();
    const splits = data.splits || {};
    const trainN = splits.train || 0;
    const valN = splits.validation || 0;
    const testN = splits.test || 0;
    document.getElementById('kpi-split-summary').innerHTML =
      `<span class="text-emerald-400">${trainN} train</span> · <span class="text-purple-400">${valN} val</span> · <span class="text-cyan-400">${testN} test</span>`;

    const z = data.redshift || {};
    document.getElementById('kpi-z-median').textContent = z.median !== undefined ? z.median.toFixed(4) : '0.000';
    document.getElementById('kpi-z-range').textContent = `${(z.min || 0).toFixed(4)} – ${(z.max || 0).toFixed(4)}`;
    document.getElementById('kpi-z-mean').textContent = z.mean !== undefined ? z.mean.toFixed(4) : '0.000';
    document.getElementById('kpi-z-std').textContent = `± ${(z.std || 0).toFixed(4)}`;

    const bax = data.baxels || {};
    if (bax.mean) {
      document.getElementById('kpi-baxels-mean').textContent = Math.round(bax.mean).toLocaleString();
      document.getElementById('kpi-total-spectra').textContent = `${(bax.total_spectra || 0).toLocaleString()} baxels`;
    } else {
      document.getElementById('kpi-baxels-mean').textContent = 'N/A';
      document.getElementById('kpi-total-spectra').textContent = 'Pendiente Etapa 1';
    }

    // 2. Redshift Histogram (Plotly)
    if (z.hist_bins && z.hist_counts) {
      const trace = {
        x: z.hist_bins,
        y: z.hist_counts,
        type: 'bar',
        marker: {
          color: '#38bdf8',
          opacity: 0.85,
          line: { color: '#0284c7', width: 1 }
        },
        name: 'Galaxias'
      };

      const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { l: 40, r: 20, t: 10, b: 40 },
        xaxis: {
          title: { text: 'Redshift (z)', font: { size: 11, color: '#94a3b8' } },
          tickfont: { color: '#94a3b8', size: 10 },
          gridcolor: '#1e293b'
        },
        yaxis: {
          title: { text: 'N° Galaxias', font: { size: 11, color: '#94a3b8' } },
          tickfont: { color: '#94a3b8', size: 10 },
          gridcolor: '#1e293b'
        },
        shapes: [{
          type: 'line',
          x0: z.median,
          x1: z.median,
          y0: 0,
          y1: Math.max(...z.hist_counts),
          line: { color: '#fbbf24', width: 2, dash: 'dot' }
        }],
        annotations: [{
          x: z.median,
          y: Math.max(...z.hist_counts),
          text: `Mediana: ${z.median.toFixed(4)}`,
          showarrow: true,
          arrowhead: 2,
          arrowcolor: '#fbbf24',
          font: { size: 10, color: '#fbbf24' }
        }]
      };

      Plotly.newPlot('chart-redshift-hist', [trace], layout, { responsive: true, displayModeBar: false });
    }

    // 3. Splits Donut Chart (Plotly)
    if (Object.keys(splits).length > 0) {
      const pieTrace = {
        values: [splits.train || 0, splits.validation || 0, splits.test || 0],
        labels: ['Train (70%)', 'Validation (15%)', 'Test (15%)'],
        type: 'pie',
        hole: 0.55,
        marker: {
          colors: ['#10b981', '#a855f7', '#06b6d4']
        },
        textinfo: 'label+percent',
        textfont: { size: 10, color: '#ffffff' }
      };

      const pieLayout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { l: 10, r: 10, t: 10, b: 10 },
        showlegend: false
      };

      Plotly.newPlot('chart-splits-pie', [pieTrace], pieLayout, { responsive: true, displayModeBar: false });
    }

    // 4. Galaxy Table
    allGalaxiesData = data.sample_galaxies || [];
    renderGalaxyTable(allGalaxiesData);

  } catch (e) {
    console.error('Error loading demographics:', e);
  }
}

function renderGalaxyTable(galaxies) {
  const tbody = document.getElementById('galaxy-table-body');
  tbody.innerHTML = '';

  if (galaxies.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="px-5 py-6 text-center text-slate-500">No hay galaxias para mostrar.</td></tr>';
    return;
  }

  galaxies.forEach((g, idx) => {
    const tr = document.createElement('tr');
    tr.className = 'hover:bg-space-700/40 transition';
    const splitColor = g.split === 'train' ? 'text-emerald-400 bg-emerald-500/10' :
                       g.split === 'validation' ? 'text-purple-400 bg-purple-500/10' :
                       'text-cyan-400 bg-cyan-500/10';

    tr.innerHTML = `
      <td class="px-5 py-2.5 font-mono text-slate-500">${idx + 1}</td>
      <td class="px-5 py-2.5 font-medium text-white">${g.plateifu}</td>
      <td class="px-5 py-2.5 text-slate-300 font-mono">${g.mangaID}</td>
      <td class="px-5 py-2.5 text-slate-200">${Number(g.z).toFixed(5)}</td>
      <td class="px-5 py-2.5">
        <span class="px-2 py-0.5 rounded-full text-[10px] font-semibold ${splitColor}">${g.split}</span>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function filterGalaxyTable() {
  const query = document.getElementById('galaxy-search').value.toLowerCase().trim();
  const splitFilter = document.getElementById('split-filter').value;

  const filtered = allGalaxiesData.filter(g => {
    const matchQuery = (g.plateifu && g.plateifu.toLowerCase().includes(query)) ||
                       (g.mangaID && g.mangaID.toLowerCase().includes(query));
    const matchSplit = splitFilter === 'all' || g.split === splitFilter;
    return matchQuery && matchSplit;
  });

  renderGalaxyTable(filtered);
}

// ============================================================================
// Diagnostics
// ============================================================================
async function loadDiagnostics() {
  try {
    const res = await fetch('/api/preprocessing/diagnostics');
    const data = await res.json();
    if (!data.error) {
      document.getElementById('diag-retention-rate').textContent = `${data.galaxy_retention_rate}%`;
      document.getElementById('diag-accepted-baxels').textContent = `${(data.total_accepted_baxels || 0).toLocaleString()} baxels`;
    }
  } catch (e) {
    console.error('Error loading diagnostics:', e);
  }
}

// ============================================================================
// Pipeline State & Execution
// ============================================================================
async function loadPipelineState() {
  try {
    const res = await fetch('/api/pipeline/state');
    const data = await res.json();
    const stages = data.stages || {};

    const updateBadge = (stageNum, stageKey) => {
      const badge = document.getElementById(`badge-stage-${stageNum}`);
      if (!badge) return;
      const st = stages[stageKey]?.status || 'pending';
      if (st === 'complete') {
        badge.className = 'text-xs px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30';
        badge.textContent = '✓ Completa';
      } else if (st === 'running') {
        badge.className = 'text-xs px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30 animate-pulse';
        badge.textContent = '⏳ Ejecutando';
      } else {
        badge.className = 'text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 border border-slate-700';
        badge.textContent = 'Pendiente';
      }
    };

    updateBadge(1, 'stage_01_spectra');
    updateBadge(2, 'stage_02_dataset');
    updateBadge(3, 'stage_03_desom1');
    updateBadge(4, 'stage_05_desom2');

    // Compute next recommended step
    const s1 = stages.stage_01_spectra?.status === 'complete';
    const s2 = stages.stage_02_dataset?.status === 'complete';
    const s3 = stages.stage_03_desom1?.status === 'complete';
    const s4 = stages.stage_05_desom2?.status === 'complete';

    let nextText = 'Comienza con la Etapa 1 (Voronoi & Extracción).';
    if (!s1) {
      nextText = 'Siguiente: Ejecutar Etapa 1 (Voronoi & Extracción).';
    } else if (!s2) {
      nextText = 'Etapa 1 completada. Siguiente: Ejecutar Etapa 2 (Auditoría & Figura 1).';
    } else if (!s3) {
      nextText = 'Etapas 1 y 2 listas. Siguiente: Ejecutar Etapa 3 (DESOM-1 & Huellas 2D).';
    } else if (!s4) {
      nextText = 'DESOM-1 listo. Siguiente: Ejecutar Etapa 4 (DESOM-2 & Mapa 10×10).';
    } else {
      nextText = '✓ Pipeline completado al 100%. Revisa todas las figuras en la pestaña Galería.';
    }

    const nextEl = document.getElementById('active-run-next-step');
    if (nextEl) nextEl.textContent = nextText;

  } catch (e) {
    console.error('Error loading pipeline state:', e);
  }
}

async function launchStage(stageNum) {
  const sampleEl = document.getElementById('cfg-sample-target') || document.getElementById('cfg-sample');
  const sample = sampleEl ? sampleEl.value : '500';
  const consoleBox = document.getElementById('console-logs');
  const indicator = document.getElementById('process-indicator');

  consoleBox.textContent = `[Iniciando Etapa ${stageNum} con muestra de ${sample} galaxias...]\n`;
  indicator.textContent = 'Ejecutando...';
  indicator.className = 'text-[11px] text-amber-400 font-semibold animate-pulse';

  try {
    // Auto-save current form settings before running
    const storageEl = document.getElementById('cfg-storage-backend');
    if (storageEl) {
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n_target: parseInt(sample) || 500,
          storage: { backend: storageEl.value }
        })
      });
    }

    const res = await fetch('/api/pipeline/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stage: stageNum, sample_size: parseInt(sample) || 500 })
    });
    const result = await res.json();

    if (result.status === 'started') {
      consoleBox.textContent += `Proceso lanzado (PID: ${result.pid})\nComando: ${result.cmd}\n\n`;
      startLogPolling(stageNum);
    } else {
      consoleBox.textContent += `Error: ${result.message}\n`;
      indicator.textContent = 'Error';
      indicator.className = 'text-[11px] text-rose-400 font-semibold';
    }
  } catch (e) {
    consoleBox.textContent += `Fallo de conexión al lanzar etapa: ${e}\n`;
    indicator.textContent = 'Fallo';
    indicator.className = 'text-[11px] text-rose-400 font-semibold';
  }
}

function startLogPolling(stageNum) {
  if (processPollInterval) clearInterval(processPollInterval);

  processPollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/api/pipeline/logs?stage=${stageNum}`);
      const text = await res.text();
      const consoleBox = document.getElementById('console-logs');
      if (text) {
        consoleBox.textContent = text;
        consoleBox.scrollTop = consoleBox.scrollHeight;
      }

      // Check status
      const statusRes = await fetch('/api/pipeline/status');
      const st = await statusRes.json();
      if (!st.is_running) {
        clearInterval(processPollInterval);
        processPollInterval = null;
        const indicator = document.getElementById('process-indicator');
        indicator.textContent = st.exit_code === 0 ? '✓ Terminado con éxito' : `Finalizado (código ${st.exit_code})`;
        indicator.className = st.exit_code === 0 ? 'text-[11px] text-emerald-400 font-semibold' : 'text-[11px] text-rose-400 font-semibold';
        refreshAll();
      }
    } catch (e) {
      console.error('Log polling error:', e);
    }
  }, 2000);
}

// ============================================================================
// Sample Spectra Viewer (Plotly Fig 1 Replicant)
// ============================================================================
async function loadSampleSpectra() {
  const container = document.getElementById('chart-spectra-viewer');
  container.innerHTML = '<p class="text-xs text-slate-500 text-center py-10">Cargando espectros normalizados...</p>';

  try {
    const res = await fetch('/api/spectra/sample');
    const data = await res.json();

    if (data.error) {
      container.innerHTML = `<p class="text-xs text-amber-400 text-center py-10">${data.error}</p>`;
      return;
    }

    const lines = {
      '[OII] 3727': 3727.0,
      'Hβ 4861': 4861.0,
      '[OIII] 5007': 5007.0,
      'Hα 6563': 6563.0,
      '[NII] 6584': 6584.0,
      '[SII] 6717': 6717.0,
    };

    const traces = [];
    const colors = ['#38bdf8', '#a855f7', '#ec4899', '#34d399', '#fbbf24'];

    data.spectra.forEach((s, idx) => {
      traces.push({
        x: data.wavelength,
        y: s.flux.map(f => f + idx * 2.0),
        mode: 'lines',
        name: `Espectro #${s.index}`,
        line: { color: colors[idx % colors.length], width: 1.2 }
      });
    });

    // Vertical lines for key emission features
    const shapes = Object.values(lines).map(w => ({
      type: 'line',
      x0: w,
      x1: w,
      y0: 0,
      y1: data.spectra.length * 2.5 + 2,
      line: { color: '#64748b', width: 1, dash: 'dot' }
    }));

    const annotations = Object.entries(lines).map(([name, w]) => ({
      x: w,
      y: data.spectra.length * 2.0 + 1.2,
      text: name,
      showarrow: false,
      textangle: -90,
      font: { size: 9, color: '#cbd5e1' }
    }));

    const layout = {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      margin: { l: 40, r: 20, t: 25, b: 40 },
      xaxis: {
        title: { text: 'Longitud de Onda en Reposo [Å]', font: { size: 11, color: '#94a3b8' } },
        tickfont: { color: '#94a3b8', size: 10 },
        gridcolor: '#1e293b'
      },
      yaxis: {
        title: { text: 'Flujo Normalizado + Offset', font: { size: 11, color: '#94a3b8' } },
        tickfont: { color: '#94a3b8', size: 10 },
        gridcolor: '#1e293b'
      },
      shapes: shapes,
      annotations: annotations,
      legend: { font: { color: '#cbd5e1', size: 10 }, orientation: 'h', y: 1.15 }
    };

    Plotly.newPlot('chart-spectra-viewer', traces, layout, { responsive: true, displayModeBar: true });

  } catch (e) {
    container.innerHTML = `<p class="text-xs text-rose-400 text-center py-10">Error cargando espectros: ${e}</p>`;
  }
}

// ============================================================================
// DESOM-1 and DESOM-2 Figure Loaders for Tabs 4 & 5
// ============================================================================
async function loadDESOM1Figures() {
  const fig4Card = document.getElementById('fig4-preview-container');
  const fig5Card = document.getElementById('fig5-preview-container');
  if (!fig4Card || !fig5Card) return;

  try {
    const res = await fetch('/api/figures');
    const figures = await res.json();
    const fig4 = figures.find(f => f.name.includes('figure_3'));
    const fig5 = figures.find(f => f.name.includes('extra_sample_fingerprints'));

    if (fig4) {
      fig4Card.innerHTML = `
        <div class="space-y-3">
          <div class="rounded-xl overflow-hidden bg-space-950 border border-space-700 cursor-pointer" onclick="window.open('${fig4.url}', '_blank')">
            <img src="${fig4.url}?t=${Date.now()}" alt="${fig4.name}" class="w-full h-80 object-contain hover:scale-105 transition duration-300">
          </div>
          <div class="flex items-center justify-between text-xs text-slate-400">
            <span>${fig4.name} (${fig4.size_kb} KB)</span>
            <a href="${fig4.url}" download class="text-astro-cyan hover:underline flex items-center space-x-1">
              <i class="fa-solid fa-download text-xs"></i>
              <span>Descargar PNG</span>
            </a>
          </div>
        </div>
      `;
    } else {
      fig4Card.innerHTML = `
        <div class="p-8 text-center bg-space-900/60 rounded-xl border border-space-700/60 text-slate-500 text-xs">
          <i class="fa-solid fa-hourglass-start text-2xl text-slate-600 mb-2"></i>
          <p>Pendiente: Ejecuta la Etapa 3 para generar la malla 15×15 de prototipos espectrales.</p>
        </div>
      `;
    }

    if (fig5) {
      fig5Card.innerHTML = `
        <div class="space-y-3">
          <div class="rounded-xl overflow-hidden bg-space-950 border border-space-700 cursor-pointer" onclick="window.open('${fig5.url}', '_blank')">
            <img src="${fig5.url}?t=${Date.now()}" alt="${fig5.name}" class="w-full h-80 object-contain hover:scale-105 transition duration-300">
          </div>
          <div class="flex items-center justify-between text-xs text-slate-400">
            <span>${fig5.name} (${fig5.size_kb} KB)</span>
            <a href="${fig5.url}" download class="text-astro-pink hover:underline flex items-center space-x-1">
              <i class="fa-solid fa-download text-xs"></i>
              <span>Descargar PNG</span>
            </a>
          </div>
        </div>
      `;
    } else {
      fig5Card.innerHTML = `
        <div class="p-8 text-center bg-space-900/60 rounded-xl border border-space-700/60 text-slate-500 text-xs">
          <i class="fa-solid fa-hourglass-start text-2xl text-slate-600 mb-2"></i>
          <p>Pendiente: Ejecuta la Etapa 3 para proyectar las huellas 2D de las galaxias.</p>
        </div>
      `;
    }
  } catch (e) {
    console.error('Error loading DESOM-1 figures:', e);
  }
}

async function loadDESOM2Figures() {
  const fig10Card = document.getElementById('fig10-preview-container');
  if (!fig10Card) return;

  try {
    const res = await fetch('/api/figures');
    const figures = await res.json();
    const fig10 = figures.find(f => f.name.includes('figure_10'));

    if (fig10) {
      fig10Card.innerHTML = `
        <div class="space-y-3">
          <div class="rounded-xl overflow-hidden bg-space-950 border border-space-700 cursor-pointer" onclick="window.open('${fig10.url}', '_blank')">
            <img src="${fig10.url}?t=${Date.now()}" alt="${fig10.name}" class="w-full max-h-[550px] object-contain hover:scale-105 transition duration-300">
          </div>
          <div class="flex items-center justify-between text-xs text-slate-400">
            <span>${fig10.name} (${fig10.size_kb} KB) — 100 Neuronas de Diversidad Morfológica/Espectral</span>
            <a href="${fig10.url}" download class="text-astro-purple hover:underline flex items-center space-x-1">
              <i class="fa-solid fa-download text-xs"></i>
              <span>Descargar PNG Alta Resolución</span>
            </a>
          </div>
        </div>
      `;
    } else {
      fig10Card.innerHTML = `
        <div class="p-10 text-center bg-space-900/60 rounded-xl border border-space-700/60 text-slate-500 text-xs">
          <i class="fa-solid fa-network-wired text-3xl text-slate-600 mb-3"></i>
          <p class="font-semibold text-slate-400">Pendiente de Ejecución</p>
          <p class="mt-1">Ejecuta la Etapa 4 para entrenar DESOM-2 y generar la cuadrícula 10×10 de diversidad de galaxias.</p>
        </div>
      `;
    }
  } catch (e) {
    console.error('Error loading DESOM-2 figures:', e);
  }
}

// ============================================================================
// Figures & Files Gallery
// ============================================================================
async function loadFiguresGallery() {
  const container = document.getElementById('figures-gallery-grid');
  try {
    const res = await fetch('/api/figures');
    const figures = await res.json();

    if (figures.length === 0) {
      container.innerHTML = '<p class="text-xs text-slate-500 col-span-3 text-center py-8">Aún no se han generado figuras en esta corrida. Ejecuta las Etapas 2, 3 o 4.</p>';
      return;
    }

    container.innerHTML = '';
    figures.forEach(fig => {
      const card = document.createElement('div');
      card.className = 'bg-space-900 rounded-xl border border-space-700 overflow-hidden hover:border-space-600 transition flex flex-col justify-between';

      let preview = '';
      if (fig.is_image) {
        preview = `<div class="h-44 bg-space-800 flex items-center justify-center overflow-hidden cursor-pointer" onclick="window.open('${fig.url}', '_blank')">
          <img src="${fig.url}" class="object-contain h-full w-full hover:scale-105 transition duration-300" alt="${fig.name}">
        </div>`;
      } else {
        preview = `<div class="h-44 bg-space-800 flex items-center justify-center text-slate-500">
          <i class="fa-solid fa-file-pdf text-4xl text-rose-400/80"></i>
        </div>`;
      }

      card.innerHTML = `
        ${preview}
        <div class="p-3 border-t border-space-700 flex items-center justify-between">
          <div>
            <span class="text-xs font-semibold text-white truncate block max-w-[180px]">${fig.name}</span>
            <span class="text-[10px] text-slate-500">${fig.size_kb} KB</span>
          </div>
          <a href="${fig.url}" target="_blank" download class="p-2 text-astro-cyan hover:bg-space-800 rounded-lg transition" title="Descargar">
            <i class="fa-solid fa-download text-xs"></i>
          </a>
        </div>
      `;
      container.appendChild(card);
    });

  } catch (e) {
    console.error('Error loading gallery:', e);
  }
}

// ============================================================================
// Accordion Toggle & Configuration Management
// ============================================================================
function toggleSection(secId) {
  const el = document.getElementById(secId);
  const icon = document.getElementById(`${secId}-icon`);
  if (!el) return;
  if (el.classList.contains('hidden')) {
    el.classList.remove('hidden');
    if (icon) icon.classList.remove('rotate-180');
  } else {
    el.classList.add('hidden');
    if (icon) icon.classList.add('rotate-180');
  }
}

async function loadFullConfig() {
  try {
    const res = await fetch('/api/config');
    const cfg = await res.json();

    // 1. Preprocessing
    const pre = cfg.preprocessing || {};
    if (pre.rest_limits) document.getElementById('cfg-pre-rest-limits').value = JSON.stringify(pre.rest_limits);
    if (pre.n_wave) document.getElementById('cfg-pre-n-wave').value = pre.n_wave;
    if (pre.target_sn) document.getElementById('cfg-pre-target-sn').value = pre.target_sn;
    if (pre.min_spaxel_sn) document.getElementById('cfg-pre-min-spaxel-sn').value = pre.min_spaxel_sn;
    if (pre.min_baxel_sn !== undefined) document.getElementById('cfg-pre-min-baxel-sn').value = pre.min_baxel_sn;
    if (pre.continuum_rest) document.getElementById('cfg-pre-continuum-rest').value = JSON.stringify(pre.continuum_rest);
    if (pre.max_interpolation_gap_A) document.getElementById('cfg-pre-max-gap').value = pre.max_interpolation_gap_A;
    if (pre.min_good_rebinned) document.getElementById('cfg-pre-min-good').value = pre.min_good_rebinned;
    if (pre.covariance_in_binning !== undefined) document.getElementById('cfg-pre-covariance').checked = pre.covariance_in_binning;

    // 2. Sample
    if (cfg.n_target) document.getElementById('cfg-sample-target').value = cfg.n_target;
    if (cfg.seed) document.getElementById('cfg-sample-seed').value = cfg.seed;
    const smp = cfg.sample || {};
    if (smp.min_redshift !== undefined) document.getElementById('cfg-sample-zmin').value = smp.min_redshift;
    if (smp.max_redshift !== undefined) document.getElementById('cfg-sample-zmax').value = smp.max_redshift;
    if (smp.split_ratios) document.getElementById('cfg-sample-splits').value = JSON.stringify(smp.split_ratios);
    if (smp.split_mode) document.getElementById('cfg-sample-split-mode').value = smp.split_mode;
    const trn = cfg.training || {};
    if (trn.map_distance) document.getElementById('cfg-train-map-distance').value = trn.map_distance;

    // 3. DESOM-1
    const d1 = cfg.desom1 || {};
    if (d1.map_size) document.getElementById('cfg-d1-map-size').value = JSON.stringify(d1.map_size);
    if (d1.latent) document.getElementById('cfg-d1-latent').value = d1.latent;
    if (d1.iterations) document.getElementById('cfg-d1-iterations').value = d1.iterations;
    if (d1.som_iterations) document.getElementById('cfg-d1-som-iterations').value = d1.som_iterations;
    if (d1.gamma !== undefined) document.getElementById('cfg-d1-gamma').value = d1.gamma;
    if (d1.lr !== undefined) document.getElementById('cfg-d1-lr').value = d1.lr;
    if (d1.batch_size) document.getElementById('cfg-d1-batch-size').value = d1.batch_size;
    if (d1.Tmax !== undefined && d1.Tmin !== undefined) document.getElementById('cfg-d1-temps').value = JSON.stringify([d1.Tmax, d1.Tmin]);

    // 4. DESOM-2
    const d2 = cfg.desom2 || {};
    if (d2.map_size) document.getElementById('cfg-d2-map-size').value = JSON.stringify(d2.map_size);
    if (d2.latent_dims) document.getElementById('cfg-d2-latent').value = d2.latent_dims;
    if (d2.iterations) document.getElementById('cfg-d2-iterations').value = d2.iterations;
    if (d2.som_iterations) document.getElementById('cfg-d2-som-iterations').value = d2.som_iterations;
    if (d2.gamma !== undefined) document.getElementById('cfg-d2-gamma').value = d2.gamma;
    if (d2.lr !== undefined) document.getElementById('cfg-d2-lr').value = d2.lr;
    if (d2.batch_size) document.getElementById('cfg-d2-batch-size').value = d2.batch_size;
    if (d2.clipping_percentile) document.getElementById('cfg-d2-clipping').value = d2.clipping_percentile;

    // 5. Storage
    const stg = cfg.storage || {};
    if (stg.backend) document.getElementById('cfg-storage-backend').value = stg.backend;
    if (stg.local_root) document.getElementById('cfg-storage-local').value = stg.local_root;
    if (stg.drive_root) document.getElementById('cfg-storage-drive').value = stg.drive_root;

  } catch (e) {
    console.error('Error loading config:', e);
  }
}

async function saveFullConfig() {
  try {
    const parseSafe = (val, fallback) => {
      try { return JSON.parse(val); } catch (e) { return fallback; }
    };

    const d1Temps = parseSafe(document.getElementById('cfg-d1-temps').value, [15.0, 0.7]);

    const updated = {
      n_target: parseInt(document.getElementById('cfg-sample-target').value) || 500,
      seed: parseInt(document.getElementById('cfg-sample-seed').value) || 20261002,
      preprocessing: {
        rest_limits: parseSafe(document.getElementById('cfg-pre-rest-limits').value, [3650.0, 8950.0]),
        n_wave: parseInt(document.getElementById('cfg-pre-n-wave').value) || 4544,
        target_sn: parseFloat(document.getElementById('cfg-pre-target-sn').value) || 20.0,
        min_spaxel_sn: parseFloat(document.getElementById('cfg-pre-min-spaxel-sn').value) || 2.0,
        min_baxel_sn: parseFloat(document.getElementById('cfg-pre-min-baxel-sn').value) || 10.0,
        continuum_rest: parseSafe(document.getElementById('cfg-pre-continuum-rest').value, [4050.0, 4150.0]),
        max_interpolation_gap_A: parseFloat(document.getElementById('cfg-pre-max-gap').value) || 50.0,
        min_good_rebinned: parseInt(document.getElementById('cfg-pre-min-good').value) || 3900,
        covariance_in_binning: document.getElementById('cfg-pre-covariance').checked,
      },
      sample: {
        min_redshift: parseFloat(document.getElementById('cfg-sample-zmin').value) || 0.0,
        max_redshift: parseFloat(document.getElementById('cfg-sample-zmax').value) || 0.15,
        split_ratios: parseSafe(document.getElementById('cfg-sample-splits').value, [0.70, 0.15, 0.15]),
        split_mode: document.getElementById('cfg-sample-split-mode').value,
      },
      training: {
        map_distance: document.getElementById('cfg-train-map-distance').value,
      },
      desom1: {
        map_size: parseSafe(document.getElementById('cfg-d1-map-size').value, [15, 15]),
        latent: parseInt(document.getElementById('cfg-d1-latent').value) || 256,
        iterations: parseInt(document.getElementById('cfg-d1-iterations').value) || 65000,
        som_iterations: parseInt(document.getElementById('cfg-d1-som-iterations').value) || 50000,
        gamma: parseFloat(document.getElementById('cfg-d1-gamma').value) || 0.0005,
        lr: parseFloat(document.getElementById('cfg-d1-lr').value) || 0.00005,
        batch_size: parseInt(document.getElementById('cfg-d1-batch-size').value) || 128,
        Tmax: parseFloat(d1Temps[0] || 15.0),
        Tmin: parseFloat(d1Temps[1] || 0.7),
      },
      desom2: {
        map_size: parseSafe(document.getElementById('cfg-d2-map-size').value, [10, 10]),
        latent_dims: parseInt(document.getElementById('cfg-d2-latent').value) || 128,
        iterations: parseInt(document.getElementById('cfg-d2-iterations').value) || 40000,
        som_iterations: parseInt(document.getElementById('cfg-d2-som-iterations').value) || 35000,
        gamma: parseFloat(document.getElementById('cfg-d2-gamma').value) || 0.000008,
        lr: parseFloat(document.getElementById('cfg-d2-lr').value) || 0.0001,
        batch_size: parseInt(document.getElementById('cfg-d2-batch-size').value) || 1024,
        clipping_percentile: parseFloat(document.getElementById('cfg-d2-clipping').value) || 99.9,
      },
      storage: {
        backend: document.getElementById('cfg-storage-backend').value,
        local_root: document.getElementById('cfg-storage-local').value,
        drive_root: document.getElementById('cfg-storage-drive').value,
      }
    };

    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updated)
    });
    if (!res.ok) {
      throw new Error(`El servidor respondió con código ${res.status}. Si el servidor ya estaba abierto antes de actualizar el código, reinícialo en la terminal con Ctrl+C y luego python3 app.py`);
    }
    const result = await res.json();
    alert('✓ Parámetros guardados y actualizados exitosamente en config.json');
    refreshAll();
  } catch (e) {
    alert(`Error guardando parámetros: ${e}`);
  }
}

async function resetConfigDefaults() {
  if (!confirm('¿Deseas restaurar todos los parámetros a los valores por defecto del Paper original?')) {
    return;
  }
  try {
    const res = await fetch('/api/config/reset', { method: 'POST' });
    if (!res.ok) {
      throw new Error(`El servidor respondió con código ${res.status}. Reinicia el servidor en la terminal.`);
    }
    const result = await res.json();
    alert('✓ Parámetros restaurados a los valores del Paper.');
    loadFullConfig();
  } catch (e) {
    alert(`Error restaurando parámetros: ${e}`);
  }
}

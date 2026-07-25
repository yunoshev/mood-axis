/* Mood Axis v2 site — all charts render from docs/site_data.json,
 * which is generated from v2/data/ by v2/build_site_data.py.
 * No numbers are hard-coded here. */

let D = null;          // site_data.json
let PG = null;         // playground.json (lazy)
const driftDetail = {}; // drift_<m>.json cache

const $ = (id) => document.getElementById(id);
const GROUP_ORDER = ["main", "extra", "giant", "twin"];
const GROUP_LABEL = { main: "Main cast", extra: "MoE extra", giant: "Giants (24–35B)", twin: "4-bit twins" };
const AXIS_COLORS = {
  warm_cold: "#f472b6", confident_cautious: "#60a5fa", empathetic_analytical: "#34d399",
  formal_casual: "#a78bfa", verbose_concise: "#fbbf24",
};
const charts = {};

function fmt(x, d = 3) { return (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d); }
function axisName(ax) { const l = D.axis_labels[ax]; return `${l[0]} ↔ ${l[1]}`; }
function poleFor(ax, v) { return D.axis_labels[ax][v >= 0 ? 0 : 1]; }
function modelsWith(field) {
  return Object.entries(D.models).filter(([, m]) => m[field]).map(([k]) => k)
    .sort((a, b) => GROUP_ORDER.indexOf(D.models[a].group) - GROUP_ORDER.indexOf(D.models[b].group));
}
function mkChart(id, cfg) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($(id).getContext("2d"), cfg);
  return charts[id];
}
function gridOpts(extra = {}) {
  return Object.assign({
    grid: { color: "rgba(51,65,85,0.5)" },
    ticks: { color: "#94a3b8", font: { size: 10 } },
  }, extra);
}
function badge(text, cls) { return `<span class="chip ${cls}">${text}</span>`; }
function sigBadge(v, yes = "significant", no = "n.s.") {
  if (v === null || v === undefined) return badge("n/a", "badge-ns");
  return v ? badge("✓ " + yes, "badge-sig") : badge(no, "badge-ns");
}

/* ---------------- Hero ---------------- */
let heroIdx = 0, heroTimer = null;
function radarData(key) {
  const m = D.models[key];
  return D.axes.map((ax) => {
    const f = (m.fingerprint || {})[ax];
    return f ? (f.mean + 1) / 2 : null;
  });
}
function renderHero() {
  const withFp = modelsWith("fingerprint");
  const orgs = new Set(Object.values(D.models).map((m) => m.org));
  $("stat-models").textContent = Object.keys(D.models).length;
  $("stat-axes").textContent = D.axes.length;
  $("stat-orgs").textContent = orgs.size;
  $("stat-version").textContent = D._meta.data_commit;
  $("footVersion").textContent = D._meta.data_commit;
  $("footBuilt").textContent = D._meta.generated_at;

  const labels = D.axes.map((ax) => D.axis_labels[ax][0]);
  const mk = (key) => ({
    label: D.models[key].name,
    data: radarData(key),
    borderColor: D.models[key].color,
    backgroundColor: D.models[key].color + "26",
    pointRadius: 3, borderWidth: 2,
  });
  const chart = mkChart("heroRadar", {
    type: "radar",
    data: { labels, datasets: [mk(withFp[0])] },
    options: {
      responsive: true, animation: { duration: 700 },
      plugins: { legend: { display: false } },
      scales: { r: {
        min: 0, max: 1, ticks: { display: false },
        grid: { color: "rgba(51,65,85,0.6)" }, angleLines: { color: "rgba(51,65,85,0.6)" },
        pointLabels: { color: "#cbd5e1", font: { size: 12 } },
      } },
    },
  });
  $("heroLabel").textContent = D.models[withFp[0]].name;
  heroTimer = setInterval(() => {
    heroIdx = (heroIdx + 1) % withFp.length;
    const key = withFp[heroIdx];
    chart.data.datasets = [mk(key)];
    chart.update();
    $("heroLabel").textContent = D.models[key].name;
  }, 2600);
}

/* ---------------- Fingerprints ---------------- */
function axisBarRow(ax, f) {
  const l = D.axis_labels[ax];
  const v = f.mean, sd = f.std;
  const pctLeft = v < 0 ? 50 + v * 50 : 50;
  const pctW = Math.abs(v) * 50;
  const sdL = Math.max(0, 50 + (v - sd) * 50), sdR = Math.min(100, 50 + (v + sd) * 50);
  const color = v >= 0 ? "#6366f1" : "#f59e0b";
  return `
    <div class="mb-2">
      <div class="flex justify-between text-[10px] text-slate-500 mb-0.5">
        <span>${l[1]}</span><span class="text-slate-400 mono">${fmt(v, 2)}</span><span>${l[0]}</span>
      </div>
      <div class="axis-bar-track">
        <div class="axis-bar-mid"></div>
        <div style="position:absolute;top:3px;bottom:3px;left:${sdL}%;width:${sdR - sdL}%;background:${color}22;border-radius:9999px"></div>
        <div class="axis-bar-fill" style="left:${pctLeft}%;width:${pctW}%;background:${color}"></div>
      </div>
    </div>`;
}
function renderFingerprints() {
  const grid = $("fpGrid");
  const keys = modelsWith("fingerprint");
  const n = D.models[keys[0]].fingerprint_n;
  $("fpN").textContent = n || "–";
  const groups = {};
  keys.forEach((k) => { (groups[D.models[k].group] ||= []).push(k); });
  let html = "";
  for (const g of GROUP_ORDER) {
    if (!groups[g]) continue;
    html += `<div class="sm:col-span-2 lg:col-span-3 mt-2 text-xs uppercase tracking-widest text-slate-500">${GROUP_LABEL[g]}</div>`;
    for (const k of groups[g]) {
      const m = D.models[k];
      const cal = m.calibration ? Object.values(m.calibration).map((c) => c.val_acc).filter((x) => x != null) : [];
      const acc = cal.length ? cal.reduce((a, b) => a + b, 0) / cal.length : null;
      const ex = (m.examples || []).map((e) =>
        `<p class="text-[11px] text-slate-500 mt-2"><span class="text-slate-400">Q: ${e.q}</span><br>${e.a}…</p>`).join("");
      html += `
      <div class="model-card bg-slate-800 rounded-xl p-4 border-slate-700/50" onclick="this.querySelector('.expandable-content').classList.toggle('expanded')">
        <div class="flex items-center justify-between mb-3">
          <div>
            <span class="font-semibold text-white text-sm">${m.name}</span>
            <span class="text-[10px] text-slate-500 ml-1">${m.org} · ${m.params}</span>
          </div>
          <span class="w-2.5 h-2.5 rounded-full" style="background:${m.color}"></span>
        </div>
        ${D.axes.map((ax) => m.fingerprint[ax] ? axisBarRow(ax, m.fingerprint[ax]) : "").join("")}
        <div class="flex gap-2 mt-3 flex-wrap">
          ${acc != null ? badge(`val acc <b>${fmt(acc, 2)}</b>`, "") : ""}
          ${m.stability ? badge(`ICC <b>${fmt(m.stability.icc, 2)}</b>`, "") : ""}
        </div>
        <div class="expandable-content">${ex}</div>
      </div>`;
    }
  }
  grid.innerHTML = html;
}

/* ---------------- Compare ---------------- */
const cmpSel = new Set();
function renderCompareChart() {
  const labels = D.axes.map((ax) => D.axis_labels[ax][0]);
  const datasets = [...cmpSel].map((key) => ({
    label: D.models[key].name, data: radarData(key),
    borderColor: D.models[key].color, backgroundColor: D.models[key].color + "1f",
    pointRadius: 3, borderWidth: 2,
  }));
  mkChart("cmpRadar", {
    type: "radar", data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 } } } },
      scales: { r: { min: 0, max: 1, ticks: { display: false },
        grid: { color: "rgba(51,65,85,0.6)" }, angleLines: { color: "rgba(51,65,85,0.6)" },
        pointLabels: { color: "#cbd5e1", font: { size: 12 } } } },
    },
  });
}
function renderCompare() {
  const keys = modelsWith("fingerprint");
  keys.slice(0, 3).forEach((k) => cmpSel.add(k));
  $("cmpChips").innerHTML = keys.map((k) => {
    const m = D.models[k];
    return `<button data-cmp="${k}" class="qtab ${cmpSel.has(k) ? "active" : ""}" style="border-color:${cmpSel.has(k) ? m.color : ""}">${m.name}</button>`;
  }).join("");
  $("cmpChips").querySelectorAll("[data-cmp]").forEach((b) => b.addEventListener("click", () => {
    const k = b.dataset.cmp;
    cmpSel.has(k) ? cmpSel.delete(k) : cmpSel.add(k);
    b.classList.toggle("active", cmpSel.has(k));
    b.style.borderColor = cmpSel.has(k) ? D.models[k].color : "";
    renderCompareChart();
  }));
  renderCompareChart();
}

/* ---------------- Stability ---------------- */
function renderStability() {
  const keys = modelsWith("stability");
  $("stabBody").innerHTML = keys.map((k) => {
    const m = D.models[k], s = m.stability;
    const cal = m.calibration ? Object.values(m.calibration) : [];
    const acc = cal.length ? cal.reduce((a, c) => a + (c.val_acc ?? 0), 0) / cal.length : null;
    const dp = cal.length ? cal.reduce((a, c) => a + (c.val_dprime ?? 0), 0) / cal.length : null;
    const cos = Object.values(s.cosine_per_axis || {});
    const mcos = cos.length ? cos.reduce((a, b) => a + b, 0) / cos.length : null;
    return `<tr class="border-b border-slate-700/40 hover:bg-slate-700/20 cursor-pointer" data-stab="${k}">
      <td class="px-4 py-2.5"><span class="inline-block w-2 h-2 rounded-full mr-2" style="background:${m.color}"></span>
        <span class="text-slate-200">${m.name}</span></td>
      <td class="px-3 py-2.5 mono">${fmt(acc, 2)}</td>
      <td class="px-3 py-2.5 mono">${fmt(dp, 1)}</td>
      <td class="px-3 py-2.5 mono">${fmt(s.icc, 2)}</td>
      <td class="px-3 py-2.5 mono">${fmt(s.rank_icc, 2)}</td>
      <td class="px-3 py-2.5 mono">${fmt(mcos, 2)}</td>
      <td class="px-3 py-2.5 mono">${fmt(s.procrustes, 3)}</td>
    </tr>`;
  }).join("");
  $("stabBody").querySelectorAll("[data-stab]").forEach((tr) => tr.addEventListener("click", () => {
    const k = tr.dataset.stab, m = D.models[k], s = m.stability;
    $("stabDetail").innerHTML = `
      <div class="bg-slate-800 rounded-xl p-4 border border-slate-700/50">
        <div class="text-sm text-white font-semibold mb-2">${m.name} — per axis</div>
        <div class="grid sm:grid-cols-5 gap-2 text-xs">
        ${D.axes.map((ax) => `
          <div class="bg-slate-900/60 rounded-lg p-2">
            <div class="text-slate-400 mb-1">${D.axis_labels[ax][0]}</div>
            <div class="text-slate-500">ICC <span class="text-slate-300 mono">${fmt((s.icc_per_axis || {})[ax], 2)}</span></div>
            <div class="text-slate-500">rank <span class="text-slate-300 mono">${fmt((s.rank_icc_per_axis || {})[ax], 2)}</span></div>
            <div class="text-slate-500">cos <span class="text-slate-300 mono">${fmt((s.cosine_per_axis || {})[ax], 2)}</span></div>
            <div class="text-slate-500">acc <span class="text-slate-300 mono">${fmt(((m.calibration || {})[ax] || {}).val_acc, 2)}</span></div>
          </div>`).join("")}
        </div>
      </div>`;
  }));
}

/* ---------------- Twins ---------------- */
function renderTwins() {
  const q = (D.analysis || {}).quant;
  if (!q) return;
  $("twinPooled").textContent = fmt(q.summary.pooled_mean, 3);
  $("twinPairs").textContent = (q.summary.n_models || 0) * D.axes.length;
  const entries = [...Object.entries(q.twins).map(([k, v]) => [k + " (bf16→4bit)", v]),
                   ...Object.entries(q.giants).map(([k, v]) => [k + " (giant)", v])];
  const labels = entries.map(([k]) => (D.models[k.split(" ")[0]] || {}).name || k.split(" ")[0]);
  const datasets = D.axes.map((ax) => ({
    label: D.axis_labels[ax][0],
    data: entries.map(([, v]) => v.axes[ax]),
    backgroundColor: AXIS_COLORS[ax] + "cc",
  }));
  mkChart("twinChart", {
    type: "bar", data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 } } } },
      scales: {
        x: gridOpts(), y: gridOpts({ min: 0.5, max: 1.0, title: { display: true, text: "axis cosine", color: "#94a3b8" } }),
      },
    },
  });
}

/* ---------------- Drift ---------------- */
let spaghettiOn = false;
function driftStatRow(label, value) {
  return `<div class="flex justify-between items-center"><span class="text-slate-500 text-xs">${label}</span><span class="mono text-slate-200 text-xs">${value}</span></div>`;
}
function renderDriftChart() {
  const mkey = $("driftModel").value, ax = $("driftAxis").value;
  const m = D.models[mkey], rec = m.drift.axes[ax];
  const turns = rec.mean_conflict_traj.map((_, i) => i + 1);
  const datasets = [];
  if (spaghettiOn && driftDetail[mkey]) {
    const det = driftDetail[mkey].axes[ax];
    det.conflict.forEach((u) => datasets.push({
      data: u.vals, borderColor: m.color + "30", borderWidth: 1, pointRadius: 0, spanGaps: true, label: "_c",
    }));
    det.control.forEach((u) => datasets.push({
      data: u.vals, borderColor: "#94a3b830", borderWidth: 1, pointRadius: 0, spanGaps: true, label: "_n",
    }));
  }
  datasets.push({
    label: `conflict mean (n=${m.drift.n_conflict})`, data: rec.mean_conflict_traj,
    borderColor: m.color, backgroundColor: m.color, borderWidth: 2.5, pointRadius: 2, spanGaps: true,
  });
  datasets.push({
    label: `control mean (n=${m.drift.n_controls})`, data: rec.mean_control_traj,
    borderColor: "#94a3b8", borderDash: [6, 4], borderWidth: 2, pointRadius: 2, spanGaps: true,
  });
  mkChart("driftChart", {
    type: "line", data: { labels: turns, datasets },
    options: {
      responsive: true, animation: { duration: 300 },
      plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 }, filter: (i) => !i.text.startsWith("_") } } },
      scales: {
        x: gridOpts({ title: { display: true, text: "turn", color: "#94a3b8" } }),
        y: gridOpts({ title: { display: true, text: axisName(ax) + "  (+" + D.axis_labels[ax][0] + ")", color: "#94a3b8" } }),
      },
    },
  });
  const dir = rec.conflict_slope == null ? "" :
    ` → toward <b>${poleFor(ax, rec.conflict_slope)}</b>`;
  $("driftStats").innerHTML = `
    ${driftStatRow("conflict slope / turn", `${fmt(rec.conflict_slope, 4)} [${fmt(rec.conflict_ci[0], 4)}, ${fmt(rec.conflict_ci[1], 4)}]`)}
    ${driftStatRow("control slope / turn", fmt(rec.control_slope, 4))}
    ${driftStatRow("slope ratio", fmt(rec.ratio, 2) + "×")}
    ${driftStatRow("|Δ| turn 1 → last (conflict)", fmt(rec.conflict_abs_delta, 3))}
    ${driftStatRow("|Δ| turn 1 → last (control)", fmt(rec.control_abs_delta, 3))}
    ${driftStatRow("recovery ratio", fmt(rec.recovery_ratio, 2))}
    ${driftStatRow("permutation p", fmt(rec.p_perm, 4))}
    ${driftStatRow("FDR q", fmt(rec.q_fdr, 4))}
    <div class="pt-2 flex flex-wrap gap-1.5">
      ${sigBadge(rec.significant, "composite", "composite n.s.")}
      ${sigBadge(rec.fdr_significant, "FDR", "FDR n.s.")}
    </div>
    <p class="text-[11px] text-slate-500 pt-1">${rec.conflict_slope != null ? "Conflict pushes this model" + dir : "No stats for this axis"}</p>`;
}
async function toggleSpaghetti() {
  const mkey = $("driftModel").value;
  if (!spaghettiOn) {
    if (!driftDetail[mkey]) {
      $("driftSpaghetti").textContent = "Loading…";
      const resp = await fetch(D.models[mkey].drift.detail_file);
      driftDetail[mkey] = await resp.json();
    }
    spaghettiOn = true;
    $("driftSpaghetti").textContent = "Hide trajectories";
  } else {
    spaghettiOn = false;
    $("driftSpaghetti").textContent = "Show all trajectories";
  }
  renderDriftChart();
}
function renderDrift() {
  const keys = modelsWith("drift");
  $("driftModel").innerHTML = keys.map((k) => `<option value="${k}">${D.models[k].name}</option>`).join("");
  $("driftAxis").innerHTML = D.axes.map((ax) => `<option value="${ax}">${axisName(ax)}</option>`).join("");
  $("driftModel").addEventListener("change", () => { spaghettiOn = false; $("driftSpaghetti").textContent = "Show all trajectories"; renderDriftChart(); });
  $("driftAxis").addEventListener("change", renderDriftChart);
  $("driftSpaghetti").addEventListener("click", toggleSpaghetti);
  renderDriftChart();
}

/* ---------------- Giants ---------------- */
function renderGiants() {
  const giants = Object.keys(D.models).filter((k) => D.models[k].group === "giant" && D.models[k].drift);
  if (!giants.length) return;
  const axes = ["warm_cold", "confident_cautious", "empathetic_analytical", "formal_casual"];
  const labels = axes.map(axisName);
  const datasets = giants.map((k) => ({
    label: D.models[k].name,
    data: axes.map((ax) => (D.models[k].drift.axes[ax] || {}).conflict_slope),
    backgroundColor: D.models[k].color + "cc",
  }));
  mkChart("giantChart", {
    type: "bar", data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 } } } },
      scales: { x: gridOpts(), y: gridOpts({ title: { display: true, text: "conflict slope / turn (+ = first pole)", color: "#94a3b8" } }) },
    },
  });
  $("giantChips").innerHTML = giants.map((k) => {
    const m = D.models[k];
    const sig = axes.filter((ax) => (m.drift.axes[ax] || {}).fdr_significant);
    return badge(`${m.name}: <b>${sig.length}/${axes.length}</b> axes FDR ✓`, sig.length >= 3 ? "badge-sig" : "");
  }).join(" ");
}

/* ---------------- Robustness ---------------- */
let fdrShowAll = false;
function renderFdr() {
  const rows = [];
  for (const [mk, m] of Object.entries(D.models)) {
    if (!m.drift) continue;
    for (const ax of D.axes) {
      const r = m.drift.axes[ax];
      if (!r || r.p_perm == null) continue;
      rows.push({ mk, ax, ...r });
    }
  }
  rows.sort((a, b) => (a.q_fdr ?? 9) - (b.q_fdr ?? 9));
  const shown = fdrShowAll ? rows : rows.filter((r) => r.significant || r.fdr_significant);
  $("fdrBody").innerHTML = shown.map((r) => `
    <tr class="border-b border-slate-700/40">
      <td class="px-4 py-2 text-slate-300">${D.models[r.mk].name}</td>
      <td class="px-3 py-2 text-slate-400">${D.axis_labels[r.ax][0]}</td>
      <td class="px-3 py-2">${sigBadge(r.significant, "yes", "no")}</td>
      <td class="px-3 py-2 mono">${fmt(r.p_perm, 4)}</td>
      <td class="px-3 py-2 mono ${r.fdr_significant ? "text-emerald-400" : "text-slate-400"}">${fmt(r.q_fdr, 4)}</td>
      <td class="px-3 py-2 mono">${fmt(r.ratio, 2)}×</td>
    </tr>`).join("");
  const nSig = rows.filter((r) => r.significant && r.fdr_significant).length;
  const mult = (D.analysis || {}).multiplicity || {};
  $("fdrSummary").innerHTML = `Benjamini–Hochberg FDR (q &lt; 0.05) over all <b class="text-slate-200">${mult.n_tests || rows.length}</b> model × axis
    permutation tests. <b class="text-emerald-400">${nSig}</b> axes pass both the composite criterion and FDR — the confirmed core.
    Rows below are sorted by q; the full table includes every test, not just the winners.`;
}
function renderFormat() {
  const fp = (D.analysis || {}).format_partialled || {};
  const rows = [];
  for (const [mk, rec] of Object.entries(fp)) {
    for (const [ax, a] of Object.entries(rec.axes)) {
      let verdict, cls;
      if (a.raw_sig && a.part_sig) { verdict = "survives"; cls = "badge-sig"; }
      else if (a.raw_sig && !a.part_sig) { verdict = "collapses"; cls = "badge-warn"; }
      else if (!a.raw_sig && a.part_sig) { verdict = "emerges"; cls = "badge-warn"; }
      else { verdict = "no drift"; cls = "badge-ns"; }
      rows.push({ mk, ax, a, verdict, cls, warn: rec.warning });
    }
  }
  rows.sort((a, b) => (a.verdict === "survives" ? -1 : 1) - (b.verdict === "survives" ? -1 : 1));
  $("fmtBody").innerHTML = rows.map((r) => `
    <tr class="border-b border-slate-700/40">
      <td class="px-4 py-2 text-slate-300">${(D.models[r.mk] || {}).name || r.mk}${r.warn ? " ⚠" : ""}</td>
      <td class="px-3 py-2 text-slate-400">${D.axis_labels[r.ax] ? D.axis_labels[r.ax][0] : r.ax}</td>
      <td class="px-3 py-2 mono">${fmt(r.a.raw_slope, 4)} ${r.a.raw_sig ? "✓" : ""}</td>
      <td class="px-3 py-2 mono">${fmt(r.a.part_slope, 4)} ${r.a.part_sig ? "✓" : ""}</td>
      <td class="px-3 py-2">${badge(r.verdict, r.cls)}</td>
    </tr>`).join("");
}
function renderLayersChart() {
  const lp = (D.analysis || {}).layer_pooling || {};
  const mk = $("lpModel").value;
  const rec = lp[mk];
  if (!rec) return;
  const nL = Object.values(rec)[0].n_layers;
  const labels = Array.from({ length: nL }, (_, i) => i);
  const datasets = Object.entries(rec).map(([ax, r]) => ({
    label: D.axis_labels[ax][0], data: r.acc_per_layer,
    borderColor: AXIS_COLORS[ax], borderWidth: 2, pointRadius: 0, tension: 0.3,
  }));
  mkChart("lpChart", {
    type: "line", data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 } } } },
      scales: {
        x: gridOpts({ title: { display: true, text: "layer", color: "#94a3b8" } }),
        y: gridOpts({ min: 0.4, max: 1.02, title: { display: true, text: "val pair accuracy", color: "#94a3b8" } }),
      },
    },
  });
  $("lpChips").innerHTML = Object.entries(rec).map(([ax, r]) =>
    badge(`${D.axis_labels[ax][0]}: <b>${r.n_acc_ge_090}/${r.n_layers}</b> layers ≥ 0.9, upper-half min <b>${fmt(r.upper_half_min_acc, 2)}</b>`,
      r.upper_half_min_acc >= 0.9 ? "badge-sig" : "")).join(" ");
}
function renderRobustness() {
  document.querySelectorAll("[data-rtab]").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("[data-rtab]").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    document.querySelectorAll(".rtab").forEach((p) => p.classList.add("hidden"));
    $("rtab-" + b.dataset.rtab).classList.remove("hidden");
  }));
  $("fdrToggle").addEventListener("click", () => {
    fdrShowAll = !fdrShowAll;
    $("fdrToggle").textContent = fdrShowAll ? "Show significant only ↑" : "Show all tests ↓";
    renderFdr();
  });
  renderFdr();
  renderFormat();
  const lp = (D.analysis || {}).layer_pooling || {};
  $("lpModel").innerHTML = Object.keys(lp).map((k) => `<option value="${k}">${(D.models[k] || {}).name || k}</option>`).join("");
  $("lpModel").addEventListener("change", renderLayersChart);
  if (Object.keys(lp).length) renderLayersChart();
}

/* ---------------- Steering ---------------- */
function renderSteerChart() {
  const mk = $("steerModel").value;
  const m = D.models[mk], st = m.steering;
  const alphaKeys = Object.keys(st.axes[Object.keys(st.axes)[0]].alpha_means).sort((a, b) => parseFloat(a) - parseFloat(b));
  const labels = alphaKeys.map((a) => parseFloat(a));
  const datasets = Object.entries(st.axes).map(([ax, rec]) => ({
    label: D.axis_labels[ax][0],
    data: alphaKeys.map((a) => rec.alpha_means[a]),
    borderColor: AXIS_COLORS[ax], backgroundColor: AXIS_COLORS[ax],
    borderWidth: 2, pointRadius: 3, tension: 0.2,
  }));
  mkChart("steerChart", {
    type: "line", data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 } } } },
      scales: {
        x: gridOpts({ title: { display: true, text: "steering dose α", color: "#94a3b8" } }),
        y: gridOpts({ title: { display: true, text: "projection on steered axis", color: "#94a3b8" } }),
      },
    },
  });
  const pct = st.layer_pct != null ? ` (${Math.round(st.layer_pct)}% depth)` : "";
  $("steerStats").innerHTML = `
    <div class="text-xs text-slate-500 mb-2">Layer <b class="text-slate-200">${st.layer}</b>/${st.n_layers}${pct} ·
      ${st.n_questions} questions per dose</div>
    ${Object.entries(st.axes).map(([ax, rec]) => {
      const sig = rec.slope_ci && rec.slope_ci[0] != null && (rec.slope_ci[0] > 0 || rec.slope_ci[1] < 0);
      return `<div class="flex justify-between items-center">
        <span class="text-xs" style="color:${AXIS_COLORS[ax]}">${D.axis_labels[ax][0]}</span>
        <span class="mono text-xs text-slate-300">Δ ${fmt(rec.total_effect, 2)} ${sig ? "✓" : ""} · leak ${fmt(rec.leakage_mean, 2)}</span>
      </div>`;
    }).join("")}
    <p class="text-[11px] text-slate-500 pt-2">Δ = total projection change across the α range. ✓ = slope CI excludes zero.
    “leak” = mean off-target movement relative to on-target.</p>`;
}
function renderSteering() {
  const keys = modelsWith("steering");
  if (!keys.length) { $("steering").style.display = "none"; return; }
  $("steerModel").innerHTML = keys.map((k) => `<option value="${k}">${D.models[k].name}</option>`).join("");
  $("steerModel").addEventListener("change", renderSteerChart);
  renderSteerChart();
}

/* ---------------- Playground ---------------- */
let pgQ = 0;
function pgState() {
  const mk = $("pgModel").value, ax = $("pgAxis").value;
  return PG.models[mk].axes[ax];
}
function renderPgText() {
  const st = pgState();
  const idx = parseInt($("pgSlider").value, 10);
  const key = st.alpha_keys[idx];
  $("pgAlpha").textContent = st.alphas[idx];
  $("pgProj").textContent = fmt((st.projections[key] || [])[pgQ], 2);
  $("pgQuestion").textContent = st.questions[pgQ];
  $("pgText").textContent = (st.texts[key] || [])[pgQ] || "(no text)";
}
function renderPgControls() {
  const st = pgState();
  const ax = $("pgAxis").value;
  $("pgSlider").max = st.alphas.length - 1;
  let zi = 0;
  st.alphas.forEach((a, i) => { if (Math.abs(a) < Math.abs(st.alphas[zi])) zi = i; });
  $("pgSlider").value = zi;
  $("pgNegLabel").textContent = "→ " + D.axis_labels[ax][1];
  $("pgPosLabel").textContent = "→ " + D.axis_labels[ax][0];
  pgQ = 0;
  $("pgQTabs").innerHTML = st.questions.map((q, i) =>
    `<button class="qtab ${i === 0 ? "active" : ""}" data-pgq="${i}">${q.length > 46 ? q.slice(0, 44) + "…" : q}</button>`).join("");
  $("pgQTabs").querySelectorAll("[data-pgq]").forEach((b) => b.addEventListener("click", () => {
    pgQ = parseInt(b.dataset.pgq, 10);
    $("pgQTabs").querySelectorAll(".qtab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    renderPgText();
  }));
  renderPgText();
}
async function renderPlayground() {
  try {
    const resp = await fetch("data/playground.json");
    PG = await resp.json();
  } catch (e) { $("playground").style.display = "none"; return; }
  const keys = Object.keys(PG.models);
  if (!keys.length) { $("playground").style.display = "none"; return; }
  $("pgModel").innerHTML = keys.map((k) => `<option value="${k}">${PG.models[k].name}</option>`).join("");
  const firstAxes = Object.keys(PG.models[keys[0]].axes);
  $("pgAxis").innerHTML = firstAxes.map((ax) => `<option value="${ax}">${axisName(ax)}</option>`).join("");
  $("pgModel").addEventListener("change", () => {
    const axes = Object.keys(PG.models[$("pgModel").value].axes);
    $("pgAxis").innerHTML = axes.map((ax) => `<option value="${ax}">${axisName(ax)}</option>`).join("");
    renderPgControls();
  });
  $("pgAxis").addEventListener("change", renderPgControls);
  $("pgSlider").addEventListener("input", renderPgText);
  renderPgControls();
}

/* ---------------- Methodology (data-driven counts) ---------------- */
function renderMethod() {
  const withDrift = modelsWith("drift").map((k) => D.models[k].drift);
  if (withDrift.length) {
    const nc = withDrift.map((d) => d.n_conflict);
    const nn = withDrift.map((d) => d.n_controls);
    const rng = (arr) => Math.min(...arr) === Math.max(...arr) ? String(arr[0]) : `${Math.min(...arr)}–${Math.max(...arr)}`;
    $("methodDrift").textContent = rng(nc);
    $("methodCtrl").textContent = rng(nn);
  }
  const accs = [];
  for (const m of Object.values(D.models)) {
    if (m.calibration) for (const c of Object.values(m.calibration)) if (c.val_acc != null) accs.push(c.val_acc);
  }
  if (accs.length) $("methodAcc").textContent = fmt(accs.reduce((a, b) => a + b, 0) / accs.length, 2);
}

/* ---------------- Section fade + init ---------------- */
function initFade() {
  const obs = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) e.target.classList.add("visible");
  }), { threshold: 0.08 });
  document.querySelectorAll(".section-fade").forEach((s) => obs.observe(s));
}

async function init() {
  try {
    const resp = await fetch("site_data.json");
    D = await resp.json();
  } catch (e) {
    document.body.innerHTML = `<div class="min-h-screen flex items-center justify-center text-slate-400">
      Could not load site_data.json — run <code class="mx-1 text-slate-300">python3 v2/build_site_data.py</code></div>`;
    return;
  }
  renderHero();
  renderFingerprints();
  renderCompare();
  renderStability();
  renderTwins();
  renderDrift();
  renderGiants();
  renderRobustness();
  renderSteering();
  renderPlayground();
  renderMethod();
  initFade();
}
init();

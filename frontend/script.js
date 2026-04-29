const API_BASE = "http://127.0.0.1:8000";
console.log("[Workbench] frontend build: 2026-04-28-r11");

let currentTrainJobId = null;
let evalPage = 1;
let evalPageSize = 20;
let analysisErrorPage = 1;
const analysisErrorPageSize = 25;

const datasetModalState = {
  category: null,
  split: "train",
  page: 1,
  pageSize: 20,
  total: 0,
};

function fmt(v, d = 4) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
  return Number(v).toFixed(d);
}

function esc(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function stateBadge(state) {
  const s = (state || "").toLowerCase();
  if (s.includes("complete")) return `<span class="state-badge state-completed">completed</span>`;
  if (s.includes("run")) return `<span class="state-badge state-running">running</span>`;
  if (s.includes("fail")) return `<span class="state-badge state-failed">failed</span>`;
  return `<span class="state-badge state-queued">${esc(state || "queued")}</span>`;
}

function errorBadge(errorType) {
  return `<span class="error-badge">${esc(errorType || "unknown")}</span>`;
}

function renderTable(el, headers, rows) {
  if (!el) return;
  const head = `<thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead>`;
  const body = `<tbody>${rows
    .map((r) => `<tr>${r.map((x) => `<td>${x}</td>`).join("")}</tr>`)
    .join("")}</tbody>`;
  el.innerHTML = head + body;
}

function buildCategoryOptionRows(summaryRows = []) {
  const byCat = {};
  summaryRows.forEach((r) => {
    byCat[r.category] = r;
  });
  const categories = [
    "Date Computation",
    "Hour Adjustment (24h)",
    "Time Zone Conversion",
    "Year Shift",
    "Month Shift",
  ];
  return categories
    .map((c) => {
      const s = byCat[c] || { train_count: 0, dev_count: 0, test_count: 0 };
      return `
        <div class="train-cat-row">
          <label class="check-item">
            <input type="checkbox" value="${esc(c)}" checked> ${esc(c)}
          </label>
          <span class="cat-count-badge">tr:${s.train_count} dv:${s.dev_count} te:${s.test_count}</span>
          <button class="mini-link-btn cat-view-btn" data-category="${esc(c)}">View Data</button>
        </div>
      `;
    })
    .join("");
}

function renderBarChart(el, items, max = 1) {
  if (!el) return;
  el.innerHTML = "";
  items.forEach((it) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-label">${esc(it.label)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, (it.value / max) * 100)}%">${fmt(it.value)}</div></div>
    `;
    el.appendChild(row);
  });
}

function normalizeDemoText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9:/\-\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenizeDemoText(text) {
  const normalized = normalizeDemoText(text);
  const tokens = normalized ? normalized.split(" ") : [];
  const bigrams = [];
  for (let i = 0; i < tokens.length - 1; i += 1) {
    bigrams.push(`${tokens[i]} ${tokens[i + 1]}`);
  }
  return { normalized, tokens, bigrams, features: [...tokens, ...bigrams] };
}

function softmax(scores) {
  const vals = Object.values(scores);
  const maxVal = Math.max(...vals);
  const exps = {};
  let sum = 0;
  Object.entries(scores).forEach(([k, v]) => {
    exps[k] = Math.exp(v - maxVal);
    sum += exps[k];
  });
  const probs = {};
  Object.entries(exps).forEach(([k, v]) => {
    probs[k] = v / sum;
  });
  return probs;
}

function renderModelInternals() {
  const data = window.researchData?.modelInternalsDemoData;
  if (!data) return;
  const input = document.getElementById("internals-question");
  if (!input) return;

  const { normalized, tokens, bigrams, features } = tokenizeDemoText(input.value);
  const idf = data.idf || {};
  const weights = data.weights || {};
  const biases = data.biases || {};
  const categories = data.categories || Object.keys(weights);

  const tfCounts = {};
  features.forEach((f) => {
    if (idf[f] != null) tfCounts[f] = (tfCounts[f] || 0) + 1;
  });

  const tfidfRows = Object.entries(tfCounts).map(([feature, tf]) => {
    const idfValue = Number(idf[feature] || 0);
    return {
      feature,
      tf,
      idf: idfValue,
      tfidf: tf * idfValue,
      type: feature.includes(" ") ? "bigram" : "unigram",
    };
  });

  const scores = {};
  categories.forEach((cat) => {
    let score = Number(biases[cat] || 0);
    tfidfRows.forEach((r) => {
      score += r.tfidf * Number(weights[cat]?.[r.feature] || 0);
    });
    scores[cat] = score;
  });
  const probs = softmax(scores);
  const ranked = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const predicted = ranked[0]?.[0] || "-";
  const confidence = ranked[0]?.[1] || 0;

  const normEl = document.getElementById("norm-output");
  if (normEl) {
    normEl.innerHTML = `
      <div><strong>Original:</strong> ${esc(input.value)}</div>
      <div><strong>Normalized:</strong> ${esc(normalized)}</div>
    `;
  }

  const tokenEl = document.getElementById("token-output");
  if (tokenEl) {
    tokenEl.innerHTML = `
      <div class="chip-section"><strong>Unigrams</strong>${tokens.map((x) => `<span class="token-chip">${esc(x)}</span>`).join("")}</div>
      <div class="chip-section"><strong>Bigrams</strong>${bigrams.map((x) => `<span class="token-chip bigram-chip">${esc(x)}</span>`).join("")}</div>
    `;
  }

  renderTable(
    document.getElementById("tfidf-table"),
    ["feature", "type", "TF", "IDF", "TF-IDF"],
    tfidfRows
      .sort((a, b) => b.tfidf - a.tfidf)
      .map((r) => [r.feature, r.type, r.tf, fmt(r.idf, 2), fmt(r.tfidf, 2)])
  );

  const contributionRows = tfidfRows
    .map((r) => {
      const w = Number(weights[predicted]?.[r.feature] || 0);
      return { ...r, weight: w, contribution: r.tfidf * w };
    })
    .filter((r) => Math.abs(r.contribution) > 0.0001)
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  renderTable(
    document.getElementById("contribution-table"),
    ["feature", "TF-IDF", `weight for ${esc(predicted)}`, "contribution"],
    contributionRows.map((r) => [
      r.feature,
      fmt(r.tfidf, 2),
      fmt(r.weight, 2),
      `<strong>${fmt(r.contribution, 2)}</strong>`,
    ])
  );

  const probEl = document.getElementById("probability-chart");
  if (probEl) {
    probEl.innerHTML = ranked
      .map(
        ([cat, p]) => `<div class="prob-row">
          <div class="prob-label">${esc(cat)}</div>
          <div class="prob-track"><div class="prob-fill" style="width:${Math.max(2, p * 100)}%">${fmt(p)}</div></div>
        </div>`
      )
      .join("");
  }

  const decisionEl = document.getElementById("internals-decision");
  if (decisionEl) {
    const fallback = confidence < Number(data.threshold || 0.95);
    decisionEl.innerHTML = `
      <div><strong>Predicted Category:</strong> ${esc(predicted)}</div>
      <div><strong>Confidence:</strong> ${fmt(confidence)} | <strong>Fallback threshold:</strong> ${fmt(data.threshold)}</div>
      <div><strong>Decision:</strong> ${fallback ? "Use fallback prompt" : "Use category-specific prompt"}</div>
      <div class="muted">This visualization uses demo weights to explain the mechanism. The production classifier uses the trained joblib model.</div>
    `;
  }
}

function initModelInternalsPage() {
  const data = window.researchData?.modelInternalsDemoData;
  const input = document.getElementById("internals-question");
  const runBtn = document.getElementById("btn-internals-run");
  const resetBtn = document.getElementById("btn-internals-reset");
  if (!data || !input || !runBtn || !resetBtn) return;
  input.value = data.defaultQuestion || "";
  runBtn.addEventListener("click", renderModelInternals);
  resetBtn.addEventListener("click", () => {
    input.value = data.defaultQuestion || "";
    renderModelInternals();
  });
  input.addEventListener("input", () => {
    window.clearTimeout(input._internalsTimer);
    input._internalsTimer = window.setTimeout(renderModelInternals, 250);
  });
  renderModelInternals();
}

function setActiveTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-page").forEach((p) => {
    p.classList.toggle("active", p.id === `page-${tabName}`);
  });
}

async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return await res.json();
}

async function apiPost(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return await res.json();
}

function initHeader() {
  const d = window.researchData.overviewData;
  document.getElementById("project-title").textContent = d.title;
  document.getElementById("project-subtitle").textContent = d.subtitle;
  document.getElementById("project-intro").textContent = d.intro;

  const p = document.getElementById("overview-pipeline");
  p.innerHTML = d.pipeline
    .map((x, i) => `<span class="pipe-node">${esc(x)}</span>${i < d.pipeline.length - 1 ? '<span class="pipe-arrow">→</span>' : ""}`)
    .join("");
}

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => setActiveTab(btn.dataset.tab));
  });
}

function getSelectedCategories() {
  const boxes = document.querySelectorAll("#train-categories input[type=checkbox]");
  return Array.from(boxes)
    .filter((b) => b.checked)
    .map((b) => b.value);
}

function renderTrainStatus(job) {
  const statusEl = document.getElementById("train-status");
  const metricsEl = document.getElementById("train-metrics");
  const modelPath = document.getElementById("train-model-path");
  const reportPath = document.getElementById("train-report-path");
  const confPath = document.getElementById("train-conf-path");

  if (!job) {
    statusEl.textContent = "No job yet.";
    metricsEl.innerHTML = "";
    modelPath.textContent = "";
    reportPath.textContent = "";
    confPath.textContent = "";
    return;
  }

  statusEl.innerHTML = `
    <div><strong>State:</strong> ${stateBadge(job.state)}</div>
    <div><strong>Stage:</strong> ${esc(job.stage || "-")}</div>
    <div><strong>Progress:</strong> ${fmt((job.progress || 0) * 100, 1)}%</div>
    <div><strong>Cache Hit:</strong> ${job.cache_hit ? "Yes" : "No"}</div>
    <div><strong>Cache Key:</strong> ${esc(job.cache_key || "-")}</div>
    <div><strong>Cache Reason:</strong> ${esc(job.cache_reason || "-")}</div>
    <div><strong>Active Model Version:</strong> ${esc(job.active_model_version || "-")}</div>
    <div><strong>Job ID:</strong> ${esc(job.job_id || "-")}</div>
    ${job.error ? `<div class="error-text"><strong>Error:</strong> ${esc(job.error)}</div>` : ""}
  `;

  const m = job.metrics || {};
  metricsEl.innerHTML = `
    <div class="metric-item"><span>Accuracy</span><strong>${m.accuracy != null ? fmt(m.accuracy) : "-"}</strong></div>
    <div class="metric-item"><span>Macro-F1</span><strong>${m.macro_f1 != null ? fmt(m.macro_f1) : "-"}</strong></div>
    <div class="metric-item"><span>N Train</span><strong>${m.n_train ?? "-"}</strong></div>
    <div class="metric-item"><span>N Test</span><strong>${m.n_test ?? "-"}</strong></div>
    <div class="metric-item"><span>Duration(s)</span><strong>${job.duration_sec != null ? fmt(job.duration_sec, 2) : "-"}</strong></div>
  `;

  modelPath.innerHTML = job.model_path ? `<strong>Model:</strong> ${esc(job.model_path)}` : "";
  reportPath.innerHTML = job.report_path ? `<strong>Report:</strong> ${esc(job.report_path)}` : "";
  confPath.innerHTML = job.confusion_matrix_path ? `<strong>Confusion Matrix:</strong> ${esc(job.confusion_matrix_path)}` : "";
  const reportPreview = document.getElementById("train-report-preview");
  const logTail = document.getElementById("train-log-tail");
  if (reportPreview) reportPreview.textContent = job.classification_report_preview || "(empty)";
  if (logTail) logTail.textContent = job.log_tail || (Array.isArray(job.log) ? job.log.join("\n") : "(empty)");

  if (job.job_id) {
    currentTrainJobId = job.job_id;
    loadEvalDetailRows();
  }
}

async function loadEvalDetailRows() {
  const statsEl = document.getElementById("eval-stats-box");
  const tableEl = document.getElementById("eval-detail-table");
  const pageEl = document.getElementById("eval-page-indicator");
  const categoryFilter = document.getElementById("eval-category-filter");
  const correctFilter = document.getElementById("eval-correct-filter");
  if (!statsEl || !tableEl || !pageEl || !categoryFilter || !correctFilter) return;

  if (!currentTrainJobId) {
    statsEl.textContent = "Run training first to load eval detail rows.";
    tableEl.innerHTML = "";
    return;
  }

  const cf = correctFilter.value || "all";
  const cat = categoryFilter.value || "all";

  try {
    const q = new URLSearchParams({
      page: String(evalPage),
      page_size: String(evalPageSize),
      correct_filter: cf,
      category_filter: cat,
    });
    const data = await apiGet(`/api/train/eval_rows/${encodeURIComponent(currentTrainJobId)}?${q.toString()}`);
    if (!data.available) {
      statsEl.innerHTML = `<span class="error-text">Evaluation details unavailable: ${esc(data.reason || "unknown")}</span>`;
      tableEl.innerHTML = "";
      pageEl.textContent = "Page 1";
      return;
    }

    statsEl.innerHTML = `
      <div><strong>Correct:</strong> ${data.stats?.correct ?? "-"}</div>
      <div><strong>Wrong:</strong> ${data.stats?.wrong ?? "-"}</div>
      <div><strong>Accuracy:</strong> ${fmt(data.stats?.accuracy ?? 0)}</div>
      <div class="muted">Sample-level correctness is sourced from eval_predictions.csv.</div>
    `;

    const rows = data.rows || [];
    const cats = ["all", ...new Set(rows.map((r) => r.true_category || "unspecified"))];
    const oldVal = categoryFilter.value;
    categoryFilter.innerHTML = cats.map((x) => `<option value="${esc(x)}">${esc(x)}</option>`).join("");
    if (cats.includes(oldVal)) categoryFilter.value = oldVal;

    renderTable(
      tableEl,
      ["id", "true_category", "pred_category", "correct"],
      rows.map((r) => [r.id, r.true_category, r.pred_category, r.correct ? "true" : "false"])
    );
    const totalPages = Math.max(1, Math.ceil((data.total || 0) / (data.page_size || evalPageSize)));
    pageEl.textContent = `Page ${data.page}/${totalPages} (total ${data.total})`;
    document.getElementById("eval-prev-page").disabled = data.page <= 1;
    document.getElementById("eval-next-page").disabled = data.page >= totalPages;
  } catch (e) {
    statsEl.innerHTML = `<span class="error-text">Failed to load evaluation details: ${esc(e.message)}</span>`;
    tableEl.innerHTML = "";
  }
}

async function openDatasetModal(category) {
  datasetModalState.category = category;
  datasetModalState.page = 1;
  const modal = document.getElementById("dataset-modal");
  const title = document.getElementById("dataset-modal-title");
  const splitSel = document.getElementById("dataset-modal-split");
  const sizeSel = document.getElementById("dataset-modal-page-size");
  if (title) title.textContent = `Category Dataset Viewer — ${category}`;
  if (splitSel) splitSel.value = datasetModalState.split;
  if (sizeSel) sizeSel.value = String(datasetModalState.pageSize);
  if (modal) modal.classList.remove("hidden");
  await loadDatasetModalRows();
}

function closeDatasetModal() {
  const modal = document.getElementById("dataset-modal");
  if (modal) modal.classList.add("hidden");
}

async function loadDatasetModalRows() {
  const meta = document.getElementById("dataset-modal-meta");
  const table = document.getElementById("dataset-modal-table");
  const indicator = document.getElementById("dataset-modal-page-indicator");
  if (!datasetModalState.category || !meta || !table || !indicator) return;

  try {
    const q = new URLSearchParams({
      category: datasetModalState.category,
      split: datasetModalState.split,
      page: String(datasetModalState.page),
      page_size: String(datasetModalState.pageSize),
    });
    const data = await apiGet(`/api/train/dataset_rows?${q.toString()}`);
    if (!data.available) {
      meta.innerHTML = `<span class="error-text">${esc(data.reason || "not available")}</span>`;
      table.innerHTML = "";
      indicator.textContent = "Page 1";
      return;
    }

    datasetModalState.total = data.total || 0;
    const totalPages = Math.max(1, Math.ceil(datasetModalState.total / (data.page_size || datasetModalState.pageSize)));
    meta.textContent = `${data.category} | ${data.split} | total ${data.total}`;
    indicator.textContent = `Page ${data.page}/${totalPages}`;
    renderTable(
      table,
      ["id", "question", "gold", "category"],
      (data.rows || []).map((r) => [r.id, `<details><summary>Expand</summary>${esc(r.question)}</details>`, r.gold, r.category])
    );
    document.getElementById("dataset-modal-prev").disabled = data.page <= 1;
    document.getElementById("dataset-modal-next").disabled = data.page >= totalPages;
  } catch (e) {
    meta.innerHTML = `<span class="error-text">Load failed: ${esc(e.message)}</span>`;
    table.innerHTML = "";
  }
}

async function initTrainPage() {
  const host = document.getElementById("train-categories");
  try {
    const summary = await apiGet("/api/train/categories_summary");
    host.innerHTML = buildCategoryOptionRows(summary.rows || []);
  } catch {
    host.innerHTML = buildCategoryOptionRows([]);
  }

  host.querySelectorAll(".cat-view-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const category = btn.dataset.category || "";
      if (!category) return;
      await openDatasetModal(category);
    });
  });

  const modalSplit = document.getElementById("dataset-modal-split");
  const modalPageSize = document.getElementById("dataset-modal-page-size");
  const modalPrev = document.getElementById("dataset-modal-prev");
  const modalNext = document.getElementById("dataset-modal-next");
  const modalClose = document.getElementById("dataset-modal-close");
  const modalBackdrop = document.getElementById("dataset-modal");

  modalSplit?.addEventListener("change", async () => {
    datasetModalState.split = modalSplit.value;
    datasetModalState.page = 1;
    await loadDatasetModalRows();
  });
  modalPageSize?.addEventListener("change", async () => {
    datasetModalState.pageSize = Number(modalPageSize.value || 20);
    datasetModalState.page = 1;
    await loadDatasetModalRows();
  });
  modalPrev?.addEventListener("click", async () => {
    datasetModalState.page = Math.max(1, datasetModalState.page - 1);
    await loadDatasetModalRows();
  });
  modalNext?.addEventListener("click", async () => {
    datasetModalState.page += 1;
    await loadDatasetModalRows();
  });
  modalClose?.addEventListener("click", closeDatasetModal);
  modalBackdrop?.addEventListener("click", (e) => {
    if (e.target === modalBackdrop) closeDatasetModal();
  });

  try {
    const spec = await apiGet("/api/train/spec");
    const specEl = document.getElementById("train-spec-card");
    if (specEl) {
      specEl.innerHTML = `
        <div><strong>Model Architecture:</strong> ${esc(spec.model_architecture || "-")}</div>
        <div><strong>Train Split:</strong> ${esc(spec.split_paths?.train || "-")} (${spec.split_counts?.train ?? "-"})</div>
        <div><strong>Dev Split:</strong> ${esc(spec.split_paths?.dev || "-")} (${spec.split_counts?.dev ?? "-"})</div>
        <div><strong>Test Split:</strong> ${esc(spec.split_paths?.test || "-")} (${spec.split_counts?.test ?? "-"})</div>
        <div><strong>Active Classifier:</strong> ${esc(spec.active_model_version || "-")}</div>
      `;
    }
  } catch (e) {
    const specEl = document.getElementById("train-spec-card");
    if (specEl) specEl.textContent = `Failed to load train spec: ${e.message}`;
  }

  const startBtn = document.getElementById("btn-train-start");

  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    startBtn.textContent = "Starting...";
    try {
      const payload = {
        categories: getSelectedCategories(),
        min_samples_per_class: Number(document.getElementById("train-min-samples").value || 20),
        seed: Number(document.getElementById("train-seed").value || 42),
      };

      const job = await apiPost("/api/train/start", payload);
      renderTrainStatus(job);

      if (job.state === "queued" || job.state === "running") {
        const timer = setInterval(async () => {
          try {
            const next = await apiGet(`/api/train/status/${job.job_id}`);
            renderTrainStatus(next);
            if (["completed", "failed", "not_found"].includes(next.state)) {
              clearInterval(timer);
            }
          } catch (e) {
            clearInterval(timer);
          }
        }, 1500);
      }
    } catch (e) {
      renderTrainStatus({ state: "failed", progress: 1, error: e.message, cache_hit: false });
    } finally {
      startBtn.disabled = false;
      startBtn.textContent = "Start One-Click Training";
    }
  });

  try {
    const latest = await apiGet("/api/train/latest");
    if (latest && latest.state !== "empty") renderTrainStatus(latest);
  } catch {
    renderTrainStatus(null);
  }

  const evalCorrect = document.getElementById("eval-correct-filter");
  const evalCategory = document.getElementById("eval-category-filter");
  const evalPrev = document.getElementById("eval-prev-page");
  const evalNext = document.getElementById("eval-next-page");
  evalCorrect?.addEventListener("change", async () => {
    evalPage = 1;
    await loadEvalDetailRows();
  });
  evalCategory?.addEventListener("change", async () => {
    evalPage = 1;
    await loadEvalDetailRows();
  });
  evalPrev?.addEventListener("click", async () => {
    evalPage = Math.max(1, evalPage - 1);
    await loadEvalDetailRows();
  });
  evalNext?.addEventListener("click", async () => {
    evalPage += 1;
    await loadEvalDetailRows();
  });
}

function renderProblemTypes(items) {
  const host = document.getElementById("problem-type-grid");
  host.innerHTML = items
    .map(
      (x) => `<article class="type-card">
        <h4>${esc(x.category)}</h4>
        <p><strong>Definition:</strong> ${esc(x.definition)}</p>
        <p><strong>Why hard:</strong> ${esc(x.why_hard)}</p>
        <p><strong>System risk:</strong> ${esc(x.risk_hint || "-")}</p>
        <p><strong>Example count:</strong> ${x.example_count}</p>
      </article>`
    )
    .join("");
}

function getFallbackProblemTypes() {
  const baseCats =
    (window.researchData?.categoryBoundaryData?.categories || []).map((x) => x.category) || [];
  const uniqueCats = [...new Set(baseCats)];
  const desc = {
    "Date Computation": {
      definition: "Compute dates across day/month/year boundaries.",
      why_hard: "Month length and leap-year rules can break naive reasoning.",
      risk_hint: "date_rollover / format mismatch",
    },
    "Hour Adjustment (24h)": {
      definition: "Add or subtract hours under 24-hour time format.",
      why_hard: "Carry/borrow around 00:00 is error-prone.",
      risk_hint: "carry_borrow_error",
    },
    "Time Zone Conversion": {
      definition: "Convert time from source zone to target zone.",
      why_hard: "Direction and day rollover are easy to invert.",
      risk_hint: "timezone_direction / day_shift",
    },
    "Year Shift": {
      definition: "Shift year value by a given offset.",
      why_hard: "Usually easy but can hide format noise.",
      risk_hint: "format_only",
    },
    "Month Shift": {
      definition: "Shift months with potential year rollover.",
      why_hard: "Month arithmetic can cross years and invalid dates.",
      risk_hint: "month_rollover",
    },
  };
  return uniqueCats.map((category) => ({
    category,
    definition: desc[category]?.definition || "Temporal reasoning category.",
    why_hard: desc[category]?.why_hard || "Requires consistent temporal normalization.",
    risk_hint: desc[category]?.risk_hint || "reasoning_error",
    example_count: 3,
  }));
}

function getFallbackExamples(category, limit = 3) {
  const bank = {
    "Date Computation": [
      { id: "fallback_dc_1", question: "What date is 10 days after 2024-02-20?", gold: "2024-03-01", source_split: "fallback" },
      { id: "fallback_dc_2", question: "What date is 1 day before 2025-01-01?", gold: "2024-12-31", source_split: "fallback" },
      { id: "fallback_dc_3", question: "Add 30 days to 2023-04-01.", gold: "2023-05-01", source_split: "fallback" },
    ],
    "Hour Adjustment (24h)": [
      { id: "fallback_h24_1", question: "What is 23:40 plus 2 hours?", gold: "01:40", source_split: "fallback" },
      { id: "fallback_h24_2", question: "What is 00:15 minus 1 hour?", gold: "23:15", source_split: "fallback" },
      { id: "fallback_h24_3", question: "What is 12:05 plus 11 hours?", gold: "23:05", source_split: "fallback" },
    ],
    "Time Zone Conversion": [
      { id: "fallback_tz_1", question: "Convert 10:00 UTC to UTC+8.", gold: "18:00", source_split: "fallback" },
      { id: "fallback_tz_2", question: "Convert 02:30 UTC+9 to UTC.", gold: "17:30", source_split: "fallback" },
      { id: "fallback_tz_3", question: "Convert 23:00 UTC-5 to UTC+1.", gold: "05:00", source_split: "fallback" },
    ],
    "Year Shift": [
      { id: "fallback_y_1", question: "What year is 5 years after 2018?", gold: "2023", source_split: "fallback" },
      { id: "fallback_y_2", question: "What year is 2 years before 2000?", gold: "1998", source_split: "fallback" },
      { id: "fallback_y_3", question: "What year is 1 year after 1999?", gold: "2000", source_split: "fallback" },
    ],
    "Month Shift": [
      { id: "fallback_m_1", question: "What month is 3 months after November 2023?", gold: "February 2024", source_split: "fallback" },
      { id: "fallback_m_2", question: "What month is 2 months before January 2025?", gold: "November 2024", source_split: "fallback" },
      { id: "fallback_m_3", question: "What month is 1 month after December 2022?", gold: "January 2023", source_split: "fallback" },
    ],
  };
  return (bank[category] || []).slice(0, Math.max(1, Number(limit || 3)));
}

function renderExamples(examples) {
  const host = document.getElementById("category-examples");
  if (!examples.length) {
    host.innerHTML = '<p class="muted">No examples found for this category.</p>';
    return;
  }
  host.innerHTML = examples
    .map(
      (e) => `<article class="example-card">
        <div><strong>ID:</strong> ${esc(e.id)}</div>
        <div><strong>Question:</strong> ${esc(e.question)}</div>
        <div><strong>Gold:</strong> ${esc(e.gold)}</div>
        <div><strong>Source:</strong> ${esc(e.source_split)}</div>
      </article>`
    )
    .join("");
}

function renderSuiteSummary(summary) {
  const host = document.getElementById("suite-summary");
  if (!summary || !Object.keys(summary).length) {
    host.innerHTML = "";
    return;
  }

  const rows = Object.entries(summary).map(([model, s]) => [
    model,
    fmt(s.accuracy),
    fmt(s.parse_rate),
    s.latency_ms != null ? fmt(s.latency_ms, 2) : "-",
    fmt(s.calls_per_query, 2),
    `${s.correct_count}/${s.sample_count}`,
  ]);

  host.innerHTML = "<table id='suite-summary-table'></table>";
  renderTable(
    document.getElementById("suite-summary-table"),
    ["System", "Accuracy", "Parse Rate", "Latency (ms)", "Calls/Query", "Correct"],
    rows
  );
}

function renderSuiteMechanism(full) {
  const host = document.getElementById("suite-mechanism");
  const linksHost = document.getElementById("suite-error-links");
  if (!host) return;
  if (!full || !full.summary) {
    host.textContent = "";
    if (linksHost) linksHost.innerHTML = "";
    return;
  }
  host.innerHTML = `
    <div><strong>Router Trigger Rate:</strong> ${full.router_trigger_rate != null ? fmt(full.router_trigger_rate) : "-"}</div>
    <div><strong>Fallback Rate:</strong> ${full.fallback_rate != null ? fmt(full.fallback_rate) : "-"}</div>
    <div><strong>Error Rows:</strong> ${(full.error_rows || []).length}</div>
    <div class="muted">Hint: a higher trigger rate is not always better; the key is whether accuracy improves.</div>
  `;

  if (linksHost) {
    const catFreq = {};
    (full.error_rows || []).forEach((r) => {
      catFreq[r.category] = (catFreq[r.category] || 0) + 1;
    });
    const topCats = Object.entries(catFreq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);
    linksHost.innerHTML = topCats
      .map(
        ([cat, n]) =>
          `<button class="mini-link-btn" data-cat="${esc(cat)}">View ${esc(cat)} Errors (${n})</button>`
      )
      .join("");
    linksHost.querySelectorAll(".mini-link-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        setActiveTab("analysis");
        const errorFilter = document.getElementById("error-filter");
        if (errorFilter) {
          errorFilter.value = btn.dataset.cat || "All";
          errorFilter.dispatchEvent(new Event("change"));
        }
      });
    });
  }
}

function renderLiveCards(results) {
  const host = document.getElementById("live-cards");
  const cards = [
    { name: "DeepSeek", key: "deepseek" },
    { name: "GPT-5-mini", key: "gpt5" },
    { name: "Doubao", key: "doubao" },
    { name: "Classifier Router + Fallback", key: "router" },
  ];

  host.innerHTML = cards
    .map((c) => {
      const r = results[c.key] || {};
      const meta = r.routerMeta
        ? `<div class="router-meta">
            <div><strong>predicted_category:</strong> ${esc(r.routerMeta.predicted_category || "-")}</div>
            <div><strong>confidence:</strong> ${r.routerMeta.confidence != null ? fmt(r.routerMeta.confidence) : "-"}</div>
            <div><strong>fallback:</strong> ${r.routerMeta.fallback_triggered ? "Yes" : "No"}</div>
            <div><strong>prompt_mode:</strong> ${esc(r.routerMeta.selected_prompt_mode || "-")}</div>
          </div>`
        : "";

      return `<article class="panel result-card">
        <h4>${esc(c.name)}</h4>
        <div class="mini-meta">Latency: ${r.latency_ms != null ? `${fmt(r.latency_ms, 2)} ms` : "-"}</div>
        <pre>${esc(r.error ? `Error: ${r.error}` : (r.response || ""))}</pre>
        ${meta}
      </article>`;
    })
    .join("");
}

async function initTestPage() {
  let problemTypes = [];
  let testCenterNote = "";
  try {
    const problemResp = await apiGet("/api/problem_types");
    problemTypes = problemResp.problem_types || [];
  } catch (e) {
    problemTypes = getFallbackProblemTypes();
    testCenterNote =
      "Backend problem-type API is unavailable. Showing local fallback category metadata and examples.";
  }

  if (!problemTypes.length) {
    problemTypes = getFallbackProblemTypes();
    testCenterNote = "No problem-type rows returned by backend. Showing local fallback metadata.";
  }

  const typeGrid = document.getElementById("problem-type-grid");
  if (testCenterNote && typeGrid) {
    typeGrid.innerHTML = `<div class="status-box">${esc(testCenterNote)}</div>`;
  }
  renderProblemTypes(problemTypes);

  const catSelect = document.getElementById("example-category");
  const suiteLimit = document.getElementById("suite-limit");

  catSelect.innerHTML = problemTypes.map((x) => `<option value="${esc(x.category)}">${esc(x.category)}</option>`).join("");

  async function loadExamples() {
    const cat = catSelect.value;
    const lim = Number(suiteLimit.value || 3);
    try {
      const res = await apiGet(`/api/problem_types/${encodeURIComponent(cat)}/examples?limit=${lim}`);
      renderExamples(res.examples || []);
    } catch {
      renderExamples(getFallbackExamples(cat, lim));
    }
  }

  catSelect.addEventListener("change", loadExamples);
  suiteLimit.addEventListener("change", loadExamples);
  await loadExamples();

  const classifyBtn = document.getElementById("btn-classify-run");
  classifyBtn.addEventListener("click", async () => {
    const prompt = (document.getElementById("classify-prompt").value || "").trim();
    const topK = Number(document.getElementById("classify-topk").value || 5);
    const resultEl = document.getElementById("classify-result");
    const probTable = document.getElementById("classify-prob-table");

    if (!prompt) {
      resultEl.textContent = "Please input a question first.";
      probTable.innerHTML = "";
      return;
    }

    classifyBtn.disabled = true;
    resultEl.textContent = "Classifying...";
    probTable.innerHTML = "";
    try {
      const res = await apiPost("/api/classify_question", { prompt, top_k: topK });
      if (res.error) {
        resultEl.innerHTML = `<span class=\"error-text\">Error: ${esc(res.error)}</span>`;
        probTable.innerHTML = "";
      } else {
        resultEl.innerHTML = `
          <div><strong>Predicted Category:</strong> ${esc(res.predicted_category || "-")}</div>
          <div><strong>Confidence:</strong> ${res.confidence != null ? fmt(res.confidence) : "-"}</div>
          <div><strong>Classifier Model:</strong> ${esc(res.model_path || "-")}</div>
        `;
        renderTable(
          probTable,
          ["Category", "Probability"],
          (res.probabilities || []).map((p) => [p.category, fmt(p.probability)])
        );
      }
    } catch (e) {
      resultEl.innerHTML = `<span class=\"error-text\">Error: ${esc(e.message)}</span>`;
      probTable.innerHTML = "";
    } finally {
      classifyBtn.disabled = false;
    }
  });

  const suiteStatus = document.getElementById("suite-status");
  const runBtn = document.getElementById("btn-run-suite");
  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    suiteStatus.textContent = "Queued";
    try {
      const payload = {
        categories: problemTypes.map((x) => x.category),
        limit_per_category: Number(document.getElementById("suite-limit").value || 3),
        threshold: Number(document.getElementById("suite-threshold").value || 0.95),
      };
      const run = await apiPost("/api/test/run_suite", payload);
      suiteStatus.innerHTML = `${stateBadge(run.state)} <span>${esc(run.run_id)}</span>`;

      const timer = setInterval(async () => {
        const s = await apiGet(`/api/test/status/${run.run_id}`);
        suiteStatus.innerHTML = `${stateBadge(s.state)} <span>${fmt((s.progress || 0) * 100, 1)}%</span>`;
        if (["completed", "failed", "not_found"].includes(s.state)) {
          clearInterval(timer);
          const full = await apiGet(`/api/test/result/${run.run_id}`);
          renderSuiteSummary(full.summary || {});
          renderSuiteMechanism(full);
          if (s.state === "failed") {
            suiteStatus.innerHTML = `${stateBadge("failed")} <span>${esc(s.error || "unknown")}</span>`;
          }
        }
      }, 1500);
    } catch (e) {
      suiteStatus.innerHTML = `${stateBadge("failed")} <span>${esc(e.message)}</span>`;
    } finally {
      runBtn.disabled = false;
    }
  });

  const liveBtn = document.getElementById("btn-live-run");
  liveBtn.addEventListener("click", async () => {
    const prompt = (document.getElementById("live-prompt").value || "").trim();
    if (!prompt) return;
    liveBtn.disabled = true;
    try {
      const threshold = Number(document.getElementById("suite-threshold").value || 0.95);
      const [baseRes, routerRes] = await Promise.all([
        apiPost("/api/query", { prompt }),
        apiPost("/api/query_with_router", { prompt, threshold }),
      ]);
      const byName = {};
      baseRes.forEach((x) => (byName[x.model] = x));

      renderLiveCards({
        deepseek: byName["DeepSeek"],
        gpt5: byName["GPT-5-mini"],
        doubao: byName["Doubao"],
        router: {
          ...routerRes,
          routerMeta: {
            predicted_category: routerRes.predicted_category,
            confidence: routerRes.confidence,
            fallback_triggered: routerRes.fallback_triggered,
            selected_prompt_mode: routerRes.selected_prompt_mode,
          },
        },
      });
    } catch (e) {
      renderLiveCards({
        deepseek: { error: e.message },
        gpt5: { error: e.message },
        doubao: { error: e.message },
        router: { error: e.message },
      });
    } finally {
      liveBtn.disabled = false;
    }
  });
}

function renderGroupedCategoryChart(el, categories) {
  if (!el) return;
  const methods = [
    { key: "fixed_corrected", label: "Fixed (Corrected)" },
    { key: "cot_corrected", label: "CoT (Corrected)" },
    { key: "router_corrected", label: "Router (Corrected)" },
    { key: "router_fallback_corrected", label: "Router+Fallback (Corrected)" },
  ];
  el.innerHTML = `
    <div class="grouped-legend">
      ${methods.map((m, i) => `<span class="legend-item legend-${i + 1}">${m.label}</span>`).join("")}
    </div>
  `;

  categories.forEach((cat) => {
    const row = document.createElement("div");
    row.className = "grouped-row";
    row.innerHTML = `
      <div class="grouped-label">${esc(cat.category)}</div>
      <div class="grouped-bars">
        ${methods
          .map(
            (m, i) => `<div class="group-bar-wrap">
              <div class="group-bar group-bar-${i + 1}" style="height:${Math.max(2, Number(cat[m.key]) * 120)}px"></div>
              <div class="group-bar-value">${fmt(cat[m.key])}</div>
            </div>`
          )
          .join("")}
      </div>
    `;
    el.appendChild(row);
  });
}

async function initAnalysisPage() {
  const policyEl = document.getElementById("analysis-policy");
  const errorFilter = document.getElementById("error-filter");
  const workflowFilter = document.getElementById("error-workflow-filter");
  const correctedFilter = document.getElementById("error-corrected-filter");
  const pageIndicator = document.getElementById("error-page-indicator");
  const prevBtn = document.getElementById("error-prev-page");
  const nextBtn = document.getElementById("error-next-page");

  const summary = await apiGet("/api/analysis/summary");
  if (!summary.available) throw new Error(`analysis summary unavailable: ${summary.error || "unknown"}`);
  const categorywise = await apiGet("/api/analysis/categorywise");
  if (!categorywise.available) throw new Error(`analysis categorywise unavailable: ${categorywise.error || "unknown"}`);

  policyEl.innerHTML = `
    <div><strong>Ruleset Version:</strong> ${esc(summary.ruleset_version || "-")}</div>
    <div><strong>Generated At:</strong> ${esc(summary.generated_at || "-")}</div>
    <div><strong>Base:</strong> original strict matching criterion</div>
    <div><strong>Corrected:</strong> rule-normalized criterion (evaluation layer only)</div>
  `;

  renderBarChart(
    document.getElementById("workflow-chart"),
    (summary.workflows || []).map((w) => ({ label: w.workflow, value: Number(w.corrected_accuracy || 0) })),
    1
  );

  renderTable(
    document.getElementById("workflow-table"),
    [
      "Workflow",
      "Base Acc",
      "Corrected Acc",
      "Delta",
      "Parse Rate",
      "Format Compliance",
      "Latency(s)",
      "Calls/Query",
    ],
    (summary.workflows || []).map((w) => [
      w.workflow,
      fmt(w.base_accuracy),
      `<strong>${fmt(w.corrected_accuracy)}</strong>`,
      fmt(w.delta),
      fmt(w.parse_rate),
      fmt(w.format_compliance),
      fmt(w.latency_sec_per_query),
      fmt(w.calls_per_query, 1),
    ])
  );

  const o = summary.oracle || {};
  document.getElementById("oracle-box").innerHTML = `
    <div class="oracle-metric">${esc(o.name || "Oracle Prompt Upper Bound")}: <strong>${fmt(o.accuracy)}</strong></div>
    <p><strong>${esc(o.definition || "Upper bound under category-best prompt selection")}</strong></p>
    <p><strong>${esc(o.interpretation || "Not a deployable online policy")}</strong></p>
    <p class="muted">This is an upper bound, not a deployable online policy.</p>
  `;

  const categories = categorywise.rows || [];
  renderGroupedCategoryChart(document.getElementById("category-chart"), categories);
  renderTable(
    document.getElementById("category-table"),
    ["Category", "Fixed", "CoT", "Router", "Router+Fallback", "Insight"],
    categories.map((c) => [
      c.category,
      `${fmt(c.fixed_base)} → <strong>${fmt(c.fixed_corrected)}</strong> (${fmt(c.fixed_delta)})`,
      `${fmt(c.cot_base)} → <strong>${fmt(c.cot_corrected)}</strong> (${fmt(c.cot_delta)})`,
      `${fmt(c.router_base)} → <strong>${fmt(c.router_corrected)}</strong> (${fmt(c.router_delta)})`,
      `${fmt(c.router_fallback_base)} → <strong>${fmt(c.router_fallback_corrected)}</strong> (${fmt(c.router_fallback_delta)})`,
      c.insight_tag,
    ])
  );

  const cats = ["all", ...new Set(categories.map((x) => x.category))];
  const wfs = ["all", ...new Set((summary.workflows || []).map((x) => x.workflow))];
  errorFilter.innerHTML = cats.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
  workflowFilter.innerHTML = wfs.map((w) => `<option value="${esc(w)}">${esc(w)}</option>`).join("");

  async function drawErrorTable() {
    const selected = errorFilter.value || "all";
    const selectedWf = workflowFilter.value || "all";
    const corrected = correctedFilter.value || "all";
    const q = new URLSearchParams({
      page: String(analysisErrorPage),
      page_size: String(analysisErrorPageSize),
      category: selected,
      workflow: selectedWf,
      corrected,
    });
    const data = await apiGet(`/api/analysis/errors?${q.toString()}`);
    if (!data.available) {
      renderTable(document.getElementById("error-table"), ["error"], [[esc(data.error || "analysis errors unavailable")]]);
      return;
    }
    const rows = data.rows || [];
    renderTable(
      document.getElementById("error-table"),
      ["sample_id", "category", "workflow", "gold", "pred_norm", "corrected_match", "error_type"],
      rows.map((r) => [
        r.sample_id,
        r.category,
        r.workflow,
        r.gold,
        r.pred_norm,
        r.corrected_match ? "true" : "false",
        errorBadge(r.error_type),
      ])
    );
    const totalPages = Math.max(1, Math.ceil((data.total || 0) / (data.page_size || analysisErrorPageSize)));
    pageIndicator.textContent = `Page ${data.page}/${totalPages} (total ${data.total})`;
    prevBtn.disabled = data.page <= 1;
    nextBtn.disabled = data.page >= totalPages;
  }

  errorFilter.addEventListener("change", async () => {
    analysisErrorPage = 1;
    await drawErrorTable();
  });
  workflowFilter.addEventListener("change", async () => {
    analysisErrorPage = 1;
    await drawErrorTable();
  });
  correctedFilter.addEventListener("change", async () => {
    analysisErrorPage = 1;
    await drawErrorTable();
  });
  prevBtn.addEventListener("click", async () => {
    analysisErrorPage = Math.max(1, analysisErrorPage - 1);
    await drawErrorTable();
  });
  nextBtn.addEventListener("click", async () => {
    analysisErrorPage += 1;
    await drawErrorTable();
  });

  await drawErrorTable();
}

async function bootstrap() {
  initHeader();
  initTabs();
  try {
    await initAnalysisPage();
  } catch (e) {
    console.error("initAnalysisPage failed:", e);
    const policyEl = document.getElementById("analysis-policy");
    if (policyEl) policyEl.innerHTML = `<span class="error-text">Analysis data unavailable: ${esc(e.message)}</span>`;
  }
  await initTrainPage();
  initModelInternalsPage();
  await initTestPage();
}

bootstrap().catch((e) => {
  console.error(e);
  alert(`Initialization failed: ${e.message}`);
});

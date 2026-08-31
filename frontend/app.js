"use strict";

const ui = {
  sessionId: document.querySelector("#sessionId"),
  turnCount: document.querySelector("#turnCount"),
  processingStatus: document.querySelector("#processingStatus"),
  pipeline: document.querySelector("#pipeline"),
  conversation: document.querySelector("#conversation"),
  recommendations: document.querySelector("#recommendations"),
  resultCount: document.querySelector("#resultCount"),
  stateContent: document.querySelector("#stateContent"),
  newSessionButton: document.querySelector("#newSessionButton"),
  errorToast: document.querySelector("#errorToast"),
  developerJson: document.querySelector("#developerJson"),
  demoModeButton: document.querySelector("#demoModeButton"),
  developerModeButton: document.querySelector("#developerModeButton"),
  developerMode: document.querySelector("#developerMode"),
  devSessionId: document.querySelector("#devSessionId"),
  devCommittedTurn: document.querySelector("#devCommittedTurn"),
  devActiveTurn: document.querySelector("#devActiveTurn"),
  devHistory: document.querySelector("#devHistory"),
  newDevSessionButton: document.querySelector("#newDevSessionButton"),
  devStageList: document.querySelector("#devStageList"),
  devMessageInput: document.querySelector("#devMessageInput"),
  startDevTurnButton: document.querySelector("#startDevTurnButton"),
  restartDevTurnButton: document.querySelector("#restartDevTurnButton"),
  runNextButton: document.querySelector("#runNextButton"),
  runAllButton: document.querySelector("#runAllButton"),
  commitTurnButton: document.querySelector("#commitTurnButton"),
  inspectorStageName: document.querySelector("#inspectorStageName"),
  inspectorImplementation: document.querySelector("#inspectorImplementation"),
  inspectorStatus: document.querySelector("#inspectorStatus"),
  devInspector: document.querySelector("#devInspector"),
  devParameters: document.querySelector("#devParameters"),
  devTiming: document.querySelector("#devTiming"),
  devActionStatus: document.querySelector("#devActionStatus"),
  devScenarioOutcome: document.querySelector("#devScenarioOutcome"),
  demoSampleSelect: document.querySelector("#demoSampleSelect"),
  devSampleSelect: document.querySelector("#devSampleSelect"),
  startDemoButton: document.querySelector("#startDemoButton"),
  nextDemoTurnButton: document.querySelector("#nextDemoTurnButton"),
  autoRunButton: document.querySelector("#autoRunButton"),
  demoRunStatus: document.querySelector("#demoRunStatus"),
  demoScenarioLabel: document.querySelector("#demoScenarioLabel"),
};

const store = {
  sessionId: null,
  turn: 0,
  busy: false,
  developer: { agent_response: null, state: null },
  debugTab: "response",
  mode: "demo",
  dev: {
    sessionId: null,
    committedTurn: 0,
    trace: null,
    currentTrace: null,
    selectedStage: "input",
    busy: false,
    history: [],
    scenario: null,
  },
  demo: {
    scenario: null,
    autoRunning: false,
    previousRecommendationIds: null,
  },
  samples: [],
};

const productIcon = `
  <svg viewBox="0 0 64 64" aria-hidden="true">
    <path d="M15 20h34l-3 33H18L15 20Z"/>
    <path d="M23 20a9 9 0 0 1 18 0"/>
    <path d="M25 34h14M28 41h8"/>
  </svg>`;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return payload;
}


function showError(message) {
  ui.errorToast.textContent = message;
  ui.errorToast.classList.add("show");
  window.clearTimeout(showError.timer);
  showError.timer = window.setTimeout(() => ui.errorToast.classList.remove("show"), 4500);
}


async function loadEvaluatorSamples() {
  try {
    const payload = await api("/api/eval/samples");
    store.samples = payload.samples || [];
    const options = store.samples.map((sample) => `<option value="${escapeHtml(sample.sample_id)}">${escapeHtml(sample.sample_id)} · ${escapeHtml(sample.scenario_type)}</option>`).join("");
    ui.demoSampleSelect.innerHTML = options;
    ui.devSampleSelect.innerHTML = options;
  } catch (error) {
    showError(`Unable to load public samples: ${error.message}`);
  }
}

function setDemoBusy(busy) {
  store.busy = busy;
  ui.startDemoButton.disabled = busy || store.demo.autoRunning;
  ui.demoSampleSelect.disabled = busy || store.demo.autoRunning;
  ui.nextDemoTurnButton.disabled = busy || !store.sessionId || store.demo.scenario?.done || store.demo.autoRunning;
  ui.autoRunButton.disabled = !store.sessionId || store.demo.scenario?.done;
  ui.newSessionButton.disabled = busy || store.demo.autoRunning;
  ui.processingStatus.textContent = busy ? "Evaluator running…" : "";
  ui.pipeline.classList.toggle("processing", busy);
}

function resetDemoPanels() {
  store.turn = 0;
  store.demo.previousRecommendationIds = null;
  store.developer = { agent_response: null, state: null };
  ui.turnCount.textContent = "0";
  ui.conversation.innerHTML = `<div class="welcome" id="welcome"><div class="welcome-icon">▶</div><h3>Evaluator scenario ready</h3><p>Click Next Turn or Auto Run. User messages are generated by the official evaluator flow.</p></div>`;
  ui.recommendations.innerHTML = `<div class="empty-state"><div class="empty-grid" aria-hidden="true"><span></span><span></span><span></span></div><h3>Waiting for Turn 1</h3><p>The real Agent will produce recommendations from the evaluator-generated message.</p></div>`;
  ui.resultCount.textContent = "EVALUATOR READY";
  ui.stateContent.innerHTML = `<div class="state-placeholder"><div class="radar" aria-hidden="true"><span></span></div><p>State will update after the first evaluator turn.</p></div>`;
  ui.developerJson.textContent = "No response yet.";
  ui.pipeline.classList.remove("complete");
}

async function startDemoSession() {
  const sampleId = ui.demoSampleSelect.value;
  if (!sampleId) return;
  store.demo.autoRunning = false;
  ui.autoRunButton.querySelector("span").textContent = "Auto Run";
  setDemoBusy(true);
  try {
    const payload = await api("/api/eval/session", {
      method: "POST",
      body: JSON.stringify({ sample_id: sampleId }),
    });
    store.sessionId = payload.session_id;
    store.demo.scenario = payload.scenario;
    ui.sessionId.textContent = payload.session_id;
    ui.sessionId.title = payload.session_id;
    ui.demoScenarioLabel.textContent = payload.scenario.scenario_type.replaceAll("_", " ");
    ui.demoRunStatus.textContent = `${payload.scenario.sample_id} · ready for Turn 1`;
    resetDemoPanels();
  } catch (error) {
    showError(error.message);
  } finally {
    setDemoBusy(false);
  }
}

async function runDemoTurn() {
  if (!store.sessionId || store.demo.scenario?.done) return false;
  setDemoBusy(true);
  try {
    const payload = await api("/api/eval/next", {
      method: "POST",
      body: JSON.stringify({ session_id: store.sessionId }),
    });
    store.turn = payload.turn;
    store.demo.scenario = payload.scenario;
    store.developer = payload.developer || { agent_response: null, state: payload.state };
    ui.turnCount.textContent = String(payload.turn);
    appendMessage("user", payload.user_message);
    appendMessage("agent", payload.agent.message, payload.agent.ask_attribute);
    renderRecommendations(payload.recommendations || []);
    const recommendationIds = (payload.recommendations || []).map((item) => item.parent_asin);
    const previousIds = store.demo.previousRecommendationIds;
    const rankingNote = previousIds
      ? (recommendationIds.join("|") === previousIds.join("|") ? "ranking unchanged" : "ranking updated")
      : "initial ranking";
    store.demo.previousRecommendationIds = recommendationIds;
    renderState(payload.state || {}, payload.agent.ask_attribute);
    renderDeveloperData();
    ui.pipeline.classList.add("complete");
    if (payload.scenario.done) {
      const outcome = payload.scenario.hit ? `TARGET HIT · RANK ${payload.scenario.hit_rank}` : "MAX TURNS · NO HIT";
      ui.demoRunStatus.textContent = `${payload.scenario.sample_id} · ${outcome} · ${rankingNote}`;
      ui.resultCount.textContent = outcome;
      store.demo.autoRunning = false;
      ui.autoRunButton.querySelector("span").textContent = "Auto Run";
    } else {
      ui.demoRunStatus.textContent = `${payload.scenario.sample_id} · Turn ${payload.turn} complete · ${rankingNote} · next message prepared`;
    }
    return !payload.scenario.done;
  } catch (error) {
    store.demo.autoRunning = false;
    ui.autoRunButton.querySelector("span").textContent = "Auto Run";
    showError(error.message);
    return false;
  } finally {
    setDemoBusy(false);
  }
}

async function toggleAutoRun() {
  if (store.demo.autoRunning) {
    store.demo.autoRunning = false;
    ui.autoRunButton.querySelector("span").textContent = "Auto Run";
    setDemoBusy(false);
    return;
  }
  store.demo.autoRunning = true;
  ui.autoRunButton.querySelector("span").textContent = "Pause";
  setDemoBusy(false);
  while (store.demo.autoRunning && !store.demo.scenario?.done) {
    const continueRunning = await runDemoTurn();
    if (!continueRunning) break;
    await new Promise((resolve) => window.setTimeout(resolve, 850));
  }
  store.demo.autoRunning = false;
  ui.autoRunButton.querySelector("span").textContent = "Auto Run";
  setDemoBusy(false);
}


function appendMessage(role, text, askAttribute = null) {
  document.querySelector("#welcome")?.remove();
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  wrapper.innerHTML = `
    <div class="message-label">${role === "user" ? "YOU" : "SHOPPING COPILOT"}</div>
    <div class="bubble">${escapeHtml(text)}</div>
    ${askAttribute ? `<div class="ask-chip">NEXT ATTRIBUTE · ${escapeHtml(askAttribute)}</div>` : ""}`;
  ui.conversation.appendChild(wrapper);
  ui.conversation.scrollTop = ui.conversation.scrollHeight;
}

function money(price) {
  return typeof price === "number" ? `$${price.toFixed(2)}` : "Price unavailable";
}

function compactDetail(value) {
  if (Array.isArray(value)) return value.filter(Boolean).slice(0, 4).join(" · ");
  if (value && typeof value === "object") {
    return Object.entries(value).slice(0, 6).map(([key, item]) => `${key}: ${item}`).join(" · ");
  }
  return String(value || "");
}

function renderRecommendations(items) {
  ui.resultCount.textContent = `${items.length} REAL CATALOG ${items.length === 1 ? "MATCH" : "MATCHES"}`;
  if (!items.length) {
    ui.recommendations.innerHTML = `
      <div class="empty-state">
        <h3>No matching products yet</h3>
        <p>Answer the agent's next question to refine the search.</p>
      </div>`;
    return;
  }

  ui.recommendations.innerHTML = items.map((entry) => {
    if (!entry.product) {
      return `<article class="product-card missing-card">
        <div class="product-visual"><span class="rank-badge">#${entry.rank}</span>${productIcon}</div>
        <div class="product-body"><div class="product-kicker">Catalog reference</div><h3 class="product-title">Product details unavailable</h3><div class="product-meta">ASIN: ${escapeHtml(entry.parent_asin)}</div></div>
      </article>`;
    }
    const product = entry.product;
    const features = (product.features || []).filter(Boolean).slice(0, 3);
    const category = (product.categories || []).filter(Boolean).slice(-2).join(" / ") || "Uncategorized";
    const description = compactDetail(product.description);
    const details = compactDetail(product.details);
    return `<article class="product-card">
      <div class="product-visual"><span class="rank-badge">#${entry.rank}</span>${productIcon}</div>
      <div class="product-body">
        <div class="product-kicker"><span>${escapeHtml(category)}</span><span class="price">${money(product.price)}</span></div>
        <h3 class="product-title">${escapeHtml(product.title || "Untitled catalog product")}</h3>
        <div class="product-meta">
          <span>${escapeHtml(product.store || "Store unavailable")}</span>
          ${typeof product.average_rating === "number" ? `<span class="rating">★ ${product.average_rating.toFixed(1)}</span>` : ""}
          ${typeof product.rating_number === "number" ? `<span>${product.rating_number.toLocaleString()} ratings</span>` : ""}
          <span>ASIN: ${escapeHtml(entry.parent_asin)}</span>
        </div>
        <div class="feature-tags">${features.map((feature) => `<span class="feature-tag">${escapeHtml(feature)}</span>`).join("")}</div>
      </div>
      ${(description || details || features.length) ? `<details class="product-details"><summary>View product details</summary><div class="detail-copy">
        ${features.length ? `<p><strong>Features:</strong> ${escapeHtml(features.join(" · "))}</p>` : ""}
        ${description ? `<p><strong>Description:</strong> ${escapeHtml(description)}</p>` : ""}
        ${details ? `<p><strong>Details:</strong> ${escapeHtml(details)}</p>` : ""}
      </div></details>` : ""}
    </article>`;
  }).join("");
}

function attributeText(value) {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value !== "object") return String(value);
  const parts = [];
  if (Array.isArray(value.values) && value.values.length) parts.push(value.values.join(", "));
  if (value.min !== null && value.min !== undefined) parts.push(`min ${value.min}`);
  if (value.max !== null && value.max !== undefined) parts.push(`max ${value.max}`);
  if (!parts.length) {
    Object.entries(value).forEach(([key, item]) => {
      if (item !== null && item !== undefined && item !== "" && !(Array.isArray(item) && !item.length)) {
        parts.push(`${key}: ${Array.isArray(item) ? item.join(", ") : item}`);
      }
    });
  }
  return parts.join(" · ");
}

function constraintMarkup(title, values, variant = "") {
  if (!values || typeof values !== "object" || !Object.keys(values).length) return "";
  return `<section class="state-section">
    <div class="state-section-title"><span>${escapeHtml(title)}</span><span>${Object.keys(values).length}</span></div>
    ${Object.entries(values).map(([name, value]) => `<div class="constraint ${variant}"><div class="constraint-name">${escapeHtml(name.replaceAll("_", " "))}</div><div class="constraint-value">${escapeHtml(attributeText(value))}</div></div>`).join("")}
  </section>`;
}

function chipSection(title, values) {
  if (!Array.isArray(values) || !values.length) return "";
  return `<section class="state-section"><div class="state-section-title"><span>${escapeHtml(title)}</span><span>${values.length}</span></div><div class="mini-chips">${values.map((value) => `<span class="mini-chip">${escapeHtml(value)}</span>`).join("")}</div></section>`;
}

function renderState(state, askAttribute) {
  const confidence = Math.max(0, Math.min(1, Number(state.intent_confidence) || 0));
  ui.stateContent.innerHTML = `
    <section class="intent-card">
      <span class="state-label">Current intent · Turn ${escapeHtml(state.turn ?? 0)}</span>
      <div class="intent-row"><span class="intent-name">${escapeHtml(state.intent || "Unknown")}</span><span class="confidence">${Math.round(confidence * 100)}% confidence</span></div>
      <div class="confidence-track"><span style="width:${confidence * 100}%"></span></div>
    </section>
    ${askAttribute ? `<section class="state-section"><div class="state-section-title">Next question</div><div class="question-card"><span class="state-label">Ask attribute</span><strong>${escapeHtml(askAttribute)}</strong></div></section>` : ""}
    ${constraintMarkup("Hard constraints", state.hard_constraint)}
    ${constraintMarkup("Soft preferences", state.soft_constraint, "soft")}
    ${constraintMarkup("Rejected values", state.rejected_values, "rejected")}
    ${chipSection("No preference", state.no_prefernce)}
    ${chipSection("Asked attributes", state.asked_attributes)}
    ${state.override_detected ? `<section class="state-section"><div class="override-alert">↻ Intent override detected on this turn</div></section>` : ""}`;
}

function renderDeveloperData() {
  const payload = store.debugTab === "response" ? store.developer.agent_response : store.developer.state;
  ui.developerJson.textContent = payload ? JSON.stringify(payload, null, 2) : "No response yet.";
}

function switchMode(mode) {
  store.mode = mode;
  const developer = mode === "developer";
  document.querySelectorAll(".demo-surface").forEach((element) => { element.hidden = developer; });
  ui.developerMode.hidden = !developer;
  ui.demoModeButton.classList.toggle("active", !developer);
  ui.developerModeButton.classList.toggle("active", developer);
  ui.newSessionButton.hidden = developer;
  document.querySelector(".pipeline").style.visibility = developer ? "hidden" : "visible";
  if (developer && !store.dev.sessionId) newDeveloperSession();
}

function setDevBusy(busy, message = "") {
  store.dev.busy = busy;
  ui.newDevSessionButton.disabled = busy;
  ui.startDevTurnButton.disabled = busy || store.dev.scenario?.done || Boolean(store.dev.trace && !store.dev.trace.committed);
  ui.restartDevTurnButton.disabled = busy || !store.dev.trace || store.dev.trace.committed;
  const responseComplete = stageByName("response")?.status === "completed";
  const hasPending = store.dev.trace && !responseComplete;
  ui.runNextButton.disabled = busy || !hasPending;
  ui.runAllButton.disabled = busy || !hasPending;
  ui.commitTurnButton.disabled = busy || !responseComplete || store.dev.trace?.committed;
  ui.devActionStatus.textContent = message || (busy ? "Executing real pipeline component…" : devStatusMessage());
}

function stageByName(name) {
  return store.dev.trace?.stages?.find((stage) => stage.name === name) || null;
}

function nextPendingStage() {
  return store.dev.trace?.stages?.find((stage) => stage.name !== "input" && stage.status !== "completed") || null;
}

function devStatusMessage() {
  if (!store.dev.trace) return "Start a turn, then advance one real pipeline stage at a time.";
  if (store.dev.scenario?.done) {
    const reason = store.dev.scenario.stop_reason === "target_hit" ? `target hit at rank ${store.dev.scenario.hit_rank}` : "10-turn limit reached";
    return `Official evaluator scenario complete: ${reason}. Start another public sample to continue.`;
  }
  if (store.dev.trace.committed) return "Turn committed. The next evaluator turn will load automatically.";
  const next = nextPendingStage();
  return next ? `Next step: ${next.label}` : "Final response ready. Commit this turn to continue the conversation.";
}

async function newDeveloperSession() {
  setDevBusy(true, "Creating isolated developer session…");
  try {
    const sampleId = ui.devSampleSelect.value;
    if (!sampleId) throw new Error("Select a public sample first.");
    const payload = await api("/api/dev/scenario", { method: "POST", body: JSON.stringify({ sample_id: sampleId }) });
    store.dev.sessionId = payload.session_id;
    store.dev.committedTurn = payload.turn;
    store.dev.trace = payload.active_trace;
    store.dev.currentTrace = payload.active_trace;
    store.dev.history = payload.history || [];
    store.dev.scenario = payload.scenario;
    store.dev.selectedStage = "input";
    ui.devSessionId.textContent = payload.session_id;
    ui.devSessionId.title = payload.session_id;
    ui.devCommittedTurn.textContent = String(payload.turn);
    ui.devMessageInput.value = payload.scenario.next_user_message || "Scenario complete";
    renderDevHistory();
    renderDeveloperTrace();
  } catch (error) {
    showError(error.message || "Unable to create developer session.");
  } finally {
    setDevBusy(false);
  }
}

async function startDeveloperTurn() {
  const message = ui.devMessageInput.value.trim();
  if (!message || store.dev.scenario?.done) {
    showError("No evaluator turn is ready.");
    return;
  }
  setDevBusy(true, "Storing developer input without running the pipeline…");
  try {
    store.dev.trace = await api("/api/dev/turn", {
      method: "POST",
      body: JSON.stringify({ session_id: store.dev.sessionId, message, top_k: 10, retrieval_k: 100 }),
    });
    store.dev.currentTrace = store.dev.trace;
    store.dev.selectedStage = "input";
    renderDeveloperTrace();
  } catch (error) {
    showError(error.message);
  } finally {
    setDevBusy(false);
  }
}

async function runDeveloperEndpoint(path, body = {}) {
  setDevBusy(true, "Executing real pipeline component…");
  try {
    const payload = await api(path, {
      method: "POST",
      body: JSON.stringify({ session_id: store.dev.sessionId, ...body }),
    });
    store.dev.trace = payload;
    store.dev.currentTrace = payload;
    if (payload.selected_stage) store.dev.selectedStage = payload.selected_stage;
    renderDeveloperTrace();
    if (payload.operation_error) showError(payload.operation_error);
  } catch (error) {
    showError(error.message);
  } finally {
    setDevBusy(false);
  }
}

async function commitDeveloperTurn() {
  setDevBusy(true, "Committing trace state and loading the next evaluator turn…");
  try {
    const payload = await api("/api/dev/commit", {
      method: "POST",
      body: JSON.stringify({ session_id: store.dev.sessionId }),
    });
    store.dev.committedTurn = payload.turn;
    store.dev.trace = payload.active_trace;
    store.dev.currentTrace = payload.active_trace;
    store.dev.history = payload.history || [];
    if (payload.scenario) store.dev.scenario = payload.scenario;
    ui.devCommittedTurn.textContent = String(payload.turn);
    ui.devMessageInput.value = payload.scenario?.next_user_message || "Evaluator scenario complete";
    renderDevHistory();
    store.dev.trace = payload.active_trace;
    store.dev.currentTrace = payload.active_trace;
    store.dev.selectedStage = payload.scenario?.done ? "response" : "input";
    renderDeveloperTrace();
  } catch (error) {
    showError(error.message);
  } finally {
    setDevBusy(false);
  }
}

function renderDevHistory() {
  ui.devHistory.innerHTML = `<option value="">Current turn</option>${store.dev.history.map((item) => `<option value="${item.turn}">Turn ${item.turn} · ${escapeHtml(item.message).slice(0, 35)}</option>`).join("")}`;
}

async function loadHistoryTrace(turn) {
  if (!turn) {
    ui.devHistory.value = "";
    store.dev.trace = store.dev.currentTrace;
    renderDeveloperTrace();
    return;
  }
  setDevBusy(true, `Loading stored trace for Turn ${turn}…`);
  try {
    store.dev.trace = await api(`/api/dev/trace/${encodeURIComponent(store.dev.sessionId)}/${turn}`);
    store.dev.selectedStage = "response";
    renderDeveloperTrace();
  } catch (error) {
    showError(error.message);
  } finally {
    setDevBusy(false);
  }
}

function renderDeveloperTrace() {
  ui.devActiveTurn.textContent = store.dev.trace?.turn ?? "—";
  renderDevScenarioOutcome();
  renderDevStages();
  renderDevParameters();
  renderDevTiming();
  renderStageInspector(store.dev.selectedStage);
  setDevBusy(store.dev.busy);
}

function renderDevScenarioOutcome() {
  const scenario = store.dev.scenario;
  const preview = store.dev.trace?.evaluation_preview;
  let text = "AWAITING RESULT";
  let style = "pending";
  if (preview?.hit) {
    text = `TARGET HIT · RANK ${preview.hit_rank}`;
    style = "hit";
  } else if (preview?.status === "not_hit" && !scenario?.done) {
    text = `TURN ${store.dev.trace?.turn || scenario?.turn || 0} · NOT HIT`;
    style = "continue";
  } else if (scenario?.done && scenario.hit) {
    text = `TARGET HIT · RANK ${scenario.hit_rank}`;
    style = "hit";
  } else if (scenario?.done) {
    text = "MAX TURNS · NO HIT";
    style = "miss";
  } else if ((scenario?.turn || 0) > 0) {
    text = `TURN ${scenario.turn} · NOT HIT YET`;
    style = "continue";
  }
  ui.devScenarioOutcome.textContent = text;
  ui.devScenarioOutcome.className = `dev-outcome ${style}`;
}

function renderDevStages() {
  const trace = store.dev.trace;
  ui.devStageList.querySelectorAll(".dev-stage").forEach((button) => {
    const stage = trace?.stages?.find((item) => item.name === button.dataset.stage);
    const status = stage?.status || "not_run";
    button.className = `dev-stage ${status.replace("_", "-")} ${store.dev.selectedStage === button.dataset.stage ? "active" : ""}`;
    button.querySelector("small").textContent = status === "completed" && stage.duration_ms !== null ? `${stage.duration_ms.toFixed(3)} ms` : status.replace("_", " ");
    button.querySelector("b").textContent = status === "completed" ? "✓" : status === "error" ? "!" : status === "running" ? "◌" : "○";
  });
}

function renderDevParameters() {
  const parameters = store.dev.trace?.parameters;
  if (!parameters) {
    ui.devParameters.innerHTML = `<div><span>Session ID</span><code>${escapeHtml(store.dev.sessionId || "Not started")}</code></div><div><span>Turn</span><strong>—</strong></div><div><span>top_k</span><strong>10</strong></div><div><span>Retrieval k</span><strong>100</strong></div>`;
    return;
  }
  ui.devParameters.innerHTML = Object.entries(parameters).map(([key, value]) => `<div><span>${escapeHtml(key.replaceAll("_", " "))}</span>${typeof value === "number" ? `<strong>${value}</strong>` : `<code title="${escapeHtml(value)}">${escapeHtml(value)}</code>`}</div>`).join("");
}

function renderDevTiming() {
  const completed = (store.dev.trace?.stages || []).filter((stage) => stage.duration_ms !== null && stage.duration_ms !== undefined && stage.name !== "input");
  ui.devTiming.innerHTML = completed.length ? completed.map((stage) => `<div class="timing-row"><span>${escapeHtml(stage.label)}</span><strong>${stage.duration_ms.toFixed(3)} ms</strong></div>`).join("") : `<p>Run stages to collect timing.</p>`;
}

function jsonBlock(title, value, open = false) {
  return `<details class="raw-json" ${open ? "open" : ""}><summary>${escapeHtml(title)}</summary><pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre></details>`;
}

function flattenDiff(before, after, path = "") {
  const changes = [];
  const keys = new Set([...Object.keys(before || {}), ...Object.keys(after || {})]);
  for (const key of keys) {
    const nextPath = path ? `${path}.${key}` : key;
    const left = before?.[key];
    const right = after?.[key];
    if (left && right && typeof left === "object" && typeof right === "object" && !Array.isArray(left) && !Array.isArray(right)) {
      changes.push(...flattenDiff(left, right, nextPath));
    } else if (JSON.stringify(left) !== JSON.stringify(right)) {
      const marker = left === undefined ? "+" : right === undefined ? "−" : "~";
      changes.push(`${marker} ${nextPath}: ${JSON.stringify(left)} → ${JSON.stringify(right)}`);
    }
  }
  return changes;
}

function inspectInput(stage) {
  const input = stage.input || {};
  return `<div class="inspect-summary"><div class="metric-box"><span>Session</span><strong>${escapeHtml(input.session_id).slice(0, 18)}…</strong></div><div class="metric-box"><span>Turn</span><strong>${input.turn}</strong></div><div class="metric-box"><span>Previous asked</span><strong>${escapeHtml(input.previous_asked_attribute || "None")}</strong></div></div>${jsonBlock("View Raw Input JSON", input, true)}`;
}

function inspectState(stage) {
  const before = stage.output?.state_before || stage.input?.state_before || {};
  const after = stage.output?.state_after || {};
  const changes = flattenDiff(before, after);
  return `<section class="inspect-section"><h3>State diff</h3><div class="change-list">${changes.length ? changes.map(escapeHtml).join("<br>") : "No structured changes detected."}</div></section><section class="inspect-section"><h3>Before / After</h3><div class="state-diff-grid"><div class="diff-panel"><span>STATE BEFORE</span><pre class="inspect-code">${escapeHtml(JSON.stringify(before, null, 2))}</pre></div><div class="diff-panel"><span>STATE AFTER</span><pre class="inspect-code">${escapeHtml(JSON.stringify(after, null, 2))}</pre></div></div></section>${jsonBlock("View Raw Stage JSON", stage)}`;
}

function inspectQuery(stage) {
  const output = stage.output || {};
  return `<div class="inspect-summary"><div class="metric-box"><span>Generated query</span><strong>${escapeHtml(output.query || "Empty")}</strong></div><div class="metric-box"><span>Source</span><strong>${escapeHtml(output.source || "—")}</strong></div><div class="metric-box"><span>Fallback used</span><strong>${output.fallback_used ? "Yes" : "No"}</strong></div></div>${jsonBlock("Input State", stage.input?.state || {})}${jsonBlock("Raw Query Output", output, true)}`;
}

function numberOrDash(value, digits = 4) {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

function candidateTable(stage, reranked = false) {
  const candidates = Array.isArray(stage.output) ? stage.output : [];
  const scores = candidates.map((item) => item.retrieval_score).filter((value) => typeof value === "number");
  const withPrice = candidates.filter((item) => typeof item.product?.price === "number").length;
  const summary = `<div class="inspect-summary"><div class="metric-box"><span>Candidates returned</span><strong>${candidates.length}</strong></div><div class="metric-box"><span>Top retrieval score</span><strong>${scores.length ? numberOrDash(Math.max(...scores)) : "—"}</strong></div><div class="metric-box"><span>Lowest retrieval score</span><strong>${scores.length ? numberOrDash(Math.min(...scores)) : "—"}</strong></div><div class="metric-box"><span>Products with price</span><strong>${withPrice} / ${candidates.length}</strong></div></div>`;
  const controls = `<div class="candidate-controls"><input type="search" data-candidate-filter placeholder="Search ASIN or title"><select data-candidate-limit>${[10,25,50,100].filter((n) => n <= Math.max(candidates.length, 10)).map((n) => `<option value="${n}">Top ${n}</option>`).join("")}</select>${reranked ? `<select data-rank-filter><option value="all">All candidates</option><option value="matched">Only matched</option><option value="violations">Only violations</option></select>` : ""}</div>`;
  const rows = candidates.map((item) => {
    const change = reranked && item.retrieval_rank && item.rerank_rank ? item.retrieval_rank - item.rerank_rank : null;
    const changeText = change === null ? "—" : change > 0 ? `↑${change}` : change < 0 ? `↓${Math.abs(change)}` : "—";
    return `<tr data-candidate-row data-search="${escapeHtml(`${item.parent_asin} ${item.product?.title || ""}`.toLowerCase())}" data-matched="${(item.matched || []).length}" data-violations="${(item.violation || []).length}"><td>${reranked ? item.rerank_rank : item.retrieval_rank}</td>${reranked ? `<td>${item.retrieval_rank ?? "—"}</td><td class="${change > 0 ? "rank-up" : change < 0 ? "rank-down" : ""}">${changeText}</td>` : ""}<td>${escapeHtml(item.parent_asin)}</td><td title="${escapeHtml(item.product?.title || "")}">${escapeHtml(item.product?.title || "Untitled")}</td><td>${item.product?.price == null ? "—" : `$${item.product.price}`}</td><td>${escapeHtml(item.product?.store || "—")}</td><td>${numberOrDash(item.retrieval_score)}</td>${reranked ? `<td>${numberOrDash(item.rerank_score)}</td><td>${escapeHtml((item.matched || []).join(", ") || "—")}</td><td>${escapeHtml((item.violation || []).join(", ") || "—")}</td>` : ""}</tr>`;
  }).join("");
  const headers = reranked ? "<th>Rerank</th><th>Retrieval</th><th>Change</th><th>ASIN</th><th>Title</th><th>Price</th><th>Store</th><th>Retrieval score</th><th>Rerank score</th><th>Matched</th><th>Violations</th>" : "<th>Rank</th><th>ASIN</th><th>Title</th><th>Price</th><th>Store</th><th>BM25 / Retrieval score</th>";
  return `${summary}${controls}<div class="candidate-table-wrap"><table class="candidate-table"><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table></div><div class="candidate-detail" data-candidate-detail><p>Click a candidate to inspect all returned diagnostics and catalog fields.</p></div>${jsonBlock(`View Raw ${reranked ? "Candidates10" : "Candidates100"} JSON`, candidates)}`;
}

function bindCandidateTable(stage) {
  const rows = [...ui.devInspector.querySelectorAll("[data-candidate-row]")];
  const input = ui.devInspector.querySelector("[data-candidate-filter]");
  const limit = ui.devInspector.querySelector("[data-candidate-limit]");
  const rankFilter = ui.devInspector.querySelector("[data-rank-filter]");
  const apply = () => {
    const query = (input?.value || "").toLowerCase();
    const max = Number(limit?.value || 100);
    const filter = rankFilter?.value || "all";
    let visible = 0;
    rows.forEach((row) => {
      const matchesText = row.dataset.search.includes(query);
      const matchesType = filter === "all" || (filter === "matched" && Number(row.dataset.matched)) || (filter === "violations" && Number(row.dataset.violations));
      const show = matchesText && matchesType && visible < max;
      row.hidden = !show;
      if (show) visible += 1;
    });
  };
  input?.addEventListener("input", apply);
  limit?.addEventListener("change", apply);
  rankFilter?.addEventListener("change", apply);
  rows.forEach((row, index) => row.addEventListener("click", () => {
    const item = stage.output[index];
    const detail = ui.devInspector.querySelector("[data-candidate-detail]");
    detail.innerHTML = `<h4>${escapeHtml(item.product?.title || item.parent_asin)}</h4><p><strong>ASIN:</strong> ${escapeHtml(item.parent_asin)} · <strong>Store:</strong> ${escapeHtml(item.product?.store || "—")} · <strong>Price:</strong> ${item.product?.price ?? "—"}</p>${jsonBlock("Candidate diagnostics and product data", item, true)}`;
  }));
  apply();
}

function inspectDialogue(stage) {
  const output = stage.output || {};
  return `<div class="dialogue-decision"><span>ask_attribute</span><strong>${escapeHtml(output.ask_attribute || "None")}</strong><span>message</span><p>${escapeHtml(output.message || "")}</p></div>${jsonBlock("Current Shopping State", stage.input?.state || {})}${jsonBlock("Raw Dialogue Decision", output, true)}`;
}

function inspectResponse(stage) {
  const output = stage.output || {};
  return `<div class="dialogue-decision"><span>Official AgentResponse</span><strong>${escapeHtml(output.ask_attribute || "No follow-up attribute")}</strong><p>${escapeHtml(output.message || "")}</p></div><div class="inspect-summary" style="margin-top:12px"><div class="metric-box"><span>Recommendations</span><strong>${output.recommendations?.length || 0}</strong></div><div class="metric-box"><span>Prompt tokens</span><strong>${output.usage?.prompt_tokens ?? 0}</strong></div><div class="metric-box"><span>Completion tokens</span><strong>${output.usage?.completion_tokens ?? 0}</strong></div></div>${jsonBlock("Official JSON", output, true)}`;
}

function renderStageInspector(name) {
  const stage = stageByName(name);
  store.dev.selectedStage = name;
  renderDevStages();
  if (!stage) {
    ui.inspectorStageName.textContent = name.toUpperCase();
    ui.inspectorImplementation.textContent = "Start a developer turn first";
    ui.inspectorStatus.textContent = "NOT RUN";
    ui.inspectorStatus.className = "stage-status not-run";
    ui.devInspector.innerHTML = `<div class="dev-empty"><span>{ }</span><h3>No stored stage data</h3><p>Start a turn and execute stages in forward order.</p></div>`;
    return;
  }
  ui.inspectorStageName.textContent = stage.label;
  ui.inspectorImplementation.textContent = stage.implementation;
  ui.inspectorStatus.textContent = stage.status.replace("_", " ").toUpperCase();
  ui.inspectorStatus.className = `stage-status ${stage.status.replace("_", "-")}`;
  if (stage.status === "not_run") {
    const next = nextPendingStage();
    ui.devInspector.innerHTML = `<div class="dev-empty"><span>○</span><h3>${next?.name === stage.name ? "Ready to execute" : "Waiting for previous stage"}</h3><p>${next?.name === stage.name ? "Click this stage or Run Next Step to call the real implementation." : "Stages can only execute in valid forward order."}</p></div>`;
    return;
  }
  if (stage.status === "error") {
    ui.devInspector.innerHTML = `<div class="error-diagnostic"><strong>${escapeHtml(stage.error?.type || "Stage error")}</strong><p>${escapeHtml(stage.error?.message || "Unknown error")}</p></div>${jsonBlock("Technical Details", stage.error?.technical_details || "")}`;
    return;
  }
  if (stage.status !== "completed") {
    ui.devInspector.innerHTML = `<div class="dev-empty"><span>◌</span><h3>Stage is running</h3><p>Waiting for the existing component to return.</p></div>`;
    return;
  }
  const renderers = { input: inspectInput, state: inspectState, query: inspectQuery, retrieval: (value) => candidateTable(value, false), reranking: (value) => candidateTable(value, true), dialogue: inspectDialogue, response: inspectResponse };
  ui.devInspector.innerHTML = renderers[name](stage);
  if (name === "retrieval" || name === "reranking") bindCandidateTable(stage);
}


ui.newSessionButton.addEventListener("click", startDemoSession);
ui.startDemoButton.addEventListener("click", startDemoSession);
ui.nextDemoTurnButton.addEventListener("click", runDemoTurn);
ui.autoRunButton.addEventListener("click", toggleAutoRun);

document.querySelectorAll("[data-debug-tab]").forEach((button) => {
  button.addEventListener("click", () => {
    store.debugTab = button.dataset.debugTab;
    document.querySelectorAll("[data-debug-tab]").forEach((tab) => tab.classList.toggle("active", tab === button));
    renderDeveloperData();
  });
});

ui.demoModeButton.addEventListener("click", () => switchMode("demo"));
ui.developerModeButton.addEventListener("click", () => switchMode("developer"));
ui.newDevSessionButton.addEventListener("click", newDeveloperSession);
ui.startDevTurnButton.addEventListener("click", startDeveloperTurn);
ui.restartDevTurnButton.addEventListener("click", () => runDeveloperEndpoint("/api/dev/restart"));
ui.runNextButton.addEventListener("click", () => runDeveloperEndpoint("/api/dev/next"));
ui.runAllButton.addEventListener("click", () => runDeveloperEndpoint("/api/dev/all"));
ui.commitTurnButton.addEventListener("click", commitDeveloperTurn);
ui.devHistory.addEventListener("change", () => loadHistoryTrace(ui.devHistory.value));
ui.devStageList.querySelectorAll(".dev-stage").forEach((button) => {
  button.addEventListener("click", () => {
    const name = button.dataset.stage;
    const stage = stageByName(name);
    if (stage?.status === "completed" || stage?.status === "error" || name === "input") {
      renderStageInspector(name);
      return;
    }
    const next = nextPendingStage();
    if (next?.name === name) runDeveloperEndpoint("/api/dev/stage", { stage: name });
    else renderStageInspector(name);
  });
});

async function initialize() {
  await loadEvaluatorSamples();
  await startDemoSession();
  if (new URLSearchParams(window.location.search).get("mode") === "developer") {
    switchMode("developer");
  }
}

initialize();
